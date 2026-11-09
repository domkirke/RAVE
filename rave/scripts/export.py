import logging
import os
import sys

logging.basicConfig(level=logging.INFO)
logging.info("library loading")
logging.info("DEBUG")
import torch


torch.set_grad_enabled(False)

import cached_conv as cc
import gin
import torch.nn as nn
from absl import flags, app

try:
    import rave
except:
    import sys, os 
    sys.path.append(os.path.abspath('.'))
    import rave
import rave.blocks
import rave.core
from rave.stft import CausalSTFT, StreamableSTFT
from rave.cachedConv2D import CachedPadding2d as CachedPadding2d_backward
from cached_conv import CachedConvTranspose2d, CachedPadding
from rave.prior import model as prior
from rave.scripting import *

FLAGS = flags.FLAGS

flags.DEFINE_string('run',
                    default=None,
                    help='Path to the run to export',
                    required=True)
flags.DEFINE_bool('streaming',
                  default=False,
                  help='Enable the model streaming mode')
flags.DEFINE_bool('use_causal_stft',
                  default=True,
                  help='Replace StreamableSTFT with CausalSTFT during streaming export')
flags.DEFINE_float(
    'fidelity',
    default=.95,
    lower_bound=.1,
    upper_bound=.999,
    help='Fidelity to use during inference (Variational mode only)')

flags.DEFINE_integer(
    'latent_size',
    default=None,
    lower_bound = 0,
    help = "Defines a fixed latent space for RAVE (overrides --fidelity)"
)

flags.DEFINE_bool(
    'no_pca', 
    default = False,
    help = "disables PCA in latent space (latent size is forced to 128)"
)

flags.DEFINE_string('name', 
                     default= None,
                     help = "custom name for the scripted model (default: run name)")
flags.DEFINE_string('out_path', 
                     default= None,
                     help = "output location of scripted model")
flags.DEFINE_bool('ema_weights',
                  default=False,
                  help='Use ema weights if avaiable')
flags.DEFINE_integer('sr',
                     default=None,
                     help='Optional resampling sample rate')
flags.DEFINE_string('prior', 
                    default=None,
                    help = "path to prior (optional)")



def get_state_dict(RUN, PRIOR):
    state_dict = torch.load(PRIOR, map_location="cpu")['state_dict']
    for k, v in RUN.state_dict().items():
        state_dict[f'synth.{k}'] = v
    return state_dict



def main(argv):
    cc.use_cached_conv(FLAGS.streaming)

    logging.info("building rave")

    # config_file = rave.core.search_for_config(FLAGS.run)
    # if config_file is None:
    #     print('Config file not found in %s'%FLAGS.run)
    # gin.parse_config_file(config_file)
    # FLAGS.run = rave.core.search_for_run(FLAGS.run)

    try:
        pretrained, model_path = rave.load_rave_checkpoint(FLAGS.run, ema=FLAGS.ema_weights)
    except: 
        pretrained, model_path = rave.load_rave_checkpoint(FLAGS.run, ema=FLAGS.ema_weights, name=None)
    pretrained.eval()
    output = FLAGS.out_path or os.path.dirname(FLAGS.run)
    model_name = FLAGS.name or rave.get_run_name(model_path)
    
    class_kwargs = {}
    if isinstance(pretrained.encoder, rave.blocks.VariationalEncoder):
        script_class = VariationalScriptedRAVE
        if FLAGS.no_pca:
            class_kwargs['use_pca'] = False
            FLAGS.fidelity = None
            FLAGS.latent_size = None
            model_name += "_nopca"
    elif isinstance(pretrained.encoder, rave.blocks.DiscreteEncoder):
        script_class = DiscreteScriptedRAVE
    elif isinstance(pretrained.encoder, rave.blocks.WasserteinEncoder):
        script_class = WasserteinScriptedRAVE
    elif isinstance(pretrained.encoder, rave.blocks.SphericalEncoder):
        script_class = SphericalScriptedRAVE
    else:
        raise ValueError(f"Encoder type {type(pretrained.encoder)} "
                         "not supported for export.")

    if isinstance(pretrained.spectrogram, CausalSTFT) and FLAGS.streaming:
        pretrained.spectrogram.stream = True
        
    if isinstance(pretrained.spectrogram, StreamableSTFT) and FLAGS.streaming:
        stft = pretrained.spectrogram
        stft_kwargs = dict(
            nfft=stft.nfft,
            hop_size=stft.hop_size,
            stream=True,
            skip_features=stft.skip_features,
            normalize=stft.normalize,
            alpha_rescale=stft.alpha_rescale,
            beta_rescale=stft.beta_rescale,
        )
        if FLAGS.use_causal_stft:
            pretrained.spectrogram = CausalSTFT(**stft_kwargs)
        else:
            pretrained.spectrogram = StreamableSTFT(
                **stft_kwargs,
                log1p=stft.log1p,
            )
    
    logging.info("warmup pass")


    logging.info("optimize model")


    # parse prior
    prior_scripted=None
    prior_expected = False
    if FLAGS.prior is not None:
        prior_expected = True
        logging.info("loading prior from checkpoint")
        prior_config_file = rave.core.search_for_config(FLAGS.prior)
        if prior_config_file is None:
            print('Config file for prior not found in %s'%FLAGS.prior)
        else:
            gin.clear_config()
            logging.info("prior config file : ", prior_config_file)
            with rave.core.GinEnv():
                gin.constant('SAMPLE_RATE', pretrained.sr)
                gin.parse_config_file(prior_config_file)
                PRIOR = rave.core.search_for_run(FLAGS.prior)
                logging.info(f"using prior model at {PRIOR}")
                prior_class = get_prior_class_from_config()
                prior_pretrained = getattr(prior, prior_class)(pretrained_vae=pretrained, n_channels=pretrained.n_channels, latent_size=FLAGS.latent_size, fidelity=FLAGS.fidelity)
                prior_state_dict = get_state_dict(prior_pretrained, PRIOR)
                try:
                    prior_pretrained.load_state_dict(prior_state_dict, strict=False)
                    prior_scripted = TraceModel(prior_pretrained, pretrained)
                except RuntimeError:
                    logging.error("""could not load prior from %s with fidelity=%f and latent_size=%s.\n"""%(PRIOR, FLAGS.fidelity, FLAGS.latent_size) \
                                  + """did you put for export the same fidelity / latent parameters as for training?""")

    for m in pretrained.modules():
        if hasattr(m, "weight_g"):
            nn.utils.remove_weight_norm(m)


    logging.info("script model")
    scripted_rave = script_class(
        pretrained=pretrained,
        channels = pretrained.n_channels,
        fidelity=FLAGS.fidelity,
        target_sr=FLAGS.sr,
        prior = prior_scripted,
        latent_size=FLAGS.latent_size,
        **class_kwargs
    )

    x = torch.zeros(1, pretrained.n_channels, 2**14)
    scripted_rave.init_cache(x)
    z = scripted_rave.encode(x)
    x = scripted_rave.decode(z)

    logging.info("save model")

    if FLAGS.streaming:
        model_name += "_streaming"
    model_name += ".ts"
    output = os.path.abspath(output)
    if not os.path.isdir(output):
        os.makedirs(output)
    if FLAGS.streaming:
        # Warmup and nn~ checks populate caches. Do not serialize that state.
        for module in scripted_rave.modules():
            if isinstance(module, (CachedPadding2d_backward, CachedPadding)):
                module.pad = module.pad[:0].clone()
            elif isinstance(module, (CachedConvTranspose2d, CachedPadding2d_backward)):
                module.cache = module.cache[:0].clone()
            elif isinstance(module, CausalSTFT):
                module.audio_buffer = module.audio_buffer[:0].clone()
                module.ola_buffer = module.ola_buffer[:0].clone()
                module.envelope_buffer = module.envelope_buffer[:0].clone()
            elif isinstance(module, StreamableSTFT):
                module.audio_buffer = module.audio_buffer[:0].clone()
                module.out_buffer = module.out_buffer[:0].clone()
                module.spec_buffer = module.spec_buffer[:0].clone()
    scripted_rave.export_to_ts(os.path.join(output, model_name))
    try:
        if pretrained.n_channels <= 2:
            # test stereo mode for VST export
            scripted_rave.set_stereo_mode(True)
            z_vst_input = torch.zeros(2, scripted_rave.full_latent_size, z.shape[-1])
            out = scripted_rave.decode(z_vst_input)
            assert out.shape[1] == 2, "model output is not stereo"
            logging.info(f"this model seems compatible with the RAVE vst.")
    except Exception as e:
        logging.warning(f"this model will not work with the RAVE VST. \n Caught error : %s"%e)

    if (prior_expected) and (not prior_scripted):
        logging.info(
            f"model exported to {os.path.join(output, model_name)}, but without prior (see error above)")
    else:
        logging.info(
            f"all good ! model exported to {os.path.join(output, model_name)}")


if __name__ == "__main__":
    app.run(main)
