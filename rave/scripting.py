import os
import torch
import torch.nn.functional as F
import math
import re
import cached_conv as cc
import gin
from types import MethodType
import torch.fx
import torch.nn as nn
from .model import RAVE
from typing import Optional, Tuple
from .prior import Prior
from .resampler import Resampler
import nn_tilde
import random
from . import blocks
from .stft import CausalSTFT, StreamableSTFT
from .cachedConv2D import CachedPadding2d
from .prior import model as prior_models

from .projections import *


def get_random_hash(n=8):
    return "".join([chr(random.randrange(97,122)) for i in range(n)])


projection_call_pattern = "\tif {{ITERATION}} == projection_idx: return self.projection_list[{{ITERATION}}].forward(z, latent_size = latent_size)\n"

inv_projection_call_pattern = "\tif {{ITERATION}} == projection_idx: return self.projection_list[{{ITERATION}}].inverse(z, latent_size = latent_size, temperature=temperature)\n"

project_pattern = """
import torch

@torch.jit.export
def project(self, z, projection_idx: int, latent_size: int | None = None):
{{PROJECT_CALL:LOOP}}
\traise ValueError("projection not valid: %s"%projection_idx)

@torch.jit.export
def project_inverse(self, z, projection_idx: int, latent_size: int | None = None, temperature: float | None = None):
{{INVERSE_CALL:LOOP}}
\traise ValueError("projection not valid: %s"%projection_idx)
"""


def _replace_placeholders(code, **kwargs):
    for k, v in kwargs.items():
        pattern = re.compile(r'\{\{%s(\:LOOP)?(\[\])?\}\}'%(k.upper()))
        iterations = list(pattern.finditer(code))
        while len(iterations) > 0:
            start, end = iterations[0].start(), iterations[0].end()
            if (iterations[0].groups()[0] or "").upper() == ":LOOP":
                n = kwargs.get(f'_{k}_loop')
                assert n is not None
                code = code[:start] + _parse_code_loop_placeholder(v, n, **kwargs)  + code[end:]
            elif iterations[0].groups()[1] == "[]":
                assert kwargs.get('iteration') is not None
                code = code[:start] + _parse_code_placeholder(v[int(kwargs['iteration'])]) + code[end:]
            else:
                code = code[:start] + _parse_code_placeholder(v) + code[end:]
            iterations = list(pattern.finditer(code))
    return code
    
def _parse_code_placeholder(v: str):
    return str(v)
def _parse_code_loop_placeholder(v: str, length: int, **values):
    val=""
    for i in range(length):
        val += _replace_placeholders(v, iteration=i, **values)
    return val

def _import_defs_from_tmpfile(code, gl=None, lo=None, tmpdir="/tmp/torchbend/jit"):
    gl = gl or globals() 
    os.makedirs(tmpdir, exist_ok=True)
    file = os.path.join(tmpdir, get_random_hash(n=32)+".py")
    with open(file, 'w+') as f:
        f.write(code)
    lo = lo or {}
    code_compiled = compile(code, file, 'exec')
    exec(code_compiled, gl, lo)
    return lo 



class ProjectionRouter(nn.Module):
    def __init__(self, projection_list):
        super().__init__()
        self.projection_list = projection_list
        self._init_callbacks()

    def __len__(self):
        return len(self.projection_list)

    def __getitem__(self, idx): 
        return self.projection_list[idx]

    def _init_callbacks(self):
        codes = _replace_placeholders(project_pattern,
                                      project_call=projection_call_pattern, _project_call_loop = len(self.projection_list), 
                                      inverse_call=inv_projection_call_pattern, _inverse_call_loop = len(self.projection_list),)

        funcs = _import_defs_from_tmpfile(codes, gl=globals(), lo=locals())
        # globals()['project'] = funcs['project']
        # globals()['project_inverse'] = funcs['project_inverse']
        project = MethodType(funcs['project'], self)
        project_inverse = MethodType(funcs['project_inverse'], self)
        setattr(self, 'project', project)
        setattr(self, 'project_inverse', project_inverse)

    def append(self, projection: RAVEProjection):
        self.projection_list.append(projection)
        self._init_callbacks()

    @torch.jit.export
    def forward(self, z, projection_idx: int, latent_size: int | None = None):
        return self.project(z, projection_idx, latent_size=latent_size)
        
    @torch.jit.export
    def inverse(self, z, projection_idx: int, latent_size: int | None = None, temperature: float | None = None):
        return self.project_inverse(z, projection_idx, latent_size=latent_size, temperature=temperature)


torch.fx._symbolic_trace._wrapped_methods_to_patch.extend([(ProjectionRouter, "forward"), (ProjectionRouter, "inverse")])


class DumbPQMF(nn.Module):
    def forward(self, x: torch.Tensor):
        return x
    def inverse(self, x: torch.Tensor):
        return x
    
class DumbPrior(nn.Module):
    def forward(self, x: torch.Tensor):
        return x

class DumbSpectrogram(nn.Module): 
    def forward(self, x: torch.Tensor):
        return x

class DumbOutputTransform(nn.Module):
    def forward(self, x: torch.Tensor):
        return x
    @torch.jit.export
    def invert(self, x: torch.Tensor):
        return x

class ScriptedRAVE(nn_tilde.Module):

    _attributes_for_tb_scripting = ['projections_names', 'projection_idx']

    def __init__(self,
                 pretrained: RAVE,
                 channels: Optional[int] = None,
                 fidelity: float = .95,
                 target_sr: bool = None, 
                 prior: Prior = None,
                 latent_size: int = None,
                 extra_projections: Optional[dict] = None) -> None:

        super().__init__()
        self.pqmf = pretrained.pqmf
        self.sr = pretrained.sr
        self.spectrogram = pretrained.spectrogram or DumbSpectrogram()
        self.resampler = None
        self.input_mode = pretrained.input_mode
        self.output_mode = pretrained.output_mode
        self.n_channels = pretrained.n_channels
        self.target_channels = channels or self.n_channels
        self.stereo_mode = False

        if pretrained.output_transform is None: 
            self.output_transform = DumbOutputTransform()
        else:
            self.output_transform = pretrained.output_transform

        # register config in torchscript for perfect recovery
        self.gin_config_str = gin.config_str()
        self.gin_operative_config_str = gin.operative_config_str()

        if target_sr is not None:
            if target_sr != self.sr:
                assert not target_sr % self.sr, "Incompatible target sampling rate"
                self.resampler = Resampler(target_sr, self.sr)
                self.sr = target_sr

        self.full_latent_size = pretrained.latent_size
        self.is_using_adain = False
        for m in self.modules():
            if isinstance(m, blocks.AdaptiveInstanceNormalization):
                self.is_using_adain = True
                break
        if self.is_using_adain and (self.n_channels != self.target_channels):
            raise ValueError("AdaIN requires the original number of channels")
        self.register_attribute('projection', 'default')
        self.register_attribute("learn_target", False)
        self.register_attribute("reset_target", False)
        self.register_attribute("learn_source", False)
        self.register_attribute("reset_source", False)
        self.register_attribute("temperature", 1.0)
        self.register_buffer("_temperature", torch.tensor(1.))

        # projection handling
        self._has_projections = hasattr(pretrained, "projections")
        self.projections_names = ['default']
        projections = nn.ModuleList([PCAProjection.from_params(pretrained.latent_mean, pretrained.fidelity, pretrained.latent_pca, name="default")])
        if self._has_projections:
            for k, v in pretrained.projections.items():
                if k.startswith('pca'):
                    projection = PCAProjection.from_buffer(v, name=k)
                elif k.startswith(tuple(LINEAR_PROJECTION_PREFIXES)):
                    projection = LinearProjection.from_buffer(v, name=k)
        #         elif k.startswith('mapper'):
        #             self.projections_types.append(2)
                else: 
                    continue
                self.projections_names.append(k)
                projections.append(projection)
        for k, projection in (extra_projections or {}).items():
            self.projections_names.append(k)
            projections.append(projection.eval())

        self.register_attribute("projection", "default")
        self.projections = ProjectionRouter(projections)

        receptive_field = getattr(pretrained, "receptive_field", None)
        self.register_attribute("receptive_field", (int(receptive_field[0]), int(receptive_field[1])))
        self.projection_idx = nn.Parameter(torch.tensor(0.), requires_grad=False)
        self.register_buffer("latent_pca", pretrained.latent_pca)
        self.register_buffer("latent_mean", pretrained.latent_mean)
        self.register_buffer("fidelity", pretrained.fidelity)

        if latent_size is None:
            if isinstance(pretrained.encoder, blocks.VariationalEncoder):
                if fidelity is None:
                    latent_size = self.full_latent_size
                else:
                    latent_size = max(
                        np.argmax(pretrained.fidelity.detach().cpu().numpy() > fidelity), 1)
                    latent_size = 2**math.ceil(math.log2(latent_size))

            elif isinstance(pretrained.encoder, blocks.DiscreteEncoder):
                latent_size = pretrained.encoder.num_quantizers

            elif isinstance(pretrained.encoder, blocks.WasserteinEncoder):
                latent_size = pretrained.latent_size

            elif isinstance(pretrained.encoder, blocks.SphericalEncoder):
                latent_size = pretrained.latent_size - 1

            else:
                raise ValueError(
                    f'Encoder type {pretrained.encoder.__class__.__name__} not supported'
                )
        else:
            assert latent_size > 0, "latent_size must be positive."

        self.latent_size = latent_size
        self.fake_adain = blocks.AdaptiveInstanceNormalization(0)
        device = pretrained.device

        # have to init cached conv before graphing
        self.encoder = pretrained.encoder
        self.decoder = pretrained.decoder
        x_len = 2**14
        x = torch.zeros(1, self.n_channels, x_len).to(device)
        z = self.encode(x)
        self.ratio_encode = x_len // z.shape[-1]

        # configure encoder
        if (pretrained.input_mode == "pqmf") or (pretrained.output_mode == "pqmf"):
            # scripting fails if cached conv is not initialized
            self.pqmf(torch.zeros(1, 1, x_len).to(device))

        encode_shape = (pretrained.n_channels, 2**14) 

        self.register_method(
            "encode",
            in_channels=self.n_channels,
            in_ratio=1,
            out_channels=self.latent_size,
            out_ratio=self.ratio_encode,
            input_labels=['(signal) Channel %d'%d for d in range(1, self.n_channels+1)],
            output_labels=[
                f'(signal) Latent dimension {i + 1}'
                for i in range(self.latent_size)
            ],
        )
        self.register_method(
            "decode",
            in_channels=self.latent_size,
            in_ratio=self.ratio_encode,
            out_channels=self.target_channels,
            out_ratio=1,
            input_labels=[
                f'(signal) Latent dimension {i+1}'
                for i in range(self.latent_size)
            ],
            output_labels=['(signal) Channel %d'%d for d in range(1, self.target_channels+1)]
        )

        self.register_method(
            "forward",
            in_channels=self.n_channels,
            in_ratio=1,
            out_channels=self.target_channels,
            out_ratio=1,
            input_labels=['(signal) Channel %d'%d for d in range(1, self.n_channels + 1)],
            output_labels=['(signal) Channel %d'%d for d in range(1, self.target_channels+1)]
        )

        # init prior in case
        self._has_prior = False
        
        if prior is not None:
            self._has_prior = True
            self.prior_module = prior
            self.register_method(
                "prior",
                in_channels=1,
                in_ratio=prior.ratio,
                out_channels = self.latent_size,
                out_ratio=prior.ratio
            )
        else:
            self.prior_module = DumbPrior()        


    def post_process_latent(self, z):
        raise NotImplementedError

    def pre_process_latent(self, z):
        raise NotImplementedError

    def load_projection(self, projection_name, projection_file):
        self.projections_names.append(projection_name)
        self.projections.append(MapperProjection.from_file(projection_file, name=projection_name))

    def init_cache(self, x):
        self.forward(x)
        if self.pqmf is not None:
            u = self.pqmf(x.reshape(-1, 1, x.shape[-1]))
            v = self.pqmf.inverse(u)

    def update_adain(self):
        for m in self.modules():
            if isinstance(m, blocks.AdaptiveInstanceNormalization):
                m.learn_x.zero_()
                m.learn_y.zero_()

                if self.learn_target[0]:
                    m.learn_y.add_(1)
                if self.learn_source[0]:
                    m.learn_x.add_(1)

                if self.reset_target[0]:
                    m.reset_y()
                if self.reset_source[0]:
                    m.reset_x()

        self.reset_source = False,
        self.reset_target = False,

    @torch.jit.export
    def set_stereo_mode(self, stereo):
        self.stereo_mode = bool(stereo);

    @torch.jit.export
    def set_projection(self, projection: str) -> int:
        # if not self._has_projections: 
            # return -1
        for i, k in enumerate(self.projections_names):
            if projection == k:
                self.projection_idx.set_(torch.tensor(float(i)))

                return 0
        return -1
        
    @torch.jit.export
    def get_projection(self) -> str:
        for i, n in enumerate(self.projections_names):
            if i == self.projection_idx.item():
                return n
        raise TypeError("projection not found")
        

    @torch.jit.export
    def encode(self, x, postprocess: bool = True):
        if self.stereo_mode:
            if self.n_channels == 1:
                x = x[:, 0].unsqueeze(0)
            elif self.n_channels > 2:
                raise RuntimeError("stereo mode is not available when n_channels > 2")

        x = self.output_transform(x)

        if self.is_using_adain:
            self.update_adain()

        if self.resampler is not None:
            x = self.resampler.to_model_sampling_rate(x)

        batch_size = x.shape[:-2]
        if self.input_mode == "pqmf":
            x = x.reshape(-1, 1, x.shape[-1])
            x = self.pqmf(x)
            x = x.reshape(batch_size + (-1, x.shape[-1]))
        elif self.input_mode == "mel":
            if self.spectrogram is not None:
                x = self.spectrogram(x)[..., :-1]
                x = torch.log1p(x).reshape(batch_size + (-1, x.shape[-1]))
            else:
                raise RuntimeError()
        z = self.encoder(x)
        if postprocess:
            z = self.post_process_latent(z)
        return z


    @torch.jit.export
    def decode(self, z, from_forward: bool = False, preprocess: bool = True):
        if self.is_using_adain and not from_forward:
            self.update_adain()
        n_batch = z.shape[0]
        if self.stereo_mode:
            n_batch = int(n_batch / 2)

        if self.target_channels > self.n_channels:
            z = z.repeat(math.ceil(self.target_channels / self.n_channels), 1, 1)[:self.target_channels]

        if preprocess:
            z = self.pre_process_latent(z)

        y = self.decoder(z)

        batch_size = z.shape[:-2]
        if self.output_mode == "pqmf":
            y = y.reshape(y.shape[0] * self.n_channels, -1, y.shape[-1])
            y = self.pqmf.inverse(y)
            y = y.reshape(batch_size+(self.n_channels, -1))

        y = self.output_transform.invert(y)

        if self.resampler is not None:
            y = self.resampler.from_model_sampling_rate(y)

        # if (output-) padding is scrambled
        if y.shape[-1] > z.shape[-1] * self.ratio_encode:
            y = y[..., :z.shape[-1] * self.ratio_encode]

        if self.stereo_mode:
            y = torch.cat([y[:n_batch], y[n_batch:]], 1)
        elif self.target_channels > self.n_channels:
            y = torch.cat(y.chunk(self.target_channels, 0), 1)
        elif self.target_channels < self.n_channels:
            y = y[:, :self.target_channels]

        return y


    @torch.jit.export
    def get_learn_target(self) -> bool:
        return self.learn_target[0]

    @torch.jit.export
    def set_learn_target(self, learn_target: bool) -> int:
        self.learn_target = (learn_target, )
        return 0

    @torch.jit.export
    def get_learn_source(self) -> bool:
        return self.learn_source[0]

    @torch.jit.export
    def set_learn_source(self, learn_source: bool) -> int:
        self.learn_source = (learn_source, )
        return 0

    @torch.jit.export
    def get_reset_target(self) -> bool:
        return self.reset_target[0]

    @torch.jit.export
    def set_reset_target(self, reset_target: bool) -> int:
        self.reset_target = (reset_target, )
        return 0

    @torch.jit.export
    def get_reset_source(self) -> bool:
        return self.reset_source[0]

    @torch.jit.export
    def set_reset_source(self, reset_source: bool) -> int:
        self.reset_source = (reset_source, )
        return 0
    
    @torch.jit.export
    def get_temperature(self) -> Tuple[float]:
        return float(self._temperature),

    @torch.jit.export 
    def set_temperature(self, temperature: float) -> int:
        if temperature < 0: 
            return -1
        self._temperature.set_(torch.tensor(temperature))
        return 0

    @torch.jit.export
    def get_receptive_field(self) -> Tuple[int, int]:
        return self.receptive_field

    @torch.jit.export 
    def set_receptive_field(self, dimred: str) -> int:
        return -1

    @torch.jit.export
    def prior(self, temp: torch.Tensor):
        if self._has_prior:
            return self.prior_module.forward(temp)
        else:
            return torch.tensor(0)
        

class VariationalScriptedRAVE(ScriptedRAVE):

    def __init__(self, *args, use_pca=True, **kwargs):
        self.use_pca = use_pca
        super(VariationalScriptedRAVE, self).__init__(*args, **kwargs)
        if self.use_pca:
            assert self.latent_size <= self.full_latent_size, "latent_size must be < %d"%(self.full_latent_size)
        self.register_method(
            "encode_full",
            in_channels=self.n_channels,
            in_ratio=self.encode_params[1],
            out_channels=self.full_latent_size,
            out_ratio=self.encode_params[3],
            input_labels=['(signal) Channel %d'%d for d in range(1, self.n_channels+1)],
            output_labels=[
                f'(signal) Latent dimension {i + 1}'
                for i in range(self.full_latent_size)
            ],
        )
        self.register_method(
            "decode_full",
            in_channels=self.full_latent_size,
            in_ratio=self.decode_params[1],
            out_channels=self.target_channels,
            out_ratio=self.decode_params[3],
            input_labels=[
                f'(signal) Latent dimension {i+1}'
                for i in range(self.full_latent_size)
            ],
            output_labels=['(signal) Channel %d'%d for d in range(1, self.target_channels+1)]
        )
        self.register_method(
            "encode_dist",
            in_channels=self.n_channels,
            in_ratio=self.encode_params[1],
            out_channels=self.latent_size * 2,
            out_ratio=self.encode_params[3],
            input_labels=['(signal) Channel %d'%d for d in range(1, self.n_channels+1)],
            output_labels=[
                f'(signal) Latent mean {i + 1}'
                for i in range(self.latent_size)
            ] + [
                f'(signal) Latent std {i + 1}'
                for i in range(self.latent_size)
            ],
        )
    
    @torch.jit.export
    def encode_full(self, x):
        if self.stereo_mode:
            if self.n_channels == 1:
                x = x[:, 0].unsqueeze(0)
            elif self.n_channels > 2:
                raise RuntimeError("stereo mode is not available when n_channels > 2")

        if self.is_using_adain:
            self.update_adain()

        if self.resampler is not None:
            x = self.resampler.to_model_sampling_rate(x)

        batch_size = x.shape[:-2]
        if self.input_mode == "pqmf":
            x = x.reshape(-1, 1, x.shape[-1])
            x = self.pqmf(x)
            x = x.reshape(batch_size + (-1, x.shape[-1]))
        elif self.input_mode =="complex_stft":
            x = self.spectrogram(x)
        elif self.input_mode == "mel":
            if self.spectrogram is not None:
                x = self.spectrogram(x)[..., :-1]
                x = torch.log1p(x).reshape(batch_size + (-1, x.shape[-1]))
            else:
                raise RuntimeError()

        z = self.encoder(x)
        z = self.post_process_latent_full(z)
        return z

    @torch.jit.export
    def decode_full(self, z, from_forward: bool = False):

        z = self.pre_process_latent_full(z)

        n_batch = z.shape[0]
        if self.stereo_mode:
            n_batch = int(n_batch / 2)

        y = self.decoder(z)

        batch_size = z.shape[:-2]
        if self.output_mode == "pqmf":
            y = y.reshape(y.shape[0] * self.n_channels, -1, y.shape[-1])
            y = self.pqmf.inverse(y)
            y = y.reshape(batch_size+(self.n_channels, -1))

        if self.resampler is not None:
            y = self.resampler.from_model_sampling_rate(y)

        # if (output-) padding is scrambled
        if y.shape[-1] > z.shape[-1] * self.ratio_encode:
            y = y[..., :z.shape[-1] * self.ratio_encode]

        if self.stereo_mode:
            y = torch.cat([y[:n_batch], y[n_batch:]], 1)
        elif self.target_channels > self.n_channels:
            y = torch.cat(y.chunk(self.target_channels, 0), 1)
        elif self.target_channels < self.n_channels:
            y = y[:, :self.target_channels]
        return y

    @torch.jit.export
    def encode_dist(self, x):
        if self.stereo_mode:
            if self.n_channels == 1:
                x = x[:, 0].unsqueeze(0)
            elif self.n_channels > 2:
                raise RuntimeError("stereo mode is not available when n_channels > 2")

        if self.is_using_adain:
            self.update_adain()

        if self.resampler is not None:
            x = self.resampler.to_model_sampling_rate(x)

        batch_size = x.shape[:-2]
        if self.input_mode == "pqmf":
            x = x.reshape(-1, 1, x.shape[-1])
            x = self.pqmf(x)
            x = x.reshape(batch_size + (-1, x.shape[-1]))
        elif self.input_mode =="complex_stft":
            x = self.spectrogram(x)
        elif self.input_mode == "mel":
            if self.spectrogram is not None:
                x = self.spectrogram(x)[..., :-1]
                x = torch.log1p(x).reshape(batch_size + (-1, x.shape[-1]))
            else:
                raise RuntimeError()
        z = self.encoder(x)

        # parse mean and scale
        z, scale = z.chunk(2, 1)
        std = nn.functional.softplus(scale) + 1e-4
        z = z - self.latent_mean.unsqueeze(-1)
        z = F.conv1d(z, self.latent_pca.unsqueeze(-1))
        z = z[:, :self.latent_size]
        std = F.conv1d(std*std, self.latent_pca.unsqueeze(-1).pow(2)).sqrt()
        std = std[:, :self.latent_size]
        return torch.cat([z, std], dim=-2)

    def post_process_latent(self, z):
        # the latent is projected with the selected projection (set_projection / nn~
        # "projection" attribute); "default" is the PCA of the model.
        z = self.encoder.reparametrize(z, temperature=self.temperature[0])[0]
        if self.use_pca:
            z = self.projections(z, int(self.projection_idx.item()), self.latent_size)
        z = z[:, :self.latent_size]
        return z
       
    def post_process_latent_full(self, z):
        # full latent space, without projection
        return self.encoder.reparametrize(z, temperature=self.temperature[0])[0]
 
    def pre_process_latent(self, z):
        if self.use_pca:
            # missing dimensions are completed by the projection (noise * temperature)
            return self.projections.inverse(z, int(self.projection_idx.item()), None, self.temperature[0])
        if z.shape[1] < self.full_latent_size:
            noise = torch.randn(
                z.shape[0],
                self.full_latent_size - z.shape[1],
                z.shape[-1],
            ).type_as(z)
            z = torch.cat([z, noise * self.temperature[0]], 1)
        return z

    def pre_process_latent_full(self, z):
        # full latent space, without projection
        return z

        
    def forward(self, x, crop_latent: bool = False):
        z = self.encode_full(x)
        if crop_latent:
            z_dec = z[..., :self.latent_size, :]
        else:
            z_dec = z
        return self.decode_full(z_dec, from_forward=True)



class DiscreteScriptedRAVE(ScriptedRAVE):

    def __init__(self, *args, **kwargs):
        super(DiscreteScriptedRAVE, self).__init__(*args,**kwargs)
        assert self.latent_size < self.full_latent_size, "latent_size must be < %d"%(self.full_latent_size)

    def post_process_latent(self, z):
        z = self.encoder.rvq.encode(z)
        return z.float()

    def pre_process_latent(self, z):
        z = torch.clamp(z, 0,
                        self.encoder.rvq.layers[0].codebook_size - 1).long()
        z = self.encoder.rvq.decode(z)
        if self.encoder.noise_augmentation:
            noise = torch.randn(z.shape[0], self.encoder.noise_augmentation,
                                z.shape[-1]).type_as(z)
            z = torch.cat([z, noise], 1)
        return z

    def forward(self, x):
        return self.decode(self.encode(x), from_forward=True)



class WasserteinScriptedRAVE(ScriptedRAVE):

    def __init__(self, *args, **kwargs):
        super(WasserteinScriptedRAVE, self).__init__(*args,**kwargs)
        assert self.latent_size < self.full_latent_size, "latent_size must be < %d"%(self.full_latent_size)

    def post_process_latent(self, z):
        return z

    def pre_process_latent(self, z):
        if self.encoder.noise_augmentation:
            noise = torch.randn(z.shape[0], self.encoder.noise_augmentation,
                                z.shape[-1]).type_as(z)
            z = torch.cat([z, noise], 1)
        return z


class SphericalScriptedRAVE(ScriptedRAVE):

    def __init__(self, *args, **kwargs):
        super(SphericalScriptedRAVE, self).__init__(*args,**kwargs)
        assert self.latent_size < self.full_latent_size, "latent_size must be < %d"%(self.full_latent_size)

    def post_process_latent(self, z):
        return blocks.unit_norm_vector_to_angles(z)

    def pre_process_latent(self, z):
        return blocks.angles_to_unit_norm_vector(z)


class TraceModel(nn.Module):
    def __init__(self, pretrained: Prior, model: RAVE):
        super().__init__()
        pretrained._jit_is_scripting = True
        self.pretrained = pretrained
        self.latent_size = pretrained.latent_size

        x = torch.zeros(1, self.pretrained.n_channels, 2**14)
        z = model.encode(x)
        z = pretrained.post_process_latent(z)
        self.ratio = x.shape[-1] // z.shape[-1]

        # self.register_buffer(
        #     "forward_params",
        #     torch.tensor([1, self.ratio, self.latent_size, self.ratio]),
        # )

        self.pretrained.synth = None

        self.register_buffer(
            "previous_step",
            self.pretrained.quantized_normal.encode(
                torch.zeros(1, self.latent_size, 1)),
        )

        self.pre_diag_cache = cc.CachedPadding(self.latent_size - 1)
        self.pre_diag_cache(z)
        self.pre_diag_cache = torch.jit.script(self.pre_diag_cache)

    def step_forward(self, temp):
        # PREDICT NEXT STEP
        x = self.pretrained.forward(self.previous_step)
        x = x / temp
        x = self.pretrained.post_process_prediction(x, argmax=False)
        # self.previous_step.copy_(x.clone())
        # self.previous_step = x.clone()

        # DECODE AND SHIFT PREDICTION
        x = self.pretrained.quantized_normal.decode(x)
        x = self.pre_diag_cache(x)
        x = self.pretrained.diagonal_shift.inverse(x)
        return x

    def forward(self, temp: torch.Tensor):
        x = torch.zeros(
            temp.shape[0],
            self.latent_size,
            temp.shape[-1],
        ).to(temp)

        temp = temp.mean(-1, keepdim=True)
        temp = nn.functional.softplus(temp) / math.log(2)

        for i in range(x.shape[-1]):
            x[..., i:i + 1] = self.step_forward(temp)

        return x


prior_classes = ['VariationalPrior']
def get_prior_class_from_config():
    prior_class = None
    for cl in prior_classes:
        try:
            gin.get_bindings(cl)
            prior_class = cl
        except:
            pass
    if prior_class is None:
        raise RuntimeError('Could not retrive Prior class from gin config')
    return prior_class


# --------------------------------------------------------------------------- #
#  Uncompiling: scripted model (.ts) -> rave.RAVE checkpoint
# --------------------------------------------------------------------------- #
#
#  A RAVE object is rebuilt from a gin configuration, then every weight / buffer
#  of the scripted model is transferred into it. Weight-norm (removed at export)
#  is re-applied by recomputing (weight_g, weight_v) from the merged weight, so
#  the resulting model is a regular, trainable RAVE (discriminator and losses are
#  freshly initialised, since they are not exported).
#
#  The configuration is taken from (in "auto" mode, first available wins):
#    1. the config embedded in the scripted model at export   ("embedded"),
#       written verbatim as config.gin
#    2. inference from the scripted architecture              ("inferred"):
#       architecture macros (capacity, ratios, dilations, latent size, causal,
#       snake / adain...) are read from the scripted modules, candidate base
#       configs from rave/configs are built with them, and the one whose
#       module structure matches the scripted model is kept.
#    3. a "shell" model wrapping the scripted sub-modules     ("shell"):
#       inference only, no config / checkpoint can be written.
#  An explicit config file (or a config.gin next to the .ts) is only used in
#  "file" mode, or when given through the `config` argument.

import datetime
import json
import logging
from pathlib import Path
from torch.nn.utils.weight_norm import WeightNorm

_log = logging.getLogger(__name__)

UNCOMPILE_METADATA_FILE = "metadata.json"
UNCOMPILE_MODES = ("auto", "embedded", "file", "infer", "shell")

# buffers that only exist in the scripted wrapper / streaming caches
_SKIP_PATTERNS = (".cache.pad", ".downsampling_delay.pad", ".pad", ".cache")
_ARCH_PREFIXES = ("encoder", "decoder", "pqmf", "spectrogram")
_IGNORED_SUBMODULES = (".cache", ".downsampling_delay", ".paddings", "rvq.")
_ENCODER_ADDONS = {"WasserteinEncoder": "wasserstein", "SphericalEncoder": "spherical"}
_MODEL_PREFIXES = ("encoder.", "decoder.", "pqmf.", "spectrogram.")


def _is_streaming_buffer(key: str) -> bool:
    return key.endswith(_SKIP_PATTERNS)


def _ts_attr(ts, name, default=None):
    try:
        return getattr(ts, name)
    except Exception:
        return default


def _raw_class_name(mod) -> str:
    name = getattr(mod, "original_name", None) if isinstance(mod, torch.jit.ScriptModule) else None
    return name or type(mod).__name__


def _module_class(mod) -> str:
    name = _raw_class_name(mod)
    return name[len("Cached"):] if name.startswith("Cached") else name


def _weight_norm_hooks(model: nn.Module):
    """yields (module_name, module, WeightNorm hook) for every weight-normed module."""
    for mname, m in model.named_modules():
        for hook in m._forward_pre_hooks.values():
            if isinstance(hook, WeightNorm):
                yield mname, m, hook


def _jsonable(v):
    if isinstance(v, torch.Tensor):
        return v.tolist()
    if isinstance(v, (list, tuple, set)):
        return [_jsonable(x) for x in v]
    if isinstance(v, dict):
        return {str(k): _jsonable(x) for k, x in v.items()}
    if isinstance(v, Path):
        return str(v)
    if v is None or isinstance(v, (bool, int, float, str)):
        return v
    return str(v)


# ---- scripted model inspection --------------------------------------------- #

def _convs(root, prefix):
    out = []
    for name, mod in root.named_modules():
        if name.startswith(prefix) and _module_class(mod) in ("Conv1d", "ConvTranspose1d") \
                and not any(p in name + "." for p in _IGNORED_SUBMODULES):
            out.append((name, mod, tuple(mod.weight.shape)))
    return out


def is_streaming(root: nn.Module) -> bool:
    """True if the (scripted or python) model contains cached convolutions."""
    for name, mod in root.named_modules():
        # CachedPQMF / CachedSequential exist in non-streaming models too
        if name.startswith(_ARCH_PREFIXES) and _raw_class_name(mod).startswith(("CachedConv", "CachedPadding")):
            return True
    return any(k.endswith(".cache.pad") for k in root.state_dict())


def is_causal(root: nn.Module) -> bool:
    """True if no encoder / decoder convolution looks at the future (works on scripted
    or python models, streaming or not)."""
    for _, mod, shape in _convs(root, "encoder") + _convs(root, "decoder"):
        pad = _ts_attr(mod, "_pad")
        if isinstance(_ts_attr(mod, "cache"), nn.Module):
            # streaming: non-causal convs have right padding, which shows up as delay
            if shape[-1] > 1 and int(mod.stride[0]) == 1 and int(_ts_attr(mod, "cumulative_delay", 0)) > 0:
                return False
        elif pad is not None and shape[-1] > 1 and pad[1] > 0:
            return False
    return True


def get_scripted_metadata(ts: torch.jit.ScriptModule) -> dict:
    """export-time information that can be read from a scripted RAVE."""
    resampler = _ts_attr(ts, "resampler")
    model_sr = _ts_attr(resampler, "model_sr") if resampler is not None else None
    sr = _ts_attr(ts, "sr")
    projections = _ts_attr(ts, "projections_names")
    return _jsonable({
        "script_class": _raw_class_name(ts),
        "streaming": is_streaming(ts),
        "causal": is_causal(ts),
        "sr": sr,
        "model_sr": model_sr if model_sr is not None else sr,
        "resampled": resampler is not None,
        "n_channels": _ts_attr(ts, "n_channels"),
        "target_channels": _ts_attr(ts, "target_channels"),
        "latent_size": _ts_attr(ts, "latent_size"),
        "full_latent_size": _ts_attr(ts, "full_latent_size"),
        "input_mode": _ts_attr(ts, "input_mode"),
        "output_mode": _ts_attr(ts, "output_mode"),
        "use_pca": _ts_attr(ts, "use_pca"),
        "ratio_encode": _ts_attr(ts, "ratio_encode"),
        "receptive_field": _ts_attr(ts, "receptive_field"),
        "projections": list(projections) if projections is not None else None,
        "has_prior": bool(_ts_attr(ts, "_has_prior", False)),
        "has_embedded_config": bool(_ts_attr(ts, "gin_config_str")),
    })


# ---- building a RAVE from a configuration ---------------------------------- #

def build_rave_from_config(configs=(), overrides=(), config_str=None):
    """builds a non-cached RAVE from gin files and / or a gin config string.
    Returns (model, operative config string)."""
    from . import core
    configs = [str(c) for c in (configs if isinstance(configs, (list, tuple)) else [configs]) if c]
    cached_state = cc.use_cached_conv.__globals__["USE_CACHED_CONV"]
    cc.use_cached_conv(False)
    try:
        with core.GinEnv():
            if config_str is not None:
                gin.parse_config(config_str)
            gin.parse_config_files_and_bindings(configs, list(overrides))
            model = RAVE()
            out_config_str = gin.config_str()
    finally:
        cc.use_cached_conv(cached_state)
    return model, out_config_str


def architecture_signature(root: nn.Module) -> dict:
    """{module name: structural description} for encoder / decoder / pqmf modules.
    Works on both scripted and python modules, cached or not, weight-normed or not."""
    sig = {}
    for name, mod in root.named_modules():
        if not name.startswith(_ARCH_PREFIXES) or any(p in name + "." for p in _IGNORED_SUBMODULES):
            continue
        own = {}
        for pname, p in list(mod.named_parameters(recurse=False)) + list(mod.named_buffers(recurse=False)):
            if pname.endswith("_g") or pname in ("pad", "cache"):
                continue
            own[pname[:-2] if pname.endswith("_v") else pname] = tuple(p.shape)
        is_leaf = len(list(mod.children())) == 0
        if (not own and not is_leaf) or _module_class(mod) in ("DumbSpectrogram", "DumbPQMF"):
            continue
        sig[name] = (_module_class(mod), tuple(sorted(own.items())),
                     _ts_attr(mod, "stride"), _ts_attr(mod, "dilation"), _total_pad(mod))
    return sig


def _total_pad(mod):
    cache = _ts_attr(mod, "cache")
    if isinstance(cache, nn.Module):  # cached (streaming) conv: padding lives in the cache
        pad = dict(cache.named_buffers()).get("pad")
        return int(pad.shape[-1]) if pad is not None else None
    pad = _ts_attr(mod, "_pad")
    return int(sum(pad)) if pad is not None else None


def signature_distance(sig_a: dict, sig_b: dict):
    diffs = [(k, sig_a.get(k), sig_b.get(k)) for k in sorted(set(sig_a) | set(sig_b))
             if sig_a.get(k) != sig_b.get(k)]
    return len(diffs), diffs


# ---- configuration inference ----------------------------------------------- #

def infer_hyperparameters(ts) -> dict:
    """reads architecture macros from the scripted model."""
    hp = {}
    n_channels = int(_ts_attr(ts, "n_channels", 1))
    hp["n_channels"] = n_channels
    resampler = _ts_attr(ts, "resampler")
    hp["SAMPLING_RATE"] = int(_ts_attr(resampler, "model_sr", None) or _ts_attr(ts, "sr"))
    hp["LATENT_SIZE"] = int(_ts_attr(ts, "full_latent_size"))

    pqmf = _ts_attr(ts, "pqmf")
    pqmf_bands = int(_ts_attr(pqmf, "n_band")) if pqmf is not None and _ts_attr(pqmf, "n_band") else None
    hp["attenuation"] = _ts_attr(pqmf, "attenuation") if pqmf is not None else None

    enc_root = "encoder.encoder" if _ts_attr(ts.encoder, "encoder") is not None else "encoder"
    enc_convs = _convs(ts, enc_root)
    hp["CAPACITY"] = enc_convs[0][2][0]
    hp["N_ENCODER_BAND"] = enc_convs[0][2][1] // n_channels
    hp["N_BAND"] = 1 if _ts_attr(ts, "output_mode") == "raw" else (pqmf_bands or hp["N_ENCODER_BAND"])

    # ratios (strided convs) and dilations (dilated convs between them)
    ratios, dilations, stage, kernel = [], [], [], None
    for name, mod, shape in enc_convs:
        stride, dil = int(mod.stride[0]), int(mod.dilation[0])
        if stride > 1:
            ratios.append(stride)
            dilations.append(stage)
            stage = []
        elif ".branches." in name and shape[-1] > 1:
            stage.append(dil)
            kernel = kernel or shape[-1]
    hp["RATIOS"] = ratios
    hp["DEC_RATIOS"] = [int(m.stride[0]) for _, m, _ in _convs(ts, "decoder")
                        if _module_class(m) == "ConvTranspose1d" or int(m.stride[0]) > 1][::-1]
    if any(dilations):
        hp["DILATIONS"] = dilations
    if kernel:
        hp["KERNEL_SIZE"] = kernel

    dec_convs = _convs(ts, "decoder")
    hp["NOISE_AUGMENTATION"] = max(dec_convs[0][2][1] - hp["LATENT_SIZE"], 0)

    # discrete models
    sd = ts.state_dict()
    layers = {k.split("rvq.layers.")[1].split(".")[0] for k in sd if "rvq.layers." in k}
    if layers:
        hp["NUM_QUANTIZERS"] = len(layers)
        embed = [v for k, v in sd.items() if "rvq.layers.0." in k and v.ndim == 2]
        if embed:
            hp["CODEBOOK_SIZE"] = int(embed[0].shape[0])

    hp["classes"] = {_module_class(m) for n, m in ts.named_modules() if n.startswith(("encoder", "decoder"))}
    hp["causal"] = is_causal(ts)
    return hp


def _config_text(files) -> str:
    """concatenated text of gin files, with their includes resolved."""
    base_path = Path(__file__).parent.resolve()
    text, seen, stack = "", set(), [Path(f) for f in files]
    while stack:
        f = stack.pop()
        if f in seen or not f.exists():
            continue
        seen.add(f)
        content = f.read_text()
        text += content
        for inc in re.findall(r'^include\s+["\'](.+?)["\']', content, flags=re.M):
            stack.append(base_path / inc)
    return text


def _hp_bindings(hp, files) -> list:
    """gin bindings for inferred macros (only macros actually used by the config files)."""
    text = _config_text(files)
    values = dict(hp, ENC_RATIOS=hp["RATIOS"])
    b = [f"{k} = {values[k]}" for k in ("SAMPLING_RATE", "LATENT_SIZE", "CAPACITY", "N_BAND", "N_ENCODER_BAND",
                                         "KERNEL_SIZE", "DILATIONS", "NOISE_AUGMENTATION", "NUM_QUANTIZERS",
                                         "CODEBOOK_SIZE", "RATIOS", "ENC_RATIOS", "DEC_RATIOS")
         if k in values and f"%{k}" in text and not (k == "RATIOS" and "%ENC_RATIOS" in text)]
    b.append(f"RAVE.n_channels = {hp['n_channels']}")
    if hp.get("attenuation") is not None:
        b.append(f"pqmf.CachedPQMF.attenuation = {hp['attenuation']}")
    return b


def _candidate_configs(ts, hp):
    """yields lists of config files, most likely first."""
    config_dir = Path(__file__).parent.resolve() / "configs"
    enc_cls = _module_class(ts.encoder)
    dec_cls = _module_class(ts.decoder)
    classes = hp["classes"]
    snake, adain = "Snake" in classes, "AdaptiveInstanceNormalization" in classes
    noise = "NoiseGeneratorV2" in classes

    if dec_cls == "Generator":
        bases = ["v1"]
    elif _ts_attr(ts, "input_mode") == "mel":
        bases = ["v2_with_augs", "hybrid"]
    elif _ts_attr(ts, "output_mode") == "raw":
        bases = ["v2_nopqmf", "v2_nopqmf_small"]
    elif enc_cls == "DiscreteEncoder":
        bases = ["discrete_v3", "discrete"] if snake else ["discrete"]
    else:
        bases = ["v3", "v2"] if (snake and adain) else ["v2"]
    for b in ["v2", "v1", "discrete", "v3", "v2_nopqmf", "v2_nopqmf_small", "v2_with_augs", "hybrid", "v4"]:
        if b not in bases:
            bases.append(b)

    addons = []
    if enc_cls in _ENCODER_ADDONS:
        addons.append(_ENCODER_ADDONS[enc_cls])
    if snake:
        addons.append("snake")
    if adain:
        addons.append("adain")
    if noise:
        addons.append("noise")

    for causal in (hp["causal"], not hp["causal"]):
        for base in bases:
            base_text = _config_text([config_dir / f"{base}.gin"])
            names = [base] + [a for a in addons + (["causal"] if causal else [])
                              if f"configs/{a}.gin" not in base_text]
            yield [str(config_dir / f"{n}.gin") for n in names]


def infer_config(ts, overrides=(), max_candidates=12):
    """builds the RAVE model whose architecture best matches the scripted model.
    Returns (model, config_str, info) where info holds the candidate files, bindings
    and the architecture distance (0 = exact match)."""
    hp = infer_hyperparameters(ts)
    _log.info("inferred hyper-parameters: " + ", ".join(f"{k}={v}" for k, v in hp.items() if k != "classes"))
    ts_sig = architecture_signature(ts)
    best = None
    for i, files in enumerate(_candidate_configs(ts, hp)):
        if i >= max_candidates:
            break
        bindings = _hp_bindings(hp, files) + list(overrides)
        try:
            model, config_str = build_rave_from_config(files, bindings)
        except Exception as e:
            _log.debug(f"candidate {files} failed: {e}")
            continue
        dist, diffs = signature_distance(ts_sig, architecture_signature(model))
        _log.info(f"candidate {[Path(f).stem for f in files]}: {dist} structural differences")
        if best is None or dist < best[2]:
            best = (model, config_str, dist, diffs, files, bindings)
        if dist == 0:
            break
    if best is None:
        raise RuntimeError("could not build any candidate configuration")
    model, config_str, dist, diffs, files, bindings = best
    if dist > 0:
        _log.warning(f"no exact architecture match; best candidate {files} has {dist} differences, e.g.:")
        for k, a, b in diffs[:8]:
            _log.warning(f"  {k}: scripted={a} / rebuilt={b}")
    else:
        _log.info(f"exact architecture match with {[Path(f).stem for f in files]}")
    header = (f"# config inferred from a scripted model\n"
              f"# candidate: {[Path(f).stem for f in files]} + bindings {bindings}\n"
              f"# architecture distance: {dist}\n\n")
    info = {"candidate": [Path(f).stem for f in files], "bindings": bindings}
    return model, header + config_str, info


# ---- weight transfer -------------------------------------------------------- #

def transfer_weights(ts: torch.jit.ScriptModule, model: RAVE) -> RAVE:
    """copies weights / buffers of a scripted model into a (weight-normed) RAVE."""
    ts_sd = {k: v.detach().clone() for k, v in ts.state_dict().items()}
    target_sd = model.state_dict()
    loaded, mismatched, pqmf_resized = set(), [], []

    # 1. weight-normed modules: recompute g / v from merged weight
    for mname, m, hook in _weight_norm_hooks(model):
        prefix = f"{mname}." if mname else ""
        w_key, g_key, v_key = (prefix + hook.name, prefix + hook.name + "_g", prefix + hook.name + "_v")
        if g_key in ts_sd and v_key in ts_sd:  # weight norm was not removed
            w_g, w_v = ts_sd[g_key], ts_sd[v_key]
        elif w_key in ts_sd:
            w_v = ts_sd[w_key]
            # w = g * v / ||v||  ->  choosing v = w gives g = ||w||
            w_g = torch.norm_except_dim(w_v, 2, hook.dim)
        else:
            continue
        if w_v.shape != target_sd[v_key].shape:
            mismatched.append((w_key, tuple(w_v.shape), tuple(target_sd[v_key].shape)))
            continue
        getattr(m, hook.name + "_g").data.copy_(w_g)
        getattr(m, hook.name + "_v").data.copy_(w_v)
        loaded.update({w_key, g_key, v_key})

    # 2. every other matching parameter / buffer
    new_sd = {}
    for k, v in ts_sd.items():
        if k in loaded or k not in target_sd:
            continue
        if target_sd[k].shape != v.shape:
            if k.startswith("pqmf.") and not _is_streaming_buffer(k):
                # PQMF rebuilds its filters / convs from the loaded tensors (e.g. other attenuation)
                pqmf_resized.append(k)
            else:
                mismatched.append((k, tuple(v.shape), tuple(target_sd[k].shape)))
                continue
        new_sd[k] = v
        loaded.add(k)
    if pqmf_resized:
        _log.info(f"pqmf filters differ in size from config ({pqmf_resized}): using the scripted ones")
        for k in ("pqmf.h", "pqmf.hk"):  # PQMF._load_from_state_dict requires both
            if k in ts_sd and k not in new_sd:
                new_sd[k] = ts_sd[k]
    model.load_state_dict(new_sd, strict=False)

    # 3. attributes stored outside the state dict
    rf = _ts_attr(ts, "receptive_field")
    if rf is not None:
        model.receptive_field.copy_(torch.tensor([int(rf[0]), int(rf[1])]))

    names = list(_ts_attr(ts, "projections_names", ["default"]))
    for i, name in enumerate(names):
        if i == 0:
            continue  # default projection == latent_pca / latent_mean
        proj = ts.projections.projection_list.__getattr__(str(i))
        psd = proj.state_dict()
        if name.startswith("pca"):
            buf = torch.cat([psd["latent_mean"][None], psd["latent_var"][None], psd["components"]], 0)
        elif name.startswith(tuple(LINEAR_PROJECTION_PREFIXES)):
            try:
                shape = psd["components"].shape
                mean = psd["latent_mean"]
                mean = mean[None].expand(shape) if mean.ndim == 1 else mean.reshape(shape)
                buf = torch.stack([mean] + [psd[k] for k in ("components", "mixing", "whitening")])
            except RuntimeError:
                _log.warning(f"could not rebuild linear projection {name}; skipped")
                continue
        else:
            _log.warning(f"projection {name} cannot be inverted; skipped")
            continue
        model.projections[name] = nn.Parameter(buf, requires_grad=False)

    # report
    missing = [k for k in target_sd if k.startswith(_MODEL_PREFIXES) and k not in loaded]
    unexpected = [k for k in ts_sd if k not in loaded and k.startswith(_MODEL_PREFIXES)
                  and not _is_streaming_buffer(k)]
    _log.info(f"transferred {len(loaded)} tensors")
    for k, s1, s2 in mismatched:
        _log.warning(f"shape mismatch for {k}: scripted {s1} / model {s2}")
    if missing:
        _log.warning(f"{len(missing)} encoder/decoder tensors not found in scripted model: {missing[:10]}...")
    if unexpected:
        _log.warning(f"{len(unexpected)} scripted tensors unused: {unexpected[:10]}...")
    return model


# ---- shell mode ------------------------------------------------------------- #

def build_rave_shell(ts: torch.jit.ScriptModule) -> RAVE:
    """inference-only RAVE wrapping the scripted sub-modules (no gin config needed)."""
    _const = lambda m: (lambda **kwargs: m)
    pqmf = _ts_attr(ts, "pqmf")
    model = RAVE(
        latent_size=int(_ts_attr(ts, "full_latent_size")),
        sampling_rate=int(_ts_attr(ts, "sr")),
        encoder=_const(ts.encoder),
        decoder=_const(ts.decoder),
        discriminator=lambda **kwargs: nn.Identity(),
        phase_1_duration=0,
        gan_loss=None,
        valid_signal_crop=False,
        feature_matching_fun=None,
        num_skipped_features=0,
        audio_distance=nn.Identity,
        multiband_audio_distance=None,
        weights={},
        pqmf=_const(pqmf) if pqmf is not None else None,
        n_channels=int(_ts_attr(ts, "n_channels", 1)),
        input_mode=_ts_attr(ts, "input_mode", "pqmf"),
        output_mode=_ts_attr(ts, "output_mode", "pqmf"),
    )
    for k in ("latent_pca", "latent_mean", "fidelity"):
        getattr(model, k).copy_(getattr(ts, k))
    if is_streaming(ts):
        _log.warning("scripted model is a streaming export: wrapped modules are stateful (cached convs)")
    return model


# ---- checks ----------------------------------------------------------------- #

def effective_state_dict(model: nn.Module) -> dict:
    """state dict with weight norm folded, i.e. what export would script."""
    sd = dict(model.state_dict())
    for mname, mod, hook in _weight_norm_hooks(model):
        prefix = f"{mname}." if mname else ""
        g = sd.pop(prefix + hook.name + "_g")
        v = sd.pop(prefix + hook.name + "_v")
        sd[prefix + hook.name] = torch._weight_norm(v, g, hook.dim).detach()
    return sd


def compare_state_dicts(sd_a, sd_b, prefixes=("encoder.", "decoder.", "pqmf."), name_a="a", name_b="b") -> float:
    """max absolute difference between common floating tensors of two state dicts."""
    keys = [k for k in sd_a if k.startswith(prefixes) and k in sd_b and not _is_streaming_buffer(k)]
    max_err, worst = 0., None
    for k in keys:
        if sd_a[k].shape != sd_b[k].shape:
            _log.warning(f"[{name_a} vs {name_b}] shape mismatch {k}")
            continue
        if not sd_a[k].is_floating_point():
            continue
        err = (sd_a[k].float() - sd_b[k].float()).abs().max().item() if sd_a[k].numel() else 0.
        if err > max_err:
            max_err, worst = err, k
    only_a = [k for k in sd_a if k.startswith(prefixes) and k not in sd_b and not _is_streaming_buffer(k)]
    _log.info(f"[{name_a} vs {name_b}] {len(keys)} common tensors, max abs diff = {max_err:.3e} ({worst})")
    if only_a:
        _log.info(f"[{name_a} vs {name_b}] {len(only_a)} tensors only in {name_a}: {only_a[:5]}")
    return max_err


@torch.no_grad()
def functional_check(ts, model, n_samples=2**15):
    """compares deterministic encode / decode of the scripted and the uncompiled model.
    Returns (encoder max abs diff, decoder max abs diff), or None if not applicable."""
    from .model import _pqmf_encode
    if model.input_mode not in ("pqmf", "raw"):
        _log.info("functional check only implemented for pqmf/raw inputs")
        return None
    if is_streaming(ts):
        _log.info("scripted model is a streaming export: skipping functional check (cached convs add delay)")
        return None
    model.eval()
    x = torch.randn(1, model.n_channels, n_samples) * .1
    x_mb = _pqmf_encode(model.pqmf, x) if model.input_mode == "pqmf" else x
    z_ts = ts.encoder(x_mb.clone())
    z_inv = model.encoder(x_mb.clone())
    err_z = (z_ts - z_inv).abs().max().item()
    mean = z_inv.chunk(2, 1)[0] if isinstance(model.encoder, blocks.VariationalEncoder) else z_inv
    y_ts = ts.decoder(mean.clone())
    y_inv = model.decoder(mean.clone())
    err_y = (y_ts - y_inv).abs().max().item()
    _log.info(f"[functional] encoder max abs diff = {err_z:.3e} | decoder max abs diff = {err_y:.3e}"
              f" (|y| max = {y_ts.abs().max().item():.3e})")
    return err_z, err_y


# ---- main entry points ------------------------------------------------------ #

def get_embedded_config(ts) -> Optional[str]:
    """gin config stored in the scripted model at export time (None for older exports)."""
    for attr in ("gin_config_str", "gin_operative_config_str"):
        config_str = _ts_attr(ts, attr)
        if config_str:
            return config_str
    return None


def _resolve_config(ts, model_path, config, mode):
    """returns (config source, config files, config string), or None if the config
    has to be inferred."""
    from . import core
    if config:
        return "file", list(config) if isinstance(config, (list, tuple)) else [config], None
    if mode in ("auto", "embedded"):
        embedded = get_embedded_config(ts)
        if embedded:
            return "embedded", [], embedded
        if mode == "embedded":
            raise RuntimeError("scripted model has no embedded gin configuration")
        _log.info("no gin config embedded in the scripted model (older export)")
        return None
    if mode == "file" and model_path is not None:
        found = core.search_for_config(str(model_path))
        if found is not None and os.path.exists(found):
            return "file", [found], None
    if mode == "file":
        raise FileNotFoundError(f"no config.gin found next to {model_path}")
    return None


def uncompile(model, out_dir=None, config=None, overrides=(), mode="auto",
              check=False, ckpt_name="last.ckpt"):
    """Converts a scripted RAVE back into a `rave.RAVE` object.

    Args:
        model: path to a .ts file, or an already loaded scripted model.
        out_dir: if given, writes {out_dir}/config.gin, {out_dir}/{ckpt_name} and
            {out_dir}/metadata.json; the folder is loadable with rave.load_rave_checkpoint.
        config: gin config file(s) to rebuild the model with (overrides the embedded one).
        overrides: additional gin bindings.
        mode: "auto" (embedded config if any, else inferred), "embedded" (config stored
            in the .ts), "file" (config.gin next to the .ts), "infer" (from the
            architecture) or "shell" (wrap scripted modules).
        check: run weight round-trip and functional checks (stored in metadata).
        ckpt_name: checkpoint file name.

    Returns:
        (model, config_str, metadata). config_str is None in shell mode.
    """
    if mode not in UNCOMPILE_MODES:
        raise ValueError(f"mode must be one of {UNCOMPILE_MODES}, got {mode}")
    if isinstance(model, torch.jit.ScriptModule):
        ts, model_path = model, None
    else:
        model_path = Path(model).resolve()
        ts = torch.jit.load(str(model_path), map_location="cpu")
    ts.eval()

    from .version import __version__
    metadata = {
        "source": str(model_path) if model_path is not None else None,
        "rave_version": __version__,
        "date": datetime.datetime.now().isoformat(timespec="seconds"),
        **get_scripted_metadata(ts),
        "config_source": None,
        "config_files": None,
        "architecture_distance": None,
    }

    if mode == "shell":
        rave_model, config_str = build_rave_shell(ts), None
        metadata["config_source"] = "shell"
    else:
        resolved = None if mode == "infer" else _resolve_config(ts, model_path, config, mode)
        if resolved is not None:
            source, files, embedded_str = resolved
            _log.info(f"using {source} config" + (f" {files}" if files else ""))
            rave_model, config_str = build_rave_from_config(files, overrides, config_str=embedded_str)
            if embedded_str is not None and not overrides:
                config_str = embedded_str  # keep the exported config as is
            metadata.update(config_source=source, config_files=files or None)
        else:
            if mode == "auto":
                _log.warning("no config found for the scripted model, inferring it from the architecture")
            try:
                rave_model, config_str, info = infer_config(ts, overrides)
                metadata.update(config_source="inferred", config_files=info["candidate"],
                                config_bindings=info["bindings"])
            except Exception as e:
                if mode == "infer":
                    raise
                _log.error(f"config inference failed ({e}), falling back to shell mode")
                rave_model, config_str = build_rave_shell(ts), None
                metadata["config_source"] = "shell"

        if config_str is not None:
            # sanity checks against what the scripted model reports
            for attr in ("n_channels", "input_mode", "output_mode"):
                a, b = _ts_attr(ts, attr), getattr(rave_model, attr)
                if a is not None and a != b:
                    _log.warning(f"{attr} differs: scripted={a}, config={b}")
            if metadata["model_sr"] is not None and metadata["model_sr"] != rave_model.sr:
                _log.warning(f"sampling rate differs: scripted={metadata['model_sr']}, config={rave_model.sr}")
            rebuilt_causal = is_causal(rave_model)
            if rebuilt_causal != metadata["causal"]:
                _log.warning(f"causality differs: scripted={metadata['causal']}, rebuilt={rebuilt_causal}; "
                             "keeping the one of the rebuilt model")
                metadata["causal"] = rebuilt_causal
            rave_model = transfer_weights(ts, rave_model)
            metadata["architecture_distance"] = signature_distance(
                architecture_signature(ts), architecture_signature(rave_model))[0]

    rave_model.eval()

    if check and config_str is not None:
        metadata["checks"] = {
            "max_weight_diff": compare_state_dicts(effective_state_dict(rave_model), ts.state_dict(),
                                                   name_a="uncompiled(folded)", name_b="scripted"),
            "functional": functional_check(ts, rave_model),
        }

    metadata = _jsonable(metadata)
    if out_dir is not None:
        save_uncompiled(rave_model, out_dir, config_str, metadata, ckpt_name=ckpt_name)
    return rave_model, config_str, metadata


def save_uncompiled(model, out_dir, config_str=None, metadata=None, ckpt_name="last.ckpt"):
    """writes {out_dir}/config.gin, {out_dir}/{ckpt_name} and {out_dir}/metadata.json,
    loadable with rave.load_rave_checkpoint(out_dir)."""
    if config_str is None:
        raise ValueError("no gin config available (shell mode): cannot write a loadable checkpoint")
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": model.state_dict(), "callbacks": {},
                "uncompile_metadata": metadata or {}}, out_dir / ckpt_name)
    (out_dir / "config.gin").write_text(config_str)
    if metadata is not None:
        (out_dir / UNCOMPILE_METADATA_FILE).write_text(json.dumps(_jsonable(metadata), indent=2))
    _log.info(f"uncompiled model written to {out_dir} ({ckpt_name}, config.gin, {UNCOMPILE_METADATA_FILE})")
    return out_dir


def load_uncompile_metadata(path) -> dict:
    """reads the metadata written by `uncompile` (from its folder, metadata.json or checkpoint)."""
    path = Path(path)
    if path.is_dir():
        path = path / UNCOMPILE_METADATA_FILE
    if path.suffix == ".ckpt":
        return torch.load(path, map_location="cpu", weights_only=False).get("uncompile_metadata", {})
    return json.loads(path.read_text())


# --------------------------------------------------------------------------- #
#  Exporting: rave.RAVE checkpoint -> scripted model (.ts)
# --------------------------------------------------------------------------- #

def _script_class_for(pretrained):
    if isinstance(pretrained.encoder, blocks.VariationalEncoder):
        return VariationalScriptedRAVE
    if isinstance(pretrained.encoder, blocks.DiscreteEncoder):
        return DiscreteScriptedRAVE
    if isinstance(pretrained.encoder, blocks.WasserteinEncoder):
        return WasserteinScriptedRAVE
    if isinstance(pretrained.encoder, blocks.SphericalEncoder):
        return SphericalScriptedRAVE
    raise ValueError(f"Encoder type {type(pretrained.encoder)} not supported for export.")


def _prepare_streaming_stft(pretrained, use_causal_stft=True):
    if isinstance(pretrained.spectrogram, CausalSTFT):
        pretrained.spectrogram.stream = True
    elif isinstance(pretrained.spectrogram, StreamableSTFT):
        stft = pretrained.spectrogram
        stft_kwargs = dict(nfft=stft.nfft, hop_size=stft.hop_size, stream=True,
                           skip_features=stft.skip_features, normalize=stft.normalize,
                           alpha_rescale=stft.alpha_rescale, beta_rescale=stft.beta_rescale)
        if use_causal_stft:
            pretrained.spectrogram = CausalSTFT(**stft_kwargs)
        else:
            pretrained.spectrogram = StreamableSTFT(**stft_kwargs, log1p=stft.log1p)


def _load_prior(prior_path, pretrained, latent_size, fidelity):
    """returns a traced prior, or None if it could not be loaded."""
    from . import core
    prior_config_file = core.search_for_config(prior_path)
    if prior_config_file is None:
        _log.error(f"config file for prior not found in {prior_path}")
        return None
    gin.clear_config()
    with core.GinEnv():
        gin.constant('SAMPLE_RATE', pretrained.sr)
        gin.parse_config_file(prior_config_file)
        prior_ckpt = core.search_for_run(prior_path)
        _log.info(f"using prior model at {prior_ckpt}")
        prior_class = get_prior_class_from_config()
        prior_pretrained = getattr(prior_models, prior_class)(
            pretrained_vae=pretrained, n_channels=pretrained.n_channels,
            latent_size=latent_size, fidelity=fidelity)
        state_dict = torch.load(prior_ckpt, map_location="cpu")['state_dict']
        for k, v in pretrained.state_dict().items():
            state_dict[f'synth.{k}'] = v
        try:
            prior_pretrained.load_state_dict(state_dict, strict=False)
            return TraceModel(prior_pretrained, pretrained)
        except RuntimeError:
            _log.error(f"could not load prior from {prior_ckpt} with fidelity={fidelity} and "
                       f"latent_size={latent_size}. Did you use the same fidelity / latent "
                       "parameters as for training?")
            return None


def _clear_streaming_caches(module):
    """warmup and nn~ checks populate caches: do not serialize that state."""
    from cached_conv import CachedConvTranspose2d, CachedPadding
    for m in module.modules():
        if isinstance(m, (CachedPadding2d, CachedPadding)):
            m.pad = m.pad[:0].clone()
        elif isinstance(m, CachedConvTranspose2d):
            m.cache = m.cache[:0].clone()
        elif isinstance(m, CausalSTFT):
            m.audio_buffer = m.audio_buffer[:0].clone()
            m.ola_buffer = m.ola_buffer[:0].clone()
            m.envelope_buffer = m.envelope_buffer[:0].clone()
        elif isinstance(m, StreamableSTFT):
            m.audio_buffer = m.audio_buffer[:0].clone()
            m.out_buffer = m.out_buffer[:0].clone()
            m.spec_buffer = m.spec_buffer[:0].clone()


DB_PARTITIONS = ("all", "train", "val")


def _export_dataset(db_path, model, partition="all", run_name=None, chunk_length=2**17):
    """dataset used to recompute latent statistics at export: the full dataset (train + val)
    or one partition of the training split (stored under the run name)."""
    import acids_dataset as ad
    if partition not in DB_PARTITIONS:
        raise ValueError(f"partition must be one of {DB_PARTITIONS}, got {partition}")
    dataset = ad.datasets.AudioDataset(str(Path(db_path).expanduser().resolve()), sr=model.sr,
                                       channels=model.n_channels, chunk_length=chunk_length,
                                       batch_by="dataset")
    if partition == "all":
        return dataset
    if run_name is None or not hasattr(dataset, "has_partition") or not dataset.has_partition(run_name):
        raise FileNotFoundError(f"no training partition {run_name} found in {db_path}; use partition='all'")
    parts = dataset.load_partition(run_name)
    return parts["train" if partition == "train" else "test"]


def compute_dataset_projections(run, db_path, partition="all", ica=False, mapper=False,
                                fidelity=.95, latent_size=None, no_pca=False, ema_weights=False,
                                ckpt_name="last.ckpt", max_chunks=None, batch_size=8,
                                chunk_length=2**17, device="cpu", mapper_steps=5000) -> dict:
    """encodes a dataset with a (non-streaming) RAVE and computes, from its latents:
      * the default PCA (latent_mean, latent_pca, fidelity), on sampled latents as during
        training (variational models only);
      * if ica: an ICA on the PCA-whitened subspace of dimension latent_size;
      * if mapper: a Mapper (MLP autoencoder) with a bottleneck of dimension latent_size.
    ica / mapper require latent_size.
    Returns {"buffers": {name: tensor}, "projections": {name: buffer},
             "extra_projections": {name: RAVEProjection}}."""
    from . import load_rave_checkpoint, get_run_name
    if (ica or mapper) and latent_size is None:
        raise ValueError("ica / mapper projections require an explicit latent_size")
    cached_state = cc.use_cached_conv.__globals__["USE_CACHED_CONV"]
    cc.use_cached_conv(False)
    try:
        try:
            model, model_path = load_rave_checkpoint(run, ema=ema_weights, name=ckpt_name)
        except FileNotFoundError:
            model, model_path = load_rave_checkpoint(run, ema=ema_weights, name=None)
    finally:
        cc.use_cached_conv(cached_state)
    model.eval()
    dataset = _export_dataset(db_path, model, partition, get_run_name(model_path), chunk_length)
    z = encode_dataset(model, dataset, batch_size, max_chunks, device, raw=True)
    out = {"buffers": {}, "projections": {}, "extra_projections": {}}

    if isinstance(model.encoder, blocks.VariationalEncoder):
        z_mean = z.chunk(2, 1)[0]
        z_sampled = model.encoder.reparametrize(z)[0]
        components, mean, var = model._compute_pca(z_sampled)
        out["buffers"] = {"latent_pca": components.float(), "latent_mean": mean.float(), "fidelity": var.float()}
        _log.info("dataset PCA fidelity: " + ", ".join(
            f"{p}: {int(np.argmax(var.numpy() > p))}" for p in (.8, .9, .95, .99)))
    else:
        z_mean = z
    k = int(latent_size) if latent_size is not None else None

    if ica:
        worth, info = ica_relevance(z_mean, n_components=k)
        _log.info(f"ICA on {k} whitened PCA components ({info['reason']})")
        if not worth:
            _log.warning("ICA does not look relevant for these latents; computing it anyway as requested")
        out["projections"]["ica"] = compute_ica(z_mean, n_components=k)
    if mapper:
        _log.info(f"training mapper {z_mean.shape[1]} -> {k} -> {z_mean.shape[1]}")
        mapper_proj = train_mapper(z_mean, k, steps=mapper_steps, device=device, name="mapper", log=_log.info)
        _log.info(f"mapper relative reconstruction error: {mapper_proj.relative_error:.3e}")
        out["extra_projections"]["mapper"] = mapper_proj
    return out


@torch.no_grad()
def export(run, out_path=None, name=None, streaming=False, fidelity=.95, latent_size=None,
           no_pca=False, ema_weights=False, sr=None, prior=None, use_causal_stft=True,
           ckpt_name="last.ckpt", db_path=None, db_partition="all", ica=False, mapper=False,
           max_chunks=None, device="cpu", mapper_steps=5000) -> str:
    """Scripts a RAVE run / checkpoint into a .ts file usable with nn~ and the RAVE VST.

    Args:
        run: run folder or checkpoint (with a config.gin in its folder or parents).
        out_path: output folder (default: folder of `run`), or full path of the .ts file.
        name: model name (default: run name); "_nopca" / "_streaming" / ".ts" are appended.
        streaming: use cached convolutions (realtime mode).
        fidelity: fidelity used to choose the latent size (variational models only).
        latent_size: fixed latent size (overrides fidelity).
        no_pca: disables PCA in latent space (variational models only).
        ema_weights: use EMA weights if available.
        sr: optional target sampling rate (resampling).
        prior: optional prior run.
        use_causal_stft: replace StreamableSTFT with CausalSTFT for streaming exports.
        ckpt_name: checkpoint name searched in run (falls back to the most recent one).
        db_path: optional dataset; its latents are used to recompute the PCA (and the
            fidelity / latent size), and to fit the ica / mapper projections.
        db_partition: "all" (train + val, default), "train" or "val".
        ica: add an ICA projection ("ica"), on the PCA-whitened subspace of dimension
            latent_size (requires db_path and latent_size).
        mapper: add a Mapper projection ("mapper"), MLP autoencoder with a bottleneck of
            dimension latent_size (requires db_path and latent_size).
        max_chunks, device: dataset encoding parameters; mapper_steps: mapper training steps.

    Returns:
        path of the exported .ts.
    """
    from . import load_rave_checkpoint, get_run_name
    if (ica or mapper) and db_path is None:
        raise ValueError("ica / mapper projections require a dataset (db_path)")
    if (ica or mapper) and latent_size is None:
        raise ValueError("ica / mapper projections require an explicit latent_size")
    dataset_projections = None
    if db_path is not None:
        dataset_projections = compute_dataset_projections(
            run, db_path, db_partition, ica=ica, mapper=mapper, fidelity=fidelity, latent_size=latent_size,
            no_pca=no_pca, ema_weights=ema_weights, ckpt_name=ckpt_name, max_chunks=max_chunks,
            device=device, mapper_steps=mapper_steps)
    from . import core
    cached_state = cc.use_cached_conv.__globals__["USE_CACHED_CONV"]
    cc.use_cached_conv(streaming)
    gin_env = core.GinEnv()  # the run config is parsed globally for scripting: restored afterwards
    gin_env.__enter__()
    try:
        try:
            pretrained, model_path = load_rave_checkpoint(run, ema=ema_weights, name=ckpt_name, keep_config=True)
        except FileNotFoundError:
            pretrained, model_path = load_rave_checkpoint(run, ema=ema_weights, name=None, keep_config=True)
        pretrained.eval()
        extra_projections = None
        if dataset_projections is not None:
            for k, v in dataset_projections["buffers"].items():
                getattr(pretrained, k).copy_(v)
            for k, v in dataset_projections["projections"].items():
                pretrained.projections[k] = nn.Parameter(v, requires_grad=False)
            extra_projections = dataset_projections["extra_projections"]

        model_name = name or get_run_name(model_path) or Path(run).stem
        script_class = _script_class_for(pretrained)
        class_kwargs = {}
        if no_pca and script_class is VariationalScriptedRAVE:
            class_kwargs['use_pca'] = False
            fidelity, latent_size = None, None
            model_name += "_nopca"
        if streaming:
            _prepare_streaming_stft(pretrained, use_causal_stft)

        prior_scripted = None
        if prior is not None:
            _log.info("loading prior from checkpoint")
            prior_scripted = _load_prior(prior, pretrained, latent_size, fidelity)

        for m in pretrained.modules():
            if hasattr(m, "weight_g"):
                nn.utils.remove_weight_norm(m)

        _log.info("script model")
        scripted_rave = script_class(pretrained=pretrained, channels=pretrained.n_channels,
                                     fidelity=fidelity, target_sr=sr, prior=prior_scripted,
                                     latent_size=latent_size, extra_projections=extra_projections,
                                     **class_kwargs)
        x = torch.zeros(1, pretrained.n_channels, 2**14)
        scripted_rave.init_cache(x)
        z = scripted_rave.encode(x)
        scripted_rave.decode(z)

        # output path
        if streaming:
            model_name += "_streaming"
        out_path = Path(out_path or os.path.dirname(os.path.abspath(run))).resolve()
        if out_path.suffix == ".ts":
            ts_path = out_path
        else:
            ts_path = out_path / (model_name + ".ts")
        ts_path.parent.mkdir(parents=True, exist_ok=True)
        if streaming:
            _clear_streaming_caches(scripted_rave)
        scripted_rave.export_to_ts(str(ts_path))

        try:
            if pretrained.n_channels <= 2:
                # test stereo mode for VST export
                scripted_rave.set_stereo_mode(True)
                out = scripted_rave.decode(torch.zeros(2, scripted_rave.full_latent_size, z.shape[-1]))
                assert out.shape[1] == 2, "model output is not stereo"
                _log.info("this model seems compatible with the RAVE vst.")
        except Exception as e:
            _log.warning(f"this model will not work with the RAVE VST.\n Caught error : {e}")
    finally:
        cc.use_cached_conv(cached_state)
        gin_env.__exit__(None, None, None)

    if prior is not None and prior_scripted is None:
        _log.info(f"model exported to {ts_path}, but without prior (see error above)")
    else:
        _log.info(f"all good ! model exported to {ts_path}")
    return str(ts_path)


# --------------------------------------------------------------------------- #
#  Custom latent projections (dimension reductions) on a .ckpt or a .ts
# --------------------------------------------------------------------------- #
#
#  Projections are stored in RAVE.projections as buffers, under keys whose prefix
#  gives their type (ScriptedRAVE only exports keys starting with "pca" / "ica"):
#      pca  -> "pca_<name>"   [mean, cumulated variance, components]
#      ica  -> "ica_<name>"   [mean, components, mixing, whitening]

import shutil
import tempfile

PROJECTION_TYPES = {
    "pca": ("pca_{name}", "PCA"),
    "ica": ("ica_{name}", "ICA"),
    "wpca": ("wpca_{name}", "whitened PCA"),
    "sfa": ("sfa_{name}", "slow feature analysis"),
    "pls": ("pls_{name}", "partial least squares (descriptors)"),
    "cca": ("cca_{name}", "canonical correlation analysis (descriptors)"),
    "lda": ("lda_{name}", "linear discriminant analysis (labels)"),
}
SUPERVISED_PROJECTIONS = {"pls": "descriptors", "cca": "descriptors", "lda": "labels"}


def projection_key(name: str, projection_type: str) -> str:
    if projection_type not in PROJECTION_TYPES:
        raise ValueError(f"unknown projection type {projection_type}; available: {list(PROJECTION_TYPES)}")
    return PROJECTION_TYPES[projection_type][0].format(name=name)


def _match_patterns(path: str, filters=None, exclude=None) -> bool:
    """glob-like (fnmatch) selection, as acids_dataset filters / exclude."""
    import fnmatch
    if filters and not any(fnmatch.fnmatch(path, f) for f in filters):
        return False
    return not any(fnmatch.fnmatch(path, e) for e in (exclude or []))


def _audio_dataset(audio, sr, n_channels, chunk_length, filters=None, exclude=None):
    """AudioDataset over a list of audio folders and / or files. filters / exclude are
    glob-like patterns (e.g. "*violin*", "takes/*_close.wav"), matched against paths
    relative to each folder, or against names of single files. Single files are selected
    in their parent folder with filters."""
    import glob
    import acids_dataset as ad
    filters, exclude = list(filters or []), list(exclude or [])
    audio = [Path(a).expanduser().resolve() for a in ([audio] if isinstance(audio, (str, Path)) else audio)]
    missing = [str(a) for a in audio if not a.exists()]
    if missing:
        raise FileNotFoundError(f"audio paths not found: {missing}")
    kwargs = dict(sr=sr, channels=n_channels, chunk_length=chunk_length, batch_by="dataset", write_index=False)
    datasets = []
    for a in (a for a in audio if a.is_dir()):
        if (filters or exclude) and ad.is_preprocessed(str(a)):
            _log.warning(f"{a} is a pre-processed dataset: --filter / --exclude are ignored for it")
            datasets.append(ad.datasets.AudioDataset(a, **kwargs))
        else:
            try:
                datasets.append(ad.datasets.AudioDataset(a, filters=filters, exclude=exclude, **kwargs))
            except RuntimeError as e:
                _log.warning(f"no audio file selected in {a} ({e})")
                continue
        datasets[-1]._rave_input = str(a)  # fallback path of chunks (e.g. pre-processed datasets)
    files_by_dir = {}
    for f in (a for a in audio if a.is_file()):
        if _match_patterns(f.name, filters, exclude):
            files_by_dir.setdefault(f.parent, []).append(glob.escape(f.name))
        else:
            _log.info(f"{f} does not match the filters: skipped")
    for parent, names in files_by_dir.items():
        datasets.append(ad.datasets.AudioDataset(parent, filters=names, **kwargs))
    if not datasets:
        raise RuntimeError("no audio file matches the given inputs / filters")
    return datasets[0] if len(datasets) == 1 else torch.utils.data.ConcatDataset(datasets)


def _item_paths(dataset, indices) -> list:
    """audio file of each dataset item (None if unknown)."""
    def path_of(ds, i):
        if isinstance(ds, torch.utils.data.Subset):
            return path_of(ds.dataset, ds.indices[i])
        if isinstance(ds, torch.utils.data.ConcatDataset):
            j = int(np.searchsorted(ds.cumulative_sizes, i, side="right"))
            return path_of(ds.datasets[j], i - (ds.cumulative_sizes[j - 1] if j > 0 else 0))
        try:
            file = ds.files[ds.chunk_position(i, random_offset=False)[0]]
            return file.get("path") or file.get("audio_path")
        except Exception:
            return getattr(ds, "_rave_input", None)
    return [path_of(dataset, i) for i in indices]


@torch.no_grad()
def encode_dataset(model: RAVE, dataset, batch_size: int = 8, max_chunks: Optional[int] = None,
                   device: str = "cpu", raw: bool = False, extras: bool = False, pitch_mode: str = "pyin"):
    """encodes a dataset of audio chunks with the RAVE encoder; returns a (n_chunks,
    latent, time) tensor of posterior means for variational models (raw encoder outputs
    if raw). If extras, returns {"latents", "descriptors" (n_chunks, len(DESCRIPTORS),
    time), "paths" (audio file of each chunk)}."""
    if isinstance(model.encoder, blocks.DiscreteEncoder):
        raise ValueError("latent projections are not supported for discrete models")
    n_items = len(dataset) if max_chunks is None else min(len(dataset), max_chunks)
    if n_items == 0:
        raise RuntimeError("no audio chunk found")
    indices = list(range(len(dataset)))
    if n_items < len(dataset):
        indices = torch.randperm(len(dataset))[:n_items].tolist()
        dataset = torch.utils.data.Subset(dataset, indices)
    _log.info(f"encoding {n_items} chunks")
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False)
    model = model.to(device).eval()
    latents, descriptors = [], []
    for x in loader:
        x = x.to(device).float()
        if x.ndim == 2:
            x = x[:, None]
        z = model.encode(x)
        if not raw and isinstance(model.encoder, blocks.VariationalEncoder):
            z = z.chunk(2, 1)[0]
        latents.append(z.cpu())
        if extras:
            descriptors.append(compute_descriptors(x, model.sr, z.shape[-1], pitch_mode=pitch_mode).cpu())
    model.cpu()
    latents = torch.cat(latents, 0).double()
    if not extras:
        return latents
    return {"latents": latents, "descriptors": torch.cat(descriptors, 0).double(),
            "paths": _item_paths(dataset, range(len(dataset)))}


def collect_latents(model: RAVE, audio, chunk_length: int = 2**17, batch_size: int = 8,
                    max_chunks: Optional[int] = None, device: str = "cpu",
                    filters=None, exclude=None, extras: bool = False, pitch_mode: str = "pyin"):
    """encodes audio (folders / files, selected with glob-like filters / exclude) with the
    RAVE encoder, and returns latents as a (n_chunks, latent_size, time) tensor (means for
    variational models), or {"latents", "descriptors", "paths"} if extras."""
    dataset = _audio_dataset(audio, model.sr, model.n_channels, chunk_length, filters, exclude)
    return encode_dataset(model, dataset, batch_size, max_chunks, device, extras=extras, pitch_mode=pitch_mode)


def _log_axis_correlations(key, projected, targets, target_names, n_axes):
    x = rearrange(projected[:, :n_axes].double(), "b c t -> (b t) c")
    y = rearrange(targets.double(), "b c t -> (b t) c")
    x = (x - x.mean(0)) / x.std(0).clamp_min(1e-12)
    y = (y - y.mean(0)) / y.std(0).clamp_min(1e-12)
    corr = (x.T @ y) / (len(x) - 1)
    lines = [f"{key}: correlation of projected dimensions with descriptors",
             "      " + "".join(f"{n:>11s}" for n in target_names)]
    lines += [f"  {i:3d} " + "".join(f"{v:11.2f}" for v in corr[i].tolist()) for i in range(corr.shape[0])]
    _log.info("\n".join(lines))


def fit_projections(model: RAVE, data, name: str, types=("pca",), dims: Optional[int] = None,
                    fidelity: float = .99, descriptors=None, labels=None) -> dict:
    """fits projections on latents, and registers them in model.projections.

    Args:
        data: latents (b, c, t), or {"latents", "descriptors" (b, len(DESCRIPTORS), t)}.
        types: among PROJECTION_TYPES.
        dims, fidelity: number of dimensions (given, or number of principal components
            reaching fidelity): whitened subspace for ica / sfa, number of components for
            pls / cca / lda (default: number of descriptors / classes - 1), fitted
            dimensions reported for pca / wpca. cca / lda use the subspace reaching fidelity.
        descriptors: descriptor names used by pls / cca (default: all DESCRIPTORS).
        labels: class label of each chunk (b,), used by lda.
    Returns {projection key: buffer}."""
    latents = data["latents"] if isinstance(data, dict) else data
    desc_names = list(descriptors or DESCRIPTORS)
    targets = None
    if any(SUPERVISED_PROJECTIONS.get(t) == "descriptors" for t in types):
        if not isinstance(data, dict) or data.get("descriptors") is None:
            raise ValueError("pls / cca projections require descriptors")
        unknown = set(desc_names) - set(DESCRIPTORS)
        if unknown:
            raise ValueError(f"unknown descriptors {unknown}; available: {DESCRIPTORS}")
        targets = data["descriptors"][:, [DESCRIPTORS.index(d) for d in desc_names]]
        # frames where a descriptor is undefined (e.g. f0 of unvoiced frames) are not used: frames
        # are flattened as (frames, latent, 1) chunks
        valid = torch.isfinite(targets).all(1)                                      # (b, t)
        frame_latents = rearrange(latents, "b c t -> (b t) c")[valid.flatten()][..., None]
        frame_targets = rearrange(targets, "b c t -> (b t) c")[valid.flatten()][..., None]
        if valid.float().mean() < 1:
            _log.info(f"pls / cca: {100 * (1 - valid.float().mean()):.1f}% of frames ignored "
                      f"(undefined descriptors, e.g. f0 of unvoiced frames)")
        if len(frame_latents) < latents.shape[1] + 2:
            raise RuntimeError(f"not enough frames with defined descriptors ({len(frame_latents)}) "
                               f"for a {latents.shape[1]}-dimensional latent space")
        if len(frame_latents) < 10 * latents.shape[1]:
            _log.warning(f"only {len(frame_latents)} frames with defined descriptors: pls / cca may be unreliable "
                         "(use more audio, or other descriptors)")
    new = {}
    for t in types:
        key = projection_key(name, t)
        _log.info(f"fitting {PROJECTION_TYPES[t][1]} projection {key}")
        if t == "pca":
            components, mean, var = model._compute_pca(latents)
            buf = torch.cat([mean[None], var[None], components], 0)
            if dims is None:
                n = int(min(len(var), int(torch.searchsorted(var.double(), torch.tensor(float(fidelity)).double())) + 1))
                _log.info(f"{key}: {n} dimensions reach a fidelity of {fidelity}")
            else:
                _log.info(f"{key}: {dims} dimensions give a fidelity of {float(var[min(dims, len(var)) - 1]):.3f}")
        elif t == "ica":
            worth, info = ica_relevance(latents, n_components=dims, fidelity=fidelity)
            _log.info(f"ICA on {info['n_components']} whitened PCA components")
            if not worth:
                _log.warning(f"ICA does not look relevant for these latents ({info['reason']}); "
                             "computing it anyway as requested")
            buf = compute_ica(latents, n_components=dims, fidelity=fidelity)
        elif t == "wpca":
            buf = compute_whitened_pca(latents, n_components=dims, fidelity=fidelity)
        elif t == "sfa":
            buf = compute_sfa(latents, n_components=dims, fidelity=fidelity)
        elif t == "pls":
            buf = compute_pls(frame_latents, frame_targets, n_components=dims)
        elif t == "cca":
            buf = compute_cca(frame_latents, frame_targets, n_components=dims, fidelity=fidelity)
        elif t == "lda":
            if labels is None:
                raise ValueError("lda projections require class labels")
            classes = sorted(set(labels))
            _log.info(f"{key}: {len(classes)} classes {classes}")
            buf = compute_lda(latents, labels, n_components=dims, fidelity=fidelity)
        else:
            raise ValueError(f"unknown projection type {t}")
        buf = buf.float().cpu()
        if t != "pca":
            info = linear_projection_info(buf)
            _log.info(f"{key}: {info['n_fitted']} fitted dimensions, scores "
                      f"{[round(v, 3) for v in info['scores'][:8]]}{' ...' if info['n_fitted'] > 8 else ''}")
            if t in ("pls", "cca"):
                projected = LinearProjection.from_buffer(buf)(frame_latents.float())
                _log_axis_correlations(key, projected, frame_targets, desc_names, info["n_fitted"])
        if key in model.projections:
            _log.warning(f"projection {key} already exists: overwritten")
        model.projections[key] = nn.Parameter(buf, requires_grad=False)
        new[key] = buf
    return new


def chunk_labels(paths, audio, label_regex: Optional[str] = None):
    """class label of each chunk, from its audio file path: the --audio input it comes from
    (default), or the first group (whole match if none) of label_regex. Chunks without
    label are None."""
    import re
    if label_regex is not None:
        pattern = re.compile(label_regex)
        labels = []
        for p in paths:
            m = pattern.search(str(p)) if p is not None else None
            labels.append((m.group(1) if m.groups() else m.group(0)) if m else None)
        return labels
    inputs = [Path(a).expanduser().resolve() for a in ([audio] if isinstance(audio, (str, Path)) else audio)]
    labels = []
    for p in paths:
        label = None
        if p is not None:
            p = Path(p).resolve()
            for a in inputs:
                if p == a or a in p.parents:
                    label = a.stem if a.is_file() else a.name
                    break
        labels.append(label)
    return labels


def _write_projections_to_ckpt(ckpt_path, projections: dict, out_path=None, remove=()):
    """adds / removes projections of a checkpoint (and of its EMA weights, if any),
    keeping everything else. Returns (output path, removed keys)."""
    from . import core
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    state_dicts = [ckpt["state_dict"]]
    if isinstance(ckpt.get("callbacks", {}).get("EMA"), dict):
        state_dicts.append(ckpt["callbacks"]["EMA"])
    removed = set()
    for sd in state_dicts:
        for k in remove:
            if sd.pop(f"projections.{k}", None) is not None:
                removed.add(k)
        for k, v in projections.items():
            sd[f"projections.{k}"] = v
    out_path = Path(out_path or ckpt_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = out_path.with_name(out_path.name + ".tmp")
    torch.save(ckpt, tmp_path)
    os.replace(tmp_path, out_path)
    # keep the run loadable at its new location
    src_config = core.search_for_config(str(ckpt_path))
    if src_config is not None and not (out_path.parent / "config.gin").exists():
        shutil.copy(src_config, out_path.parent / "config.gin")
    return str(out_path), sorted(removed)


def _resolve_ckpt(model_path, ckpt_name="last.ckpt") -> Path:
    from . import core
    ckpt_path = core.search_for_run(model_path, name=ckpt_name) or core.search_for_run(model_path, name=None)
    if ckpt_path is None:
        raise FileNotFoundError(f"no checkpoint found in {model_path}")
    return Path(ckpt_path).resolve()


def _load_ckpt_model(ckpt_path) -> RAVE:
    from . import load_rave_checkpoint
    cached_state = cc.use_cached_conv.__globals__["USE_CACHED_CONV"]
    cc.use_cached_conv(False)
    try:
        model, _ = load_rave_checkpoint(ckpt_path)
    finally:
        cc.use_cached_conv(cached_state)
    return model


def _edit_model_projections(model_path, edit, out=None, overwrite=False, ckpt_name="last.ckpt"):
    """applies edit(model) -> (projections to add {key: buffer}, keys to remove) to a
    run / .ckpt (in place, or to out) or to a .ts (uncompiled, edited and re-exported to
    out, or replacing it if overwrite). Returns (output path, added keys, removed keys)."""
    model_path = Path(model_path).resolve()
    if model_path.suffix != ".ts":
        ckpt_path = _resolve_ckpt(model_path, ckpt_name)
        to_add, to_remove = edit(ckpt_path)
        out_path, removed = _write_projections_to_ckpt(ckpt_path, to_add, out, remove=to_remove)
        return out_path, list(to_add), removed

    target = Path(out).resolve() if out is not None else model_path
    if target.is_dir() or target.suffix != ".ts":
        target = target / model_path.name
    if target == model_path and not overwrite:
        raise FileExistsError(f"{model_path} would be replaced; give an output path or overwrite=True")
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        model, _, metadata = uncompile(model_path, out_dir=tmp / "run")
        if metadata["has_prior"]:
            _log.warning("the prior of the scripted model cannot be recovered: it will not be exported")
        to_add, to_remove = edit(model)
        _, removed = _write_projections_to_ckpt(tmp / "run" / "last.ckpt", to_add, remove=to_remove)
        ts_path = export(tmp / "run",
                         out_path=tmp / "export.ts",
                         streaming=bool(metadata["streaming"]),
                         latent_size=metadata["latent_size"] if metadata["use_pca"] is not False else None,
                         no_pca=metadata["use_pca"] is False,
                         sr=metadata["sr"] if metadata["resampled"] else None)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(ts_path, target)
    return str(target), list(to_add), removed


# ---- latents cache ---------------------------------------------------------- #
#
#  latents are cached in the package, in
#      rave/.dimred_cache/{model_name}_{path_hash}/{version}/{name}/{id_hash}/
#  with
#      latents.pt   (n_chunks, latent_size, time) tensor
#      meta.json    inputs, encoding parameters and encoder fingerprint
#  path_hash identifies the model file (models with the same name in different folders),
#  id_hash the inputs (sorted paths, size / mtime of their audio files) and the encoding
#  parameters; the encoder fingerprint, checked with a tolerance, prevents reusing latents
#  of a model whose weights changed.

LATENTS_CACHE_ROOT = Path(__file__).parent.resolve() / ".dimred_cache"


def model_cache_identity(model_path, ckpt_name="last.ckpt"):
    """(model folder name, version) used for the cache folders of a run / .ckpt / .ts."""
    import hashlib
    model_path = Path(model_path).resolve()
    if model_path.suffix == ".ts":
        path_hash = hashlib.sha1(str(model_path).encode()).hexdigest()[:8]
        return f"{model_path.stem}_{path_hash}", "export"
    name, version = _run_identity(model_path, ckpt_name)
    path_hash = hashlib.sha1(str(_resolve_ckpt(model_path, ckpt_name)).encode()).hexdigest()[:8]
    return f"{name}_{path_hash}", version


def _run_identity(model_path, ckpt_name="last.ckpt"):
    from . import get_run_name
    ckpt_path = _resolve_ckpt(model_path, ckpt_name)
    version = next((p for p in reversed(ckpt_path.parent.parts) if p.startswith("version_")), "default")
    name = get_run_name(ckpt_path)
    if not name:
        parts = ckpt_path.parent.parts
        idx = len(parts) - 1 - parts[::-1].index(version) if version in parts else None
        name = parts[idx - 1] if idx else ckpt_path.parent.name
    return name, version


def _audio_inputs_signature(audio, filters=None, exclude=None) -> list:
    from acids_dataset.writers.utils import _VALID_EXTS
    exts = {e.lower() for e in _VALID_EXTS}
    paths = sorted({str(Path(a).expanduser().resolve()) for a in ([audio] if isinstance(audio, (str, Path)) else audio)})
    signature = []
    for p in paths:
        p = Path(p)
        if p.is_dir():
            files = sorted(f for f in p.rglob("*") if f.is_file() and f.suffix.lower() in exts
                           and _match_patterns(str(f.relative_to(p)), filters, exclude))
        else:
            files = [p] if _match_patterns(p.name, filters, exclude) else []
        signature.append([str(p), [[str(f), f.stat().st_size, f.stat().st_mtime] for f in files]])
    return signature


def latents_cache_dir(model_path, name, audio, root=None, ckpt_name="last.ckpt", filters=None,
                      exclude=None, **params) -> Path:
    """cache folder of the latents of model_path for the given audio inputs and encoding params."""
    import hashlib
    model_name, version = model_cache_identity(model_path, ckpt_name)
    key = json.dumps({"audio": _audio_inputs_signature(audio, filters, exclude), "params": params,
                      "filters": list(filters or []), "exclude": list(exclude or [])}, sort_keys=True)
    id_hash = hashlib.sha1(key.encode()).hexdigest()[:16]
    return Path(root or LATENTS_CACHE_ROOT) / model_name / version / name / id_hash


def _encoder_fingerprint(model) -> list:
    """per-tensor mean / std of the (weight-norm folded) encoder weights."""
    sd = effective_state_dict(model)
    fp = []
    for k in sorted(sd):
        if k.startswith("encoder.") and sd[k].is_floating_point() and sd[k].numel() > 1:
            v = sd[k].double()
            fp += [v.mean().item(), v.std().item()]
    return fp


def _load_latents_cache(cache_dir: Path, model):
    latents_file, extras_file, meta_file = cache_dir / "latents.pt", cache_dir / "extras.pt", cache_dir / "meta.json"
    if not (latents_file.exists() and extras_file.exists() and meta_file.exists()):
        return None
    meta = json.loads(meta_file.read_text())
    fp, cached_fp = np.array(_encoder_fingerprint(model)), np.array(meta.get("encoder_fingerprint", []))
    if fp.shape != cached_fp.shape or not np.allclose(fp, cached_fp, rtol=1e-3, atol=1e-6):
        _log.info(f"cache {cache_dir} was computed with other encoder weights: recomputing")
        return None
    extras = torch.load(extras_file, map_location="cpu")
    if list(extras.get("descriptor_names", [])) != list(DESCRIPTORS):
        _log.info(f"cache {cache_dir} has other descriptors: recomputing")
        return None
    _log.info(f"loading cached latents from {cache_dir}")
    return {"latents": torch.load(latents_file, map_location="cpu"), "descriptors": extras["descriptors"],
            "paths": extras["paths"]}


def _save_latents_cache(cache_dir: Path, data: dict, model, audio, params):
    cache_dir.mkdir(parents=True, exist_ok=True)
    torch.save(data["latents"], cache_dir / "latents.pt")
    torch.save({"descriptors": data["descriptors"], "paths": data["paths"], "descriptor_names": list(DESCRIPTORS)},
               cache_dir / "extras.pt")
    meta = {"audio": sorted(str(Path(a).expanduser().resolve()) for a in ([audio] if isinstance(audio, (str, Path)) else audio)),
            "params": params, "shape": list(data["latents"].shape),
            "date": datetime.datetime.now().isoformat(timespec="seconds"),
            "encoder_fingerprint": _encoder_fingerprint(model)}
    (cache_dir / "meta.json").write_text(json.dumps(meta, indent=2))
    _log.info(f"latents cached in {cache_dir}")


def get_latents(model, audio, cache_dir=None, force=False, chunk_length: int = 2**17,
                batch_size: int = 8, max_chunks: Optional[int] = None, device: str = "cpu",
                filters=None, exclude=None, pitch_mode: str = "pyin") -> dict:
    """{"latents", "descriptors", "paths"} of audio, loaded from cache_dir if valid (and
    not force), else computed (and cached in cache_dir if given)."""
    if cache_dir is not None and not force:
        data = _load_latents_cache(Path(cache_dir), model)
        if data is not None:
            return data
    data = collect_latents(model, audio, chunk_length, batch_size, max_chunks, device, filters, exclude,
                           extras=True, pitch_mode=pitch_mode)
    if cache_dir is not None:
        params = dict(chunk_length=chunk_length, max_chunks=max_chunks, sr=model.sr, n_channels=model.n_channels,
                      filters=list(filters or []), exclude=list(exclude or []), pitch_mode=pitch_mode)
        _save_latents_cache(Path(cache_dir), data, model, audio, params)
    return data


def latents_cache_size(model_path=None, root=None, ckpt_name="last.ckpt") -> int:
    """size in bytes of the dimred latent caches (all of them, or those of model_path)."""
    root = Path(root or LATENTS_CACHE_ROOT)
    if model_path is not None:
        root = root / model_cache_identity(model_path, ckpt_name)[0]
    if not root.exists():
        return 0
    return sum(f.stat().st_size for f in root.rglob("*") if f.is_file())


def format_size(n_bytes: int) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if n_bytes < 1024 or unit == "GB":
            return f"{n_bytes:.0f} {unit}" if unit == "B" else f"{n_bytes:.1f} {unit}"
        n_bytes /= 1024


def purge_latents_cache(model_path=None, root=None, ckpt_name="last.ckpt") -> list:
    """removes the dimred latent caches (default root: rave/.dimred_cache): all of them,
    or those of model_path."""
    root = Path(root or LATENTS_CACHE_ROOT)
    if not root.exists():
        candidates = []
    elif model_path is not None:
        candidates = [root / model_cache_identity(model_path, ckpt_name)[0]]
    else:
        candidates = [p for p in root.iterdir() if p.is_dir()]
    removed = []
    for c in candidates:
        if c.exists():
            shutil.rmtree(c)
            removed.append(str(c))
    _log.info(f"removed {len(removed)} cache folder(s)" + (f": {removed}" if removed else ""))
    return removed


# ---- main entry points ------------------------------------------------------ #

def add_projections(model_path, name: str, audio, types=("pca",), out=None, overwrite: bool = False,
                    chunk_length: int = 2**17, batch_size: int = 8, max_chunks: Optional[int] = None,
                    device: str = "cpu", ckpt_name: str = "last.ckpt",
                    dims: Optional[int] = None, fidelity: float = .99,
                    cache: bool = False, force: bool = False, cache_root=None,
                    filters=None, exclude=None, descriptors=None, label_regex: Optional[str] = None,
                    pitch_mode: str = "pyin") -> str:
    """Computes custom latent projections of a RAVE model on a set of audio files, and
    writes them into the model.

    Args:
        model_path: a run folder / .ckpt, or a scripted model (.ts).
        name: name of the projections (keys are given by `projection_key`).
        audio: audio folder(s) and / or file(s).
        filters, exclude: glob-like patterns selecting audio files (paths relative to each
            folder, or names of single files), e.g. "*violin*", "takes/*_close.wav".
        types: projection types, among PROJECTION_TYPES ("pca", "ica").
        out: output path. Default: the checkpoint is updated in place; a .ts is
            replaced only if `overwrite` is True.
        overwrite: allow replacing the original .ts.
        chunk_length, batch_size, max_chunks, device: encoding parameters.
        dims, fidelity: number of dimensions (given, or number of principal components
            reaching fidelity): PCA-whitened subspace for ICA, reported for PCA.
        descriptors: descriptors used by pls / cca (default: all, see DESCRIPTORS).
        pitch_mode: acids_dataset F0 tracker of the f0 descriptor (pyin, yin, crepe, ...).
        label_regex: lda class labels: first group of this regular expression on audio file
            paths (default: one class per audio input).
        cache: cache latents in {cache_root}/{model_name}_{path_hash}/{version}/{name}/{id_hash}
            (cache_root: rave/.dimred_cache), and load them from there if valid.
        force: recompute latents even if a valid cache exists (and re-write it).

    .ckpt: projections are added to the checkpoint state dict.
    .ts: the model is uncompiled in a temporary folder, projections are added to the
         checkpoint, and the model is re-exported with its original export settings
         (streaming, latent size, pca, sampling rate). A prior cannot be recovered.

    Returns:
        path of the written model.
    """
    types = [types] if isinstance(types, str) else list(types)
    for t in types:
        projection_key(name, t)  # validates types early
    cache_dir = None

    def edit(model_or_ckpt):
        nonlocal cache_dir
        model = model_or_ckpt if isinstance(model_or_ckpt, RAVE) else _load_ckpt_model(model_or_ckpt)
        if cache:
            params = dict(chunk_length=chunk_length, max_chunks=max_chunks, sr=model.sr, n_channels=model.n_channels,
                          pitch_mode=pitch_mode)
            cache_dir = latents_cache_dir(model_path, name, audio, root=cache_root, ckpt_name=ckpt_name,
                                          filters=filters, exclude=exclude, **params)
        data = get_latents(model, audio, cache_dir=cache_dir, force=force, chunk_length=chunk_length,
                           filters=filters, exclude=exclude, pitch_mode=pitch_mode,
                           batch_size=batch_size, max_chunks=max_chunks, device=device)
        labels = None
        if "lda" in types:
            labels = chunk_labels(data["paths"], audio, label_regex)
            keep = [i for i, l in enumerate(labels) if l is not None]
            if len(keep) < len(labels):
                _log.warning(f"{len(labels) - len(keep)} chunks without label are ignored by lda")
            lda_data = {"latents": data["latents"][keep], "descriptors": data["descriptors"][keep]}
            lda_proj = fit_projections(model, lda_data, name, ["lda"], dims, fidelity,
                                       labels=[labels[i] for i in keep])
        else:
            lda_proj = {}
        other = [t for t in types if t != "lda"]
        projections = fit_projections(model, data, name, other, dims, fidelity, descriptors=descriptors) if other else {}
        return {**projections, **lda_proj}, []

    out_path, added, _ = _edit_model_projections(model_path, edit, out, overwrite, ckpt_name)
    _log.info(f"projections {added} written to {out_path}")
    return out_path


def list_projections(model_path, ckpt_name: str = "last.ckpt") -> list:
    """names of the projections stored in a run / .ckpt or a .ts."""
    model_path = Path(model_path).resolve()
    if model_path.suffix == ".ts":
        ts = torch.jit.load(str(model_path), map_location="cpu")
        return [n for n in _ts_attr(ts, "projections_names", []) if n != "default"]
    sd = torch.load(_resolve_ckpt(model_path, ckpt_name), map_location="cpu", weights_only=False)["state_dict"]
    return [k[len("projections."):] for k in sd if k.startswith("projections.")]


def _describe_buffer(name, buf, latent_size=None) -> dict:
    """type / dimensionality of a projection stored as a buffer (see PROJECTION_TYPES)."""
    info = {"name": name, "type": "unknown", "input_dim": None, "output_dim": None}
    if name.startswith("pca") or name == "default":
        info.update(type="PCA", input_dim=int(buf.shape[-1]), output_dim=int(buf.shape[0] - 2))
        var = buf[1].double()
        if var.max() > 0:
            info["fidelity_dims"] = {str(f): int(min(len(var), int((var > f).double().argmax()) + 1)) for f in (.9, .95, .99)}
        else:
            info["note"] = "fidelity not computed"
    elif name.startswith(tuple(LINEAR_PROJECTION_PREFIXES)):
        prefix = next(p for p in LINEAR_PROJECTION_PREFIXES if name.startswith(p))
        info.update(type=LINEAR_PROJECTION_PREFIXES[prefix], input_dim=int(buf.shape[-1]), output_dim=int(buf.shape[-2]))
        stored = linear_projection_info(buf)
        if stored is not None:
            info["n_fitted"] = stored["n_fitted"]
            info["scores"] = stored["scores"]
    return info


def describe_projections(model_path, ckpt_name: str = "last.ckpt") -> dict:
    """registered projections of a .ts (or run / .ckpt): name, type and dimensionality
    (input -> projected). Returns {"latent_size", "full_latent_size", "current", "projections": [...]}."""
    model_path = Path(model_path).resolve()
    if model_path.suffix != ".ts":
        sd = torch.load(_resolve_ckpt(model_path, ckpt_name), map_location="cpu", weights_only=False)["state_dict"]
        default = torch.cat([sd["latent_mean"][None], sd["fidelity"][None], sd["latent_pca"]], 0)
        projections = [_describe_buffer("default", default)]
        projections += [_describe_buffer(k[len("projections."):], v) for k, v in sd.items() if k.startswith("projections.")]
        return {"latent_size": None, "full_latent_size": int(sd["latent_pca"].shape[0]), "current": None,
                "projections": projections}

    ts = torch.jit.load(str(model_path), map_location="cpu")
    names = list(_ts_attr(ts, "projections_names", []))
    projections = []
    for i, name in enumerate(names):
        proj = getattr(ts.projections.projection_list, str(i))
        cls = _raw_class_name(proj)
        if cls == "PCAProjection":
            buf = torch.cat([proj.latent_mean[None], proj.latent_var[None], proj.components], 0)
            info = _describe_buffer("default" if i == 0 else name, buf)
            info["name"] = name
        elif cls in ("ICAProjection", "LinearProjection"):
            buf = torch.stack([proj.latent_mean[None].expand_as(proj.components), proj.components,
                               proj.mixing, proj.whitening]) if proj.whitening.dim() == 2 else proj.components[None]
            info = _describe_buffer(name, buf)
        elif cls == "MapperProjection":
            info = {"name": name, "type": "Mapper (MLP)", "input_dim": int(proj.input_size),
                    "output_dim": int(proj.output_size)}
        else:
            info = {"name": name, "type": cls, "input_dim": None, "output_dim": None}
        projections.append(info)
    current = None
    try:
        current = ts.get_projection()
    except Exception:
        pass
    return {"latent_size": _ts_attr(ts, "latent_size"), "full_latent_size": _ts_attr(ts, "full_latent_size"),
            "current": current, "projections": projections}


def format_projections(description: dict) -> str:
    """human-readable table of describe_projections."""
    rows = [("name", "type", "dims (in -> out)", "notes")]
    for p in description["projections"]:
        dims = f"{p['input_dim']} -> {p['output_dim']}" if p["input_dim"] is not None else "?"
        notes = []
        if "fidelity_dims" in p:
            notes.append("dims for fidelity " + ", ".join(f"{k}: {v}" for k, v in p["fidelity_dims"].items()))
        if "note" in p:
            notes.append(p["note"])
        if "n_fitted" in p:
            label = {"ICA": "sources", "SFA": "slow features", "PLS": "components", "CCA": "canonical dims",
                     "LDA": "discriminant dims", "whitened PCA": "dims at fidelity"}.get(p["type"], "fitted dims")
            notes.append(f"{p['n_fitted']} {label}")
        if description.get("current") == p["name"]:
            notes.append("selected")
        rows.append((p["name"], p["type"], dims, "; ".join(notes)))
    widths = [max(len(r[i]) for r in rows) for i in range(3)]
    lines = ["  ".join(r[i].ljust(widths[i]) for i in range(3)) + "  " + r[3] for r in rows]
    lines.insert(1, "  ".join("-" * w for w in widths) + "  " + "-" * 5)
    header = []
    if description.get("latent_size") is not None:
        header.append(f"exported latent size: {description['latent_size']} (full: {description['full_latent_size']})")
    else:
        header.append(f"full latent size: {description['full_latent_size']}")
    return "\n".join(header + [""] + lines)


def remove_projections(model_path, name: str, types=None, out=None, overwrite: bool = False,
                       ckpt_name: str = "last.ckpt") -> Tuple[str, list]:
    """Removes the projections `name` of the given types (default: all types) from a
    run / .ckpt or a .ts (see add_projections for output handling).
    Returns (path of the written model, removed keys)."""
    types = list(PROJECTION_TYPES) if types is None else ([types] if isinstance(types, str) else list(types))
    keys = [projection_key(name, t) for t in types]
    existing = list_projections(model_path, ckpt_name)
    if not set(keys) & set(existing):
        _log.warning(f"no projection {keys} found in {model_path} (available: {existing}): nothing written")
        return str(model_path), []

    def edit(model_or_ckpt):
        if isinstance(model_or_ckpt, RAVE):
            for k in keys:
                if k in model_or_ckpt.projections:
                    del model_or_ckpt.projections[k]
        return {}, keys

    out_path, _, removed = _edit_model_projections(model_path, edit, out, overwrite, ckpt_name)
    missing = sorted(set(keys) - set(removed))
    if not removed:
        _log.warning(f"no projection {keys} found in {model_path}")
    elif missing:
        _log.info(f"not found (skipped): {missing}")
    _log.info(f"projections {removed} removed, model written to {out_path}")
    return out_path, removed
