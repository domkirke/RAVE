"""
Invert an exported TorchScript RAVE model (.ts) back into a `rave.RAVE` object.

Two modes:

  * config mode (when a config.gin is given or found next to the .ts):
      the RAVE object is rebuilt from the gin config, then every weight / buffer
      of the scripted model is transferred into it. Weight-norm (removed at
      export) is re-applied by recomputing (weight_g, weight_v) from the merged
      weight, so the resulting model is a regular, trainable RAVE (discriminator
      and losses are freshly initialised, since they are not exported).

  * inferred config (--infer_config, or when no config.gin is found):
      architecture macros (capacity, ratios, dilations, n_band, latent size, causal,
      snake / adain...) are read from the scripted modules, candidate base configs
      from rave/configs are built with them, and the one whose module structure
      matches the scripted model exactly is kept. Training-only settings
      (discriminator, losses) cannot be recovered and come from the base config.

  * shell mode (--shell, or fallback when inference fails):
      a RAVE object is built around the scripted encoder / decoder / pqmf
      submodules. Good for inference / scripting experiments only.

Usage:
    rave uncompile --model model.ts [--config config.gin] [--out out_dir]
                   [--reference path/to/run_or_ckpt] [--check]
"""
import logging
import os
import re
from pathlib import Path

import torch
import torch.nn as nn
import cached_conv as cc
import gin
from absl import app, flags

import rave
import rave.core
from torch.nn.utils.weight_norm import WeightNorm

logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
log = logging.getLogger("invert_scripted")

FLAGS = flags.FLAGS
flags.DEFINE_string("model", None, help="exported .ts model", required=True)
flags.DEFINE_multi_string("config", None, help="gin config of the run (default: searched next to the .ts)")
flags.DEFINE_multi_string("override", [], help="additional gin bindings")
flags.DEFINE_bool("shell", False, help="wrap scripted modules instead of rebuilding from config")
flags.DEFINE_bool("infer_config", False,
                  help="infer the gin config from the scripted architecture (default when no config.gin is found)")
flags.DEFINE_string("out", None, help="output folder: writes config.gin + last.ckpt (loadable with rave.load_rave_checkpoint)")
flags.DEFINE_string("reference", None, help="original run / ckpt to compare weights against")
flags.DEFINE_bool("check", False, help="run round-trip and functional checks")

# buffers that only exist in the scripted wrapper / streaming caches
_SKIP_PATTERNS = (".cache.pad", ".downsampling_delay.pad", ".pad", ".cache")


def _is_streaming_buffer(key: str) -> bool:
    return key.endswith(_SKIP_PATTERNS)


def _ts_attr(ts, name, default=None):
    try:
        return getattr(ts, name)
    except Exception:
        return default


def _weight_norm_hooks(model: nn.Module):
    """yields (module_name, module, WeightNorm hook) for every weight-normed module."""
    for mname, m in model.named_modules():
        for hook in m._forward_pre_hooks.values():
            if isinstance(hook, WeightNorm):
                yield mname, m, hook


# --------------------------------------------------------------------------- #
#  config mode
# --------------------------------------------------------------------------- #

def build_rave_from_config(config, overrides=()):
    configs = [str(c) for c in (config if isinstance(config, (list, tuple)) else [config])]
    cc.use_cached_conv(False)
    with rave.core.GinEnv():
        gin.parse_config_files_and_bindings(configs, list(overrides))
        model = rave.RAVE()
        config_str = gin.config_str()
    return model, config_str


# --------------------------------------------------------------------------- #
#  config inference
# --------------------------------------------------------------------------- #

_ARCH_PREFIXES = ("encoder", "decoder", "pqmf", "spectrogram")
_IGNORED_SUBMODULES = (".cache", ".downsampling_delay", ".paddings", "rvq.")
_ENCODER_ADDONS = {"WasserteinEncoder": "wasserstein", "SphericalEncoder": "spherical"}


def _module_class(mod) -> str:
    name = getattr(mod, "original_name", None) if isinstance(mod, torch.jit.ScriptModule) else None
    name = name or type(mod).__name__
    return name[len("Cached"):] if name.startswith("Cached") else name


def _total_pad(mod):
    cache = _ts_attr(mod, "cache")
    if isinstance(cache, nn.Module):  # cached (streaming) conv: padding lives in the cache
        pad = dict(cache.named_buffers()).get("pad")
        return int(pad.shape[-1]) if pad is not None else None
    pad = _ts_attr(mod, "_pad")
    return int(sum(pad)) if pad is not None else None


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


def signature_distance(sig_a: dict, sig_b: dict):
    diffs = [(k, sig_a.get(k), sig_b.get(k)) for k in sorted(set(sig_a) | set(sig_b))
             if sig_a.get(k) != sig_b.get(k)]
    return len(diffs), diffs


def _convs(ts, prefix):
    out = []
    for name, mod in ts.named_modules():
        if name.startswith(prefix) and _module_class(mod) in ("Conv1d", "ConvTranspose1d") \
                and not any(p in name + "." for p in _IGNORED_SUBMODULES):
            out.append((name, mod, tuple(mod.weight.shape)))
    return out


def _is_causal(ts) -> bool:
    for _, mod, shape in _convs(ts, "encoder") + _convs(ts, "decoder"):
        pad = _ts_attr(mod, "_pad")
        if isinstance(_ts_attr(mod, "cache"), nn.Module):
            # streaming: non-causal convs have right padding, which shows up as delay
            if shape[-1] > 1 and int(mod.stride[0]) == 1 and int(_ts_attr(mod, "cumulative_delay", 0)) > 0:
                return False
        elif pad is not None and shape[-1] > 1 and pad[1] > 0:
            return False
    return True


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

    classes = {_module_class(m) for n, m in ts.named_modules() if n.startswith(("encoder", "decoder"))}
    hp["classes"] = classes
    hp["causal"] = _is_causal(ts)
    return hp


def _config_text(files) -> str:
    """concatenated text of gin files, with their includes resolved."""
    text, seen, stack = "", set(), [Path(f) for f in files]
    while stack:
        f = stack.pop()
        if f in seen or not f.exists():
            continue
        seen.add(f)
        content = f.read_text()
        text += content
        for inc in re.findall(r'^include\s+["\'](.+?)["\']', content, flags=re.M):
            stack.append(rave.BASE_PATH / inc)
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
            base_text = _config_text([rave.BASE_PATH / "configs" / f"{base}.gin"])
            names = [base] + [a for a in addons + (["causal"] if causal else [])
                              if f"configs/{a}.gin" not in base_text]
            files = [str(rave.BASE_PATH / "configs" / f"{n}.gin") for n in names]
            yield files


def infer_config(ts, overrides=(), max_candidates=12):
    """builds the RAVE model whose architecture matches the scripted model.
    Returns (model, config_str, distance)."""
    hp = infer_hyperparameters(ts)
    log.info("inferred hyper-parameters: " + ", ".join(f"{k}={v}" for k, v in hp.items() if k != "classes"))
    ts_sig = architecture_signature(ts)
    best = None
    for i, files in enumerate(_candidate_configs(ts, hp)):
        if i >= max_candidates:
            break
        bindings = _hp_bindings(hp, files) + list(overrides)
        try:
            model, config_str = build_rave_from_config(files, bindings)
        except Exception as e:
            log.debug(f"candidate {files} failed: {e}")
            continue
        dist, diffs = signature_distance(ts_sig, architecture_signature(model))
        log.info(f"candidate {[Path(f).stem for f in files]}: {dist} structural differences")
        if best is None or dist < best[2]:
            best = (model, config_str, dist, diffs, files, bindings)
        if dist == 0:
            break
    if best is None:
        raise RuntimeError("could not build any candidate configuration")
    model, config_str, dist, diffs, files, bindings = best
    if dist > 0:
        log.warning(f"no exact architecture match; best candidate {files} has {dist} differences, e.g.:")
        for k, a, b in diffs[:8]:
            log.warning(f"  {k}: scripted={a} / rebuilt={b}")
    else:
        log.info(f"exact architecture match with {[Path(f).stem for f in files]}")
    header = (f"# config inferred by invert_scripted.py from a scripted model\n"
              f"# candidate: {[Path(f).stem for f in files]} + bindings {bindings}\n"
              f"# architecture distance: {dist}\n\n")
    return model, header + config_str, dist


def save_run(model, out_dir, config_str=None, ckpt_name="last.ckpt"):
    """writes {out_dir}/config.gin + {out_dir}/{ckpt_name}, loadable with rave.load_rave_checkpoint(out_dir)."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": model.state_dict(), "callbacks": {}}, out_dir / ckpt_name)
    log.info(f"saved inverted state dict to {out_dir / ckpt_name}")
    if config_str is None:
        log.warning("no gin config available (shell mode): config.gin not written")
    else:
        (out_dir / "config.gin").write_text(config_str)
        log.info(f"saved gin config to {out_dir / 'config.gin'}")


def transfer_weights(ts: torch.jit.ScriptModule, model: rave.RAVE):
    ts_sd = {k: v.detach().clone() for k, v in ts.state_dict().items()}
    target_sd = model.state_dict()
    loaded, mismatched = set(), []

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
            mismatched.append((k, tuple(v.shape), tuple(target_sd[k].shape)))
            continue
        new_sd[k] = v
        loaded.add(k)
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
        elif name.startswith("ica"):
            try:
                buf = torch.stack([psd[k].reshape(psd["components"].shape)
                                   for k in ("latent_mean", "components", "mixing", "whitening")])
            except RuntimeError:
                log.warning(f"could not rebuild ICA projection {name}; skipped")
                continue
        else:
            log.warning(f"projection {name} cannot be inverted; skipped")
            continue
        model.projections[name] = nn.Parameter(buf, requires_grad=False)

    # report
    model_modules = ("encoder.", "decoder.", "pqmf.", "spectrogram.")
    missing = [k for k in target_sd if k.startswith(model_modules) and k not in loaded]
    unexpected = [k for k in ts_sd if k not in loaded and k.startswith(model_modules)
                  and not _is_streaming_buffer(k)]
    log.info(f"transferred {len(loaded)} tensors")
    for k, s1, s2 in mismatched:
        log.warning(f"shape mismatch for {k}: scripted {s1} / model {s2}")
    if missing:
        log.warning(f"{len(missing)} encoder/decoder tensors not found in scripted model: {missing[:10]}...")
    if unexpected:
        log.warning(f"{len(unexpected)} scripted tensors unused: {unexpected[:10]}...")
    return model


# --------------------------------------------------------------------------- #
#  shell mode
# --------------------------------------------------------------------------- #

def build_rave_shell(ts: torch.jit.ScriptModule):
    """inference-only RAVE wrapping the scripted sub-modules (no gin config needed)."""
    _const = lambda m: (lambda **kwargs: m)
    pqmf = _ts_attr(ts, "pqmf")
    model = rave.RAVE(
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
    if any(_is_streaming_buffer(k) and v.ndim > 0 and v.shape[0] > 0 or k.endswith(".cache.pad")
           for k, v in ts.state_dict().items()):
        log.warning("scripted model looks like a streaming export: wrapped modules are stateful (cached convs)")
    return model


# --------------------------------------------------------------------------- #
#  checks
# --------------------------------------------------------------------------- #

def _effective_state_dict(model: nn.Module):
    """state dict with weight norm folded, i.e. what export.py would script."""
    sd = dict(model.state_dict())
    for mname, mod, hook in _weight_norm_hooks(model):
        prefix = f"{mname}." if mname else ""
        g = sd.pop(prefix + hook.name + "_g")
        v = sd.pop(prefix + hook.name + "_v")
        sd[prefix + hook.name] = torch._weight_norm(v, g, hook.dim).detach()
    return sd


def compare_state_dicts(sd_a, sd_b, prefixes=("encoder.", "decoder.", "pqmf."), name_a="a", name_b="b"):
    keys = [k for k in sd_a if k.startswith(prefixes) and k in sd_b and not _is_streaming_buffer(k)]
    max_err, worst = 0., None
    for k in keys:
        if sd_a[k].shape != sd_b[k].shape:
            log.warning(f"[{name_a} vs {name_b}] shape mismatch {k}")
            continue
        if not sd_a[k].is_floating_point():
            continue
        err = (sd_a[k].float() - sd_b[k].float()).abs().max().item() if sd_a[k].numel() else 0.
        if err > max_err:
            max_err, worst = err, k
    only_a = [k for k in sd_a if k.startswith(prefixes) and k not in sd_b and not _is_streaming_buffer(k)]
    log.info(f"[{name_a} vs {name_b}] {len(keys)} common tensors, max abs diff = {max_err:.3e} ({worst})")
    if only_a:
        log.info(f"[{name_a} vs {name_b}] {len(only_a)} tensors only in {name_a}: {only_a[:5]}")
    return max_err


@torch.no_grad()
def functional_check(ts, model, n_samples=2**15):
    """compares deterministic encode/decode of the scripted model and the inverted one."""
    model.eval()
    x = torch.randn(1, model.n_channels, n_samples) * .1
    x_mb = rave.model._pqmf_encode(model.pqmf, x) if model.input_mode == "pqmf" else x
    if model.input_mode not in ("pqmf", "raw"):
        log.info("functional check only implemented for pqmf/raw inputs")
        return
    z_ts = ts.encoder(x_mb.clone())
    z_inv = model.encoder(x_mb.clone())
    err_z = (z_ts - z_inv).abs().max().item()
    mean = z_inv.chunk(2, 1)[0] if isinstance(model.encoder, rave.blocks.VariationalEncoder) else z_inv
    y_ts = ts.decoder(mean.clone())
    y_inv = model.decoder(mean.clone())
    err_y = (y_ts - y_inv).abs().max().item()
    log.info(f"[functional] encoder max abs diff = {err_z:.3e} | decoder max abs diff = {err_y:.3e}"
             f" (|y| max = {y_ts.abs().max().item():.3e})")
    return err_z, err_y


# --------------------------------------------------------------------------- #

def invert(model_path, config=None, overrides=(), shell=False, infer=False):
    ts = torch.jit.load(str(model_path), map_location="cpu").eval()
    if shell:
        return build_rave_shell(ts), ts, None
    if config is None and not infer:
        config = rave.core.search_for_config(str(model_path))
        if config is None or not os.path.exists(config):
            log.warning("no config.gin found next to the scripted model, inferring it from the architecture")
            infer = True
        else:
            log.info(f"using config {config}")
    if infer:
        try:
            model, config_str, _ = infer_config(ts, overrides)
        except Exception as e:
            log.error(f"config inference failed ({e}), falling back to shell mode")
            return build_rave_shell(ts), ts, None
    else:
        model, config_str = build_rave_from_config(config, overrides)
    # sanity checks against what the scripted model reports
    for attr in ("n_channels", "input_mode", "output_mode"):
        a, b = _ts_attr(ts, attr), getattr(model, attr)
        if a is not None and a != b:
            log.warning(f"{attr} differs: scripted={a}, config={b}")
    sr = _ts_attr(ts, "sr")
    if sr is not None and _ts_attr(ts, "resampler") is None and sr != model.sr:
        log.warning(f"sampling rate differs: scripted={sr}, config={model.sr}")
    model = transfer_weights(ts, model)
    model.eval()
    return model, ts, config_str


def main(argv):
    model, ts, config_str = invert(FLAGS.model, FLAGS.config, FLAGS.override, FLAGS.shell, FLAGS.infer_config)
    print(f"\ninverted model: {type(model).__name__} | encoder: {type(model.encoder).__name__} "
          f"| decoder: {type(model.decoder).__name__} | sr={model.sr} | n_channels={model.n_channels} "
          f"| latent_size={model.latent_size}\n")

    if FLAGS.check and not FLAGS.shell:
        ts_sd = ts.state_dict()
        compare_state_dicts(_effective_state_dict(model), ts_sd, name_a="inverted(folded)", name_b="scripted")
        streaming = any(k.endswith(".cache.pad") for k in ts_sd)
        if streaming:
            log.info("scripted model is a streaming export: skipping functional check (cached convs add delay)")
        else:
            functional_check(ts, model)

    if FLAGS.reference is not None:
        ref, _ = rave.load_rave_checkpoint(FLAGS.reference)
        compare_state_dicts(model.state_dict(), ref.state_dict(), prefixes=("encoder.", "decoder.", "pqmf."),
                            name_a="inverted", name_b="reference")
        compare_state_dicts(_effective_state_dict(model), _effective_state_dict(ref),
                            name_a="inverted(folded)", name_b="reference(folded)")
        for k in ("latent_pca", "latent_mean", "fidelity"):
            log.info(f"[inverted vs reference] {k} max abs diff = "
                     f"{(getattr(model, k) - getattr(ref, k)).abs().max().item():.3e}")

    if FLAGS.out is not None:
        save_run(model, FLAGS.out, config_str)


if __name__ == "__main__":
    app.run(main)
