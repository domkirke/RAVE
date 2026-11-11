"""
Builds .ts models holding every available latent projection, to try them in Max/MSP (nn~).

    pytest tests/test_dimred_export.py -s

Exported models contain the projections
    default          PCA of the model (recomputed on the dataset, export --db_path)
    pca_<name>       PCA added with dimred
    ica_<name>       ICA added with dimred (on --dims whitened components)
    ica              ICA added at export (on latent_size whitened components)
    mapper           Mapper (MLP autoencoder, bottleneck of latent_size) trained at export
and are written, in offline and streaming versions, to $RAVE_TEST_OUT (default:
tests/outs/dimred).

Environment variables:
    RAVE_TEST_MODEL     run folder / .ckpt (or .ts, uncompiled first) of a trained model
                        (default: an untrained model, only useful to test the patches)
    RAVE_TEST_AUDIO     audio folder used to fit projections (default: tests/datasets/mono)
    RAVE_TEST_OUT       output folder (default: tests/outs/dimred)
    RAVE_TEST_LATENT    exported latent size (default: 8)
    RAVE_TEST_CHUNKS    maximum number of encoded audio chunks (default: 256)
    RAVE_TEST_DEVICE    encoding / mapper training device (default: cpu)
"""
import os
import shutil
from pathlib import Path

import gin
import pytest
import torch
import cached_conv as cc

import rave
from rave.scripting import add_projections, export, list_projections, uncompile

TESTS_DIR = Path(__file__).parent
NAME = "test"
LATENT_SIZE = int(os.environ.get("RAVE_TEST_LATENT", 8))
MAX_CHUNKS = int(os.environ.get("RAVE_TEST_CHUNKS", 256))
DEVICE = os.environ.get("RAVE_TEST_DEVICE", "cpu")
AUDIO = Path(os.environ.get("RAVE_TEST_AUDIO", TESTS_DIR / "datasets" / "mono"))
OUT = Path(os.environ.get("RAVE_TEST_OUT", TESTS_DIR / "outs" / "dimred"))
EXPECTED_PROJECTIONS = ["default", f"pca_{NAME}", f"ica_{NAME}", "ica", "mapper"]


def _untrained_run(path: Path) -> Path:
    """small untrained v2 run (config.gin + last.ckpt)."""
    path.mkdir(parents=True, exist_ok=True)
    cc.use_cached_conv(False)
    with rave.core.GinEnv():
        gin.parse_config_files_and_bindings([str(rave.BASE_PATH / "configs" / "v2.gin")], ["CAPACITY = 16"])
        model = rave.RAVE()
        (path / "config.gin").write_text(gin.config_str())
    torch.save({"state_dict": model.state_dict(), "callbacks": {}}, path / "last.ckpt")
    return path


def _working_run(tmp_path: Path) -> Path:
    """copy of the model to edit (the original model is never modified)."""
    model = os.environ.get("RAVE_TEST_MODEL")
    run = tmp_path / "run"
    if model is None:
        return _untrained_run(run)
    model = Path(model).expanduser().resolve()
    if model.suffix == ".ts":
        uncompile(model, out_dir=run)
        return run
    ckpt = rave.core.search_for_run(model) or rave.core.search_for_run(model, name=None)
    config = rave.core.search_for_config(str(model))
    assert ckpt is not None and config is not None, f"no checkpoint / config found in {model}"
    run.mkdir(parents=True)
    shutil.copy(ckpt, run / "last.ckpt")
    shutil.copy(config, run / "config.gin")
    return run


@pytest.fixture(scope="module")
def exported_models(tmp_path_factory):
    tmp_path = tmp_path_factory.mktemp("dimred_export")
    run = _working_run(tmp_path)
    model_name = Path(os.environ.get("RAVE_TEST_MODEL", "untrained")).stem
    # 1. dimred on the checkpoint: pca_<name> / ica_<name>
    add_projections(run, NAME, [AUDIO], types=["pca", "ica"], dims=LATENT_SIZE,
                    max_chunks=MAX_CHUNKS, device=DEVICE)
    # 2. export with dataset PCA, ica and mapper (after dimred: re-exporting a .ts drops the mapper)
    paths = {}
    for streaming in (False, True):
        paths[streaming] = export(run, out_path=OUT, name=f"{model_name}_projections", streaming=streaming,
                                  latent_size=LATENT_SIZE, db_path=AUDIO, ica=True, mapper=True,
                                  max_chunks=MAX_CHUNKS, device=DEVICE)
    return paths


@pytest.mark.parametrize("streaming", [False, True], ids=["offline", "streaming"])
def test_export_all_projections(exported_models, streaming):
    path = exported_models[streaming]
    assert os.path.exists(path)
    ts = torch.jit.load(path)
    names = list(ts.projections_names)
    assert names == EXPECTED_PROJECTIONS, names
    assert list_projections(path) == EXPECTED_PROJECTIONS[1:]
    assert ts.latent_size == LATENT_SIZE

    # every projection is selectable and invertible
    z = torch.randn(1, ts.full_latent_size, 32)
    for i, name in enumerate(names):
        assert ts.set_projection(name) == 0
        assert ts.get_projection() == name
        projected = ts.projections.forward(z, i, latent_size=LATENT_SIZE)
        assert projected.shape == (1, LATENT_SIZE, 32), (name, projected.shape)
        assert ts.projections.inverse(projected, i).shape == z.shape, name
    assert ts.set_projection("default") == 0

    # offline model: audio round trip (streaming caches are only initialised by nn~)
    if not streaming:
        x = torch.randn(1, ts.n_channels, 2**15) * .1
        y = ts.decode(ts.encode(x))
        assert y.shape[-1] == x.shape[-1] and torch.isfinite(y).all()
    print(f"\n{'streaming' if streaming else 'offline'} model with projections {names}: {path}")
