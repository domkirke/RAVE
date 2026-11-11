import math
import numpy as np
import torch.nn.functional as F
import torch, torch.nn as nn
from einops import rearrange
from sklearn.decomposition import PCA, FastICA

        
@torch.fx.wrap
def get_noise(z: torch.Tensor, full_latent_size: int):
    return torch.randn(
            z.shape[0],
            full_latent_size - z.shape[1],
            z.shape[-1],
    ).type_as(z)


class RAVEProjection(nn.Module):

    def __init__(self, name: str | None = None):
        super().__init__()
        self.name = torch.jit.Attribute(name or "", str)

    def __repr__(self): 
        return "%s(name=%s)"%(type(self).__name__, self.name.value)
    
    def forward(self, x): 
        return x

    @torch.jit.export
    def inverse(self, x):
        return x



class PCAProjection(RAVEProjection): 
    def __init__(self, name: str | None = None, seed: int = 0):
        super().__init__(name=name)
        self.seed = seed
        self.latent_mean = nn.Parameter(torch.tensor(0), requires_grad=False)
        self.latent_var = nn.Parameter(torch.tensor(0), requires_grad=False)
        self.components = nn.Parameter(torch.tensor(0), requires_grad=False)

    @staticmethod
    def from_params(latent_mean, latent_var, components, name=None):
        self = PCAProjection(name=name)
        self.latent_mean.data = latent_mean
        self.latent_var.data = latent_var
        self.components.data = components
        return self

    @staticmethod
    def from_buffer(buffer, name=None):
        self = PCAProjection(name=name)
        latent_mean, latent_var, components = torch.split(buffer, (1, 1, buffer.shape[0]-2), dim=0)
        self.latent_mean.data = latent_mean[0]
        self.latent_var.data = latent_var[0]
        self.components.data = components
        return self


    def fit(self, z):
        z = rearrange(z, "b c t -> (b t) c")
        z_mean = z.mean(0)
        z = z - z_mean
        
        pca = PCA(z.shape[-1], random_state = self.seed).fit(z)
        components = pca.components_
        components = torch.from_numpy(components)

        var = pca.explained_variance_ / np.sum(pca.explained_variance_)
        var = torch.from_numpy(np.cumsum(var))

        self.latent_mean = z_mean
        self.latent_var = var
        self.components = components
        return components, z_mean, var

    def forward(self, z: torch.Tensor, latent_size: int | None = None):
        z = z - self.latent_mean.unsqueeze(-1).to(z)
        z = F.conv1d(z, self.components.unsqueeze(-1).to(z))
        if latent_size is not None: 
            z = z[:, :latent_size]
        return z

    def inverse(self, z: torch.Tensor, latent_size: int | None = None, temperature: float | None = None):
        proj_dim = self.components.shape[0]
        if temperature is None :
            temperature = 1.
        if z.shape[1] < proj_dim:
            noise = get_noise(z, proj_dim)
            z = torch.cat([z, noise * temperature], 1)
        z = F.conv1d(z, self.components.T.unsqueeze(-1).to(z))
        z = z + self.latent_mean.unsqueeze(-1).to(z)
        return z



# ICA (not computed during training: see rave dimred)
#
# ICA does not reduce dimensionality by itself: latents are first whitened with a PCA
# truncated to a target dimension k, then ICA finds the rotation of this whitened
# subspace maximising the independence of the sources. The projection is completed
# with the remaining principal components, so that it stays a (c, c) invertible map:
#     projected dims [0, k)  : ICA sources (unit variance, sorted by contribution)
#     projected dims [k, c)  : residual principal components (sorted by variance)

def _pca_whitening(z, n_components: int | None = None, fidelity: float = .99):
    """returns (x (n, c), mean (c,), eigenvectors (c, c), eigenvalues (c,), k)."""
    x = rearrange(z, "b c t -> (b t) c").detach().cpu().double().numpy()
    mean = x.mean(0)
    pca = PCA(x.shape[-1]).fit(x)
    eigvals = np.maximum(pca.explained_variance_, 0)
    if n_components is None:
        var = np.cumsum(eigvals) / max(eigvals.sum(), 1e-12)
        n_components = int(np.searchsorted(var, fidelity) + 1)
    n_components = int(min(max(n_components, 1), x.shape[-1]))
    return x, mean, pca.components_, eigvals, n_components


def ica_relevance(z, n_components: int | None = None, fidelity: float = .99,
                  min_relative_var: float = 1e-4, skew_threshold: float = .2,
                  kurtosis_threshold: float = .3):
    """Preliminary test telling whether an ICA of latents (b, c, t), on the PCA-whitened
    subspace of dimension k (n_components, or reaching `fidelity`), is worth computing.

    ICA requires
      * enough samples;
      * a non-degenerate whitened subspace (k-th eigenvalue above min_relative_var times
        the first one);
      * non-gaussian whitened dimensions (gaussian sources cannot be separated): tested
        with |skewness| and |excess kurtosis|, whose thresholds are raised by 3 standard
        errors of their gaussian estimates.

    Returns (worth: bool, info: dict)."""
    x, mean, eigvecs, eigvals, k = _pca_whitening(z, n_components, fidelity)
    n = x.shape[0]
    info = {"n_dims": x.shape[-1], "n_components": k}
    if n < 10 * k:
        info["reason"] = f"not enough samples ({n} frames for {k} components)"
        return False, info
    if eigvals[k - 1] <= min_relative_var * eigvals[0]:
        info["reason"] = (f"degenerate whitening: component {k} has a relative variance of "
                          f"{eigvals[k - 1] / max(eigvals[0], 1e-12):.1e}")
        return False, info
    y = (x - mean) @ eigvecs[:k].T / np.sqrt(eigvals[:k])
    skew = np.abs((y ** 3).mean(0))
    kurt = np.abs((y ** 4).mean(0) - 3)
    skew_thr = skew_threshold + 3 * math.sqrt(6 / n)
    kurt_thr = kurtosis_threshold + 3 * math.sqrt(24 / n)
    n_nongaussian = int(((skew > skew_thr) | (kurt > kurt_thr)).sum())
    info.update(n_nongaussian=n_nongaussian, max_skew=float(skew.max()), max_kurtosis=float(kurt.max()))
    if n_nongaussian == 0:
        info["reason"] = "whitened dimensions are gaussian (no independent sources to separate)"
        return False, info
    info["reason"] = f"{n_nongaussian} non-gaussian dimensions out of {k}"
    return True, info


def compute_ica(z, n_components: int | None = None, fidelity: float = .99,
                seed: int = 0, max_iter: int = 1000):
    """ICA of latents (b, c, t) on their PCA-whitened subspace of dimension k
    (n_components, or the number of principal components reaching `fidelity`).

    Returns a (4, c, c) buffer [mean (repeated on rows), components, mixing, whitening]
    such that projected = (z - mean) @ components.T and z = projected @ mixing.T + mean
    (see ICAProjection); dims [0, k) are ICA sources, dims [k, c) residual principal
    components."""
    x, mean, eigvecs, eigvals, k = _pca_whitening(z, n_components, fidelity)
    n_dims = x.shape[-1]
    scale = np.sqrt(np.maximum(eigvals[:k], 1e-12))
    whitening = eigvecs[:k] / scale[:, None]                     # (k, c)
    y = (x - mean) @ whitening.T                                 # whitened, (n, k)
    if k > 1:
        rotation = FastICA(whiten=False, max_iter=max_iter, random_state=seed).fit(y).components_
    else:
        rotation = np.ones((1, 1))
    # rotation is orthogonal: sources = y @ rotation.T, y = sources @ rotation
    components = rotation @ whitening                            # (k, c)
    mixing = eigvecs[:k].T @ (scale[:, None] * rotation.T)       # (c, k)
    order = np.argsort(-np.linalg.norm(mixing, axis=0))          # largest contribution first
    components, mixing = components[order], mixing[:, order]
    # complete with residual principal components
    components = np.concatenate([components, eigvecs[k:]], 0)
    mixing = np.concatenate([mixing, eigvecs[k:].T], 1)
    whitening = np.concatenate([whitening, eigvecs[k:]], 0)
    mean = np.repeat(mean[None], n_dims, 0)
    return torch.from_numpy(np.stack([mean, components, mixing, whitening])).float()


class ICAProjection(RAVEProjection): 
    def __init__(self, name: str | None = None, seed: int = 0):
        super().__init__(name=name)
        self.seed = seed
        self.latent_mean = nn.Parameter(torch.tensor(0), requires_grad=False)
        self.components = nn.Parameter(torch.tensor(0), requires_grad=False)
        self.mixing = nn.Parameter(torch.tensor(0), requires_grad=False)
        self.whitening = nn.Parameter(torch.tensor(0), requires_grad=False)

    @staticmethod
    def from_params(latent_mean, components, mixing, whitening, name=None):
        self = ICAProjection(name=name)
        self.latent_mean.data = latent_mean
        self.components.data = components
        self.mixing.data = mixing 
        self.whitening.data = whitening
        return self

    @staticmethod
    def from_buffer(buffer, name=None):
        """buffer (see RAVE._compute_ica): [mean (repeated on rows), components, mixing, whitening],
        or [components, mixing, whitening] (zero mean)."""
        self = ICAProjection(name=name)
        if buffer.shape[0] == 4:
            latent_mean, components, mixing, whitening = buffer.unbind(0)
            latent_mean = latent_mean[0]
        else:
            components, mixing, whitening = buffer.unbind(0)
            latent_mean = torch.zeros(components.shape[-1])
        self.latent_mean.data = latent_mean
        self.components.data = components
        self.mixing.data = mixing
        self.whitening.data = whitening
        return self

    def fit(self, z, n_components: int | None = None, fidelity: float = .99):
        """fits the projection on latents (b, c, t); see compute_ica."""
        buffer = compute_ica(z, n_components=n_components, fidelity=fidelity, seed=self.seed)
        fitted = ICAProjection.from_buffer(buffer)
        for k in ("latent_mean", "components", "mixing", "whitening"):
            getattr(self, k).data = getattr(fitted, k).data
        return buffer

    def forward(self, z: torch.Tensor, latent_size: int | None = None):
        # sources = (z - mean) @ components.T
        z = z - self.latent_mean.unsqueeze(-1).to(z)
        z = F.conv1d(z, self.components.unsqueeze(-1).to(z))
        if latent_size is not None: 
            z = z[:, :latent_size]
        return z

    def inverse(self, z: torch.Tensor, latent_size: int | None = None, temperature: float | None = None):
        # z = sources @ mixing.T + mean
        proj_dim = self.components.shape[0]
        if temperature is None :
            temperature = 1.
        if z.shape[1] < proj_dim:
            noise = get_noise(z, proj_dim)
            z = torch.cat([z, noise * temperature], 1)
        z = F.conv1d(z, self.mixing.unsqueeze(-1).to(z))
        z = z + self.latent_mean.unsqueeze(-1).to(z)
        return z

    

# LATENT MAPPER

class Mapper(nn.Module):
    def __init__(self, input_size, output_size, hidden_size=300, hidden_layers=3):
        super().__init__()
        layer_shapes = [input_size] + [hidden_size] * hidden_layers + [output_size]
        layers = []
        for i in range(len(layer_shapes) - 1):
            layers.append(nn.Linear(layer_shapes[i], layer_shapes[i+1]))
            if i != len(layer_shapes) - 2:
                layers.append(nn.BatchNorm1d(layer_shapes[i+1]))
                layers.append(nn.ReLU())
        self._module = nn.Sequential(*layers)

    def forward(self, x):
        return self._module(x)



class MapperProjection(RAVEProjection):
    """non-linear projection: MLP autoencoder latent (input_size) -> output_size -> latent."""

    def __init__(self, input_size: int = 128, output_size: int = 8, hidden_size: int = 300,
                 hidden_layers: int = 3, name: str | None = None):
        super().__init__(name=name)
        self.input_size = input_size
        self.output_size = output_size
        self.encoder = Mapper(input_size, output_size, hidden_size, hidden_layers)
        self.decoder = Mapper(output_size, input_size, hidden_size, hidden_layers)

    @staticmethod
    def from_file(mapper_file, name=None, input_size: int = 128, output_size: int = 8,
                  hidden_size: int = 300, hidden_layers: int = 3):
        """loads a mapper checkpoint ({'state_dict': {encoder.*, decoder.*}})."""
        self = MapperProjection(input_size, output_size, hidden_size, hidden_layers, name=name)
        state_dict = torch.load(mapper_file, map_location="cpu")['state_dict']
        encoder_state_dict, decoder_state_dict = {}, {}
        for k, v in state_dict.items():
            module, param_name = k.split('.')[0], ".".join(k.split('.')[1:])
            if module == "encoder": encoder_state_dict[param_name] = v
            if module == "decoder": decoder_state_dict[param_name] = v
        self.encoder.load_state_dict(encoder_state_dict)
        self.decoder.load_state_dict(decoder_state_dict)
        return self.eval()

    def _encode(self, z: torch.Tensor) -> torch.Tensor:
        n_batch, n_channels, n_steps = z.shape
        z = self.encoder(z.permute(0, 2, 1).reshape(-1, n_channels))
        return z.reshape(n_batch, n_steps, -1).permute(0, 2, 1)

    def _decode(self, z: torch.Tensor) -> torch.Tensor:
        n_batch, n_channels, n_steps = z.shape
        z = self.decoder(z.permute(0, 2, 1).reshape(-1, n_channels))
        return z.reshape(n_batch, n_steps, -1).permute(0, 2, 1)

    def forward(self, z: torch.Tensor, latent_size: int | None = None):
        z = self._encode(z)
        if latent_size is not None:
            if latent_size < z.shape[1]:
                z = z[:, :latent_size]
            elif latent_size > z.shape[1]:
                z = torch.cat([z, torch.zeros(z.shape[0], latent_size - z.shape[1], z.shape[2],
                                              dtype=z.dtype, device=z.device)], dim=1)
        return z

    def inverse(self, z: torch.Tensor, latent_size: int | None = None, temperature: float | None = None):
        # as other projections, missing bottleneck dims are filled with noise * temperature
        # (zero if temperature is None); extra ones are ignored
        if z.shape[1] < self.output_size:
            fill = torch.zeros(z.shape[0], self.output_size - z.shape[1], z.shape[2], dtype=z.dtype, device=z.device)
            if temperature is not None:
                fill = torch.randn_like(fill) * temperature
            z = torch.cat([z, fill], dim=1)
        elif z.shape[1] > self.output_size:
            z = z[:, :self.output_size]
        return self._decode(z)


def train_mapper(z, output_size: int, hidden_size: int = 300, hidden_layers: int = 3, steps: int = 5000,
                 batch_size: int = 1024, lr: float = 1e-3, device: str = "cpu", seed: int = 0,
                 name: str | None = None, log=None) -> "MapperProjection":
    """trains a MapperProjection (MLP autoencoder, MSE) on latents z (b, c, t).
    Returns the projection in eval mode; its relative reconstruction error is stored in
    `.relative_error`."""
    generator = torch.Generator().manual_seed(seed)
    torch.manual_seed(seed)
    x = rearrange(z, "b c t -> (b t) c").float()
    mapper = MapperProjection(x.shape[-1], output_size, hidden_size, hidden_layers, name=name).to(device)
    optimizer = torch.optim.Adam(mapper.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, steps)
    mapper.train()
    batch_size = min(batch_size, x.shape[0])
    with torch.enable_grad():  # export runs under no_grad
        _train_mapper_loop(mapper, x, steps, batch_size, optimizer, scheduler, generator, device, log)
    mapper = mapper.eval().cpu()
    with torch.no_grad():
        idx = torch.randperm(x.shape[0], generator=generator)[:100000]
        xv = x[idx]
        err = (mapper.decoder(mapper.encoder(xv)) - xv).pow(2).mean() / xv.var(0).sum().div(x.shape[-1])
    mapper.relative_error = float(err)
    return mapper


def _train_mapper_loop(mapper, x, steps, batch_size, optimizer, scheduler, generator, device, log):
    for step in range(steps):
        idx = torch.randint(0, x.shape[0], (batch_size,), generator=generator)
        xb = x[idx].to(device)
        loss = F.mse_loss(mapper.decoder(mapper.encoder(xb)), xb)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()
        if log is not None and (step % max(steps // 10, 1) == 0 or step == steps - 1):
            log(f"mapper step {step + 1}/{steps}: mse = {loss.item():.4e}")
