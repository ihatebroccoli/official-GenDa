import jax
import jax.numpy as jnp
import optax
import flax.linen as nn
from typing import Sequence, Tuple
from functools import partial
import numpy as np

import distrax
from typing import Sequence, Callable, Optional
import time

def get_ensemble(ensemble_size, model_cls, out_axes=0, in_axes=None, methods=['__call__']):
    ensemble = nn.vmap(
            target=model_cls,
            in_axes=in_axes,
            out_axes=out_axes,
            variable_axes={"params": 0},
            split_rngs={"params": True},
            axis_size=ensemble_size,
            methods=methods
        )
    return ensemble


class Decoder(nn.Module):
    """State → latent Ψ(s)"""
    recon_dim: int
    use_encoder: bool = False
    pix_latent_dim: Optional[int] = None
    pixel_shape: Tuple[int, int, int] = (64, 64, 3)
    pixel_dim: int = None
    @nn.compact
    def __call__(self, x):
        x = nn.Dense(1024, name='fc1')(x)
        x = nn.relu(x)
        x = nn.Dense(1024, name='fc2')(x)
        x = nn.relu(x)
        if self.use_encoder:
            x = PixelDecoder(name='pix_enc', pixel_shape=self.pixel_shape)(x)
            x = x.reshape((x.shape[0], -1))
        else:
            x = nn.Dense(self.recon_dim, name='fc3')(x)
        return x

    
class Encoder(nn.Module):
    """State → latent Ψ(s)"""
    latent_dim: int
    
    use_encoder: bool = False
    pix_latent_dim: Optional[int] = None
    pixel_shape: Tuple[int, int, int] = (64, 64, 3)
    pixel_dim: int = None
    @nn.compact
    def __call__(self, s):
        if self.use_encoder:
            s = s[:, :self.pixel_dim]
            x = PixelEncoder(self.pix_latent_dim, name='pix_enc', pixel_shape=self.pixel_shape)(s)
        else:
            x = s
        x = nn.Dense(1024, name='fc1', kernel_init=nn.initializers.xavier_uniform())(x)
        x = nn.relu(x)
        x = nn.Dense(1024, name='fc2', kernel_init=nn.initializers.xavier_uniform())(x)
        x = nn.relu(x)
        return nn.Dense(self.latent_dim, name='fc3')(x)


class DynamicsModel(nn.Module):
    """State → latent Ψ(s)"""
    latent_dim: int
    @nn.compact
    def __call__(self, s, a):
        x = jnp.concatenate([s, a], axis=-1)
        x = nn.Dense(1024, name="fc1")(x)
        x = nn.relu(x)
        x = nn.Dense(1024, name="fc2")(x)
        x = nn.relu(x)
        return nn.Dense(self.latent_dim, name="fc3")(x)



class DistributionalModel(nn.Module):
    """State → latent Ψ(s)"""
    output_dim: int
    use_encoder: bool = False
    pix_latent_dim: Optional[int] = None
    pixel_shape: Tuple[int, int, int] = (64, 64, 3)
    pixel_dim: int = None
    log_std_min: float = -5
    log_std_max: float = 2.
    
    @nn.compact
    def __call__(self, s):
        if self.use_encoder:
            s = s[:, :self.pixel_dim]
            x = PixelEncoder(self.pix_latent_dim, name='pix_enc', pixel_shape=self.pixel_shape)(s)
        else:
            x = s
        x = nn.Dense(1024, name="fc1")(x)
        x = nn.relu(x)
        x = nn.Dense(1024, name="fc2")(x)
        x = nn.relu(x)
        mu = nn.Dense(self.output_dim, name="fc_mu")(x)
        log_std = nn.Dense(self.output_dim, name="fc_std")(x)
        log_std = jnp.tanh(log_std)
        log_std = self.log_std_min + 0.5 * (self.log_std_max - self.log_std_min) * (log_std + 1)

        return mu, log_std
    
    @nn.compact
    def sample(self, s, key):
        if self.use_encoder:
            s = s[:, :self.pixel_dim]
            x = PixelEncoder(self.pix_latent_dim, name='pix_enc', pixel_shape=self.pixel_shape)(s)
        else:
            x = s
        x = nn.Dense(1024, name="fc1")(x)
        x = nn.relu(x)
        x = nn.Dense(1024, name="fc2")(x)
        x = nn.relu(x)
        mu = nn.Dense(self.output_dim, name="fc_mu")(x)
        log_std = nn.Dense(self.output_dim, name="fc_std")(x)
        log_std = jnp.tanh(log_std)
        log_std = self.log_std_min + 0.5 * (self.log_std_max - self.log_std_min) * (log_std + 1)
        
        std = jnp.exp(log_std)
        eps = jax.random.normal(key, mu.shape)
        x_t = mu + eps * std
        return mu, log_std, x_t



class PixelEncoder(nn.Module):
    pix_latent_dim: int
    act: Callable = nn.elu
    norm: str = "none"                 # 'none' | 'layer' | 'group' | 'batch'
    cnn_depth: int = 48
    cnn_kernels: Tuple[int, ...] = (4, 4, 4, 4)
    pixel_shape: Tuple[int, int, int] = (64, 64, 3)

    @nn.compact
    def __call__(self, x):
        x = x.reshape(-1, *self.pixel_shape) * (1./255.0)# (B, H, W, C)
        for i, k in enumerate(self.cnn_kernels):
            depth = (2 ** i) * self.cnn_depth
            x = nn.Conv(
                features=depth, kernel_size=(k, k), strides=(2, 2),
                padding="VALID", name=f"conv{i+1}"
            )(x)
            x = self.act(x)
        x = x.reshape((x.shape[0], -1))
        return x



class PixelDecoder(nn.Module):
    act: Callable = nn.elu
    norm: str = "none"                 # 'none' | 'layer' | 'group' | 'batch'
    cnn_depth: int = 48
    pixel_shape: Tuple[int, int, int] = (64, 64, 3)

    @nn.compact
    def __call__(self, z):
        start_h, start_w = 2, 2
        depths = [(2 ** i) * self.cnn_depth for i in range(4)]  # [48, 96, 192, 384]
        start_ch = depths[-1]  # 384

        # latent → (2,2, start_ch)
        x = nn.Dense(start_h * start_w * start_ch, name="proj")(z)
        x = self.act(x)
        x = x.reshape((-1, start_h, start_w, start_ch))

        kernels = [4, 4, 4, 6]

        out_channels = depths[::-1]  # [384, 192, 96, 48]

        for i, (k, out_ch) in enumerate(zip(kernels, out_channels[1:]+[self.pixel_shape[-1]])):
            x = nn.ConvTranspose(
                features=out_ch,
                kernel_size=(k, k),
                strides=(2, 2),
                padding="VALID",
                name=f"deconv{i+1}"
            )(x)
            if i < len(kernels) - 1:
                x = self.act(x)

        return x


class GaussianActor(nn.Module):
    hidden_dims: Sequence[int]
    action_dim: int
    log_std_min: float = -5
    log_std_max: float = 2.
    use_encoder: bool = False
    pix_latent_dim: Optional[int] = None
    pixel_dim: Optional[int] = None
    pixel_shape: Tuple[int, int, int] = (64, 64, 3)
    use_local: bool = False

    def setup(self):
        # Pixel encoder
        if self.use_encoder and not self.use_local:
            self.pix_enc = PixelEncoder(
                self.pix_latent_dim,
                pixel_shape=self.pixel_shape,
                name="pix_enc"
            )
        else:
            self.pix_enc = None

        # Hidden layers as Sequential
        layers = []
        for i, h in enumerate(self.hidden_dims):
            layers.append(
                nn.Dense(h, kernel_init=nn.initializers.xavier_uniform(), name=f"fc{i}")
            )
            layers.append(nn.relu)
        self.net = nn.Sequential(layers)

        # Output heads
        self.fc_mu = nn.Dense(self.action_dim, name="fc_mu")
        self.fc_std = nn.Dense(self.action_dim, name="fc_std")

    def _encode(self, sz):
        if self.pix_enc is not None:
            s = sz[:, :self.pixel_dim]
            z = sz[:, self.pixel_dim:]
            s = self.pix_enc(s)
            sz = jnp.concatenate([s, z], axis=-1)
        return sz

    def __call__(self, sz, key):
        sz = self._encode(sz)
        x = self.net(sz)

        mu = self.fc_mu(x)
        log_std = self.fc_std(x)

        # CleanRL std transform
        log_std = jnp.tanh(log_std)
        log_std = self.log_std_min + 0.5 * (self.log_std_max - self.log_std_min) * (log_std + 1)
        std = jnp.exp(log_std)

        # Reparameterization trick
        eps = jax.random.normal(key, mu.shape)
        x_t = mu + eps * std
        y_t = jnp.tanh(x_t)
        action = y_t

        # Log prob
        var = std ** 2
        log_prob = -(((x_t - mu) ** 2) / (2 * var) + log_std + jnp.log(jnp.sqrt(2 * jnp.pi)))
        log_prob -= jnp.log((1 - y_t ** 2) + 1e-6)
        log_prob = jnp.sum(log_prob, axis=-1, keepdims=True)

        return action, log_prob

    def get_intermediate(self, sz):
        sz = self._encode(sz)
        x = self.net(sz)
        return x

    def eval_action(self, sz):
        sz = self._encode(sz)
        x = self.net(sz)
        mu = self.fc_mu(x)
        return jnp.tanh(mu)



class Critic(nn.Module):
    hidden_dims: Sequence[int]
    output_dim: int = 1
    use_encoder: bool = False
    use_local: bool = False
    pix_latent_dim: Optional[int] = None
    pixel_dim: Optional[int] = None
    pixel_shape: Tuple[int, int, int] = (64, 64, 3)
    use_ln: bool = True
    @nn.compact
    def __call__(self, sz: jnp.ndarray, a: jnp.ndarray) -> jnp.ndarray:
        if self.use_encoder and not self.use_local:
            s = sz[:, :self.pixel_dim]
            z = sz[:, self.pixel_dim:]
            s = PixelEncoder(self.pix_latent_dim, name='pix_enc', pixel_shape=self.pixel_shape)(s)
            sz = jnp.concatenate([s, z], axis=-1)
        x = jnp.concatenate([sz, a], axis=-1)
        for i, h in enumerate(self.hidden_dims):
            x = nn.Dense(h, name=f"fc{i}", kernel_init=nn.initializers.xavier_uniform())(x)
            x = nn.relu(x)
            if self.use_ln:
                x = nn.LayerNorm(name=f"ln{i}")(x)
        return nn.Dense(self.output_dim, "fc_out")(x).squeeze()  # [B, 1] -> [B]



class PseudoLocalEncoder(nn.Module):
    """State → latent Ψ(s)"""
    mask_dim: Sequence[int]
    @nn.compact
    def __call__(self, s):
        return s[:, 2:].copy(), None



class MLPDiscriminator(nn.Module): # For counting
    hidden: Sequence[int] = (64, 64)
    @nn.compact
    def __call__(self, x):
        # x: (B, d)
        for h in self.hidden:
            x = nn.Dense(h)(x)
            x = nn.silu(x)
        x = nn.Dense(1)(x)  # logits
        x = jnp.squeeze(x, axis=-1)  # (B,)
        return x  # logits



def init_lsh_preprocess(option_dim: int, key: jax.Array):
    """
    Returns a function f(z) -> (onehot_256, bin_idx)
    Uses 8 random hyperplanes => 8-bit code => 256 bins.
    """
    # 8 hyperplanes (8, option_dim)
    # IMPORTANT: keep this matrix fixed across training for stability.
    H = jax.random.normal(key, (8, option_dim), dtype=jnp.float32)
    # normalize hyperplanes (optional but nice)
    H = H / (jnp.linalg.norm(H, axis=-1, keepdims=True) + 1e-8)
    return H

@jax.jit
def preprocess_lsh(H, z, eps=1e-8):
    # z: (B, d)
    proj = z @ H.T                      # (B, 8)
    bits = (proj > 0).astype(jnp.int32) # (B, 8)

    bit_weights = (2 ** jnp.arange(8, dtype=jnp.int32))[None, :]  # (1,8)
    idx = jnp.sum(bits * bit_weights, axis=-1)                    # (B,)

    dz = jax.nn.one_hot(idx, 256, dtype=jnp.float32)              # (B,256)
    return dz
