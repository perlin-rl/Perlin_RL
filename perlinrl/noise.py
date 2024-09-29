import numpy as np
import torch as th
import colorednoise as cn
from perlin_noise import PerlinNoise
from torch.distributions import Normal

PI = 3.1415926535897932384626433

class Colored_Noise():
    def __init__(self, known_shape=None, beta=1, num_samples=2**14, random_state=None):
        assert known_shape, 'known_shape need to be defined for Colored Noise'
        self.known_shape = known_shape
        self.compact_shape = np.prod(list(known_shape))
        self.beta = beta
        self.num_samples = num_samples  # Actually very cheap...
        self.index = 0
        self.reset(random_state=random_state)

    def __call__(self, shape, latent: th.Tensor = None) -> th.Tensor:
        assert shape == self.known_shape or (shape[1:] == self.known_shape[1:] and shape[0] <= self.known_shape[0])
        sample = self.samples[:, self.index]
        self.index = (self.index+1) % self.num_samples
        return th.Tensor(sample).view(self.known_shape)[:shape[0]]

    def reset(self, random_state=None):
        self.samples = cn.powerlaw_psd_gaussian(
            self.beta, (self.compact_shape, self.num_samples), random_state=random_state)


class Pink_Noise(Colored_Noise):
    def __init__(self, known_shape=None, num_samples=2**14, random_state=None):
        super().__init__(known_shape=known_shape, beta=1, num_samples=num_samples, random_state=random_state)


class shortPink_Noise(Colored_Noise):
    def __init__(self, known_shape=None, num_samples=1000, random_state=None):
        super().__init__(known_shape=known_shape, beta=1, num_samples=num_samples, random_state=random_state)


class White_Noise():
    def __init__(self, known_shape=None):
        self.known_shape = known_shape

    def __call__(self, shape=None, latent: th.Tensor = None) -> th.Tensor:
        if shape == None:
            shape = self.known_shape
        return th.Tensor(np.random.normal(0, 1, shape))

    def reset(self):
        pass


def get_colored_noise(beta, known_shape=None):
    if beta == 0:
        return White_Noise(known_shape)
    elif beta == 1:
        return Pink_Noise(known_shape)
    else:
        return Colored_Noise(known_shape, beta=beta)


class SDE_Noise():
    def __init__(self, shape, latent_sde_dim=64, Base_Noise=White_Noise):
        raise Exception('Not implemented yet. Just use SB3s gSDE...')
        self.shape = shape
        self.latent_sde_dim = latent_sde_dim
        self.Base_Noise = Base_Noise

        batch_size = self.shape[0]
        self.weights_dist = self.Base_Noise(
            (self.latent_sde_dim,) + self.shape)
        self.weights_dist_batch = self.Base_Noise(
            (batch_size, self.latent_sde_dim,) + self.shape)

    def sample_weights(self):
        # Reparametrization trick to pass gradients
        self.exploration_mat = self.weights_dist.sample()
        # Pre-compute matrices in case of parallel exploration
        self.exploration_matrices = self.weights_dist_batch.sample()

    def __call__(self, latent: th.Tensor) -> th.Tensor:
        latent_sde = latent.detach()
        latent_sde = latent_sde[..., -self.sde_latent_dim:]
        latent_sde = th.nn.functional.normalize(latent_sde, dim=-1)

        p = self.distribution
        if isinstance(p, th.distributions.Normal) or isinstance(p, th.distributions.Independent):
            chol = th.diag_embed(self.distribution.stddev)
        elif isinstance(p, th.distributions.MultivariateNormal):
            chol = p.scale_tril

        # Default case: only one exploration matrix
        if len(latent_sde) == 1 or len(latent_sde) != len(self.exploration_matrices):
            return (th.mm(latent_sde, self.exploration_mat) @ chol)[0]

        # Use batch matrix multiplication for efficient computation
        # (batch_size, n_features) -> (batch_size, 1, n_features)
        latent_sde = latent_sde.unsqueeze(dim=1)
        # (batch_size, 1, n_actions)
        noise = th.bmm(th.bmm(latent_sde, self.exploration_matrices), chol)
        return noise.squeeze(dim=1)

class Perlin_Noise():
    def __init__(self, known_shape=None, scale=0.1, octave=1):
        self.known_shape = known_shape
        self.scale = scale
        self.octave = octave
        self.magic = PI  # Axis offset, should be (kinda) irrational
        # We want to genrate samples, that approx ~N(0,1)
        self.normal_factor = PI/20
        self.clear_cache_every = 128
        self.reset()

    def __call__(self, shape=None):
        if shape == None:
            shape = self.known_shape
        self.index += 1
        noise = [self.noise([self.index*self.scale, self.magic+(2*a)]) / self.normal_factor
                 for a in range(np.prod(shape))]
        if self.index % self.clear_cache_every == 0:
            self.noise.cache = {}
        return th.Tensor(noise).view(shape)

    def reset(self):
        self.index = 0
        self.noise = PerlinNoise(octaves=self.octave)


class Sync_Perlin_Noise():
    def __init__(self, known_shape=None, scale=0.1, octave=1):
        self.known_shape = known_shape
        self.scale = scale
        self.octave = octave
        self.magic = PI  # Axis offset, should be (kinda) irrational
        # We want to genrate samples, that approx ~N(0,1)
        self.normal_factor = PI/20
        self.clear_cache_every = 128
        self.reset()

    def __call__(self, shape=None):
        if shape == None:
            shape = self.known_shape
        self.index += 1
        noise = [self.noise([self.index*self.scale, self.magic+(2*a)]) / self.normal_factor
                 for a in range(shape[-1])]
        if self.index % self.clear_cache_every == 0:
            self.noise.cache = {}
        return th.Tensor(noise)

    def reset(self):
        self.index = 0
        self.noise = PerlinNoise(octaves=self.octave)

class Harmonic_Perlin_Noise():
    def __init__(self, known_shape=None, scale=0.1, octaves=8):
        self.known_shape = known_shape
        self.scale = scale
        assert octaves >= 1
        if type(octaves) in [int, float]:
            int_octaves = int(octaves)
            octaves_arr = [1/(i+1) for i in range(int_octaves)]
            if int_octaves != octaves:
                octaves_arr += [1/(int_octaves+2)*(octaves-int_octaves)]
        octaves_arr = np.array(octaves_arr)
        self.octaves = octaves_arr / np.linalg.norm(octaves_arr)
        self.clear_cache_every = 1024
        self.reset()

    def __call__(self, shape=None):
        if shape == None:
            shape = self.known_shape
        harmonics = [noise(shape)*self.octaves[i] for i, noise in enumerate(self.noises)]
        if self.index % self.clear_cache_every == 0:
            for i, noise in enumerate(self.noises):
                noise.cache = {}
        return sum(harmonics)

    def reset(self):
        self.index = 0
        self.noises = []
        for octave, amplitude in enumerate(self.octaves):
            self.noises += [Perlin_Noise(known_shape=self.known_shape, scale=self.scale, octave=(octave+1))]


class Dirty_Perlin_Noise():
    def __init__(self, known_shape=None, scale=0.1, dirty_ratio=1/3):
        self.known_shape = known_shape
        self.scale = scale
        self.dirty_ratio = dirty_ratio
        self.reset()

    def __call__(self, shape=None):
        if shape == None:
            shape = self.known_shape
        return self.perlin(shape)*(1-self.dirty_ratio) + self.white(shape)*self.dirty_ratio

    def reset(self):
        self.perlin = Perlin_Noise(known_shape=self.known_shape, scale=self.scale, octave=1)
        self.white = White_Noise(known_shape=self.known_shape)

class Rayleigh_Perlin_Noise(): # Ignore Me (was to lazy to write about it in the paper)
    def __init__(self, known_shape=None, sigma=0.1):
        self.known_shape = known_shape
        self.sigma = sigma
        self.magic = PI  # Axis offset, should be (kinda) irrational
        # We want to genrate samples, that approx ~N(0,1)
        self.normal_factor = PI/20
        self.clear_cache_every = 128
        self.reset()

    def __call__(self, shape=None):
        assert shape == self.known_shape or (shape[1:] == self.known_shape[1:] and shape[0] <= self.known_shape[0])
        self.index += 1
        noise = [self.noise([self.index*self.scales[a%np.prod(self.known_shape[:-1])], self.magic+(2*a)]) / self.normal_factor
                 for a in range(np.prod(shape))]
        if self.index % self.clear_cache_every == 0:
            self.noise.cache = {}
        return th.Tensor(noise).view(shape)

    def reset(self):
        self.index = 0
        self.scales = np.random.rayleigh(scale=self.sigma, size=np.prod(self.known_shape[:-1]))
        self.noise = PerlinNoise(octaves=1)

class Sync_Rayleigh_Perlin_Noise():
    def __init__(self, known_shape=None, sigma=0.1):
        self.known_shape = known_shape
        self.sigma = sigma
        self.magic = PI  # Axis offset, should be (kinda) irrational
        # We want to genrate samples, that approx ~N(0,1)
        self.normal_factor = PI/20
        self.clear_cache_every = 128
        self.reset()

    def __call__(self, shape=None):
        if shape == None:
            shape = self.known_shape
        self.index += 1
        noise = [self.noise([self.index*self.scale, self.magic+(2*a)]) / self.normal_factor
                 for a in range(shape[-1])]
        if self.index % self.clear_cache_every == 0:
            self.noise.cache = {}
        return th.Tensor(noise)

    def reset(self):
        self.index = 0
        self.scale = np.random.rayleigh(scale=self.sigma, size=(1,))[0]
        self.noise = PerlinNoise(octaves=1)

