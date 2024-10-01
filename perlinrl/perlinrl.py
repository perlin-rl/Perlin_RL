from enum import Enum
import numpy as np
import torch as th
import scipy.spatial
from torch import nn
from stable_baselines3.common.distributions import Distribution as SB3_Distribution
from stable_baselines3.common.distributions import sum_independent_dims
from torch.distributions import Normal, MultivariateNormal
import torch.nn.functional as F
from perlinrl import noise
from perlinrl.tensor_ops import fill_triangular, fill_triangular_inverse


class Par_Strength(Enum):
    SCALAR = 'SCALAR'
    DIAG = 'DIAG'
    FULL = 'FULL'
    CONT_SCALAR = 'CONT_SCALAR'
    CONT_DIAG = 'CONT_DIAG'
    CONT_HYBRID = 'CONT_HYBRID'
    CONT_FULL = 'CONT_FULL'

class EnforcePositiveType(Enum):
    # This need to be implemented in this ugly fashion,
    # because cloudpickle does not like more complex enums

    NONE = 0
    SOFTPLUS = 1
    ABS = 2
    RELU = 3
    LOG = 4

    def apply(self, x):
        # aaaaaa
        return [nn.Identity(), nn.Softplus(beta=10, threshold=2), th.abs, nn.ReLU(inplace=False), th.log][self.value](x)

class Avaible_Noise_Funcs(Enum):
    WHITE = 0
    PINK = 1
    COLOR = 2
    PERLIN = 3
    HARMONICPERLIN = 4
    DIRTYPERLIN = 5
    SDE = 6
    SHORTPINK = 7
    SYNCPERLIN = 8
    RAYLEIGHPERLIN = 9


    def get_func(self):
        # stil aaaaaaaa
        return [noise.White_Noise, noise.Pink_Noise, noise.Colored_Noise, noise.Perlin_Noise, noise.Harmonic_Perlin_Noise, noise.Dirty_Perlin_Noise, noise.SDE_Noise, noise.shortPink_Noise, noise.Sync_Perlin_Noise, noise.Rayleigh_Perlin_Noise][self.value]

def cast_to_enum(inp, Class):
    if isinstance(inp, Enum):
        return inp
    else:
        return Class[inp]

def cast_to_Noise(Inp, known_shape):
    if callable(Inp):  # TODO: Allow instantiated?
        return Inp(known_shape)
    else:
        func, *pars = Inp.split('_')
        pars = [float(par) for par in pars]
        return Avaible_Noise_Funcs[func].get_func()(known_shape, *pars)

class PerlinRL_Distribution(SB3_Distribution):
    def __init__(
        self,
        action_dim: int,
        n_envs: int=1,
        par_strength: Par_Strength = Par_Strength.CONT_DIAG,
        init_std: float = 1,
        epsilon: float = 1e-6,
        msqrt_induces_full: bool = False, # IGNORE ME (for precise trust region stuff)
        Base_Noise=noise.White_Noise,
    ):
        super().__init__()

        self.action_dim = action_dim
        self.par_strength = cast_to_enum(par_strength, Par_Strength)
        self.init_std = init_std
        self.epsilon = epsilon
        self.msqrt_induces_full = msqrt_induces_full

        self.base_noise = cast_to_Noise(Base_Noise, (n_envs, action_dim))

    def proba_distribution_net(self, latent_dim: int, return_log_std: bool = False):
        mu_net = nn.Linear(latent_dim, self.action_dim)
        std_net = StdNet(latent_dim, self.action_dim, self.init_std, self.par_strength, self.epsilon, return_log_std)

        return mu_net, std_net

    def proba_distribution(
            self, mean_actions: th.Tensor, std_actions: th.Tensor) -> SB3_Distribution:
        if self.is_full():
            self.distribution = MultivariateNormal(mean_actions, scale_tril=std_actions, validate_args=not self.msqrt_induces_full)
            self.distribution._mark_mSqrt = self.msqrt_induces_full
        else:
            self.distribution = Normal(
                mean_actions, std_actions)
        return self

    def log_prob(self, actions: th.Tensor) -> th.Tensor:
        return self._log_prob(actions, self.distribution)

    def _log_prob(self, actions: th.Tensor, dist: Normal):
        return sum_independent_dims(dist.log_prob(actions.to(dist.mean.device)))

    def entropy(self) -> th.Tensor:
        return sum_independent_dims(self.distribution.entropy())

    def get_actions(self, deterministic: bool = False) -> th.Tensor:
        """
        Return actions according to the probability distribution.

        :param deterministic:
        :return:
        """
        if deterministic:
            return self.mode()
        return self.sample()

    def sample(self, epsilon=None) -> th.Tensor:
        pi_mean, pi_decomp = self.distribution.mean.cpu(), self.distribution.scale_tril.cpu() if self.is_full() else self.distribution.scale.cpu()
        if epsilon == None:
            epsilon = self.base_noise(pi_mean.shape)
        eta =  epsilon.detach()
        # reparameterization with rigged samples
        if self.is_full():
            actions = pi_mean + th.matmul(pi_decomp, eta.unsqueeze(-1)).squeeze(-1)
        else:
            actions = pi_mean + pi_decomp * eta

        self.gaussian_actions = actions # SB3 quirk
        return actions

    def is_contextual(self):
        return self.par_strength in [Par_Strength.CONT_SCALAR, Par_Strength.CONT_DIAG, Par_Strength.CONT_HYBRID, Par_Strength.CONT_FULL]

    def is_full(self):
        return self.par_strength in [Par_Strength.FULL, Par_Strength.CONT_FULL]

    def mode(self) -> th.Tensor:
        return self.distribution.mean

    def actions_from_params(self, mean: th.Tensor, std: th.Tensor, deterministic: bool = False) -> th.Tensor:
        self.proba_distribution(mean, std)
        return self.get_actions(deterministic=deterministic)

    def log_prob_from_params(self, mean: th.Tensor, std: th.Tensor):
        actions = self.actions_from_params(mean, std)
        log_prob = self.log_prob(actions)
        return actions, log_prob

class StdNet(nn.Module):
    def __init__(self, latent_dim: int, action_dim: int, std_init: float, par_strength: bool, epsilon: float, return_log_std: bool):
        super().__init__()
        self.action_dim = action_dim
        self.latent_dim = latent_dim
        self.std_init = std_init
        self.par_strength = par_strength
        self.enforce_positive_type = EnforcePositiveType.SOFTPLUS

        self.epsilon = epsilon
        self.return_log_std = return_log_std

        if self.par_strength == Par_Strength.SCALAR:
            self.param = nn.Parameter(
                th.Tensor([std_init]), requires_grad=True)
        elif self.par_strength == Par_Strength.DIAG:
            self.param = nn.Parameter(
                th.Tensor(th.ones(action_dim)*std_init), requires_grad=True)
        elif self.par_strength == Par_Strength.FULL:
            ident = th.eye(action_dim)*std_init
            ident_chol = fill_triangular_inverse(ident)
            self.param = nn.Parameter(
                th.Tensor(ident_chol), requires_grad=True)
        elif self.par_strength == Par_Strength.CONT_SCALAR:
            self.net = nn.Linear(latent_dim, 1)
        elif self.par_strength == Par_Strength.CONT_HYBRID:
            self.net = nn.Linear(latent_dim, 1)
            self.param = nn.Parameter(
                th.Tensor(th.ones(action_dim)*std_init), requires_grad=True)
        elif self.par_strength == Par_Strength.CONT_DIAG:
            self.net = nn.Linear(latent_dim, self.action_dim)
            self.bias = th.ones(action_dim)*self.std_init
        elif self.par_strength == Par_Strength.CONT_FULL:
            self.net = nn.Linear(latent_dim, action_dim * (action_dim + 1) // 2)
            self.bias = fill_triangular_inverse(th.eye(action_dim)*self.std_init)


    def forward(self, x: th.Tensor) -> th.Tensor:
        if self.par_strength == Par_Strength.SCALAR:
            return self._ensure_positive_func(
                th.ones(self.action_dim) * self.param[0])
        elif self.par_strength == Par_Strength.DIAG:
            return self._ensure_positive_func(self.param)
        elif self.par_strength == Par_Strength.FULL:
            return self._chol_from_flat(self.param)
        elif self.par_strength == Par_Strength.CONT_SCALAR:
            cont = self.net(x)
            diag_chol = th.ones(self.action_dim, device=cont.device) * cont * self.std_init
            return self._ensure_positive_func(diag_chol)
        elif self.par_strength == Par_Strength.CONT_HYBRID:
            cont = self.net(x)
            return self._ensure_positive_func(self.param * cont)
        elif self.par_strength == Par_Strength.CONT_DIAG:
            cont = self.net(x)
            bias = self.bias.to(cont.device)
            diag_chol = cont + bias
            return self._ensure_positive_func(diag_chol)
        elif self.par_strength == Par_Strength.CONT_FULL:
            cont = self.net(x)
            bias = self.bias.to(device=cont.device)
            return self._chol_from_flat(cont + bias)

        raise Exception()

    def _ensure_positive_func(self, x):
        return self.enforce_positive_type.apply(x) + self.epsilon

    def _chol_from_flat(self, flat_chol):
        chol = fill_triangular(flat_chol)
        return self._ensure_diagonal_positive(chol)

    def _ensure_diagonal_positive(self, chol):
        if len(chol.shape) == 1:
            # If our chol is a vector (representing a diagonal chol)
            return self._ensure_positive_func(chol)
        return chol.tril(-1) + self._ensure_positive_func(chol.diagonal(dim1=-2,
                                                                        dim2=-1)).diag_embed() + chol.triu(1)
    def string(self):
        return '<StdNet />'
