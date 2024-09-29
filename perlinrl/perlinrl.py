from enum import Enum
import numpy as np
import torch as th
import scipy.spatial
from torch import nn
from stable_baselines3.common.distributions import Distribution as SB3_Distribution
from stable_baselines3.common.distributions import sum_independent_dims
from torch.distributions import Normal, MultivariateNormal
import torch.nn.functional as F
from perlinrl import noise, kernel
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


class Avaible_Kernel_Funcs(Enum):
    RBF = 0
    SE = 1
    BROWN = 2
    PINK = 3

    def get_func(self):
        # stil aaaaaaaa
        return [kernel.rbf, kernel.se, kernel.brown, kernel.pink][self.value]


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


def cast_to_kernel(inp):
    if callable(inp):
        return inp
    else:
        func, *pars = inp.split('_')
        pars = [float(par) for par in pars]
        return Avaible_Kernel_Funcs[func].get_func()(*pars)


def cast_to_Noise(Inp, known_shape):
    if callable(Inp):  # TODO: Allow instantiated?
        return Inp(known_shape)
    else:
        func, *pars = Inp.split('_')
        pars = [float(par) for par in pars]
        return Avaible_Noise_Funcs[func].get_func()(known_shape, *pars)

# NOTE:
# Please ignore anything related to 'conditioning' or 'pca', it is not related to the Perlin RL paper.

class PerlinRL_Distribution(SB3_Distribution):
    def __init__(
        self,
        action_dim: int,
        n_envs: int=1,
        par_strength: Par_Strength = Par_Strength.CONT_DIAG,
        kernel_func=kernel.rbf(), # IGNORE ME
        init_std: float = 1,
        cond_noise: float = 0, # IGNORE ME
        window: int = 64, # IGNORE ME
        epsilon: float = 1e-6,
        skip_conditioning: bool = True, # IGNORE ME
        temporal_gradient_emission: bool = False, # IGNORE ME
        msqrt_induces_full: bool = False, # IGNORE ME (for precise trust region stuff)
        Base_Noise=noise.White_Noise,
    ):
        super().__init__()

        self.action_dim = action_dim
        self.kernel_func = cast_to_kernel(kernel_func)
        self.par_strength = cast_to_enum(par_strength, Par_Strength)
        self.init_std = init_std
        self.cond_noise = cond_noise
        self.window = window
        self.epsilon = epsilon
        self.skip_conditioning = skip_conditioning
        self.temporal_gradient_emission = temporal_gradient_emission
        self.msqrt_induces_full = msqrt_induces_full

        self.base_noise = cast_to_Noise(Base_Noise, (n_envs, action_dim))

        assert not (not skip_conditioning and self.is_full()), 'Conditioning full Covariances not yet implemented'

        # Premature optimization is the root of all evil
        self._build_conditioner()
        # *Optimizes it anyways*

    def proba_distribution_net(self, latent_dim: int, return_log_std: bool = False):
        mu_net = nn.Linear(latent_dim, self.action_dim)
        std_net = StdNet(latent_dim, self.action_dim, self.init_std, self.par_strength, self.epsilon, return_log_std)

        return mu_net, std_net

    def proba_distribution(
            self, mean_actions: th.Tensor, std_actions: th.Tensor) -> SB3_Distribution:
        if self.is_full():
            self.distribution = MultivariateNormal(mean_actions, scale_tril=std_actions, validate_args=not self.msqrt_induces_full)
            #self.distribution.scale = th.diagonal(std_actions, dim1=-2, dim2=-1)
            self.distribution._mark_mSqrt = self.msqrt_induces_full
        else:
            self.distribution = Normal(
                mean_actions, std_actions)
        return self

    def log_prob(self, actions: th.Tensor) -> th.Tensor:
        return self._log_prob(actions, self.distribution)

    def conditioned_log_prob(self, actions: th.Tensor, trajectory: th.Tensor = None) -> th.Tensor:
        pi_mean, pi_std = self.distribution.mean.cpu(), self.distribution.scale.cpu()
        rho_mean, rho_std = self._conditioning_engine(trajectory, pi_mean, pi_std)
        dist = Normal(rho_mean, rho_std)
        return self._log_prob(dist)

    def _log_prob(self, actions: th.Tensor, dist: Normal):
        return sum_independent_dims(dist.log_prob(actions.to(dist.mean.device)))

    def entropy(self) -> th.Tensor:
        return sum_independent_dims(self.distribution.entropy())

    def get_actions(self, deterministic: bool = False, trajectory: th.Tensor = None) -> th.Tensor:
        """
        Return actions according to the probability distribution.

        :param deterministic:
        :return:
        """
        if deterministic:
            return self.mode()
        return self.sample(traj=trajectory)

    def sample(self, traj: th.Tensor, f_sigma: float = 1.0, epsilon=None) -> th.Tensor:
        assert self.skip_conditioning or type(traj) != type(None), 'A past trajectory has to be supplied if conditinoning is performed'
        pi_mean, pi_decomp = self.distribution.mean.cpu(), self.distribution.scale_tril.cpu() if self.is_full() else self.distribution.scale.cpu()
        rho_mean, rho_std = self._conditioning_engine(traj, pi_mean, pi_decomp)
        rho_std = rho_std * f_sigma
        eta = self._get_rigged(pi_mean, pi_decomp,
                               rho_mean, rho_std,
                               epsilon)
        # reparameterization with rigged samples
        if self.is_full():
            actions = pi_mean + th.matmul(pi_decomp, eta.unsqueeze(-1)).squeeze(-1)
        else:
            actions = pi_mean + pi_decomp * eta

        self.gaussian_actions = actions
        return actions

    def is_contextual(self):
        return self.par_strength in [Par_Strength.CONT_SCALAR, Par_Strength.CONT_DIAG, Par_Strength.CONT_HYBRID, Par_Strength.CONT_FULL]

    def is_full(self):
        return self.par_strength in [Par_Strength.FULL, Par_Strength.CONT_FULL]


    def _get_rigged(self, pi_mean, pi_std, rho_mean, rho_std, epsilon=None):
        # Ugly function to ensure, that the gradients flow as intended for each modus operandi
        if not self.temporal_gradient_emission or self.skip_conditioning:
            with th.no_grad():
                return self._get_emitting_rigged(pi_mean, pi_std, rho_mean, rho_std, epsilon=epsilon).detach()
        return self._get_emitting_rigged(pi_mean.detach(), pi_std.detach(), rho_mean, rho_std, epsilon=epsilon)

    def _get_emitting_rigged(self, pi_mean, pi_std, rho_mean, rho_std, epsilon=None):
        if epsilon == None:
            epsilon = self.base_noise(pi_mean.shape)

        if self.skip_conditioning:
            return epsilon.detach()

        Delta = rho_mean - pi_mean
        Pi_mu = 1 / pi_std
        Pi_sigma = rho_std / pi_std

        eta = Pi_mu * Delta + Pi_sigma * epsilon

        return eta

    def _pad_and_cut_trajectory(self, traj, value=0):
        if traj.shape[-2] < self.window:
            if traj.shape[-2] == 0:
                shape = list(traj.shape)
                shape[-2] = 1
                traj = th.ones(shape)*value
            missing = self.window - traj.shape[-2]
            return F.pad(input=traj, pad=(0, 0, missing, 0, 0, 0), value=value)
        return traj[:, -self.window:, :]

    def _conditioning_engine(self, trajectory, pi_mean, pi_std):
        if self.skip_conditioning:
            return pi_mean, pi_std

        traj = self._pad_and_cut_trajectory(trajectory)
        y = th.cat((traj.transpose(-1, -2), pi_mean.unsqueeze(-1).unsqueeze(0).repeat(traj.shape[0], 1, traj.shape[-2])), dim=1)

        with th.no_grad():
            conditioners = th.Tensor(self._adapt_conditioner(pi_std))

            S = th.cholesky_solve(self.Sig12.expand(conditioners.shape[:-1]).unsqueeze(-1), conditioners).squeeze(-1)

            rho_std = self.Sig22 - (S @ self.Sig12)
        rho_mean = th.einsum('bai,bai->ba', S, y)

        return rho_mean, rho_std

    def _build_conditioner(self):
        # Precomputes the Cholesky decomp of the cov matrix to be used as a pseudoinverse.
        # Also precomputes some auxilary stuff for _adapt_conditioner.
        w = self.window
        Z = np.linspace(0, w, w+1).reshape(-1, 1)
        X = np.array([w]).reshape(-1, 1)

        Sig11 = self.kernel_func(
            Z, Z) + np.diag(np.hstack((np.repeat(self.cond_noise**2, w), 0)))
        self.Sig12 = th.Tensor(self.kernel_func(Z, X)).squeeze(-1)
        self.Sig22 = th.Tensor(self.kernel_func(
            X, X)).squeeze(-1).squeeze(-1)
        self.conditioner = np.linalg.cholesky(Sig11)
        self.adapt_norm = np.linalg.norm(
            self.conditioner[-1, :][:-1], axis=-1)**2

    def _adapt_conditioner(self, pi_std):
        # We can not actually precompute the cov inverse completely,
        # since it also depends on the current policies sigma.
        # But, because of the way the Cholesky Decomp works,
        # we can use the precomputed L (conditioner)
        # (which is computed by an efficient LAPACK implementation)
        # and adapt it for our new k(x_w+1,x_w+1) value (in python)
        # (Which is dependent on pi)
        # S_{ij} = \frac{1}{D_j} \left( A_{ij} - \sum_{k=1}^{j-1} S_{ik} S_{jk} D_k \right), \qquad\text{for } i>j
        # https://martin-thoma.com/images/2012/07/cholesky-zerlegung-numerik.png
        # This way conditioning of the GP can be done in O(dim(A)) time.
        if not self.is_contextual() and False:
            # Always assuming contextual will merely waste cpu cycles
            # TODO: fix, this does not work
            # safe inplace
            self.conditioner[-1, -
                             1] = np.sqrt(pi_std**2 + self.Sig22 - self.adapt_norm)
            return np.expand_dims(np.expand_dims(self.conditioner, 0), 0)
        else:
            conditioner = np.zeros(
                (pi_std.shape[0], pi_std.shape[1]) + self.conditioner.shape)
            conditioner[:, :] = self.conditioner
            conditioner[:, :, -1, -
                        1] = np.sqrt(pi_std**2 + self.Sig22 - self.adapt_norm)
            return conditioner

    def mode(self) -> th.Tensor:
        return self.distribution.mean

    def actions_from_params(
        self, mean: th.Tensor, std: th.Tensor, deterministic: bool = False, trajectory: th.Tensor = None
    ) -> th.Tensor:
        self.proba_distribution(mean, std)
        return self.get_actions(deterministic=deterministic, trajectory=trajectory)

    def log_prob_from_params(self, mean: th.Tensor, std: th.Tensor, trajectory: th.Tensor = None):
        actions = self.actions_from_params(mean, std, trajectory=trajectory)
        log_prob = self.log_prob(actions)
        return actions, log_prob

    def print_info(self, traj: th.Tensor):
        pi_mean, pi_std = self.distribution.mean, self.distribution.scale,
        rho_mean, rho_std = self._conditioning_engine(traj, pi_mean, pi_std)
        eta = self._get_rigged(pi_mean, pi_std,
                               rho_mean, rho_std)
        print('pi  ~ N('+str(pi_mean)+','+str(pi_std)+')')
        print('rho ~ N('+str(rho_mean)+','+str(rho_std)+')')


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
