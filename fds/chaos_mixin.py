"""
chaos_mixin.py

FDS perturbation logic aligned with the paper.

The main reproduction path uses the shuffled-Lorenz driver. This shared mixin
also keeps driver sources needed by the public ablation and control scripts.

Implementation notes:
1. _sl_init builds a burn-in shuffled-Lorenz state pool.
2. _sl_noise samples the pool i.i.d. and shuffles spatial positions.
3. _chaos_apply_kick records c_k = u_k - u_tilde_k before and after clipping.
4. get_clip_stats() exposes clipping-error statistics.

Theory mapping in thEORY.tex:
  u_k = sqrt(h) * tau_k * sigma(t_k) * xi_k
      code uses kick = tau * base_scale * noise.
  u_tilde_k = Clip(u_k)
      ratio_cap-clipped kick.
  c_k = u_k - u_tilde_k
      clipping error recorded by this mixin.
  sigma(t_k) = base_scale * I
      scalar simplification for the d by d identity case.
"""
from typing import Optional
import numpy as np
import torch

# Burn-in steps for the shuffled-Lorenz pool.
_LORENZ_BURNIN = 10000
_SHUFFLED_LORENZ_POOL_SIZE = 500_000


class ChaosMixin:
    def set_chaos_params(
        self,
        tau0: float = 0.3,
        beta: float = 0.01,
        p_start: float = 0.3,
        p_end: float = 0.7,
        seed: int = 0,
        ratio_cap: float = 0.09,
        ratio_cap_end: Optional[float] = 0.045,
        xi_source: str = "shuffled_lorenz",
        lorenz_sigma: float = 10.0,
        lorenz_rho: float = 28.0,
        lorenz_beta: float = 8.0 / 3.0,
        lorenz_dt: float = 0.005,
        lorenz_burnin: int = _LORENZ_BURNIN,
    ) -> None:
        self.chaos_tau0 = float(tau0)
        self.chaos_beta = float(beta)
        self.chaos_p_start = float(p_start)
        self.chaos_p_end = float(p_end)
        self.chaos_seed = int(seed)
        self.chaos_ratio_cap = float(ratio_cap)
        self.chaos_ratio_cap_end = None if ratio_cap_end is None else float(ratio_cap_end)
        self.chaos_xi_source = str(xi_source)
        allowed_sources = {"shuffled_lorenz", "lorenz", "gaussian", "rossler"}
        if self.chaos_xi_source not in allowed_sources:
            raise ValueError(f"Unknown xi_source={self.chaos_xi_source!r}.")
        self.chaos_lorenz_sigma = float(lorenz_sigma)
        self.chaos_lorenz_rho = float(lorenz_rho)
        self.chaos_lorenz_beta = float(lorenz_beta)
        self.chaos_lorenz_dt = float(lorenz_dt)
        self.chaos_lorenz_burnin = int(lorenz_burnin)
        self._shuffle_counter = 0  # Per-sample and per-channel shuffle counter.
        self._pool_size = 2_000_000
        self._lorenz_pool = None
        self._lorenz_pool_pos = 0
        # Shuffled-Lorenz state pool with i.i.d. random sampling.
        self._sl_pool = None
        self._sl_rng = None
        self._lorenz_state = self._chaos_init_lorenz(int(seed))
        if self.chaos_xi_source == "shuffled_lorenz":
            self._sl_init(int(seed))
        # C_n(t) statistics: per-step clipping error norm.
        self._clip_stats: list[float] = []

    def _sl_init(self, seed: int):
        """Initialize the shuffled-Lorenz state pool with 500k states."""
        _SL_POOL_SIZE = 500_000
        sigma_L, rho_L, beta_L = 10.0, 28.0, 8.0 / 3.0
        dt = 0.005
        y = np.array([float((seed % 100) + 1) / 100.0 + 0.1, 0.0, 0.0])
        for _ in range(10000):  # burn-in
            dx = sigma_L * (y[1] - y[0])
            dy = y[0] * (rho_L - y[2]) - y[1]
            dz = y[0] * y[1] - beta_L * y[2]
            y = y + dt * np.array([dx, dy, dz])
        pool = np.zeros((_SL_POOL_SIZE, 3), dtype=np.float32)
        for i in range(_SL_POOL_SIZE):
            dx = sigma_L * (y[1] - y[0])
            dy = y[0] * (rho_L - y[2]) - y[1]
            dz = y[0] * y[1] - beta_L * y[2]
            y = y + dt * np.array([dx, dy, dz])
            pool[i] = y
        self._sl_pool = pool
        self._sl_pool_size = _SL_POOL_SIZE
        self._sl_rng = np.random.RandomState((seed * 7 + 13) % 100000)

    def _sl_noise(self, shape):
        """Sample i.i.d. noise from the shuffled-Lorenz pool."""
        numel = 1
        for s in shape:
            numel *= s
        n_states = (numel + 2) // 3
        idx = self._sl_rng.randint(0, self._sl_pool_size, size=n_states)
        states = self._sl_pool[idx]
        values = states.flatten()[:numel]
        noise = torch.tensor(values, dtype=torch.float32).view(shape)
        if noise.ndim == 4:
            B, channels, H, W = noise.shape
            for b in range(B):
                for c in range(channels):
                    gen = torch.Generator()
                    gen.manual_seed(self._shuffle_counter * 1000 + b * channels + c)
                    perm = torch.randperm(H * W, generator=gen)
                    noise[b, c] = noise[b, c].reshape(-1)[perm].reshape(H, W)
        self._shuffle_counter += 1
        return noise

    def _chaos_init_lorenz(self, seed: int):
        """Initialize the sequential Lorenz driver after burn-in."""
        base = float((seed % 10000) + 1) / 10000.0
        state = (0.1 + base, 0.0 + base * 0.5, 0.0 - base * 0.25)
        burnin = int(getattr(self, "chaos_lorenz_burnin", _LORENZ_BURNIN))
        for _ in range(burnin):
            state = self._chaos_lorenz_step(state)
        return state

    def _chaos_lorenz_deriv(self, state):
        x, y, z = state
        dx = self.chaos_lorenz_sigma * (y - x)
        dy = x * (self.chaos_lorenz_rho - z) - y
        dz = x * y - self.chaos_lorenz_beta * z
        return (dx, dy, dz)

    def _chaos_lorenz_step(self, state):
        dt = self.chaos_lorenz_dt
        dx, dy, dz = self._chaos_lorenz_deriv(state)
        return (state[0] + dx * dt, state[1] + dy * dt, state[2] + dz * dt)

    def _chaos_lorenz_noise(self, shape):
        """Generate sequential Lorenz states and apply the same spatial shuffle."""
        numel = 1
        for s in shape:
            numel *= s
        n_steps = (numel + 2) // 3 + 1
        values = []
        state = self._lorenz_state
        for _ in range(n_steps):
            state = self._chaos_lorenz_step(state)
            values.extend([state[0], state[1], state[2]])
        self._lorenz_state = state

        noise = torch.tensor(values[:numel], dtype=torch.float32).view(shape)
        if noise.ndim == 4:
            B, channels, H, W = noise.shape
            for b in range(B):
                for c in range(channels):
                    gen = torch.Generator()
                    gen.manual_seed(self._shuffle_counter * 1000 + b * channels + c)
                    perm = torch.randperm(H * W, generator=gen)
                    noise[b, c] = noise[b, c].reshape(-1)[perm].reshape(H, W)
        self._shuffle_counter += 1
        return noise

    def get_clip_stats(self) -> list[float]:
        """
        Return per-step clipping error norms [||c_0||, ||c_1||, ...].
        Used to validate Proposition 1.8: sup_t ||C_n(t)|| ->^P 0.
        """
        return list(self._clip_stats)

    def reset_clip_stats(self) -> None:
        self._clip_stats = []

    def set_timesteps(self, num_inference_steps, device=None):
        """Wrap set_timesteps and keep alphas_cumprod as a NumPy array."""
        if hasattr(self, "alphas_cumprod") and torch.is_tensor(self.alphas_cumprod):
            self.alphas_cumprod = self.alphas_cumprod.detach().cpu().float().numpy()
        if device is not None:
            super().set_timesteps(num_inference_steps, device=device)
        else:
            super().set_timesteps(num_inference_steps)
        # Parent schedulers may convert alphas_cumprod back to a torch tensor.
        if hasattr(self, "alphas_cumprod") and torch.is_tensor(self.alphas_cumprod):
            self.alphas_cumprod = self.alphas_cumprod.detach().cpu().float().numpy()

    def _chaos_apply_kick(
        self,
        sample: torch.Tensor,
        sigma_t: float,
        sigma_next: float,
        step_index: int,
        num_steps: int,
    ) -> torch.Tensor:
        if not hasattr(self, "chaos_tau0") or self.chaos_tau0 <= 0.0:
            return sample

        i = step_index + 1
        progress = i / float(num_steps)

        if progress < self.chaos_p_start or progress > self.chaos_p_end:
            return sample

        sigma_t = float(sigma_t)
        sigma_next = float(sigma_next)
        if sigma_next <= 0:
            return sample

        # base_scale corresponds to sigma(t_k) = base_scale * I.
        alpha_t = 1.0 / (1.0 + sigma_t ** 2) ** 0.5
        alpha_next = 1.0 / (1.0 + sigma_next ** 2) ** 0.5
        lambda_t = torch.log(torch.tensor(alpha_t / max(sigma_t, 1e-12)))
        lambda_next = torch.log(torch.tensor(alpha_next / max(sigma_next, 1e-12)))
        h = max(float(lambda_next - lambda_t), 0.0)
        base_scale = sigma_next * (max(1.0 - torch.exp(torch.tensor(-2.0 * h)).item(), 0.0)) ** 0.5
        if base_scale <= 0:
            return sample

        tau = self.chaos_tau0 * ((1.0 - self.chaos_beta) ** (i - 1))
        if tau <= 0:
            return sample

        if self.chaos_xi_source == "shuffled_lorenz":
            noise = self._sl_noise(sample.shape).to(device=sample.device, dtype=torch.float32)
        elif self.chaos_xi_source == "lorenz":
            noise = self._chaos_lorenz_noise(sample.shape).to(device=sample.device, dtype=torch.float32)
        elif self.chaos_xi_source == "gaussian":
            seed = self.chaos_seed + i * 10007
            gen = torch.Generator(device=sample.device).manual_seed(seed % (1 << 32))
            noise = torch.randn(sample.shape, generator=gen, device=sample.device, dtype=torch.float32)
        else:
            raise RuntimeError(f"xi_source={self.chaos_xi_source!r} requires a source-specific scheduler.")
        dims = tuple(range(1, noise.ndim))
        noise = noise - noise.mean(dim=dims, keepdim=True)
        rms = torch.sqrt((noise ** 2).mean(dim=dims, keepdim=True) + 1e-8)
        noise = noise / rms

        # u_k before clipping.
        kick_raw = (tau * base_scale) * noise

        # ratio_cap clipping: u_tilde_k = Clip(u_k), Definition 1.4.
        kick = kick_raw
        if self.chaos_ratio_cap > 0.0:
            p_start = self.chaos_p_start
            p_end = self.chaos_p_end
            gate_span = max(p_end - p_start, 1e-6)
            t = max(0.0, min(1.0, (progress - p_start) / gate_span))
            if self.chaos_ratio_cap_end is not None:
                ratio_cap_eff = self.chaos_ratio_cap + (self.chaos_ratio_cap_end - self.chaos_ratio_cap) * t
            else:
                ratio_cap_eff = self.chaos_ratio_cap

            dims = tuple(range(1, sample.ndim))
            kick_norm = torch.sqrt((kick_raw.float() ** 2).mean(dim=dims))
            sample_norm = torch.sqrt((sample.float() ** 2).mean(dim=dims)) + 1e-12
            ratio = kick_norm / sample_norm
            scale = torch.clamp(ratio_cap_eff / ratio, max=1.0)
            kick = kick_raw * scale.view(-1, *([1] * (kick_raw.ndim - 1)))

        # Record clipping error c_k = u_k - u_tilde_k for Definition 1.7.
        clip_err = kick_raw - kick
        dims = tuple(range(1, clip_err.ndim))
        clip_err_norm = float(torch.sqrt((clip_err.float() ** 2).mean(dim=dims)).mean().item())
        self._clip_stats.append(clip_err_norm)

        return sample + kick.to(dtype=sample.dtype)


class P1ShuffledLorenzMixin(ChaosMixin):
    """P1-compatible shuffled-Lorenz driver.

    The public DiT FDS path uses this mixin so that the main release code and
    the P1 reproduction code share the same driver implementation. Public
    callers should request ``xi_source="shuffled_lorenz"``. Internally this
    mixin routes that request through the scheduler's Lorenz hook, which is
    overridden here with i.i.d. samples from a burn-in shuffled-Lorenz pool.
    """

    def set_chaos_params(self, *args, **kwargs) -> None:
        public_xi_source = kwargs.get("xi_source", "shuffled_lorenz")
        if public_xi_source == "shuffled_lorenz":
            kwargs["xi_source"] = "lorenz"
        super().set_chaos_params(*args, **kwargs)
        self.chaos_public_xi_source = public_xi_source

    def _chaos_init_lorenz(self, seed: int):
        sigma_l, rho_l, beta_l = 10.0, 28.0, 8.0 / 3.0
        dt = 0.005
        y = np.array([float((seed % 100) + 1) / 100.0 + 0.1, 0.0, 0.0])
        for _ in range(10000):
            dx = sigma_l * (y[1] - y[0])
            dy = y[0] * (rho_l - y[2]) - y[1]
            dz = y[0] * y[1] - beta_l * y[2]
            y = y + dt * np.array([dx, dy, dz])

        pool = np.zeros((_SHUFFLED_LORENZ_POOL_SIZE, 3), dtype=np.float32)
        for i in range(_SHUFFLED_LORENZ_POOL_SIZE):
            dx = sigma_l * (y[1] - y[0])
            dy = y[0] * (rho_l - y[2]) - y[1]
            dz = y[0] * y[1] - beta_l * y[2]
            y = y + dt * np.array([dx, dy, dz])
            pool[i] = y

        self._lorenz_pool = pool
        self._pool_rng = np.random.RandomState((seed * 7 + 13) % 100000)
        return (0.0, 0.0, 0.0)

    def _chaos_lorenz_noise(self, shape):
        numel = 1
        for s in shape:
            numel *= s
        n_states = (numel + 2) // 3
        idx = self._pool_rng.randint(0, _SHUFFLED_LORENZ_POOL_SIZE, size=n_states)
        values = self._lorenz_pool[idx].flatten()[:numel]
        noise = torch.tensor(values, dtype=torch.float32).view(shape)

        if noise.ndim == 4:
            _, channels, height, width = noise.shape
            for c in range(channels):
                gen = torch.Generator()
                gen.manual_seed(self._shuffle_counter * 1000 + c)
                perm = torch.randperm(height * width, generator=gen)
                noise[0, c] = noise[0, c].reshape(-1)[perm].reshape(height, width)

        self._shuffle_counter += 1
        return noise
