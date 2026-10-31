"""
schedulers.py

FDS scheduler wrappers for DPM-Solver++ and DDIM.
Both wrappers share the same shuffled-Lorenz FDS implementation in
``chaos_mixin.py``.
"""
from typing import Optional, Union

import torch
from diffusers import DDIMScheduler, DPMSolverMultistepScheduler
from diffusers.schedulers.scheduling_utils import SchedulerOutput

from .chaos_mixin import ChaosMixin, P1ShuffledLorenzMixin


class ChaosDPMSolverScheduler(ChaosMixin, DPMSolverMultistepScheduler):
    """DPM-Solver++ scheduler with generic FDS driver support."""

    def step(
        self,
        model_output: torch.Tensor,
        timestep: torch.Tensor,
        sample: torch.Tensor,
        generator: Optional[torch.Generator] = None,
        return_dict: bool = True,
    ) -> SchedulerOutput:
        if self._step_index is None:
            self._init_step_index(timestep)
        step_index = self._step_index
        num_steps = len(self.timesteps)

        sigma_t = float(self.sigmas[step_index]) if step_index < len(self.sigmas) else 1.0
        sigma_next = float(self.sigmas[step_index + 1]) if step_index + 1 < len(self.sigmas) else 0.0

        out = super().step(
            model_output=model_output,
            timestep=timestep,
            sample=sample,
            generator=generator,
            return_dict=return_dict,
        )
        prev_sample = out.prev_sample if return_dict else out[0]
        prev_sample = self._chaos_apply_kick(prev_sample, sigma_t, sigma_next, step_index, num_steps)

        if return_dict:
            return SchedulerOutput(prev_sample=prev_sample)
        return (prev_sample,)


class P1ShuffledLorenzDPMSolverScheduler(P1ShuffledLorenzMixin, DPMSolverMultistepScheduler):
    """DPM-Solver++ FDS scheduler using the P1 shuffled-Lorenz implementation."""

    def step(
        self,
        model_output: torch.Tensor,
        timestep: torch.Tensor,
        sample: torch.Tensor,
        generator: Optional[torch.Generator] = None,
        return_dict: bool = True,
    ) -> SchedulerOutput:
        if self._step_index is None:
            self._init_step_index(timestep)
        step_index = self._step_index
        num_steps = len(self.timesteps)

        sigma_t = float(self.sigmas[step_index]) if step_index < len(self.sigmas) else 1.0
        sigma_next = float(self.sigmas[step_index + 1]) if step_index + 1 < len(self.sigmas) else 0.0

        out = super().step(
            model_output=model_output,
            timestep=timestep,
            sample=sample,
            generator=generator,
            return_dict=return_dict,
        )
        prev_sample = out.prev_sample if return_dict else out[0]
        prev_sample = self._chaos_apply_kick(prev_sample, sigma_t, sigma_next, step_index, num_steps)

        if return_dict:
            return SchedulerOutput(prev_sample=prev_sample)
        return (prev_sample,)


class ChaosDDIMScheduler(ChaosMixin, DDIMScheduler):
    """DDIM scheduler with generic FDS driver support."""

    def step(
        self,
        model_output: torch.Tensor,
        timestep: Union[float, torch.Tensor],
        sample: torch.Tensor,
        eta: float = 0.0,
        use_clipped_model_output: bool = False,
        generator=None,
        variance_noise: Optional[torch.Tensor] = None,
        return_dict: bool = True,
    ) -> Union[SchedulerOutput, tuple]:
        step_index, sigma_t, sigma_next = _ddim_step_sigmas(self, timestep)

        out = super().step(
            model_output=model_output,
            timestep=timestep,
            sample=sample,
            eta=eta,
            use_clipped_model_output=use_clipped_model_output,
            generator=generator,
            variance_noise=variance_noise,
            return_dict=return_dict,
        )
        prev_sample = out.prev_sample if return_dict else out[0]
        prev_sample = self._chaos_apply_kick(prev_sample, sigma_t, sigma_next, step_index, len(self.timesteps))

        if return_dict:
            return SchedulerOutput(prev_sample=prev_sample)
        return (prev_sample,)


class P1ShuffledLorenzDDIMScheduler(P1ShuffledLorenzMixin, DDIMScheduler):
    """DDIM FDS scheduler using the P1 shuffled-Lorenz implementation."""

    def step(
        self,
        model_output: torch.Tensor,
        timestep: Union[float, torch.Tensor],
        sample: torch.Tensor,
        eta: float = 0.0,
        use_clipped_model_output: bool = False,
        generator=None,
        variance_noise: Optional[torch.Tensor] = None,
        return_dict: bool = True,
    ) -> Union[SchedulerOutput, tuple]:
        step_index, sigma_t, sigma_next = _ddim_step_sigmas(self, timestep)

        out = super().step(
            model_output=model_output,
            timestep=timestep,
            sample=sample,
            eta=eta,
            use_clipped_model_output=use_clipped_model_output,
            generator=generator,
            variance_noise=variance_noise,
            return_dict=return_dict,
        )
        prev_sample = out.prev_sample if return_dict else out[0]
        prev_sample = self._chaos_apply_kick(prev_sample, sigma_t, sigma_next, step_index, len(self.timesteps))

        if return_dict:
            return SchedulerOutput(prev_sample=prev_sample)
        return (prev_sample,)


def _ddim_step_sigmas(scheduler: DDIMScheduler, timestep: Union[float, torch.Tensor]) -> tuple[int, float, float]:
    """Return step index and DDIM sigma values for the current transition."""
    t_int = int(timestep)
    matches = (scheduler.timesteps == t_int).nonzero(as_tuple=True)[0]
    step_index = int(matches[0]) if len(matches) > 0 else 0
    num_steps = len(scheduler.timesteps)

    alpha_t = float(scheduler.alphas_cumprod[t_int])
    sigma_t = ((1.0 - alpha_t) / alpha_t) ** 0.5

    if step_index + 1 < num_steps:
        t_next = int(scheduler.timesteps[step_index + 1])
        alpha_next = float(scheduler.alphas_cumprod[t_next])
    else:
        alpha_next = 1.0
    sigma_next = ((1.0 - alpha_next) / alpha_next) ** 0.5

    return step_index, sigma_t, sigma_next
