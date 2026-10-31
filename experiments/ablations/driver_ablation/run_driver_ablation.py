"""
Driver ablation.
This corresponds to Assumptions 2 and 3 on fast-system properties.
Drivers satisfying the assumptions should be effective, and the driver family itself is secondary.

Compared drivers:
  - shuffled_lorenz: FDS driver, i.i.d. samples from a burn-in Lorenz pool.
  - lorenz: sequential Lorenz control.
  - gaussian: i.i.d. white noise, memoryless baseline.
  - ar1: AR(1) momentum noise, random driver with memory.
  - logistic: logistic map, deterministic chaos.
  - sin: sine wave, satisfies Assumption 2 but fails Assumption 3 with Q=0.

Fixed setting: sampler=dpmpp, NFE=8, shared parameters, 50k samples, 5 seeds.
Output: exp_c_results.json.

Usage:
"""
import argparse
import glob
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image

_orig_from_numpy = torch.from_numpy
def _safe_from_numpy(x):
    try:
        return _orig_from_numpy(x)
    except TypeError:
        return torch.as_tensor(x)
torch.from_numpy = _safe_from_numpy

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from diffusers import DiTPipeline, DPMSolverMultistepScheduler
from fds.schedulers import ChaosDPMSolverScheduler, P1ShuffledLorenzDPMSolverScheduler
from fds.chaos_mixin import ChaosMixin

DEFAULT_DIT_MODEL = "/root/autodl-tmp/01_models/DiT-XL-2-256/DiT-XL-2-256"
DEFAULT_INCEPTION = "/root/autodl-tmp/classify_image_graph_def.pb"
DEFAULT_REF_NPZ   = "/root/autodl-tmp/VIRTUAL_imagenet256_labeled.npz"
DEFAULT_REF_POOL  = "/root/autodl-tmp/ref_pool_virtual10k.npy"

BASE_CONFIG_OVERRIDES = {
    "algorithm_type": "dpmsolver++", "solver_order": 3,
    "solver_type": "midpoint", "use_karras_sigmas": False,
    "use_lu_lambdas": False, "lower_order_final": False, "euler_at_final": False,
}

# Driver list for the appendix driver-ablation table.
SOURCES = ["shuffled_lorenz", "lorenz", "gaussian", "ar1", "logistic", "sin"]


class AR1Mixin(ChaosMixin):
    """AR(1) momentum noise, y_{k+1}=rho*y_k+sqrt(1-rho^2)*eps, with Q=(1+rho)/(1-rho)*I."""
    _AR1_RHO = 0.9

    def _chaos_init_lorenz(self, seed: int):
        self._ar1_rng = np.random.RandomState(seed % 100000)
        self._ar1_state = self._ar1_rng.randn(1).astype(np.float32)  # Initialize from the stationary distribution.
        return (0.0, 0.0, 0.0)

    def _chaos_lorenz_noise(self, shape):
        numel = 1
        for s in shape:
            numel *= s
        rho = self._AR1_RHO
        scale = np.sqrt(1.0 - rho ** 2)
        vals = []
        x = float(self._ar1_state[0])
        for _ in range(numel):
            x = rho * x + scale * float(self._ar1_rng.randn())
            vals.append(x)
        self._ar1_state[0] = x
        return torch.tensor(vals, dtype=torch.float32).view(shape)


class TentMixin(ChaosMixin):
    """Logistic map driver, y_{k+1}=4*y_k*(1-y_k), mapped to [-1,1] with zero mean."""

    def _chaos_init_lorenz(self, seed: int):
        rng = np.random.RandomState(seed % 100000)
        # Avoid unstable fixed points 0 and 1.
        x = rng.uniform(0.1, 0.9)
        self._logistic_state = np.array([x], dtype=np.float64)
        return (0.0, 0.0, 0.0)

    def _chaos_lorenz_noise(self, shape):
        numel = 1
        for s in shape:
            numel *= s
        x = float(self._logistic_state[0])
        vals = []
        for _ in range(numel):
            x = 4.0 * x * (1.0 - x)
            vals.append(2.0 * x - 1.0)  # Map to [-1,1] with zero mean.
        self._logistic_state[0] = x
        return torch.tensor(vals, dtype=torch.float32).view(shape)


class SinMixin(ChaosMixin):
    """Sine-wave phase rotation driver. It has zero mean but bounded partial sums, so Q=0."""
    _SIN_OMEGA = 2.0 * np.pi * (np.sqrt(5) - 1) / 2  # Golden-ratio irrational rotation.

    def _chaos_init_lorenz(self, seed: int):
        self._sin_phase = float(seed % 1000) / 1000.0 * 2.0 * np.pi
        return (0.0, 0.0, 0.0)

    def _chaos_lorenz_noise(self, shape):
        numel = 1
        for s in shape:
            numel *= s
        phase = self._sin_phase
        vals = []
        for _ in range(numel):
            phase = (phase + self._SIN_OMEGA) % (2.0 * np.pi)
            vals.append(np.sin(phase))
        self._sin_phase = phase
        return torch.tensor(vals, dtype=torch.float32).view(shape)


class ChaosDPMAR1(AR1Mixin, DPMSolverMultistepScheduler):
    def step(self, model_output, timestep, sample, generator=None, return_dict=True):
        from diffusers.schedulers.scheduling_utils import SchedulerOutput
        if self._step_index is None:
            self._init_step_index(timestep)
        step_index = self._step_index
        sigma_t    = float(self.sigmas[step_index])     if step_index < len(self.sigmas)     else 1.0
        sigma_next = float(self.sigmas[step_index + 1]) if step_index + 1 < len(self.sigmas) else 0.0
        out = super().step(model_output=model_output, timestep=timestep, sample=sample,
                           generator=generator, return_dict=return_dict)
        prev_sample = out.prev_sample if return_dict else out[0]
        prev_sample = self._chaos_apply_kick(prev_sample, sigma_t, sigma_next, step_index, len(self.timesteps))
        return SchedulerOutput(prev_sample=prev_sample) if return_dict else (prev_sample,)


class ChaosDPMTent(TentMixin, DPMSolverMultistepScheduler):
    def step(self, model_output, timestep, sample, generator=None, return_dict=True):
        from diffusers.schedulers.scheduling_utils import SchedulerOutput
        if self._step_index is None:
            self._init_step_index(timestep)
        step_index = self._step_index
        sigma_t    = float(self.sigmas[step_index])     if step_index < len(self.sigmas)     else 1.0
        sigma_next = float(self.sigmas[step_index + 1]) if step_index + 1 < len(self.sigmas) else 0.0
        out = super().step(model_output=model_output, timestep=timestep, sample=sample,
                           generator=generator, return_dict=return_dict)
        prev_sample = out.prev_sample if return_dict else out[0]
        prev_sample = self._chaos_apply_kick(prev_sample, sigma_t, sigma_next, step_index, len(self.timesteps))
        return SchedulerOutput(prev_sample=prev_sample) if return_dict else (prev_sample,)


class ChaosDPMSin(SinMixin, DPMSolverMultistepScheduler):
    def step(self, model_output, timestep, sample, generator=None, return_dict=True):
        from diffusers.schedulers.scheduling_utils import SchedulerOutput
        if self._step_index is None:
            self._init_step_index(timestep)
        step_index = self._step_index
        sigma_t    = float(self.sigmas[step_index])     if step_index < len(self.sigmas)     else 1.0
        sigma_next = float(self.sigmas[step_index + 1]) if step_index + 1 < len(self.sigmas) else 0.0
        out = super().step(model_output=model_output, timestep=timestep, sample=sample,
                           generator=generator, return_dict=return_dict)
        prev_sample = out.prev_sample if return_dict else out[0]
        prev_sample = self._chaos_apply_kick(prev_sample, sigma_t, sigma_next, step_index, len(self.timesteps))
        return SchedulerOutput(prev_sample=prev_sample) if return_dict else (prev_sample,)


def _save_images(images, out_dir):
    images = (np.asarray(images) * 255.0).round().astype(np.uint8)
    if images.ndim == 3:
        images = images[None]
    for i, img in enumerate(images):
        Image.fromarray(img).save(os.path.join(out_dir, f"sample_{i+1}.png"))


def _run_eval(sample_dir, eval_out, ref_npz, inception_path, ref_pool_cache):
    cmd = [sys.executable, "/root/AFLoPS/eval.py", ref_npz, sample_dir,
           "--save_path", eval_out, "--inception_path", inception_path,
           "--image_size", "256", "--ref_pool_cache", ref_pool_cache]
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = ""
    subprocess.run(cmd, check=True, env=env)


def _read_eval_json(eval_dir):
    files = glob.glob(os.path.join(eval_dir, "eval_results_*.json"))
    if not files:
        return {}
    with open(files[0]) as f:
        return json.load(f)


def _count_pngs(root_dir):
    total = 0
    stack = [root_dir]
    while stack:
        cur = stack.pop()
        with os.scandir(cur) as entries:
            for entry in entries:
                if entry.is_dir():
                    stack.append(entry.path)
                elif entry.is_file() and entry.name.endswith(".png"):
                    total += 1
    return total


def _normalize_alphas_cumprod(scheduler):
    if hasattr(scheduler, "alphas_cumprod") and torch.is_tensor(scheduler.alphas_cumprod):
        scheduler.alphas_cumprod = scheduler.alphas_cumprod.detach().cpu().float().numpy()


def build_scheduler_for_source(source: str, base_config: dict, nfe: int, seed: int = 0):
    cfg = dict(base_config)
    cfg.update(BASE_CONFIG_OVERRIDES)
    if source == "ode":
        s = DPMSolverMultistepScheduler.from_config(cfg)
        _normalize_alphas_cumprod(s)
        return s
    # All sources use identical shared parameters, changing only the noise source.
    # Couple seed with seed_base so different runs use independent FDS perturbation sequences.
    common_kw = dict(tau0=0.3, beta=0.01, p_start=0.3, p_end=0.7, seed=seed,
                     ratio_cap=0.09, ratio_cap_end=0.045,
                     lorenz_sigma=10.0, lorenz_rho=28.0, lorenz_beta=8.0/3.0, lorenz_dt=0.005)
    if source == "shuffled_lorenz":
        s = P1ShuffledLorenzDPMSolverScheduler.from_config(cfg)
        s.set_chaos_params(**common_kw, xi_source="shuffled_lorenz")
    elif source == "lorenz":
        s = ChaosDPMSolverScheduler.from_config(cfg)
        s.set_chaos_params(**common_kw, xi_source="lorenz")
    elif source == "gaussian":
        s = ChaosDPMSolverScheduler.from_config(cfg)
        s.set_chaos_params(**common_kw, xi_source="gaussian")
    elif source == "ar1":
        s = ChaosDPMAR1.from_config(cfg)
        s.set_chaos_params(**common_kw, xi_source="lorenz")  # xi_source is overridden by AR1Mixin.
    elif source == "logistic":
        s = ChaosDPMTent.from_config(cfg)
        s.set_chaos_params(**common_kw, xi_source="lorenz")  # xi_source is overridden by TentMixin.
    elif source == "sin":
        s = ChaosDPMSin.from_config(cfg)
        s.set_chaos_params(**common_kw, xi_source="lorenz")  # xi_source is overridden by SinMixin.
    else:
        raise ValueError(f"Unknown source: {source}")
    _normalize_alphas_cumprod(s)
    return s


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--nfe", type=int, default=8)
    parser.add_argument("--num_seeds", type=int, default=5)
    parser.add_argument("--seed_base", type=int, default=42)
    parser.add_argument("--num_classes", type=int, default=1000)
    parser.add_argument("--images_per_class", type=int, default=50,
                        help="Images per class, default 50 for 50k samples")
    parser.add_argument("--sources", nargs="+", default=None,
                        help="Subset of noise sources to run, for example --sources shuffled_lorenz gaussian")
    parser.add_argument("--save_dir", default="/root/autodl-tmp/exp_c_source_ablation")
    parser.add_argument("--dit_model", default=DEFAULT_DIT_MODEL)
    parser.add_argument("--ref_npz", default=DEFAULT_REF_NPZ)
    parser.add_argument("--ref_pool_cache", default=DEFAULT_REF_POOL)
    parser.add_argument("--inception_path", default=DEFAULT_INCEPTION)
    parser.add_argument("--run_eval", action="store_true", help="Compatibility flag; formal runs evaluate by default")
    parser.add_argument("--skip_eval", action="store_true", help="Use for smoke tests with too few samples for FID")
    args = parser.parse_args()

    os.makedirs(args.save_dir, exist_ok=True)
    results_path = os.path.join(args.save_dir, "exp_c_results.json")
    all_results = json.load(open(results_path)) if os.path.exists(results_path) else []
    done_keys = {r["key"] for r in all_results}

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pipe = DiTPipeline.from_pretrained(args.dit_model, torch_dtype=torch.float16, local_files_only=True, use_safetensors=False)
    base_config = dict(pipe.scheduler.config)

    sources_to_run = args.sources if args.sources else SOURCES
    for source in sources_to_run:
        for seed_offset in range(args.num_seeds):
            seed = args.seed_base + seed_offset
            key = f"{source}_seed{seed}"
            if key in done_keys:
                print(f"[SKIP] {key}")
                continue
            print(f"[RUN] {key}")

            exp_dir = os.path.join(args.save_dir, source, f"seed{seed}")
            os.makedirs(exp_dir, exist_ok=True)

            pipe.scheduler = build_scheduler_for_source(source, base_config, args.nfe, seed)
            pipe = pipe.to(device)

            t0 = time.time()
            with torch.no_grad():
                for class_label in range(args.num_classes):
                    class_dir = os.path.join(exp_dir, str(class_label))
                    if os.path.exists(class_dir) and \
                            len([f for f in os.listdir(class_dir) if f.endswith(".png")]) >= args.images_per_class:
                        continue
                    os.makedirs(class_dir, exist_ok=True)
                    gen = torch.manual_seed(seed * 10000 + class_label)
                    imgs = pipe(class_labels=[class_label] * args.images_per_class,
                                num_inference_steps=args.nfe, generator=gen,
                                guidance_scale=1.5, output_type="np").images
                    _save_images(imgs, class_dir)

            if args.skip_eval:
                metrics = {"smoke_png_count": _count_pngs(exp_dir), "eval_skipped": True}
            else:
                eval_out = os.path.join(exp_dir, "eval")
                os.makedirs(eval_out, exist_ok=True)
                _run_eval(exp_dir, eval_out, args.ref_npz, args.inception_path, args.ref_pool_cache)
                metrics = _read_eval_json(eval_out)

                for item in os.listdir(exp_dir):
                    item_path = os.path.join(exp_dir, item)
                    if os.path.isdir(item_path) and item != "eval":
                        shutil.rmtree(item_path)

            record = {"key": key, "source": source, "seed": seed,
                      "elapsed_sec": time.time() - t0, **metrics}
            # Re-read JSON before writing to avoid overwrites from concurrent processes.
            import fcntl
            lock_path = results_path + ".lock"
            with open(lock_path, "w") as lf:
                fcntl.flock(lf, fcntl.LOCK_EX)
                try:
                    cur = json.load(open(results_path)) if os.path.exists(results_path) else []
                    cur_keys = {r["key"] for r in cur}
                    if key not in cur_keys:
                        cur.append(record)
                    with open(results_path, "w") as f:
                        json.dump(cur, f, indent=2)
                finally:
                    fcntl.flock(lf, fcntl.LOCK_UN)
            done_keys.add(key)
            print(f"  FID={metrics.get('fid', 'N/A')}  PNG={metrics.get('smoke_png_count', 'N/A')}")

    print(f"\nDriver ablation done. Results: {results_path}")


if __name__ == "__main__":
    main()
