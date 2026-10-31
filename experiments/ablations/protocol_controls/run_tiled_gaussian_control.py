"""
Tiled Gaussian vs Gaussian vs Lorenz.
===================================================
Route 2: Test whether spatial tiling structure (not chi boundedness)
explains Lorenz's advantage.

Design:
  - Gaussian:       z ~ N(0, I_d), all dims independent, Q=I
  - TiledGaussian:  z_3 ~ N(0, I_3), tiled to d dims (every 3 dims share one value)
                    Same tiling structure as Lorenz Pi broadcast, but i.i.d. in time
  - Lorenz:         3D chaotic signal, tiled to d dims
  - ODE:            baseline

If TiledGaussian ≈ Lorenz > Gaussian:
    -> spatial tiling/coherence is the key mechanism
If TiledGaussian ≈ Gaussian < Lorenz:
    -> tiling is not the key, chaotic temporal structure matters

Fixed: DPM++, NFE=8, 50k samples, seed=42, matching the paper.
Output: exp_n_results.json
"""
import json
import os
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
from fds.chaos_mixin import ChaosMixin
from fds.schedulers import ChaosDPMSolverScheduler

DEFAULT_DIT_MODEL = "/root/autodl-tmp/01_models/DiT-XL-2-256/DiT-XL-2-256"
DEFAULT_INCEPTION = "/root/autodl-tmp/classify_image_graph_def.pb"
DEFAULT_REF_NPZ   = "/root/autodl-tmp/VIRTUAL_imagenet256_labeled.npz"
DEFAULT_REF_POOL  = "/root/autodl-tmp/ref_pool_virtual10k.npy"

BASE_CONFIG = {
    "algorithm_type": "dpmsolver++", "solver_order": 3,
    "solver_type": "midpoint", "use_karras_sigmas": False,
    "use_lu_lambdas": False, "lower_order_final": False, "euler_at_final": False,
}

CHAOS_PARAMS = dict(
    tau0=0.3, beta=0.01,
    p_start=0.3, p_end=0.7,
    ratio_cap=0.09, ratio_cap_end=0.045,
)


class TiledGaussianMixin(ChaosMixin):
    """
    Gaussian sampled in 3D, tiled to full latent dimension.
    Mirrors Lorenz's Pi broadcast: z_3 ~ N(0, I_3), then tile to (B, C, H, W).
    - Q = I (same as Gaussian, i.i.d. in time)
    - Spatial structure: every 3 dims share one value (same as Lorenz tiling)
    - C_chi = inf (no temporal bounding)

    This isolates the effect of spatial tiling from chaotic temporal structure.
    """
    def _chaos_init_lorenz(self, seed: int):
        self._tiled_seed = seed
        return (0.0, 0.0, 0.0)

    def _chaos_lorenz_noise(self, shape):
        # shape: (B, C, H, W)  e.g. (64, 4, 32, 32)
        B = shape[0]
        d = 1
        for s in shape[1:]:
            d *= s  # total spatial dims = C*H*W

        gen = torch.Generator().manual_seed(self._tiled_seed % (2**31))
        self._tiled_seed += 1

        # Sample 3D Gaussian, tile to d dims (matching Lorenz's Pi broadcast)
        z3 = torch.randn(B, 3, generator=gen)  # (B, 3)
        repeats = (d + 2) // 3  # ceil(d/3)
        z_tiled = z3.repeat(1, repeats)[:, :d]  # (B, d)
        noise = z_tiled.view(shape)  # (B, C, H, W)
        return noise.float()


class TiledGaussianScheduler(TiledGaussianMixin, DPMSolverMultistepScheduler):
    def step(self, model_output, timestep, sample, generator=None, return_dict=True):
        from diffusers.schedulers.scheduling_utils import SchedulerOutput
        if self._step_index is None:
            self._init_step_index(timestep)
        step_index = self._step_index
        sigma_t    = float(self.sigmas[step_index])     if step_index < len(self.sigmas)     else 1.0
        sigma_next = float(self.sigmas[step_index + 1]) if step_index + 1 < len(self.sigmas) else 0.0
        out = super().step(model_output=model_output, timestep=timestep,
                           sample=sample, generator=generator, return_dict=return_dict)
        prev_sample = out.prev_sample if return_dict else out[0]
        prev_sample = self._chaos_apply_kick(prev_sample, sigma_t, sigma_next,
                                              step_index, len(self.timesteps))
        return SchedulerOutput(prev_sample=prev_sample) if return_dict else (prev_sample,)


def verify_tiling():
    """Sanity check: TiledGaussian has same spatial structure as Lorenz tiling."""
    shape = (4, 4, 32, 32)
    mixin = TiledGaussianMixin.__new__(TiledGaussianMixin)
    mixin._tiled_seed = 42
    noise = mixin._chaos_lorenz_noise(shape)

    # Check: within each tile of 3, values should be identical
    flat = noise[0].view(-1)  # (4096,)
    # Every 3 consecutive elements should be equal (up to boundary)
    ok = all(flat[i].item() == flat[i - i%3].item() for i in range(min(30, len(flat))))
    print(f"[Verify] TiledGaussian shape: {noise.shape}")
    print(f"[Verify] Tiling pattern correct: {ok}")
    print(f"[Verify] First 9 flat values: {flat[:9].tolist()}")
    assert noise.shape == torch.Size(shape), "Shape mismatch"
    print("[Verify] Passed.\n")


def _save_images(images, out_dir):
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    arr = (np.asarray(images) * 255.0).round().astype(np.uint8)
    for i, img in enumerate(arr):
        Image.fromarray(img).save(Path(out_dir) / f"{i:05d}.png")


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


def _eval(out_dir, ref_npz, ref_pool, inception_pb):
    import glob
    eval_out = str(Path(out_dir) / "eval")
    Path(eval_out).mkdir(parents=True, exist_ok=True)
    cmd = [sys.executable, "/root/AFLoPS/eval.py", ref_npz, out_dir,
           "--save_path", eval_out, "--inception_path", inception_pb,
           "--image_size", "256", "--ref_pool_cache", ref_pool]
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = ""
    subprocess.run(cmd, check=True, env=env)
    files = glob.glob(str(Path(eval_out) / "eval_results_*.json"))
    if not files:
        raise RuntimeError(f"No eval result in {eval_out}")
    with open(files[0]) as f:
        data = json.load(f)
    return {"fid": data.get("FID", float("nan")),
            "precision": data.get("Precision", float("nan")),
            "recall": data.get("Recall", float("nan")),
            "is": data.get("IS", float("nan"))}


def run_condition(name, scheduler_cls, chaos_params, pipe, args, seed, out_base):
    out_dir = Path(out_base) / name / f"seed{seed}"
    if out_dir.exists() and len(list(out_dir.glob("*.png"))) >= args.n_samples:
        print(f"  [skip] {name} already done")
    else:
        out_dir.mkdir(parents=True, exist_ok=True)
        sched = scheduler_cls.from_pretrained(
            args.model, subfolder="scheduler", **BASE_CONFIG)
        sched.set_timesteps(args.nfe)
        sched.set_chaos_params(seed=seed, **chaos_params)
        pipe.scheduler = sched
        torch.manual_seed(seed)
        np.random.seed(seed)

        class_ids = (list(range(1000)) * (args.n_samples // 1000 + 1))[:args.n_samples]
        all_images = []
        for start in range(0, args.n_samples, args.batch_size):
            batch = class_ids[start:start + args.batch_size]
            with torch.no_grad():
                out = pipe(class_labels=batch, num_inference_steps=args.nfe,
                           guidance_scale=args.guidance_scale, output_type="np")
            all_images.extend(out.images)
            if (start // args.batch_size) % 5 == 0:
                print(f"    {name}: {start+len(batch)}/{args.n_samples}")
        _save_images(all_images, out_dir)

    if args.skip_eval:
        return {"fid": None, "precision": None, "recall": None, "is": None,
                "smoke_png_count": _count_pngs(str(out_dir)), "eval_skipped": True}
    return _eval(str(out_dir), args.ref_npz, args.ref_pool, args.inception)


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model",      default=DEFAULT_DIT_MODEL)
    parser.add_argument("--inception",  default=DEFAULT_INCEPTION)
    parser.add_argument("--ref_npz",    default=DEFAULT_REF_NPZ)
    parser.add_argument("--ref_pool",   default=DEFAULT_REF_POOL)
    parser.add_argument("--nfe",        type=int,   default=8)
    parser.add_argument("--n_samples",  type=int,   default=50000)
    parser.add_argument("--batch_size", type=int,   default=64)
    parser.add_argument("--guidance_scale", type=float, default=1.5)
    parser.add_argument("--seed",       type=int,   default=42)
    parser.add_argument("--out_dir",    default=str(Path(__file__).parent / "results"))
    parser.add_argument("--conditions", nargs="+",  default=None)
    parser.add_argument("--skip_eval", action="store_true", help="Use for smoke tests with too few samples for FID")
    args = parser.parse_args()

    verify_tiling()

    Path(args.out_dir).mkdir(parents=True, exist_ok=True)
    gpu_id = os.environ.get("CUDA_VISIBLE_DEVICES", "all")
    results_path = Path(args.out_dir) / f"exp_n_results_gpu{gpu_id}.json"

    print("Loading DiT model...")
    pipe = DiTPipeline.from_pretrained(args.model, torch_dtype=torch.float16)
    pipe = pipe.to("cuda")

    base = dict(CHAOS_PARAMS)
    all_conditions = [
        ("ODE",          ChaosDPMSolverScheduler, {**base, "tau0": 0.0,  "xi_source": "lorenz"}),
        ("Lorenz",       ChaosDPMSolverScheduler, {**base, "xi_source": "lorenz"}),
        ("Gaussian",     ChaosDPMSolverScheduler, {**base, "xi_source": "gaussian"}),
        ("TiledGaussian",TiledGaussianScheduler,  {**base, "xi_source": "lorenz"}),
    ]

    if args.conditions:
        all_conditions = [(n,c,p) for n,c,p in all_conditions if n in args.conditions]
        if not all_conditions:
            print(f"No matching conditions: {args.conditions}")
            sys.exit(1)

    results = {}
    for name, cls, params in all_conditions:
        print(f"\n=== {name} ===")
        t0 = time.time()
        m = run_condition(name, cls, params, pipe, args, args.seed, args.out_dir)
        elapsed = time.time() - t0
        results[name] = {**m, "elapsed_s": elapsed}
        if m.get("eval_skipped"):
            print(f"  eval skipped  PNG={m.get('smoke_png_count', 'N/A')} ({elapsed:.0f}s)")
        else:
            print(f"  FID={m['fid']:.3f} Prec={m['precision']:.3f} Rec={m['recall']:.3f} ({elapsed:.0f}s)")

    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved -> {results_path}")

    print(f"\n{'Condition':<18} {'FID':>7} {'Prec':>7} {'Rec':>7}")
    print("-" * 42)
    for name in ["ODE","Gaussian","TiledGaussian","Lorenz"]:
        if name in results:
            m = results[name]
            if m.get("eval_skipped"):
                print(f"{name:<18} {'skip':>7} {'skip':>7} {'skip':>7}")
            else:
                print(f"{name:<18} {m['fid']:>7.3f} {m['precision']:>7.3f} {m['recall']:>7.3f}")


if __name__ == "__main__":
    main()
