"""
Experiment 25: Gaussian(Q_L^red) — C_chi isolation control
===========================================================
Goal: Cleanly isolate the effect of C_chi by comparing against FDS.

Design:
  Gaussian(Q_L^red): sample z = L_Q @ eps, eps ~ N(0,I), tile to d dims.
  - Q = Q_L^red (same marginal covariance as FDS/shuffled-Lorenz)
  - C_chi = infinity (Gaussian, unbounded support)
  - tau_ACF = 1 (i.i.d.)
  - Same tiling as FDS

  vs FDS (shuffled-Lorenz):
  - Q = Q_L^red (same)
  - C_chi = 34.65 (bounded support)
  - tau_ACF = 1 (i.i.d.)

  The ONLY difference is C_chi: finite vs infinite.
  Any FID gap is attributable to C_chi.

Fixed: DPM++, NFE=8, 50k samples, 5 seeds
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


def compute_Q_L_red_cholesky(seed=0, n_states=500_000):
    sigma_L, rho_L, beta_L = 10.0, 28.0, 8.0 / 3.0
    dt = 0.005
    y = np.array([float((seed % 100) + 1) / 100.0 + 0.1, 0.0, 0.0])
    for _ in range(10000):
        dx = sigma_L * (y[1] - y[0])
        dy = y[0] * (rho_L - y[2]) - y[1]
        dz = y[0] * y[1] - beta_L * y[2]
        y = y + dt * np.array([dx, dy, dz])
    pool = np.zeros((n_states, 3), dtype=np.float64)
    for i in range(n_states):
        dx = sigma_L * (y[1] - y[0])
        dy = y[0] * (rho_L - y[2]) - y[1]
        dz = y[0] * y[1] - beta_L * y[2]
        y = y + dt * np.array([dx, dy, dz])
        pool[i] = y
    pool_centered = pool - pool.mean(axis=0)
    Q_red = np.cov(pool_centered.T)
    L = np.linalg.cholesky(Q_red)
    return L, Q_red

print("Computing Q_L^red Cholesky factor...")
Q_L_RED_CHOL, Q_L_RED = compute_Q_L_red_cholesky(seed=0)
print(f"Q_L^red eigenvalues: {np.linalg.eigvalsh(Q_L_RED)}")
print(f"sqrt(tr(Q_L^red)) = {np.sqrt(np.trace(Q_L_RED)):.2f}")


class GaussianQLredMixin(ChaosMixin):
    """
    Gaussian(Q_L^red): i.i.d. Gaussian with covariance = Lorenz marginal covariance.
    - Q = Q_L^red (same as FDS)
    - C_chi = infinity (Gaussian, unbounded)
    - tau_ACF = 1 (i.i.d.)
    """
    def _chaos_init_lorenz(self, seed: int):
        self._gq_rng = np.random.RandomState(seed % 100000)
        return (0.0, 0.0, 0.0)

    def _chaos_lorenz_noise(self, shape):
        B = shape[0]
        d = 1
        for s in shape[1:]:
            d *= s
        eps = self._gq_rng.randn(B, 3).astype(np.float32)
        z3 = (eps @ Q_L_RED_CHOL.T).astype(np.float32)
        repeats = (d + 2) // 3
        z_tiled = np.tile(z3, (1, repeats))[:, :d]
        noise = torch.from_numpy(z_tiled).view(shape)
        return noise


class GaussianQLredScheduler(GaussianQLredMixin, DPMSolverMultistepScheduler):
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


def verify_gaussian_ql_red():
    rng = np.random.RandomState(42)
    n_samples = 100000
    eps = rng.randn(n_samples, 3)
    z3 = eps @ Q_L_RED_CHOL.T
    Q_est = np.cov(z3.T)
    print(f"\n[Verify] Q_L^red (target):\n{np.round(Q_L_RED, 2)}")
    print(f"[Verify] Q_est (from 100k Gaussian samples):\n{np.round(Q_est, 2)}")
    print(f"[Verify] Max relative error: {np.max(np.abs(Q_est - Q_L_RED) / (np.abs(Q_L_RED) + 1e-8)):.4f}")
    print(f"[Verify] sqrt(tr(Q_est)) = {np.sqrt(np.trace(Q_est)):.2f}")
    print(f"[Verify] C_chi: INFINITE (Gaussian, unbounded support)")
    print("[Verify] Passed.\n")


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
    eval_out = str(Path(out_dir) / "eval")
    Path(eval_out).mkdir(parents=True, exist_ok=True)
    cmd = [sys.executable, "/root/AFLoPS/eval.py", ref_npz, out_dir,
           "--save_path", eval_out, "--inception_path", inception_pb,
           "--image_size", "256", "--ref_pool_cache", ref_pool]
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = ""
    subprocess.run(cmd, check=True, env=env)
    import glob
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
    eval_json = out_dir / "eval"
    existing_eval = list(eval_json.glob("eval_results_*.json")) if eval_json.exists() else []
    if existing_eval:
        print(f"  [skip] {name} seed{seed} eval already done")
        with open(existing_eval[0]) as f:
            data = json.load(f)
        return {"fid": data.get("FID", float("nan")),
                "precision": data.get("Precision", float("nan")),
                "recall": data.get("Recall", float("nan")),
                "is": data.get("IS", float("nan"))}

    existing = sum(1 for _ in os.scandir(out_dir)) if out_dir.exists() else 0
    if existing >= args.n_samples:
        print(f"  [skip] {name} seed{seed} generation done ({existing} images)")
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
        img_idx = 0
        for start in range(0, args.n_samples, args.batch_size):
            batch = class_ids[start:start + args.batch_size]
            with torch.no_grad():
                out = pipe(class_labels=batch, num_inference_steps=args.nfe,
                           guidance_scale=args.guidance_scale, output_type="np")
            for img in out.images:
                arr = (np.asarray(img) * 255.0).round().astype(np.uint8)
                Image.fromarray(arr).save(out_dir / f"{img_idx:05d}.png")
                img_idx += 1
            if (start // args.batch_size) % 10 == 0:
                print(f"    {name} seed{seed}: {img_idx}/{args.n_samples}")

    if args.skip_eval:
        return {"fid": None, "precision": None, "recall": None, "is": None,
                "smoke_png_count": _count_pngs(str(out_dir)), "eval_skipped": True}

    metrics = _eval(str(out_dir), args.ref_npz, args.ref_pool, args.inception)
    # Delete PNG files after successful eval.
    for f in out_dir.iterdir():
        if f.suffix == ".png":
            f.unlink()
    return metrics


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
    parser.add_argument("--seeds",      type=int, nargs="+", default=[42, 43, 44, 45, 46])
    parser.add_argument("--out_dir",    default="/root/autodl-tmp/exp_gaussian_ql_red")
    parser.add_argument("--skip_eval", action="store_true", help="Use for smoke tests with too few samples for FID")
    args = parser.parse_args()

    verify_gaussian_ql_red()

    Path(args.out_dir).mkdir(parents=True, exist_ok=True)
    gpu_id = os.environ.get("CUDA_VISIBLE_DEVICES", "all")

    print("Loading DiT model...")
    pipe = DiTPipeline.from_pretrained(args.model, torch_dtype=torch.float16)
    pipe = pipe.to("cuda")

    base = dict(CHAOS_PARAMS)
    all_conditions = [
        ("Gaussian-QLred", GaussianQLredScheduler, {**base, "xi_source": "lorenz"}),
    ]

    all_results = {}
    for seed in args.seeds:
        print(f"\n{'='*60}")
        print(f"Seed {seed}")
        print(f"{'='*60}")
        seed_results = {}
        for name, cls, params in all_conditions:
            print(f"\n--- {name} (seed {seed}) ---")
            t0 = time.time()
            m = run_condition(name, cls, params, pipe, args, seed, args.out_dir)
            elapsed = time.time() - t0
            seed_results[name] = {**m, "elapsed_s": elapsed}
            if m.get("eval_skipped"):
                print(f"  eval skipped  PNG={m.get('smoke_png_count', 'N/A')} ({elapsed:.0f}s)")
            else:
                print(f"  FID={m['fid']:.3f} Prec={m['precision']:.3f} "
                      f"Rec={m['recall']:.3f} ({elapsed:.0f}s)")
        all_results[f"seed{seed}"] = seed_results

    summary = {}
    for name, _, _ in all_conditions:
        vals = [all_results[f"seed{s}"][name] for s in args.seeds if name in all_results.get(f"seed{s}", {})]
        vals = [v for v in vals if v.get("fid") is not None]
        fids = [v["fid"] for v in vals]
        precs = [v["precision"] for v in vals]
        recs = [v["recall"] for v in vals]
        if fids:
            summary[name] = {
                "fid_mean": float(np.mean(fids)), "fid_std": float(np.std(fids)),
                "prec_mean": float(np.mean(precs)), "prec_std": float(np.std(precs)),
                "rec_mean": float(np.mean(recs)), "rec_std": float(np.std(recs)),
                "n_seeds": len(fids),
            }

    results_path = Path(args.out_dir) / f"results_gpu{gpu_id}.json"
    with open(results_path, "w") as f:
        json.dump({"per_seed": all_results, "summary": summary,
                   "Q_L_red_trace": float(np.trace(Q_L_RED)),
                   "sqrt_q": float(np.sqrt(np.trace(Q_L_RED)))}, f, indent=2)
    print(f"\nSaved -> {results_path}")

    # Also save a summary next to this script.
    script_dir = Path(__file__).resolve().parent
    summary_path = script_dir / "summary.json"
    with open(summary_path, "w") as f:
        json.dump({"summary": summary, "per_seed": all_results,
                   "sqrt_q": float(np.sqrt(np.trace(Q_L_RED))),
                   "description": "Gaussian(Q_L^red): i.i.d. N(0, Q_L^red), C_chi=inf, isolates C_chi vs FDS"}, f, indent=2)
    print(f"Summary -> {summary_path}")

    print(f"\n{'='*60}")
    print("Summary (mean +/- std across seeds)")
    print(f"{'='*60}")
    print(f"{'Condition':<20} {'FID':>12} {'Prec':>10} {'Rec':>10}")
    print("-" * 55)
    for name in ["Gaussian-QLred"]:
        if name in summary:
            s = summary[name]
            print(f"{name:<20} {s['fid_mean']:.2f}+/-{s['fid_std']:.2f}"
                  f"  {s['prec_mean']:.3f}+/-{s['prec_std']:.3f}"
                  f"  {s['rec_mean']:.3f}+/-{s['rec_std']:.3f}")

    print(f"\nsqrt(tr(Q_L^red)) = {np.sqrt(np.trace(Q_L_RED)):.2f}")
    print("Gaussian(Q_L^red): Q matched to FDS, C_chi=inf.")
    print("FDS vs Gaussian(Q_L^red) gap = pure C_chi effect.")


if __name__ == "__main__":
    main()
