"""
Comparison: sequential R?ssler versus shuffled R?ssler.
Purpose: test whether shuffling itself, which changes C_chi from infinite to finite, is effective independent of the driver family.

Design:
  - sequential R?ssler: read states in temporal order, tau_ACF >> N, C_chi = inf.
  - shuffled R?ssler: i.i.d. sampling from a random pool, tau_ACF = 0, C_chi = 22.75.
  - both use identical R?ssler parameters and the same injection protocol.
  - ODE/FDS baselines can be reused from exp_l96v2.

Parameters: NFE=8, DPM++, 50k samples, 5 seeds, guidance_scale=1.5.
Output path: /root/autodl-tmp/exp_rossler_compare/.

Usage:
"""

import argparse, json, math, os, subprocess, sys, time
from pathlib import Path

import numpy as np
import torch

THIS_DIR = Path(__file__).resolve().parent
FL_ROOT  = THIS_DIR.parents[2]
sys.path.insert(0, str(FL_ROOT))

from diffusers import DiTPipeline, DPMSolverMultistepScheduler
from diffusers.schedulers.scheduling_utils import SchedulerOutput
from fds.chaos_mixin import ChaosMixin
from fds.schedulers import ChaosDPMSolverScheduler as FDSScheduler

DEFAULT_MODEL     = "/root/autodl-tmp/01_models/DiT-XL-2-256/DiT-XL-2-256"
DEFAULT_INCEPTION = "/root/autodl-tmp/classify_image_graph_def.pb"
DEFAULT_REF_NPZ   = "/root/autodl-tmp/VIRTUAL_imagenet256_labeled.npz"
DEFAULT_REF_POOL  = "/root/autodl-tmp/ref_pool_virtual10k.npy"
DEFAULT_OUT       = "/root/autodl-tmp/exp_rossler_compare"
BASELINE_DIR      = "/root/autodl-tmp/exp_l96v2"

BASE_SCHED = {
    "algorithm_type": "dpmsolver++", "solver_order": 3,
    "solver_type": "midpoint", "use_karras_sigmas": False,
    "use_lu_lambdas": False, "lower_order_final": False, "euler_at_final": False,
}

CHAOS_PARAMS_BASE = dict(
    tau0=0.3, beta=0.01,
    p_start=0.3, p_end=0.7,
    ratio_cap=0.09, ratio_cap_end=0.045,
)

# R?ssler parameters, identical to exp_rossler.
_A, _B, _C = 0.2, 0.2, 5.7
_DT        = 0.05
_BURNIN    = 50000
_POOL_SZ   = 500_000


def _rossler_deriv(s):
    x, y, z = s
    return np.array([-y - z, x + _A * y, _B + z * (x - _C)])


def _rossler_build_pool(seed):
    s = np.array([1.0, 0.0, 0.0], dtype=np.float64)
    rng = np.random.default_rng(seed)
    s += rng.standard_normal(3) * 0.1
    for _ in range(_BURNIN):
        k1 = _rossler_deriv(s)
        k2 = _rossler_deriv(s + 0.5*_DT*k1)
        k3 = _rossler_deriv(s + 0.5*_DT*k2)
        k4 = _rossler_deriv(s + _DT*k3)
        s  = s + _DT/6*(k1 + 2*k2 + 2*k3 + k4)
    pool = np.zeros((_POOL_SZ, 3), dtype=np.float32)
    for i in range(_POOL_SZ):
        pool[i] = s
        k1 = _rossler_deriv(s)
        k2 = _rossler_deriv(s + 0.5*_DT*k1)
        k3 = _rossler_deriv(s + 0.5*_DT*k2)
        k4 = _rossler_deriv(s + _DT*k3)
        s  = s + _DT/6*(k1 + 2*k2 + 2*k3 + k4)
    return pool


# Shuffled R?ssler mixin, reusing exp_rossler logic.
class ShuffledRosslerMixin(ChaosMixin):
    """i.i.d. sampling from the R?ssler pool, tau_ACF=0, C_chi=22.75."""

    def set_chaos_params(self, tau0=0.3, beta=0.01, p_start=0.3, p_end=0.7,
                         seed=0, ratio_cap=0.09, ratio_cap_end=None,
                         xi_source="rossler", **kwargs):
        super().set_chaos_params(tau0=tau0, beta=beta, p_start=p_start, p_end=p_end,
                                 seed=seed, ratio_cap=ratio_cap,
                                 ratio_cap_end=ratio_cap_end,
                                 xi_source=xi_source, **kwargs)
        print(f"  [ShuffledRossler] initializing pool seed={seed}...")
        self._sr_pool    = _rossler_build_pool(seed)
        self._sr_rng     = np.random.default_rng(seed + 88888)
        self._sr_counter = 0

    def _rossler_noise(self, shape):
        numel    = 1
        for s in shape: numel *= s
        n_needed = math.ceil(numel / 3)
        idx      = self._sr_rng.integers(0, _POOL_SZ, size=n_needed)
        flat     = self._sr_pool[idx].flatten()
        flat     = flat[:numel] if len(flat) >= numel else np.tile(flat, math.ceil(numel/len(flat)))[:numel]
        noise    = torch.tensor(flat, dtype=torch.float32).view(shape)
        if noise.ndim == 4:
            B, C, H, W = noise.shape
            for b in range(B):
                for c in range(C):
                    gen  = torch.Generator()
                    gen.manual_seed(self._sr_counter * 1000 + b * C + c)
                    perm = torch.randperm(H * W, generator=gen)
                    noise[b, c] = noise[b, c].reshape(-1)[perm].reshape(H, W)
        self._sr_counter += 1
        return noise

    def _chaos_apply_kick(self, sample, sigma_t, sigma_next, step_index, num_steps):
        if not hasattr(self, "chaos_tau0") or self.chaos_tau0 <= 0.0:
            return sample
        i        = step_index + 1
        progress = i / float(num_steps)
        if progress < self.chaos_p_start or progress > self.chaos_p_end:
            return sample
        sigma_t    = float(sigma_t)
        sigma_next = float(sigma_next)
        if sigma_next <= 0: return sample
        alpha_t    = 1.0 / (1.0 + sigma_t**2)**0.5
        alpha_next = 1.0 / (1.0 + sigma_next**2)**0.5
        lambda_t   = math.log(alpha_t / max(sigma_t, 1e-12))
        lambda_next= math.log(alpha_next / max(sigma_next, 1e-12))
        h          = max(lambda_next - lambda_t, 0.0)
        base_scale = sigma_next * math.sqrt(max(1.0 - math.exp(-2.0*h), 0.0))
        if base_scale <= 0: return sample
        tau = self.chaos_tau0 * ((1.0 - self.chaos_beta)**(i - 1))
        if tau <= 0: return sample
        noise = self._rossler_noise(sample.shape).to(device=sample.device, dtype=torch.float32)
        dims  = tuple(range(1, noise.ndim))
        noise = noise - noise.mean(dim=dims, keepdim=True)
        rms   = torch.sqrt((noise**2).mean(dim=dims, keepdim=True) + 1e-8)
        noise = noise / rms
        kick_raw = (tau * base_scale) * noise
        if self.chaos_ratio_cap > 0.0:
            gate_span     = max(self.chaos_p_end - self.chaos_p_start, 1e-6)
            t             = max(0.0, min(1.0, (progress - self.chaos_p_start) / gate_span))
            ratio_cap_eff = (self.chaos_ratio_cap + (self.chaos_ratio_cap_end - self.chaos_ratio_cap) * t
                             if self.chaos_ratio_cap_end is not None else self.chaos_ratio_cap)
            kick_norm   = torch.sqrt((kick_raw.float()**2).mean(dim=dims))
            sample_norm = torch.sqrt((sample.float()**2).mean(dim=dims)) + 1e-12
            scale = torch.clamp(ratio_cap_eff / (kick_norm / sample_norm), max=1.0)
            kick_raw = kick_raw * scale.view(-1, *([1] * (kick_raw.ndim - 1)))
        return sample + kick_raw.to(dtype=sample.dtype)


# Sequential R?ssler mixin, temporal order, tau_ACF >> N, C_chi = inf.
class SeqRosslerMixin(ShuffledRosslerMixin):
    """Read R?ssler states sequentially without shuffling, tau_ACF >> N, C_chi = inf."""

    def set_chaos_params(self, **kwargs):
        super().set_chaos_params(**kwargs)
        # Sequential cursor, read from the beginning of the pool.
        self._sr_seq_idx = 0

    def _rossler_noise(self, shape):
        numel = 1
        for s in shape: numel *= s
        n_needed = math.ceil(numel / 3)
        # Sequential read without random sampling.
        idx_start = self._sr_seq_idx % _POOL_SZ
        idx_end   = idx_start + n_needed
        if idx_end <= _POOL_SZ:
            flat = self._sr_pool[idx_start:idx_end].flatten()
        else:
            flat = np.concatenate([
                self._sr_pool[idx_start:],
                self._sr_pool[:(idx_end - _POOL_SZ)]
            ]).flatten()
        self._sr_seq_idx = (self._sr_seq_idx + n_needed) % _POOL_SZ
        flat  = flat[:numel]
        noise = torch.tensor(flat, dtype=torch.float32).view(shape)
        # No spatial shuffle.
        return noise


# Scheduler classes.
class ShuffledRosslerScheduler(ShuffledRosslerMixin, DPMSolverMultistepScheduler):
    def step(self, model_output, timestep, sample, generator=None, return_dict=True) -> SchedulerOutput:
        if self._step_index is None:
            self._init_step_index(timestep)
        step_index = self._step_index
        num_steps  = len(self.timesteps)
        sigma_t    = float(self.sigmas[step_index])     if step_index < len(self.sigmas)     else 1.0
        sigma_next = float(self.sigmas[step_index + 1]) if step_index + 1 < len(self.sigmas) else 0.0
        out = super(ShuffledRosslerMixin, self).step(
            model_output=model_output, timestep=timestep,
            sample=sample, generator=generator, return_dict=True)
        prev_sample = self._chaos_apply_kick(
            out.prev_sample, sigma_t, sigma_next, step_index, num_steps)
        return SchedulerOutput(prev_sample=prev_sample)


class SeqRosslerScheduler(SeqRosslerMixin, DPMSolverMultistepScheduler):
    def step(self, model_output, timestep, sample, generator=None, return_dict=True) -> SchedulerOutput:
        if self._step_index is None:
            self._init_step_index(timestep)
        step_index = self._step_index
        num_steps  = len(self.timesteps)
        sigma_t    = float(self.sigmas[step_index])     if step_index < len(self.sigmas)     else 1.0
        sigma_next = float(self.sigmas[step_index + 1]) if step_index + 1 < len(self.sigmas) else 0.0
        out = super(SeqRosslerMixin, self).step(
            model_output=model_output, timestep=timestep,
            sample=sample, generator=generator, return_dict=True)
        prev_sample = self._chaos_apply_kick(
            out.prev_sample, sigma_t, sigma_next, step_index, num_steps)
        return SchedulerOutput(prev_sample=prev_sample)


# ── eval helpers ──────────────────────────────────────────────────────────────
def _eval_if_done(out_dir):
    eval_out = Path(out_dir) / "eval"
    existing = list(eval_out.glob("eval_results_*.json")) if eval_out.exists() else []
    if not existing: return None
    try:
        with open(existing[0]) as f:
            data = json.load(f)
        fid = data.get("FID", float("nan"))
        if fid and not math.isnan(float(fid)):
            return {"fid": float(fid),
                    "precision": float(data.get("Precision", float("nan"))),
                    "recall":    float(data.get("Recall",    float("nan")))}
    except Exception:
        pass
    return None


def _run_eval(out_dir, args):
    eval_out = Path(out_dir) / "eval"
    eval_out.mkdir(parents=True, exist_ok=True)
    cmd = [sys.executable, "/root/AFLoPS/eval.py",
           args.ref_npz, str(out_dir),
           "--save_path", str(eval_out),
           "--inception_path", args.inception,
           "--image_size", "256",
           "--ref_pool_cache", args.ref_pool]
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = ""
    env.pop("CUDA_DEVICE_ORDER", None)
    subprocess.run(cmd, check=True, env=env)
    files = list(eval_out.glob("eval_results_*.json"))
    if not files:
        raise RuntimeError(f"No eval result in {eval_out}")
    with open(files[0]) as f:
        data = json.load(f)
    for png in Path(out_dir).glob("*.png"):
        png.unlink()
    return {"fid": float(data.get("FID", float("nan"))),
            "precision": float(data.get("Precision", float("nan"))),
            "recall":    float(data.get("Recall",    float("nan")))}


def run_condition(name, sched_cls, chaos_params, pipe, args, seed):
    out_dir = Path(args.out_dir) / name / f"seed{seed}"
    done = _eval_if_done(out_dir)
    if done is not None:
        print(f"  [skip] {name} seed={seed} FID={done['fid']:.3f}")
        return done

    n_existing = sum(1 for f in os.scandir(out_dir) if f.name.endswith(".png")) \
                 if out_dir.exists() else 0
    if n_existing < args.n_samples:
        out_dir.mkdir(parents=True, exist_ok=True)
        sched = sched_cls.from_pretrained(args.model, subfolder="scheduler", **BASE_SCHED)
        sched.set_timesteps(args.nfe)
        sched.set_chaos_params(seed=seed, **chaos_params)
        pipe.scheduler = sched

        from PIL import Image as PILImage
        images_per_class = max(1, args.n_samples // 1000)
        img_idx = n_existing
        for class_label in range(1000):
            if img_idx >= args.n_samples: break
            gen = torch.Generator(device="cuda")
            gen.manual_seed(seed * 10000 + class_label)
            labels = torch.tensor([class_label] * images_per_class,
                                  dtype=torch.long, device="cuda")
            with torch.no_grad():
                out = pipe(class_labels=labels,
                           num_inference_steps=args.nfe,
                           guidance_scale=args.guidance_scale,
                           generator=gen, output_type="np")
            arr = (np.asarray(out.images) * 255.).round().astype(np.uint8)
            for img in arr:
                PILImage.fromarray(img).save(out_dir / f"{img_idx:05d}.png")
                img_idx += 1
            if class_label % 200 == 0:
                print(f"    {name} seed={seed}: {img_idx}/{args.n_samples}")
        print(f"  [gen done] {name} seed={seed}: {img_idx} images")
    else:
        print(f"  [skip gen] {name} seed={seed}: {n_existing} images exist")

    return _run_eval(str(out_dir), args)


def _load_baseline(cond, seed, baseline_dir):
    r = _eval_if_done(Path(baseline_dir) / cond / f"seed{seed}")
    if r is not None:
        print(f"  [reuse baseline] {cond} seed={seed} FID={r['fid']:.3f}")
    return r


# Summary.
def summarize(out_dir, baseline_dir, seeds):
    conditions = ["ODE", "FDS", "SHUF_ROSSLER", "SEQ_ROSSLER"]
    summary    = {}
    for cond in conditions:
        fids, precs, recs = [], [], []
        for seed in seeds:
            if cond in ("ODE", "FDS"):
                r = _eval_if_done(Path(baseline_dir) / cond / f"seed{seed}")
            else:
                r = _eval_if_done(Path(out_dir) / cond / f"seed{seed}")
            if r:
                fids.append(r["fid"]); precs.append(r["precision"]); recs.append(r["recall"])
        if fids:
            summary[cond] = {
                "fid_mean":  float(np.mean(fids)),
                "fid_std":   float(np.std(fids, ddof=1)) if len(fids) > 1 else 0.0,
                "prec_mean": float(np.mean(precs)),
                "rec_mean":  float(np.mean(recs)),
                "n_seeds":   len(fids),
            }

    print(f"\n{'Condition':<16} {'N':>3} {'FID':>8} {'±':>6} {'Prec':>6} {'Rec':>6}  C_chi")
    print("-" * 60)
    chi_map = {"ODE": "—", "FDS": "<∞(34.65)", "SHUF_ROSSLER": "<∞(22.75)", "SEQ_ROSSLER": "∞"}
    for cond in conditions:
        if cond in summary:
            s = summary[cond]
            print(f"{cond:<16} {s['n_seeds']:>3} "
                  f"{s['fid_mean']:>8.3f} {s['fid_std']:>6.3f} "
                  f"{s['prec_mean']:>6.3f} {s['rec_mean']:>6.3f}  {chi_map[cond]}")

    out_path = Path(out_dir) / "summary.json"
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSaved: {out_path}")
    return summary


# ── main ──────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model",          default=DEFAULT_MODEL)
    parser.add_argument("--inception",      default=DEFAULT_INCEPTION)
    parser.add_argument("--ref_npz",        default=DEFAULT_REF_NPZ)
    parser.add_argument("--ref_pool",       default=DEFAULT_REF_POOL)
    parser.add_argument("--out_dir",        default=DEFAULT_OUT)
    parser.add_argument("--baseline_dir",   default=BASELINE_DIR)
    parser.add_argument("--nfe",            type=int,   default=8)
    parser.add_argument("--n_samples",      type=int,   default=50000)
    parser.add_argument("--guidance_scale", type=float, default=1.5)
    parser.add_argument("--seed",           type=int,   default=42)
    parser.add_argument("--conditions",     nargs="+",
                        choices=["ODE", "FDS", "SHUF_ROSSLER", "SEQ_ROSSLER"],
                        default=["SHUF_ROSSLER", "SEQ_ROSSLER"])
    parser.add_argument("--summarize",      action="store_true")
    parser.add_argument("--seeds",          type=str,   default="42,43,44,45,46")
    args = parser.parse_args()

    Path(args.out_dir).mkdir(parents=True, exist_ok=True)

    if args.summarize:
        seeds = [int(s) for s in args.seeds.split(",")]
        summarize(args.out_dir, args.baseline_dir, seeds)
        return

    sched_map = {
        "ODE":         (FDSScheduler,             {**CHAOS_PARAMS_BASE, "tau0": 0.0, "xi_source": "shuffled_lorenz"}),
        "FDS":         (FDSScheduler,             {**CHAOS_PARAMS_BASE, "xi_source": "shuffled_lorenz"}),
        "SHUF_ROSSLER":(ShuffledRosslerScheduler, {**CHAOS_PARAMS_BASE, "xi_source": "rossler"}),
        "SEQ_ROSSLER": (SeqRosslerScheduler,      {**CHAOS_PARAMS_BASE, "xi_source": "rossler"}),
    }

    print("Loading DiT model...")
    pipe = DiTPipeline.from_pretrained(args.model, torch_dtype=torch.float16)
    pipe = pipe.to("cuda")

    results = {}
    for cond in args.conditions:
        print(f"\n=== {cond} (seed={args.seed}) ===")
        if cond in ("ODE", "FDS"):
            r = _load_baseline(cond, args.seed, args.baseline_dir)
            if r is not None:
                results[cond] = {**r, "seed": args.seed, "reused": True}
                continue
        sched_cls, params = sched_map[cond]
        t0 = time.time()
        r  = run_condition(cond, sched_cls, params, pipe, args, args.seed)
        elapsed = time.time() - t0
        results[cond] = {**r, "seed": args.seed, "elapsed_s": elapsed}
        print(f"  FID={r['fid']:.3f} Prec={r['precision']:.3f} Rec={r['recall']:.3f} ({elapsed:.0f}s)")

    out_path = Path(args.out_dir) / f"results_seed{args.seed}.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
