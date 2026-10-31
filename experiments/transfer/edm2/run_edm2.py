"""
run_edm2.py, EDM2 ImageNet-64 experiment.
Compared methods:
  - ode: hand-written DPM++ 2M order=2, standard algorithm.
  - sde: official SDE-DPMSolver++ order=2 midpoint, EDM parameterization with Gaussian noise.
  - fds: DPM++ 2M ODE plus injection protocol with shuffled Lorenz.

Usage:
"""
import os, sys, math, json, pickle, subprocess, warnings, time, statistics
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
import numpy as np
import torch
import PIL.Image

def _resolve_tada_root():
    env_root = os.environ.get("FDS_TADA_ROOT") or os.environ.get("TADA_ROOT")
    candidates = []
    if env_root:
        candidates.append(env_root)
    candidates.extend([
        "/root/ml-tada",
        os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../third_party/tada")),
        os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")),
    ])
    for candidate in candidates:
        sample_py = os.path.join(candidate, "codebase", "edm2", "sample.py")
        third_party = os.path.join(candidate, "third_party")
        if os.path.exists(sample_py) and os.path.isdir(third_party):
            return os.path.abspath(candidate)
    checked = ", ".join(os.path.abspath(item) for item in candidates)
    raise FileNotFoundError(
        "Cannot find the TADA/EDM2 dependency root. Set FDS_TADA_ROOT to the "
        f"TADA repository root. Checked: {checked}"
    )


_TADA_ROOT = _resolve_tada_root()
sys.path.insert(0, _TADA_ROOT)
sys.path.insert(0, os.path.join(_TADA_ROOT, "third_party/edm2"))
sys.path.insert(0, os.path.join(_TADA_ROOT, "third_party/dpm_solver_v3"))

from third_party.edm2 import dnnlib
from third_party.dpm_solver_v3.codebases.edm.samplers.utils import (
    NoiseScheduleEDM, model_wrapper,
)

warnings.filterwarnings("ignore", "`resume_download` is deprecated")
warnings.filterwarnings("ignore", "You are using `torch.load` with `weights_only=False`")

# Noise schedule.
def edm_rho7_schedule(nfe, sigma_max=80.0, sigma_min=0.002, rho=7.0):
    si = torch.arange(nfe + 1, dtype=torch.float64)
    return (sigma_max ** (1/rho) + si / nfe
            * (sigma_min ** (1/rho) - sigma_max ** (1/rho))) ** rho


# Lorenz pool for FDS.
_SL_POOL_SIZE = 500_000

def _build_lorenz_pool(seed=42, sigma=10.0, rho=28.0, beta=8.0/3.0,
                       dt=0.005, burnin=10000):
    y = np.array([float((seed % 100) + 1) / 100.0 + 0.1, 0.0, 0.0])
    for _ in range(burnin):
        dx = sigma * (y[1] - y[0])
        dy = y[0] * (rho - y[2]) - y[1]
        dz = y[0] * y[1] - beta * y[2]
        y = y + dt * np.array([dx, dy, dz])
    pool = np.zeros((_SL_POOL_SIZE, 3), dtype=np.float32)
    for i in range(_SL_POOL_SIZE):
        dx = sigma * (y[1] - y[0])
        dy = y[0] * (rho - y[2]) - y[1]
        dz = y[0] * y[1] - beta * y[2]
        y = y + dt * np.array([dx, dy, dz])
        pool[i] = y
    return pool

def _sl_noise(pool, rng, shape, shuffle_counter):
    numel = 1
    for s in shape:
        numel *= s
    n_states = (numel + 2) // 3
    values = pool[rng.randint(0, _SL_POOL_SIZE, size=n_states)].flatten()[:numel]
    noise = torch.tensor(values, dtype=torch.float32).view(shape)
    if noise.ndim == 4:
        B, channels, H, W = noise.shape
        for b in range(B):
            for c in range(channels):
                gen = torch.Generator()
                gen.manual_seed(shuffle_counter * 1000 + b * channels + c)
                perm = torch.randperm(H * W, generator=gen)
                noise[b, c] = noise[b, c].reshape(-1)[perm].reshape(H, W)
    return noise


# ratio_cap follows the paper EDM2 schedule, with ratio_cap_end = ratio_cap / 2.
NFE_RATIO_CAP = {
    5: (0.09, 0.045),
    6: (0.09, 0.045),
    7: (0.09, 0.045),
    8: (0.09, 0.045),
    9: (0.09, 0.045),
    10: (0.09, 0.045),
    11: (0.06, 0.03),
    12: (0.06, 0.03),
    15: (0.03, 0.015),
    20: (0.03, 0.015),
}
DEFAULT_MULTI_JOBS = {
    5: 0.09,
    6: 0.09,
    7: 0.09,
    8: 0.09,
    9: 0.09,
    10: 0.09,
    11: 0.06,
    12: 0.06,
    15: 0.03,
    20: 0.03,
}
DEFAULT_SEEDS = (42, 43, 44, 45, 46)
DEFAULT_METHODS = ("ode", "sde", "fds")


def default_jobs_text():
    return ",".join(f"{nfe}:{rc}" for nfe, rc in DEFAULT_MULTI_JOBS.items())


def default_seeds_text():
    return ",".join(str(seed) for seed in DEFAULT_SEEDS)


def parse_jobs(text):
    if not text.strip():
        return dict(DEFAULT_MULTI_JOBS)
    jobs = {}
    for item in text.split(","):
        item = item.strip()
        if not item:
            continue
        nfe_s, rc_s = item.split(":", 1)
        jobs[int(nfe_s)] = float(rc_s)
    return dict(sorted(jobs.items()))


def parse_ints(text):
    values = [int(item.strip()) for item in text.split(",") if item.strip()]
    if not values:
        raise ValueError("seeds cannot be empty")
    return values


def parse_methods(text, allowed=("ode", "sde", "fds")):
    methods = [item.strip() for item in text.split(",") if item.strip()]
    bad = [m for m in methods if m not in allowed]
    if bad:
        raise ValueError(f"Unsupported methods: {bad}")
    if not methods:
        raise ValueError("methods cannot be empty")
    return methods


def parse_gpus(text):
    gpus = [item.strip() for item in text.split(",") if item.strip()]
    if not gpus:
        raise ValueError("--gpus cannot be empty")
    return gpus


def _apply_kick(sample, sigma_t, sigma_next, step_index, num_steps,
                pool, rng, shuffle_counter,
                chaos_seed=0,
                tau0=0.3, beta_decay=0.01,
                p_start=0.3, p_end=0.7,
                ratio_cap=0.09, ratio_cap_end=0.045):
    """FDS injection protocol: gate window + τ decay + ratio-cap clipping"""
    i = step_index + 1
    progress = i / float(num_steps)
    if progress < p_start or progress > p_end:
        return sample, shuffle_counter
    sigma_t = float(sigma_t)
    sigma_next = float(sigma_next)
    if sigma_next <= 0:
        return sample, shuffle_counter
    # EDM base_scale
    base_scale = sigma_t - sigma_next
    if base_scale <= 0:
        return sample, shuffle_counter
    tau = tau0 * ((1.0 - beta_decay) ** (i - 1))
    if tau <= 0:
        return sample, shuffle_counter

    # Lorenz pool for FDS.
    noise = _sl_noise(pool, rng, sample.shape, shuffle_counter)
    noise = noise.to(device=sample.device, dtype=torch.float32)
    shuffle_counter += 1

    # Shuffled Lorenz noise.
    dims = tuple(range(1, noise.ndim))
    noise = noise - noise.mean(dim=dims, keepdim=True)
    rms = torch.sqrt((noise ** 2).mean(dim=dims, keepdim=True) + 1e-8)
    noise = noise / rms
    kick_raw = (tau * base_scale) * noise

    # ratio-cap clipping
    if ratio_cap > 0.0:
        gate_span = max(p_end - p_start, 1e-6)
        t = max(0.0, min(1.0, (progress - p_start) / gate_span))
        ratio_cap_eff = ratio_cap + (ratio_cap_end - ratio_cap) * t if ratio_cap_end is not None else ratio_cap
        kick_norm = torch.sqrt((kick_raw.float() ** 2).mean(dim=dims))
        sample_norm = torch.sqrt((sample.float() ** 2).mean(dim=dims)) + 1e-12
        ratio = kick_norm / sample_norm
        scale = torch.clamp(ratio_cap_eff / ratio, max=1.0)
        kick_raw = kick_raw * scale.view(-1, *([1] * (kick_raw.ndim - 1)))

    return sample + kick_raw.to(dtype=sample.dtype), shuffle_counter


# ══════════════════════════════════════════════════════════════════
# Method 1: ODE, DPM++ 2M order=2, standard algorithm.
# ══════════════════════════════════════════════════════════════════
def dpmpp_2m_ode(model_fn, x, sigmas):
    """Pure ODE DPM++ 2M."""
    nfe = len(sigmas) - 1
    old_denoised = None
    for i in range(nfe):
        sigma = sigmas[i]
        sigma_next = sigmas[i + 1]
        sigma_tensor = torch.tensor([sigma], device=x.device,
                                    dtype=torch.float64).expand(x.shape[0])
        eps = model_fn(x, sigma_tensor)
        denoised = x - sigma * eps

        t, t_next = sigma, sigma_next
        h = math.log(t_next / t) if t > 0 and t_next > 0 else 0.0

        if old_denoised is None or i == 0:
            x = (t_next / t) * x + (1 - t_next / t) * denoised
        else:
            h_prev = math.log(t / sigmas[i - 1]) if sigmas[i - 1] > 0 else 0.0
            r = h_prev / h if abs(h) > 1e-12 else 0.0
            d = (1 + 0.5 / r) * denoised - (0.5 / r) * old_denoised
            x = (t_next / t) * x + (1 - t_next / t) * d

        old_denoised = denoised
    return x


# ══════════════════════════════════════════════════════════════════
# Method 2: SDE, official SDE-DPMSolver++ order=2 midpoint.
# ══════════════════════════════════════════════════════════════════
# Under EDM parameterization: r = sigma_next / sigma_cur, r^2 = r*r.
# First order update.
# Second order update.
#              + σ_next * sqrt(1 - r²) * noise

def sde_dpmpp_2m(model_fn, x, sigmas, seed):
    """Official SDE-DPMSolver++ order=2 midpoint with Gaussian noise."""
    nfe = len(sigmas) - 1
    model_outputs = []

    for i in range(nfe):
        s_cur = sigmas[i]
        s_next = sigmas[i + 1]

        sigma_tensor = torch.tensor([s_cur], device=x.device,
                                    dtype=torch.float64).expand(x.shape[0])
        eps = model_fn(x, sigma_tensor)
        denoised = x - s_cur * eps
        model_outputs.append(denoised)

        r = s_next / s_cur
        r2 = r * r

        # Noise, skipped at the final step.
        if s_next > 0 and i < nfe - 1:
            noise_seed = seed + i * 10007
            gen = torch.Generator(device=x.device).manual_seed(noise_seed % (1 << 32))
            noise = torch.randn(x.shape, generator=gen, device=x.device, dtype=x.dtype)
            noise_coeff = s_next * math.sqrt(1.0 - r2)
        else:
            noise = torch.zeros_like(x)
            noise_coeff = 0.0

        D0 = model_outputs[-1]

        if len(model_outputs) < 2 or i == 0:
            x = r2 * x + (1.0 - r2) * D0 + noise_coeff * noise
        else:
            h = math.log(s_cur / s_next)
            h_prev = math.log(sigmas[i - 1] / s_cur)
            r0 = h_prev / h if abs(h) > 1e-12 else 0.0
            D1 = (1.0 / r0) * (D0 - model_outputs[-2]) if abs(r0) > 1e-12 else torch.zeros_like(D0)
            x = r2 * x + (1.0 - r2) * D0 + 0.5 * (1.0 - r2) * D1 + noise_coeff * noise

        if len(model_outputs) > 2:
            model_outputs = model_outputs[-2:]

    return x


# ══════════════════════════════════════════════════════════════════
# Lorenz pool for FDS.
# ══════════════════════════════════════════════════════════════════
def dpmpp_2m_fds(model_fn, x, sigmas, seed, ratio_cap=0.09, ratio_cap_end=0.045):
    """DPM++ 2M ODE backbone + FDS injection protocol"""
    nfe = len(sigmas) - 1
    pool = _build_lorenz_pool(seed=seed)
    rng = np.random.RandomState((seed * 7 + 13) % 100000)
    shuffle_counter = 0

    old_denoised = None
    for i in range(nfe):
        sigma = sigmas[i]
        sigma_next = sigmas[i + 1]
        sigma_tensor = torch.tensor([sigma], device=x.device,
                                    dtype=torch.float64).expand(x.shape[0])
        eps = model_fn(x, sigma_tensor)
        denoised = x - sigma * eps

        t, t_next = sigma, sigma_next
        h = math.log(t_next / t) if t > 0 and t_next > 0 else 0.0

        if old_denoised is None or i == 0:
            x = (t_next / t) * x + (1 - t_next / t) * denoised
        else:
            h_prev = math.log(t / sigmas[i - 1]) if sigmas[i - 1] > 0 else 0.0
            r = h_prev / h if abs(h) > 1e-12 else 0.0
            d = (1 + 0.5 / r) * denoised - (0.5 / r) * old_denoised
            x = (t_next / t) * x + (1 - t_next / t) * d

        old_denoised = denoised

        # FDS kick
        x, shuffle_counter = _apply_kick(
            x, sigma, sigma_next, i, nfe,
            pool, rng, shuffle_counter,
            chaos_seed=seed,
            ratio_cap=ratio_cap, ratio_cap_end=ratio_cap_end,
        )
    return x


# Utility classes.
class StackedRandomGenerator:
    def __init__(self, device, seeds):
        self.generators = [
            torch.Generator(device).manual_seed(int(s) % (1 << 32))
            for s in seeds
        ]
    def randn(self, size, **kwargs):
        assert size[0] == len(self.generators)
        return torch.stack(
            [torch.randn(size[1:], generator=g, **kwargs)
             for g in self.generators]
        )
    def randint(self, *args, size, **kwargs):
        assert size[0] == len(self.generators)
        return torch.stack(
            [torch.randint(*args, size=size[1:], generator=g, **kwargs)
             for g in self.generators]
        )


# Generation.
def generate(net, encoder, method, nfe, num_images, seed, outdir,
             max_batch_size=64, sigma_max=80.0, sigma_min=0.002,
             device=torch.device("cuda"),
             ratio_cap=0.09, ratio_cap_end=0.045):
    os.makedirs(outdir, exist_ok=True)
    existing = sum(1 for e in os.scandir(outdir) if e.name.endswith(".png"))
    if existing >= num_images:
        print(f"  [skip] {existing} images exist")
        return

    sigmas = edm_rho7_schedule(nfe, sigma_max, sigma_min).tolist()
    ns = NoiseScheduleEDM()

    all_seeds = list(range(seed, seed + num_images))
    img_idx = existing
    for batch_start in range(existing, len(all_seeds), max_batch_size):
        batch_seeds = all_seeds[batch_start:batch_start + max_batch_size]
        bs = len(batch_seeds)
        rnd = StackedRandomGenerator(device, batch_seeds)
        z = rnd.randn([bs, net.img_channels, net.img_resolution,
                        net.img_resolution], device=device)
        labels = None
        if net.label_dim > 0:
            labels = torch.eye(net.label_dim, device=device)[
                rnd.randint(net.label_dim, size=[bs], device=device)]
        mfn = model_wrapper(net, ns, labels)
        with torch.no_grad():
            if method == "ode":
                x = dpmpp_2m_ode(mfn, z * sigma_max, sigmas)
            elif method == "sde":
                x = sde_dpmpp_2m(mfn, z * sigma_max, sigmas, seed)
            elif method == "fds":
                x = dpmpp_2m_fds(mfn, z * sigma_max, sigmas, seed,
                                 ratio_cap=ratio_cap, ratio_cap_end=ratio_cap_end)
            else:
                raise ValueError(f"Unknown method: {method}")
        imgs = encoder.decode(x)
        for t in imgs:
            img_np = t.permute(1, 2, 0).clamp(0, 255).to(torch.uint8).cpu().numpy()
            PIL.Image.fromarray(img_np, "RGB").save(
                os.path.join(outdir, f"{img_idx:06d}.png"))
            img_idx += 1
        if batch_start % (max_batch_size * 10) == 0:
            print(f"  {img_idx}/{num_images}")
    print(f"  Done: {img_idx} images")


def eval_metrics(image_dir, ref_path):
    env = os.environ.copy()
    env["PYTHONPATH"] = f"{_TADA_ROOT}:{os.path.join(_TADA_ROOT, 'third_party/edm2')}:{os.path.join(_TADA_ROOT, 'third_party/dpm_solver_v3')}"
    cmd = [
        "torchrun", "--standalone", "--nproc_per_node=1",
        "codebase/edm2/sample.py", "calc",
        f"--images={image_dir}", f"--ref={ref_path}",
        "--metrics", "fid,fd_dinov2",
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, cwd=_TADA_ROOT, env=env)
    output = result.stdout + result.stderr
    metrics = {}
    for line in output.split("\n"):
        line = line.strip()
        for key in ["fid", "fd_dinov2"]:
            if line.startswith(key):
                parts = line.split("=")
                if len(parts) == 2:
                    try:
                        metrics[key] = float(parts[1].strip())
                    except ValueError:
                        pass
    if result.returncode != 0 or not all(
        isinstance(metrics.get(key), (int, float)) and math.isfinite(metrics[key])
        for key in ("fid", "fd_dinov2")
    ):
        raise RuntimeError(f"EDM2 evaluation failed or returned incomplete metrics.\n{output[-1600:]}")
    return metrics


def png_count(directory):
    if not os.path.exists(directory):
        return 0
    with os.scandir(directory) as entries:
        return sum(1 for entry in entries if entry.is_file() and entry.name.endswith(".png"))


def delete_pngs(directory):
    if not os.path.exists(directory):
        return 0
    removed = 0
    with os.scandir(directory) as entries:
        for entry in entries:
            if entry.is_file() and entry.name.endswith(".png"):
                os.remove(entry.path)
                removed += 1
    return removed


def metrics_complete(path):
    if not os.path.exists(path):
        return False
    try:
        with open(path, "r") as f:
            data = json.load(f)
    except json.JSONDecodeError:
        return False
    return all(isinstance(data.get(key), (int, float)) and math.isfinite(data[key]) for key in ("fid", "fd_dinov2"))


def load_model(net_path, device):
    print(f"Loading model from {net_path} ...")
    with dnnlib.util.open_url(net_path) as f:
        data = pickle.load(f)
    net = data["ema"].to(device)
    encoder = data.get("encoder", None)
    if encoder is None:
        encoder = dnnlib.util.construct_class_by_name(
            class_name="training.encoders.StandardRGBEncoder")
    encoder.init(device)
    return net, encoder


def summarize_records(records, title, summary_path):
    with open(summary_path, "w") as f:
        json.dump(records, f, indent=2)

    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}")
    methods = sorted({r["method"] for r in records})
    nfes = sorted({r["nfe"] for r in records})
    for nfe in nfes:
        for method in methods:
            fids = [r["fid"] for r in records
                    if r["method"] == method and r["nfe"] == nfe and r.get("fid") is not None]
            dinos = [r["fd_dinov2"] for r in records
                     if r["method"] == method and r["nfe"] == nfe and r.get("fd_dinov2") is not None]
            if fids:
                print(f"  nfe={nfe:2d} {method:10s}  FID={np.mean(fids):.3f}+-{np.std(fids):.3f}  "
                      f"FD_DINOv2={np.mean(dinos):.3f}+-{np.std(dinos):.3f}")
    print(f"\n  Saved to {summary_path}")


def run_single(args):
    if args.nfe is None:
        raise ValueError("--nfe is required unless --jobs is used.")
    seeds = parse_ints(args.seeds)
    seed_offsets = {s: s * 100000 for s in seeds}
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    net, encoder = load_model(args.net, device)

    methods = parse_methods(args.methods)
    rc, rce = NFE_RATIO_CAP[args.nfe]
    all_results = []

    for method in methods:
        for seed in seeds:
            tag = f"{method}_nfe{args.nfe}_seed{seed}"
            outdir = os.path.join(args.base_dir, tag)
            json_path = os.path.join(outdir, "metrics.json")

            if metrics_complete(json_path):
                with open(json_path) as f:
                    cached = json.load(f)
                print(f"  [cached] {tag}: fid={cached.get('fid')}, fd_dinov2={cached.get('fd_dinov2')}")
                all_results.append(cached)
                continue

            print(f"\n{'='*60}")
            print(f"  {tag} (img_seed_start={seed_offsets[seed]})")
            print(f"{'='*60}")

            generate(net, encoder, method, args.nfe, args.num, seed_offsets[seed],
                     outdir, max_batch_size=args.batch, device=device,
                     ratio_cap=rc, ratio_cap_end=rce)

            if args.no_eval:
                all_results.append({"method": method, "nfe": args.nfe, "seed": seed,
                                    "num": args.num, "eval_skipped": True})
                continue
            net.cpu(); torch.cuda.empty_cache()
            metrics = eval_metrics(outdir, args.ref)
            net.to(device)
            print(f"  fid={metrics.get('fid')}, fd_dinov2={metrics.get('fd_dinov2')}")

            result = {"method": method, "nfe": args.nfe, "seed": seed,
                      "img_seed_start": seed_offsets[seed], "num": args.num, **metrics}
            all_results.append(result)

            with open(json_path, "w") as f:
                json.dump(result, f, indent=2)

            # Clean PNG files after successful eval.
            if metrics_complete(json_path) and args.delete_after_eval:
                delete_pngs(outdir)

    summary_path = os.path.join(args.base_dir, f"summary_nfe{args.nfe}.json")
    summarize_records(all_results, f"SUMMARY NFE={args.nfe} ({args.num} images, {len(seeds)} seeds)", summary_path)


def run_multi_worker(args):
    jobs = parse_jobs(args.jobs)
    methods = parse_methods(args.methods, allowed=DEFAULT_METHODS)
    seeds = parse_ints(args.seeds)
    nfe = int(args.worker_nfe)
    if nfe not in jobs:
        raise RuntimeError(f"worker nfe={nfe} is not in jobs")
    rc = jobs[nfe]
    rc_end = rc / 2.0

    NFE_RATIO_CAP[nfe] = (rc, rc_end)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] worker nfe={nfe} rc={rc} rc_end={rc_end} device={device}", flush=True)
    print(f"[INFO] methods={methods} seeds={seeds}", flush=True)

    net, encoder = load_model(args.net, device)
    all_results = []
    os.makedirs(args.base_dir, exist_ok=True)

    for method in methods:
        for seed in seeds:
            img_seed_start = seed * 100000
            tag = f"{method}_nfe{nfe}_rc{rc:.3f}_seed{seed}".replace(".", "p")
            outdir = os.path.join(args.base_dir, tag)
            os.makedirs(outdir, exist_ok=True)
            json_path = os.path.join(outdir, "metrics.json")

            if metrics_complete(json_path):
                with open(json_path, "r") as f:
                    cached = json.load(f)
                removed = delete_pngs(outdir) if args.delete_after_eval else 0
                print(f"[SKIP] {tag} metrics already exist, removed PNG files: {removed}", flush=True)
                all_results.append(cached)
                continue

            existing = png_count(outdir)
            if existing >= args.num:
                print(f"[INFO] {tag} PNG files are complete, skipping generation and running eval", flush=True)
            else:
                print(f"[INFO] {tag} starting generation, existing PNG {existing}/{args.num}", flush=True)
                generate(
                    net,
                    encoder,
                    method,
                    nfe,
                    args.num,
                    img_seed_start,
                    outdir,
                    max_batch_size=args.batch,
                    device=device,
                    ratio_cap=rc,
                    ratio_cap_end=rc_end,
                )

            if args.no_eval:
                result = {
                    "method": method,
                    "nfe": nfe,
                    "ratio_cap": rc,
                    "ratio_cap_end": rc_end,
                    "seed": seed,
                    "img_seed_start": img_seed_start,
                    "num": args.num,
                    "eval_skipped": True,
                }
                with open(json_path, "w") as f:
                    json.dump(result, f, indent=2)
                all_results.append(result)
                continue

            net.cpu()
            torch.cuda.empty_cache()
            metrics = eval_metrics(outdir, args.ref)
            net.to(device)
            print(f"[INFO] {tag} metrics={metrics}", flush=True)

            result = {
                "method": method,
                "nfe": nfe,
                "ratio_cap": rc,
                "ratio_cap_end": rc_end,
                "seed": seed,
                "img_seed_start": img_seed_start,
                "num": args.num,
                **metrics,
            }
            with open(json_path, "w") as f:
                json.dump(result, f, indent=2)
            all_results.append(result)

            if metrics_complete(json_path) and args.delete_after_eval:
                removed = delete_pngs(outdir)
                print(f"[INFO] {tag} removed PNG files: {removed}", flush=True)

    summary_path = os.path.join(args.base_dir, f"summary_nfe{nfe}.json")
    summarize_records(all_results, f"SUMMARY NFE={nfe}", summary_path)


def collect_multi_summary(args):
    jobs = parse_jobs(args.jobs)
    methods = parse_methods(args.methods, allowed=DEFAULT_METHODS)
    seeds = parse_ints(args.seeds)
    rows = []

    for nfe, rc in jobs.items():
        for method in methods:
            values = []
            for seed in seeds:
                tag = f"{method}_nfe{nfe}_rc{rc:.3f}_seed{seed}".replace(".", "p")
                path = os.path.join(args.base_dir, tag, "metrics.json")
                if os.path.exists(path):
                    with open(path, "r") as f:
                        values.append(json.load(f))
            fids = [float(v["fid"]) for v in values if v.get("fid") is not None]
            dinos = [float(v["fd_dinov2"]) for v in values if v.get("fd_dinov2") is not None]
            rows.append({
                "nfe": nfe,
                "method": method,
                "ratio_cap": rc if method == "fds" else None,
                "ratio_cap_end": rc / 2.0 if method == "fds" else None,
                "seeds_requested": seeds,
                "seeds_done": [v.get("seed") for v in values if v.get("fid") is not None],
                "num_per_seed": args.num,
                "fid_mean": statistics.fmean(fids) if fids else None,
                "fid_std_pop": statistics.pstdev(fids) if len(fids) > 1 else 0.0 if fids else None,
                "fd_dinov2_mean": statistics.fmean(dinos) if dinos else None,
                "fd_dinov2_std_pop": statistics.pstdev(dinos) if len(dinos) > 1 else 0.0 if dinos else None,
                "records": values,
            })

    return {
        "experiment": "edm2",
        "created_by": str(Path(__file__).resolve()),
        "base_dir": args.base_dir,
        "net": args.net,
        "ref": args.ref,
        "jobs": {str(k): [v, v / 2.0] for k, v in jobs.items()},
        "methods": methods,
        "seeds": seeds,
        "rows": rows,
    }


def write_multi_summary(args):
    summary = collect_multi_summary(args)
    script_dir = Path(__file__).resolve().parent
    base_dir = Path(args.base_dir)
    base_dir.mkdir(parents=True, exist_ok=True)
    for path in (script_dir / "summary_edm2.json", base_dir / "summary_edm2.json"):
        path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        print(f"[INFO] saved {path}", flush=True)


def run_multi_controller(args):
    jobs = parse_jobs(args.jobs)
    methods = parse_methods(args.methods, allowed=DEFAULT_METHODS)
    seeds = parse_ints(args.seeds)
    gpus = parse_gpus(args.gpus)

    print(f"[INFO] jobs={jobs} methods={methods} seeds={seeds} gpus={gpus}", flush=True)
    if args.dry_run:
        print("[DRY-RUN] no jobs launched", flush=True)
        return

    queues = {gpu: [] for gpu in gpus}
    for idx, item in enumerate(jobs.items()):
        queues[gpus[idx % len(gpus)]].append(item)

    script_dir = Path(__file__).resolve().parent
    log_dir = script_dir / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)

    def run_gpu_queue(gpu, queue):
        gpu_results = []
        for nfe, rc in queue:
            log_path = log_dir / f"nfe{nfe}_rc{rc:.3f}_gpu{gpu}.log".replace(".", "p")
            cmd = [
                sys.executable,
                str(Path(__file__).resolve()),
                "--worker",
                "--worker_nfe",
                str(nfe),
                "--base_dir",
                args.base_dir,
                "--net",
                args.net,
                "--ref",
                args.ref,
                "--jobs",
                args.jobs,
                "--methods",
                args.methods,
                "--seeds",
                args.seeds,
                "--num",
                str(args.num),
                "--batch",
                str(args.batch),
            ]
            if args.no_eval:
                cmd.append("--no_eval")
            if args.delete_after_eval:
                cmd.append("--delete_after_eval")
            env = os.environ.copy()
            env["CUDA_VISIBLE_DEVICES"] = gpu
            env["PYTHONUTF8"] = "1"
            print(f"[LAUNCH] gpu={gpu} nfe={nfe} rc={rc} log={log_path}", flush=True)
            with log_path.open("w", encoding="utf-8") as f:
                ret = subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT, env=env, check=False)
            gpu_results.append({"gpu": gpu, "nfe": nfe, "rc": rc, "returncode": ret.returncode, "log": str(log_path)})
            if ret.returncode != 0:
                raise RuntimeError(f"worker failed gpu={gpu} nfe={nfe} rc={rc} log={log_path}")
        return gpu_results

    all_worker_results = []
    with ThreadPoolExecutor(max_workers=len(gpus)) as executor:
        futures = [executor.submit(run_gpu_queue, gpu, queue) for gpu, queue in queues.items() if queue]
        for future in as_completed(futures):
            all_worker_results.extend(future.result())

    worker_summary = {
        "finished_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "workers": all_worker_results,
    }
    (script_dir / "worker_summary.json").write_text(
        json.dumps(worker_summary, indent=2),
        encoding="utf-8",
    )
    write_multi_summary(args)


def build_parser():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--net", default="/root/autodl-tmp/edm2-img64-s.pkl")
    parser.add_argument("--ref", default="/root/autodl-tmp/img64-ref.pkl")
    parser.add_argument("--nfe", type=int, default=None,
                        help="Single-NFE mode. Use --jobs for multi-NFE mode.")
    parser.add_argument("--jobs", default=None,
                        help="Multi-NFE jobs as nfe:ratio_cap pairs, for example 6:0.09,7:0.09.")
    parser.add_argument("--methods", default="ode,sde,fds",
                        help="Comma-separated methods. Multi mode supports ode,sde,fds.")
    parser.add_argument("--num", type=int, default=50000)
    parser.add_argument("--seeds", type=str, default=default_seeds_text())
    parser.add_argument("--batch", type=int, default=64)
    parser.add_argument("--base_dir", default="/root/autodl-tmp/edm2")
    parser.add_argument("--gpus", default="0,1")
    parser.add_argument("--no_eval", action="store_true")
    parser.add_argument("--delete_after_eval", action="store_true", default=True)
    parser.add_argument("--dry_run", action="store_true")
    parser.add_argument("--worker", action="store_true")
    parser.add_argument("--worker_nfe", type=int, default=None)
    parser.add_argument("--collect_only", action="store_true")
    return parser


def main():
    args = build_parser().parse_args()
    if args.jobs is None:
        args.jobs = default_jobs_text()
    if args.collect_only:
        write_multi_summary(args)
        return
    if args.worker:
        if args.worker_nfe is None:
            raise RuntimeError("--worker requires --worker_nfe")
        run_multi_worker(args)
        return
    if args.nfe is not None and "--jobs" not in sys.argv:
        run_single(args)
        return
    run_multi_controller(args)


if __name__ == "__main__":
    main()
