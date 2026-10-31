"""
run_ratio_cap_sensitivity.py
Ratio_cap sensitivity scan for dpmpp_fds and ddim_fds.
Fixes best hyperparams from Exp-B (tau0=0.3, beta=0.01, w=0.3-0.7),
sweeps ratio_cap across the reported NFE grid for both samplers.

Scan:
  sampler   in {dpmpp, ddim}
  ratio_cap in {0.01, 0.02, 0.04, 0.06, 0.09, 0.12, 0.15}
  NFE       in {5, 6, 7, 8, 9, 10, 11, 12, 15, 20}
  Fixed: tau0=0.3, beta=0.01, p_start=0.3, p_end=0.7, seed=42
  5000 images (1000 classes x 5), consistent with Exp-B

Usage:
    python run_ratio_cap_sensitivity.py                              # run all (dpmpp + ddim)
    python run_ratio_cap_sensitivity.py --sampler dpmpp              # dpmpp only
    python run_ratio_cap_sensitivity.py --sampler ddim               # ddim only
    python run_ratio_cap_sensitivity.py --task_key ddim_rc0.05_nfe8  # single task (for launcher)
"""
import argparse
import fcntl
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
from diffusers import DiTPipeline, DPMSolverMultistepScheduler, DDIMScheduler
from fds.schedulers import P1ShuffledLorenzDDIMScheduler, P1ShuffledLorenzDPMSolverScheduler

DEFAULT_DIT_MODEL = "/root/autodl-tmp/01_models/DiT-XL-2-256/DiT-XL-2-256"
DEFAULT_INCEPTION = "/root/autodl-tmp/classify_image_graph_def.pb"
DEFAULT_REF_NPZ   = "/root/autodl-tmp/VIRTUAL_imagenet256_labeled.npz"
DEFAULT_REF_POOL  = "/root/autodl-tmp/ref_pool_virtual10k.npy"

DPMPP_CONFIG_OVERRIDES = {
    "algorithm_type": "dpmsolver++", "solver_order": 3,
    "solver_type": "midpoint", "use_karras_sigmas": False,
    "use_lu_lambdas": False, "lower_order_final": False, "euler_at_final": False,
}

# Best hyperparams from Exp-B
FIXED_TAU0    = 0.3
FIXED_BETA    = 0.01
FIXED_P_START = 0.3
FIXED_P_END   = 0.7

RATIO_CAPS = [0.01, 0.02, 0.04, 0.06, 0.09, 0.12, 0.15]
NFES       = [5, 6, 7, 8, 9, 10, 11, 12, 15, 20]
SAMPLERS   = ["dpmpp", "ddim"]

# All tasks: (sampler, ratio_cap, nfe) pairs + one ODE baseline per sampler×NFE
ALL_TASKS = [(s, rc, nfe) for s in SAMPLERS for nfe in NFES for rc in RATIO_CAPS]


def task_key(sampler, rc, nfe):
    return f"{sampler}_rc{rc}_nfe{nfe}"


def ode_key(sampler, nfe):
    return f"{sampler}_ode_baseline_nfe{nfe}"


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


def _append_result(results_path, record):
    """Atomic append with file lock, safe for multi-process writes."""
    lock_path = results_path + ".lock"
    with open(lock_path, "w") as lf:
        fcntl.flock(lf, fcntl.LOCK_EX)
        if os.path.exists(results_path):
            with open(results_path) as f:
                data = json.load(f)
        else:
            data = []
        data.append(record)
        with open(results_path, "w") as f:
            json.dump(data, f, indent=2)
        fcntl.flock(lf, fcntl.LOCK_UN)


def run_one(pipe, base_config, device, args, key, sampler, nfe, ratio_cap=None):
    """Generate images and evaluate for one task. ratio_cap=None means ODE baseline."""
    exp_dir = os.path.join(args.save_dir, key)
    os.makedirs(exp_dir, exist_ok=True)

    cfg = dict(base_config)

    if sampler == "dpmpp":
        cfg.update(DPMPP_CONFIG_OVERRIDES)
        if ratio_cap is None:
            s = DPMSolverMultistepScheduler.from_config(cfg)
        else:
            s = P1ShuffledLorenzDPMSolverScheduler.from_config(cfg)
            s.set_chaos_params(
                tau0=FIXED_TAU0, beta=FIXED_BETA,
                p_start=FIXED_P_START, p_end=FIXED_P_END,
                seed=args.seed, ratio_cap=ratio_cap, ratio_cap_end=ratio_cap / 2,
                xi_source="shuffled_lorenz", lorenz_sigma=10.0, lorenz_rho=28.0,
                lorenz_beta=8.0 / 3.0, lorenz_dt=0.005,
            )
    else:  # ddim
        if ratio_cap is None:
            s = DDIMScheduler.from_config(cfg)
        else:
            s = P1ShuffledLorenzDDIMScheduler.from_config(cfg)
            s.set_chaos_params(
                tau0=FIXED_TAU0, beta=FIXED_BETA,
                p_start=FIXED_P_START, p_end=FIXED_P_END,
                seed=args.seed, ratio_cap=ratio_cap, ratio_cap_end=ratio_cap / 2,
                xi_source="shuffled_lorenz", lorenz_sigma=10.0, lorenz_rho=28.0,
                lorenz_beta=8.0 / 3.0, lorenz_dt=0.005,
            )

    if hasattr(s, "alphas_cumprod") and torch.is_tensor(s.alphas_cumprod):
        s.alphas_cumprod = s.alphas_cumprod.detach().cpu().float().numpy()
    pipe.scheduler = s
    pipe = pipe.to(device)

    t0 = time.time()
    with torch.no_grad():
        for class_label in range(args.num_classes):
            class_dir = os.path.join(exp_dir, str(class_label))
            if os.path.exists(class_dir) and \
                    len([f for f in os.listdir(class_dir) if f.endswith(".png")]) >= args.images_per_class:
                continue
            os.makedirs(class_dir, exist_ok=True)
            gen = torch.manual_seed(args.seed * 10000 + class_label)
            imgs = pipe(class_labels=[class_label] * args.images_per_class,
                        num_inference_steps=nfe, generator=gen,
                        guidance_scale=1.5, output_type="np").images
            _save_images(imgs, class_dir)

    if args.skip_eval:
        metrics = {"smoke_png_count": _count_pngs(exp_dir), "eval_skipped": True}
    else:
        eval_out = os.path.join(exp_dir, "eval")
        os.makedirs(eval_out, exist_ok=True)
        _run_eval(exp_dir, eval_out, args.ref_npz, args.inception_path, args.ref_pool_cache)
        metrics = _read_eval_json(eval_out)

        # Delete images to save disk space
        for item in os.listdir(exp_dir):
            item_path = os.path.join(exp_dir, item)
            if os.path.isdir(item_path) and item != "eval":
                shutil.rmtree(item_path)

    record = {
        "key": key, "sampler": sampler, "nfe": nfe, "ratio_cap": ratio_cap,
        "tau0": FIXED_TAU0 if ratio_cap is not None else None,
        "beta": FIXED_BETA if ratio_cap is not None else None,
        "p_start": FIXED_P_START if ratio_cap is not None else None,
        "p_end": FIXED_P_END if ratio_cap is not None else None,
        "fds_driver": "p1_shuffled_lorenz" if ratio_cap is not None else None,
        "elapsed_sec": time.time() - t0,
        **metrics,
    }
    return record


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num_classes", type=int, default=1000)
    parser.add_argument("--images_per_class", type=int, default=5)
    parser.add_argument("--save_dir", default="/root/autodl-tmp/exp_b_ratio_cap")
    parser.add_argument("--dit_model", default=DEFAULT_DIT_MODEL)
    parser.add_argument("--ref_npz", default=DEFAULT_REF_NPZ)
    parser.add_argument("--ref_pool_cache", default=DEFAULT_REF_POOL)
    parser.add_argument("--inception_path", default=DEFAULT_INCEPTION)
    parser.add_argument("--sampler", default=None, choices=["dpmpp", "ddim"],
                        help="Run only this sampler (default: both)")
    parser.add_argument("--task_key", default=None,
                        help="Run only this task key (for multi-GPU launcher)")
    parser.add_argument("--skip_eval", action="store_true", help="Use for smoke tests with too few samples for FID")
    args = parser.parse_args()

    os.makedirs(args.save_dir, exist_ok=True)
    results_path = os.path.join(args.save_dir, "exp_b_ratio_cap_results.json")

    done_keys = set()
    if os.path.exists(results_path):
        with open(results_path) as f:
            done_keys = {r["key"] for r in json.load(f)}

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pipe = DiTPipeline.from_pretrained(
        args.dit_model, torch_dtype=torch.float16,
        local_files_only=True, use_safetensors=False,
    )
    base_config = dict(pipe.scheduler.config)

    # Filter samplers if --sampler specified
    samplers = [args.sampler] if args.sampler else SAMPLERS

    # Build full task list: ODE baselines + FDS ratio_cap sweep
    tasks = []
    for s in samplers:
        for nfe in NFES:
            tasks.append((ode_key(s, nfe), s, nfe, None))
    for s, rc, nfe in ALL_TASKS:
        if s in samplers:
            tasks.append((task_key(s, rc, nfe), s, nfe, rc))

    for key, sampler, nfe, rc in tasks:
        if args.task_key is not None and args.task_key != key:
            continue
        if key in done_keys:
            print(f"[SKIP] {key}")
            continue

        print(f"[RUN] {key}  sampler={sampler}  nfe={nfe}  ratio_cap={rc}")
        record = run_one(pipe, base_config, device, args, key, sampler, nfe, rc)
        _append_result(results_path, record)
        fid = record.get("FID", record.get("fid"))
        rec = record.get("Recall", record.get("recall"))
        if fid is None or rec is None:
            print(f"  eval skipped  PNG={record.get('smoke_png_count', 'N/A')}  -> {results_path}")
        else:
            print(f"  FID={fid:.4f}  Recall={rec:.4f}  -> {results_path}")

    print(f"\nDone. Results: {results_path}")


if __name__ == "__main__":
    main()
