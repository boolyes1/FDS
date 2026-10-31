"""
Hyperparameter sensitivity sweep.
This verifies that FDS is stable over a range of hyperparameters.

Sweep parameters:
  tau0 in {0.1, 0.2, 0.3, 0.5}
  beta in {0.005, 0.01, 0.02}
  window in {(0.2,0.8), (0.3,0.7), (0.4,0.6)}

Fixed setting: sampler=dpmpp, NFE=8, ratio_cap=0.09 to 0.045, 5k samples, 1 seed.
Output: results/exp_b_results.json.

Usage:
    python run_hyperparam_sensitivity.py                          # Run all tasks serially.
    python run_hyperparam_sensitivity.py --nfe 10 --images_per_class 10
    python run_hyperparam_sensitivity.py --task_key ode_baseline  # Single-task mode for launchers.
    python run_hyperparam_sensitivity.py --task_key tau0.3_beta0.01_w0.3-0.7
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
from itertools import product
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
from fds.schedulers import P1ShuffledLorenzDPMSolverScheduler

DEFAULT_DIT_MODEL = "/root/autodl-tmp/01_models/DiT-XL-2-256/DiT-XL-2-256"
DEFAULT_INCEPTION = "/root/autodl-tmp/classify_image_graph_def.pb"
DEFAULT_REF_NPZ   = "/root/autodl-tmp/VIRTUAL_imagenet256_labeled.npz"
DEFAULT_REF_POOL  = "/root/autodl-tmp/ref_pool_virtual10k.npy"

BASE_CONFIG_OVERRIDES = {
    "algorithm_type": "dpmsolver++", "solver_order": 3,
    "solver_type": "midpoint", "use_karras_sigmas": False,
    "use_lu_lambdas": False, "lower_order_final": False, "euler_at_final": False,
}

SWEEP = {
    "tau0":   [0.1, 0.2, 0.3, 0.5],
    "beta":   [0.005, 0.01, 0.02],
    "window": [(0.2, 0.8), (0.3, 0.7), (0.4, 0.6)],
}


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


def _append_result(results_path: str, record: dict):
    """Atomically append under a file lock, safe for concurrent writers."""
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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--nfe", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num_classes", type=int, default=1000)
    parser.add_argument("--images_per_class", type=int, default=5)
    parser.add_argument("--save_dir", default="/root/autodl-tmp/exp_b_hyperparam")
    parser.add_argument("--dit_model", default=DEFAULT_DIT_MODEL)
    parser.add_argument("--ref_npz", default=DEFAULT_REF_NPZ)
    parser.add_argument("--ref_pool_cache", default=DEFAULT_REF_POOL)
    parser.add_argument("--inception_path", default=DEFAULT_INCEPTION)
    parser.add_argument("--task_key", default=None,
                        help="Run only one task key, for multi-GPU launchers")
    parser.add_argument("--skip_eval", action="store_true", help="Use for smoke tests with too few samples for FID")
    args = parser.parse_args()

    os.makedirs(args.save_dir, exist_ok=True)
    results_path = os.path.join(args.save_dir, "exp_b_results.json")

    # Load existing results for resume.
    if os.path.exists(results_path):
        with open(results_path) as f:
            all_results = json.load(f)
    else:
        all_results = []

    done_keys = {r["key"] for r in all_results}

    # Single-task mode, run only the group specified by --task_key.
    task_filter = args.task_key

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pipe = DiTPipeline.from_pretrained(args.dit_model, torch_dtype=torch.float16, local_files_only=True, use_safetensors=False)
    base_config = dict(pipe.scheduler.config)

    # ODE baseline without FDS, used as the hyperparameter reference.
    ode_key = "ode_baseline"
    if (task_filter is None or task_filter == ode_key) and ode_key not in done_keys:
        print(f"[RUN] {ode_key}")
        exp_dir = os.path.join(args.save_dir, ode_key)
        os.makedirs(exp_dir, exist_ok=True)
        cfg = dict(base_config)
        cfg.update(BASE_CONFIG_OVERRIDES)
        s = DPMSolverMultistepScheduler.from_config(cfg)
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
        record = {"key": ode_key, "tau0": None, "beta": None, "p_start": None, "p_end": None,
                  "elapsed_sec": time.time() - t0, **metrics}
        all_results.append(record)
        done_keys.add(ode_key)
        _append_result(results_path, record)
        print(f"  ODE baseline FID={metrics.get('fid', 'N/A')}  PNG={metrics.get('smoke_png_count', 'N/A')}")

    for tau0, beta, (p_start, p_end) in product(SWEEP["tau0"], SWEEP["beta"], SWEEP["window"]):
        key = f"tau{tau0}_beta{beta}_w{p_start}-{p_end}"
        if task_filter is not None and task_filter != key:
            continue
        if key in done_keys:
            print(f"[SKIP] {key}")
            continue

        print(f"[RUN] {key}")
        exp_dir = os.path.join(args.save_dir, key)
        os.makedirs(exp_dir, exist_ok=True)

        cfg = dict(base_config)
        cfg.update(BASE_CONFIG_OVERRIDES)
        s = P1ShuffledLorenzDPMSolverScheduler.from_config(cfg)
        s.set_chaos_params(
            tau0=tau0, beta=beta, p_start=p_start, p_end=p_end,
            seed=args.seed, ratio_cap=0.09, ratio_cap_end=0.045,
            xi_source="shuffled_lorenz", lorenz_sigma=10.0, lorenz_rho=28.0,
            lorenz_beta=8.0/3.0, lorenz_dt=0.005,
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

            # Delete images to save disk.
            for item in os.listdir(exp_dir):
                item_path = os.path.join(exp_dir, item)
                if os.path.isdir(item_path) and item != "eval":
                    shutil.rmtree(item_path)

        record = {"key": key, "tau0": tau0, "beta": beta,
                  "p_start": p_start, "p_end": p_end,
                  "fds_driver": "p1_shuffled_lorenz",
                  "elapsed_sec": time.time() - t0, **metrics}
        all_results.append(record)
        done_keys.add(key)

        _append_result(results_path, record)
        print(f"  FID={metrics.get('fid', 'N/A')}  saved → {results_path}")

    print(f"\nHyperparameter sensitivity done. Total records: {len(all_results)}, results: {results_path}")


if __name__ == "__main__":
    main()
