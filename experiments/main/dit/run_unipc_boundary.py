#!/usr/bin/env python3
"""
UniPC boundary-case experiment for DiT-XL/2 ImageNet-256.

The script reproduces the appendix UniPC check with three methods:

- ode: vanilla UniPC
- post: FDS kick after the full predictor-corrector step
- x0: FDS kick inside UniPC's x0 conversion path

It uses the same shuffled-Lorenz FDS protocol and AFLoPS evaluation interface as
the main DiT release script.
"""
import argparse
import gc
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
from diffusers import DiTPipeline, UniPCMultistepScheduler
from diffusers.schedulers.scheduling_utils import SchedulerOutput

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from fds.chaos_mixin import P1ShuffledLorenzMixin


DEFAULT_DIT_MODEL = "/root/autodl-tmp/01_models/DiT-XL-2-256/DiT-XL-2-256"
DEFAULT_INCEPTION = "/root/autodl-tmp/classify_image_graph_def.pb"
DEFAULT_REF_NPZ = "/root/autodl-tmp/VIRTUAL_imagenet256_labeled.npz"
DEFAULT_REF_POOL = "/root/autodl-tmp/ref_pool_virtual10k.npy"

NFE_RATIO_CAP = {
    5: (0.09, 0.045),
    6: (0.09, 0.045),
    7: (0.09, 0.045),
    8: (0.09, 0.045),
    9: (0.06, 0.03),
    10: (0.04, 0.02),
    11: (0.04, 0.02),
    12: (0.04, 0.02),
    15: (0.03, 0.015),
    20: (0.02, 0.01),
}

FDS_PARAMS = dict(
    tau0=0.3,
    beta=0.01,
    p_start=0.3,
    p_end=0.7,
    xi_source="shuffled_lorenz",
    lorenz_sigma=10.0,
    lorenz_rho=28.0,
    lorenz_beta=8.0 / 3.0,
    lorenz_dt=0.005,
)


def _sigma_pair(scheduler, timestep):
    if isinstance(timestep, torch.Tensor):
        timestep = int(timestep.item())
    matches = (scheduler.timesteps == timestep).nonzero(as_tuple=True)[0]
    step_index = int(matches[0]) if len(matches) else len(scheduler.timesteps) - 1
    prev_timestep = 0 if step_index == len(scheduler.timesteps) - 1 else int(scheduler.timesteps[step_index + 1])
    sigma_t = float(scheduler.sigma_t[int(timestep)])
    sigma_next = 0.0 if step_index == len(scheduler.timesteps) - 1 else float(scheduler.sigma_t[int(prev_timestep)])
    return step_index, sigma_t, sigma_next


class PostFDSUniPCScheduler(P1ShuffledLorenzMixin, UniPCMultistepScheduler):
    """UniPC with the standard post-step FDS trajectory kick."""

    def step(self, model_output, timestep, sample, return_dict=True):
        step_index, sigma_t, sigma_next = _sigma_pair(self, timestep)
        out = super().step(model_output=model_output, timestep=timestep, sample=sample, return_dict=return_dict)
        prev_sample = out.prev_sample if return_dict else out[0]
        prev_sample = self._chaos_apply_kick(prev_sample, sigma_t, sigma_next, step_index, len(self.timesteps))
        if return_dict:
            return SchedulerOutput(prev_sample=prev_sample)
        return (prev_sample,)


class X0FDSUniPCScheduler(P1ShuffledLorenzMixin, UniPCMultistepScheduler):
    """UniPC with FDS injected into the converted x0 prediction."""

    def convert_model_output(self, model_output, *args, sample=None, **kwargs):
        if sample is None and len(args) > 1:
            sample = args[1]
        converted = super().convert_model_output(model_output, sample=sample)
        if not getattr(self, "predict_x0", False):
            return converted
        timestep = self.timesteps[self.step_index]
        step_index, sigma_t, sigma_next = _sigma_pair(self, timestep)
        return self._chaos_apply_kick(converted, sigma_t, sigma_next, step_index, len(self.timesteps))


def build_scheduler(method, base_config, nfe, seed, solver_order):
    cfg = dict(base_config)
    cfg.update({
        "solver_order": solver_order,
        "solver_type": "bh2",
        "predict_x0": True,
        "lower_order_final": True,
    })
    if method == "ode":
        scheduler = UniPCMultistepScheduler.from_config(cfg)
    elif method == "post":
        scheduler = PostFDSUniPCScheduler.from_config(cfg)
    elif method == "x0":
        scheduler = X0FDSUniPCScheduler.from_config(cfg)
    else:
        raise ValueError(f"Unknown method: {method}")

    if method != "ode":
        ratio_cap, ratio_cap_end = NFE_RATIO_CAP[nfe]
        scheduler.set_chaos_params(**FDS_PARAMS, ratio_cap=ratio_cap, ratio_cap_end=ratio_cap_end, seed=seed)
    return scheduler


def save_flat_images(images, out_dir, start_index):
    images = np.asarray(images)
    if images.ndim == 3:
        images = images[None]
    images = (images * 255.0).round().clip(0, 255).astype(np.uint8)
    for i, image in enumerate(images):
        Image.fromarray(image).save(os.path.join(out_dir, f"{start_index + i:05d}.png"))
    return start_index + len(images)


def png_count(path):
    if not os.path.exists(path):
        return 0
    with os.scandir(path) as entries:
        return sum(1 for entry in entries if entry.is_file() and entry.name.endswith(".png"))


def eval_done(seed_dir):
    eval_dir = Path(seed_dir) / "eval"
    if not eval_dir.exists():
        return False
    with os.scandir(eval_dir) as entries:
        return any(entry.is_file() and entry.name.startswith("eval_results_") and entry.name.endswith(".json") for entry in entries)


def delete_pngs(seed_dir):
    removed = 0
    with os.scandir(seed_dir) as entries:
        for entry in entries:
            if entry.is_file() and entry.name.endswith(".png"):
                os.remove(entry.path)
                removed += 1
    return removed


def run_eval(sample_dir, eval_out, args):
    cmd = [
        sys.executable,
        "/root/AFLoPS/eval.py",
        args.ref_npz,
        sample_dir,
        "--save_path",
        eval_out,
        "--inception_path",
        args.inception_path,
        "--image_size",
        "256",
        "--ref_pool_cache",
        args.ref_pool_cache,
    ]
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = ""
    env["TF_ENABLE_ONEDNN_OPTS"] = "1"
    subprocess.run(cmd, check=True, env=env)


def read_eval(seed_dir):
    eval_dir = Path(seed_dir) / "eval"
    if not eval_dir.exists():
        return None
    files = sorted(eval_dir.glob("eval_results_*.json"))
    if not files:
        return None
    return json.loads(files[-1].read_text(encoding="utf-8"))


def summarize(save_path, records):
    groups = {}
    for rec in records:
        key = (rec["method"], rec["nfe"])
        groups.setdefault(key, []).append(rec)
    summary = []
    for (method, nfe), items in sorted(groups.items()):
        fids = [item.get("FID") or item.get("fid") for item in items if item.get("FID") is not None or item.get("fid") is not None]
        precs = [item.get("Precision") or item.get("precision") for item in items if item.get("Precision") is not None or item.get("precision") is not None]
        recs = [item.get("Recall") or item.get("recall") for item in items if item.get("Recall") is not None or item.get("recall") is not None]
        summary.append({
            "method": method,
            "nfe": nfe,
            "num_seeds": len(items),
            "fid_mean": float(np.mean(fids)) if fids else None,
            "fid_std": float(np.std(fids)) if fids else None,
            "precision_mean": float(np.mean(precs)) if precs else None,
            "recall_mean": float(np.mean(recs)) if recs else None,
        })
    out = {"records": records, "summary": summary}
    out_path = Path(save_path) / "summary_unipc_boundary.json"
    out_path.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"[SUMMARY] {out_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--methods", nargs="+", default=["ode", "post", "x0"], choices=["ode", "post", "x0"])
    parser.add_argument("--nfe_list", type=int, nargs="+", default=[5, 8])
    parser.add_argument("--seed_base", type=int, default=42)
    parser.add_argument("--num_seeds", type=int, default=5)
    parser.add_argument("--n_samples", type=int, default=50000)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--guidance_scale", type=float, default=1.5)
    parser.add_argument("--solver_order", type=int, default=2)
    parser.add_argument("--dit_model", default=DEFAULT_DIT_MODEL)
    parser.add_argument("--save_dir", default="/root/autodl-tmp/unipc_boundary")
    parser.add_argument("--run_eval", action="store_true")
    parser.add_argument("--delete_after_eval", action="store_true")
    parser.add_argument("--ref_npz", default=DEFAULT_REF_NPZ)
    parser.add_argument("--ref_pool_cache", default=DEFAULT_REF_POOL)
    parser.add_argument("--inception_path", default=DEFAULT_INCEPTION)
    args = parser.parse_args()

    for nfe in args.nfe_list:
        if nfe not in NFE_RATIO_CAP:
            raise ValueError(f"Unsupported NFE {nfe}. Valid values: {sorted(NFE_RATIO_CAP)}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float16 if device.type == "cuda" else torch.float32
    save_path = Path(args.save_dir)
    save_path.mkdir(parents=True, exist_ok=True)

    pipe = DiTPipeline.from_pretrained(args.dit_model, torch_dtype=dtype, local_files_only=True, use_safetensors=False)
    pipe = pipe.to(device)
    pipe.set_progress_bar_config(disable=True)
    base_config = dict(pipe.scheduler.config)

    records = []
    seeds = [args.seed_base + i for i in range(args.num_seeds)]
    for nfe in args.nfe_list:
        for method in args.methods:
            for seed in seeds:
                tag = f"unipc_{method}_nfe{nfe}_seed{seed}"
                seed_dir = save_path / tag
                seed_dir.mkdir(parents=True, exist_ok=True)
                if args.run_eval and args.delete_after_eval and eval_done(seed_dir):
                    print(f"[SKIP] {tag} eval exists")
                    metrics = read_eval(seed_dir) or {}
                    records.append({"key": tag, "method": method, "nfe": nfe, "seed": seed, **metrics})
                    continue

                pipe.scheduler = build_scheduler(method, base_config, nfe, seed, args.solver_order)
                existing = png_count(seed_dir)
                if existing >= args.n_samples:
                    print(f"[GEN SKIP] {tag} PNG complete")
                else:
                    if existing > 0:
                        print(f"[REGEN] {tag} partial PNG {existing}/{args.n_samples}")
                        delete_pngs(seed_dir)
                    torch.manual_seed(seed)
                    np.random.seed(seed)
                    class_ids = (list(range(1000)) * (args.n_samples // 1000 + 1))[:args.n_samples]
                    image_index = 0
                    t0 = time.time()
                    with torch.inference_mode():
                        for start in range(0, args.n_samples, args.batch_size):
                            labels = class_ids[start:start + args.batch_size]
                            out = pipe(
                                class_labels=labels,
                                num_inference_steps=nfe,
                                guidance_scale=args.guidance_scale,
                                output_type="np",
                            )
                            image_index = save_flat_images(out.images, seed_dir, image_index)
                            del out
                            gc.collect()
                            torch.cuda.empty_cache()
                            if image_index % 5000 == 0:
                                print(f"[GEN] {tag} {image_index}/{args.n_samples}")
                    elapsed = time.time() - t0
                    (seed_dir / "timing.json").write_text(
                        json.dumps({"key": tag, "method": method, "nfe": nfe, "seed": seed, "elapsed_sec": elapsed}, indent=2),
                        encoding="utf-8",
                    )

                metrics = {}
                if args.run_eval:
                    pipe.to("cpu")
                    gc.collect()
                    torch.cuda.empty_cache()
                    eval_out = seed_dir / "eval"
                    eval_out.mkdir(exist_ok=True)
                    run_eval(str(seed_dir), str(eval_out), args)
                    metrics = read_eval(seed_dir) or {}
                    pipe.to(device)
                    gc.collect()
                    torch.cuda.empty_cache()
                    if args.delete_after_eval:
                        removed = delete_pngs(seed_dir)
                        print(f"[CLEAN] {tag} removed_png={removed}")
                records.append({"key": tag, "method": method, "nfe": nfe, "seed": seed, **metrics})
                summarize(save_path, records)

    summarize(save_path, records)


if __name__ == "__main__":
    main()
