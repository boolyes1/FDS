"""
run_dit.py

Public DiT reproduction script for DPM++ and DDIM.
DPM++ and DDIM support the full reported low-NFE grid.

Smoke-test example:
    python run_dit.py --sampler dpmpp --nfe 8 --fds \
        --num_classes 10 --images_per_class 5 --show_progress

Full experiment examples. Delete PNGs after eval to save disk.
    CUDA_VISIBLE_DEVICES=0 python run_dit.py --sampler dpmpp --nfe 5  --run_eval --delete_after_eval
    CUDA_VISIBLE_DEVICES=1 python run_dit.py --sampler dpmpp --nfe 5  --sde --run_eval --delete_after_eval
    CUDA_VISIBLE_DEVICES=2 python run_dit.py --sampler dpmpp --nfe 8  --fds --run_eval --delete_after_eval
    CUDA_VISIBLE_DEVICES=3 python run_dit.py --sampler ddim  --nfe 8  --run_eval --delete_after_eval
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

_orig_from_numpy = torch.from_numpy
def _safe_from_numpy(x):
    try:
        return _orig_from_numpy(x)
    except TypeError:
        return torch.as_tensor(x)
torch.from_numpy = _safe_from_numpy

from diffusers import (
    DiTPipeline,
    DPMSolverMultistepScheduler,
    DDIMScheduler,
)

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))  # Release root with shared schedulers.
from fds.schedulers import (
    ChaosDDIMScheduler,
    ChaosDPMSolverScheduler,
    P1ShuffledLorenzDDIMScheduler,
    P1ShuffledLorenzDPMSolverScheduler,
)

DEFAULT_DIT_MODEL = "/root/autodl-tmp/01_models/DiT-XL-2-256/DiT-XL-2-256"
DEFAULT_INCEPTION = "/root/autodl-tmp/classify_image_graph_def.pb"
DEFAULT_REF_NPZ   = "/root/autodl-tmp/VIRTUAL_imagenet256_labeled.npz"
DEFAULT_REF_POOL  = "/root/autodl-tmp/ref_pool_virtual10k.npy"

# Sampler by NFE to (ratio_cap, ratio_cap_end).
# DDIM uses a different sigma scale than DPM++, so it keeps its own caps.
NFE_RATIO_CAP = {
    "dpmpp": {
        5:  (0.09,    0.045),
        6:  (0.09,    0.045),
        7:  (0.09,    0.045),
        8:  (0.09,    0.045),
        9:  (0.06,    0.03),
        10: (0.04,    0.02),
        11: (0.04,    0.02),
        12: (0.04,    0.02),
        15: (0.03,    0.015),
        20: (0.02,    0.01),
    },
    "ddim": {
        5:  (0.09,  0.045),
        6:  (0.09,  0.045),
        7:  (0.09,  0.045),
        8:  (0.09,  0.045),
        9:  (0.06,  0.03),
        10: (0.09,  0.045),
        11: (0.04,  0.02),
        12: (0.04,  0.02),
        15: (0.03,  0.015),
        20: (0.02,  0.01),
    },
}

BASE_CONFIG_OVERRIDES_ODE = {
    "algorithm_type":    "dpmsolver++",
    "solver_order":      3,
    "solver_type":       "midpoint",
    "use_karras_sigmas": False,
    "use_lu_lambdas":    False,
    "lower_order_final": False,
    "euler_at_final":    False,
}

BASE_CONFIG_OVERRIDES_SDE = {
    "algorithm_type":    "sde-dpmsolver++",
    "solver_order":      2,
    "solver_type":       "midpoint",
    "use_karras_sigmas": False,
    "use_lu_lambdas":    False,
    "lower_order_final": False,
    "euler_at_final":    False,
}

SHUFFLED_LORENZ_FDS_PARAMS = dict(
    tau0=0.3, beta=0.01,
    p_start=0.3, p_end=0.7,
    xi_source="shuffled_lorenz",
    lorenz_sigma=10.0, lorenz_rho=28.0,
    lorenz_beta=8.0/3.0, lorenz_dt=0.005,
)


def _normalize_alphas_cumprod(scheduler):
    if hasattr(scheduler, "alphas_cumprod") and torch.is_tensor(scheduler.alphas_cumprod):
        scheduler.alphas_cumprod = scheduler.alphas_cumprod.detach().cpu().float().numpy()


def build_scheduler(sampler: str, use_fds: bool, nfe: int, base_config: dict, seed: int = 0,
                    use_sde: bool = False):
    cfg = dict(base_config)
    if sampler == "dpmpp":
        if use_sde:
            cfg.update(BASE_CONFIG_OVERRIDES_SDE)
            s = DPMSolverMultistepScheduler.from_config(cfg)
        elif use_fds:
            cfg.update(BASE_CONFIG_OVERRIDES_ODE)
            ratio_cap, ratio_cap_end = NFE_RATIO_CAP["dpmpp"][nfe]
            fds_kw = dict(**SHUFFLED_LORENZ_FDS_PARAMS, ratio_cap=ratio_cap, ratio_cap_end=ratio_cap_end, seed=seed)
            s = P1ShuffledLorenzDPMSolverScheduler.from_config(cfg)
            s.set_chaos_params(**fds_kw)
        else:
            cfg.update(BASE_CONFIG_OVERRIDES_ODE)
            s = DPMSolverMultistepScheduler.from_config(cfg)
        _normalize_alphas_cumprod(s)

    elif sampler == "ddim":
        ratio_cap, ratio_cap_end = NFE_RATIO_CAP["ddim"][nfe]
        fds_kw = dict(**SHUFFLED_LORENZ_FDS_PARAMS, ratio_cap=ratio_cap, ratio_cap_end=ratio_cap_end, seed=seed)
        if use_fds:
            s = P1ShuffledLorenzDDIMScheduler.from_config(base_config)
            s.set_chaos_params(**fds_kw)
        else:
            s = DDIMScheduler.from_config(base_config)
        _normalize_alphas_cumprod(s)

    else:
        raise ValueError(f"Unknown sampler: {sampler}")

    return s


def _save_numpy_images(images: np.ndarray, out_dir: str):
    if images.ndim == 3:
        images = images[None]
    images = (images * 255.0).round().astype(np.uint8)
    for i in range(images.shape[0]):
        Image.fromarray(images[i]).save(os.path.join(out_dir, f"sample_{i+1}.png"))


def _save_flat_images(images: np.ndarray, out_dir: str, start_index: int) -> int:
    if images.ndim == 3:
        images = images[None]
    images = (images * 255.0).round().astype(np.uint8)
    os.makedirs(out_dir, exist_ok=True)
    for i in range(images.shape[0]):
        Image.fromarray(images[i]).save(os.path.join(out_dir, f"{start_index + i:05d}.png"))
    return start_index + images.shape[0]


def _flat_png_count(seed_dir: str) -> int:
    if not os.path.exists(seed_dir):
        return 0
    with os.scandir(seed_dir) as entries:
        return sum(1 for entry in entries if entry.is_file() and entry.name.endswith(".png"))


def _run_eval(sample_dir, eval_out, ref_npz, inception_path, ref_pool_cache):
    cmd = [
        sys.executable, "/root/AFLoPS/eval.py",
        ref_npz, sample_dir,
        "--save_path", eval_out,
        "--inception_path", inception_path,
        "--image_size", "256",
        "--ref_pool_cache", ref_pool_cache,
    ]
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = ""
    env["TF_ENABLE_ONEDNN_OPTS"] = "1"
    subprocess.run(cmd, check=True, env=env)


def _delete_seed_images(seed_dir: str):
    """Delete all PNGs under one seed directory while preserving eval."""
    for root, dirs, files in os.walk(seed_dir):
        if os.path.basename(root) == "eval":
            dirs.clear()
            continue
        for f in files:
            if f.endswith(".png"):
                os.remove(os.path.join(root, f))
    # Delete empty class directories.
    with os.scandir(seed_dir) as entries:
        class_dirs = [
            entry.path for entry in entries
            if entry.is_dir() and entry.name != "eval"
        ]
    for item_path in class_dirs:
        try:
            shutil.rmtree(item_path)
        except Exception:
            pass


def _seed_eval_done(seed_dir: str) -> bool:
    """Return whether this seed already has an eval result JSON."""
    eval_dir = Path(seed_dir) / "eval"
    if not eval_dir.exists():
        return False
    with os.scandir(eval_dir) as entries:
        return any(
            entry.is_file()
            and entry.name.startswith("eval_results_")
            and entry.name.endswith(".json")
            for entry in entries
        )


def _class_done(seed_dir: str, class_label: int, images_per_class: int) -> bool:
    """Return whether one class already has enough PNG files."""
    class_dir = os.path.join(seed_dir, str(class_label))
    if not os.path.exists(class_dir):
        return False
    with os.scandir(class_dir) as entries:
        png_count = sum(
            1 for entry in entries
            if entry.is_file() and entry.name.endswith(".png")
        )
    return png_count >= images_per_class


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sampler", required=True, choices=["dpmpp", "ddim"])
    parser.add_argument("--nfe", type=int, default=8)
    parser.add_argument("--fds", action="store_true", help="Enable FDS with shuffled-Lorenz.")
    parser.add_argument("--sde", action="store_true", help="Use the SDE DPM++ variant. Only valid for dpmpp.")
    parser.add_argument("--seed_base", type=int, default=42)
    parser.add_argument("--num_seeds", type=int, default=5)
    parser.add_argument("--dit_model", default=DEFAULT_DIT_MODEL)
    parser.add_argument("--save_dir", default="/root/autodl-tmp/sampler_comparison")
    parser.add_argument("--num_classes", type=int, default=1000)
    parser.add_argument("--images_per_class", type=int, default=50)
    parser.add_argument("--n_samples", type=int, default=None,
                        help="Total images for the P1-style FDS path. Defaults to num_classes * images_per_class.")
    parser.add_argument("--batch_size", type=int, default=64,
                        help="Batch size for the P1-style FDS path.")
    parser.add_argument("--guidance_scale", type=float, default=1.5)
    parser.add_argument("--show_progress", action="store_true")
    parser.add_argument("--batch_size_gen", type=int, default=200,
                        help="Number of images generated per pipe() call across classes.")
    parser.add_argument("--run_eval", action="store_true")
    parser.add_argument("--delete_after_eval", action="store_true",
                        help="Delete PNGs after each seed eval to save disk.")
    parser.add_argument("--ref_npz", default=DEFAULT_REF_NPZ)
    parser.add_argument("--ref_pool_cache", default=DEFAULT_REF_POOL)
    parser.add_argument("--inception_path", default=DEFAULT_INCEPTION)
    args = parser.parse_args()

    if args.fds and args.sde:
        raise ValueError("--fds and --sde are mutually exclusive.")
    if args.sde and args.sampler != "dpmpp":
        raise ValueError("--sde is only defined for dpmpp in the main release path.")
    if args.fds and args.nfe not in NFE_RATIO_CAP[args.sampler]:
        valid = sorted(NFE_RATIO_CAP[args.sampler])
        raise ValueError(f"FDS caps for sampler={args.sampler} are available only for NFE {valid}.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch_dtype = torch.float16 if device.type == "cuda" else torch.float32

    variant_tag = "_fds_shuffled_lorenz" if args.fds else ("_sde" if args.sde else "")
    exp_name = f"{args.sampler}{variant_tag}_nfe{args.nfe}"
    save_path = os.path.join(args.save_dir, exp_name)
    os.makedirs(save_path, exist_ok=True)

    print(f"[INFO] sampler={args.sampler}  fds={args.fds}  sde={args.sde}  nfe={args.nfe}")
    print(f"[INFO] seeds={list(range(args.seed_base, args.seed_base + args.num_seeds))}")
    print(f"[INFO] save_dir={save_path}")
    print(f"[INFO] delete_after_eval={args.delete_after_eval}")

    pipe = DiTPipeline.from_pretrained(
        args.dit_model, torch_dtype=torch_dtype, local_files_only=True, use_safetensors=False
    )
    pipe.set_progress_bar_config(disable=not args.show_progress)

    base_config = dict(pipe.scheduler.config)
    pipe = pipe.to(device)

    # Optional speed optimization when xformers is installed.
    try:
        pipe.enable_xformers_memory_efficient_attention()
        print("[INFO] xformers memory-efficient attention enabled")
    except Exception:
        pass  # Silently skip when xformers is unavailable.

    t0_total = time.time()

    for seed_offset in range(args.num_seeds):
        seed = args.seed_base + seed_offset
        seed_dir = os.path.join(save_path, f"seed{seed}")

        # Skip completed seeds when delete_after_eval is enabled.
        if args.run_eval and args.delete_after_eval and _seed_eval_done(seed_dir):
            print(f"[INFO] seed={seed} eval already exists, skipping")
            continue

        # Reinitialize the scheduler for every seed so chaos_seed follows seed.
        pipe.scheduler = build_scheduler(args.sampler, args.fds, args.nfe, base_config, seed,
                                          use_sde=args.sde)

        os.makedirs(seed_dir, exist_ok=True)

        if args.fds:
            t0 = time.time()
            n_samples = args.n_samples or (args.num_classes * args.images_per_class)
            existing = _flat_png_count(seed_dir)
            total_images = existing
            if existing >= n_samples:
                print(f"[INFO] seed={seed} P1-style PNG complete, skipping generation")
            else:
                if existing > 0:
                    print(f"[INFO] seed={seed} partial P1-style PNG {existing}/{n_samples}; regenerating")
                    _delete_seed_images(seed_dir)
                    os.makedirs(seed_dir, exist_ok=True)
                torch.manual_seed(seed)
                np.random.seed(seed)
                class_ids = (list(range(1000)) * (n_samples // 1000 + 1))[:n_samples]
                img_idx = 0
                with torch.inference_mode():
                    for start in range(0, n_samples, args.batch_size):
                        batch = class_ids[start:start + args.batch_size]
                        out = pipe(
                            class_labels=batch,
                            num_inference_steps=args.nfe,
                            guidance_scale=args.guidance_scale,
                            output_type="np",
                        )
                        img_idx = _save_flat_images(out.images, seed_dir, img_idx)
                        total_images = img_idx
                        del out
                        gc.collect()
                        torch.cuda.empty_cache()
                        if (start // args.batch_size) % 20 == 0:
                            print(f"[INFO] seed={seed} P1-style {total_images}/{n_samples}")

            elapsed = time.time() - t0
            print(f"[INFO] seed={seed} generated {total_images} P1-style images in {elapsed:.1f}s")

            timing = {
                "sampler": args.sampler,
                "fds": args.fds,
                "sde": args.sde,
                "fds_driver": "p1_shuffled_lorenz" if args.fds else None,
                "nfe": args.nfe,
                "seed": seed,
                "elapsed_sec": elapsed,
                "num_images": total_images,
                "batch_size": args.batch_size,
            }
            with open(os.path.join(seed_dir, "timing.json"), "w") as f:
                json.dump(timing, f, indent=2)

            if args.run_eval:
                pipe.to("cpu")
                gc.collect()
                torch.cuda.empty_cache()

                eval_out = os.path.join(seed_dir, "eval")
                os.makedirs(eval_out, exist_ok=True)
                print(f"[INFO] seed={seed} starting eval after releasing GPU memory...")
                _run_eval(seed_dir, eval_out, args.ref_npz, args.inception_path, args.ref_pool_cache)
                print(f"[INFO] seed={seed} eval finished")

                pipe.to(device)
                gc.collect()
                torch.cuda.empty_cache()

                if args.delete_after_eval:
                    _delete_seed_images(seed_dir)
                    print(f"[INFO] seed={seed} PNG files deleted")
            continue

        # Generation.
        t0 = time.time()
        total_images = 0
        ipc = args.images_per_class
        bsg = args.batch_size_gen  # Total images per pipe() call.
        # bsg should be an integer multiple of ipc. Otherwise it is rounded down.
        classes_per_call = max(1, bsg // ipc)

        with torch.inference_mode():
            for start in range(0, args.num_classes, classes_per_call):
                class_batch = list(range(start, min(start + classes_per_call, args.num_classes)))

                # Checkpointing: filter completed classes.
                pending = [c for c in class_batch
                           if not _class_done(seed_dir, c, ipc)]
                total_images += (len(class_batch) - len(pending)) * ipc
                if not pending:
                    continue

                for c in pending:
                    os.makedirs(os.path.join(seed_dir, str(c)), exist_ok=True)

                # Use a different latent seed for each image.
                # This seed formula is unique for ipc <= 100 and num_classes <= 1000.
                gen_list = [
                    torch.Generator().manual_seed(seed * 10000 + c * 100 + i)
                    for c in pending
                    for i in range(ipc)
                ]
                labels = [c for c in pending for _ in range(ipc)]

                images = pipe(
                    class_labels=labels,
                    num_inference_steps=args.nfe,
                    generator=gen_list,
                    guidance_scale=1.5,
                    output_type="np",
                ).images
                images_np = np.asarray(images)
                del images
                gc.collect()
                torch.cuda.empty_cache()

                # Split and save images by class.
                for idx, c in enumerate(pending):
                    class_imgs = images_np[idx * ipc:(idx + 1) * ipc]
                    _save_numpy_images(class_imgs, os.path.join(seed_dir, str(c)))
                    total_images += ipc

        elapsed = time.time() - t0
        print(f"[INFO] seed={seed} generated {total_images} images in {elapsed:.1f}s")

        timing = {"sampler": args.sampler, "fds": args.fds, "sde": args.sde,
                  "fds_driver": "shuffled_lorenz" if args.fds else None, "nfe": args.nfe,
                  "seed": seed, "elapsed_sec": elapsed, "num_images": total_images}
        with open(os.path.join(seed_dir, "timing.json"), "w") as f:
            json.dump(timing, f, indent=2)

        # Eval.
        if args.run_eval:
            # Release GPU memory before launching the TF eval subprocess.
            pipe.to("cpu")
            gc.collect()
            torch.cuda.empty_cache()

            eval_out = os.path.join(seed_dir, "eval")
            os.makedirs(eval_out, exist_ok=True)
            print(f"[INFO] seed={seed} starting eval after releasing GPU memory...")
            _run_eval(seed_dir, eval_out, args.ref_npz, args.inception_path, args.ref_pool_cache)
            print(f"[INFO] seed={seed} eval finished")

            # Move the model back to GPU after eval.
            pipe.to(device)
            gc.collect()
            torch.cuda.empty_cache()

            # Delete images after successful eval.
            if args.delete_after_eval:
                _delete_seed_images(seed_dir)
                print(f"[INFO] seed={seed} PNG files deleted")

    print(f"[INFO] all done in {time.time() - t0_total:.1f}s")


if __name__ == "__main__":
    main()
