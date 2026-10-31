from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parent
DIT_SCRIPT = REPO_ROOT / "experiments" / "main" / "dit" / "run_dit.py"
DEFAULT_SAVE_DIR = Path("/root/autodl-tmp/fds_smoke_test")
DEFAULT_DIT_MODEL = Path("/root/autodl-tmp/01_models/DiT-XL-2-256/DiT-XL-2-256")


VARIANT_COMMANDS = {
    "ode": [],
    "sde": ["--sde"],
    "fds": ["--fds"],
    "ddim_ode": ["--sampler", "ddim"],
    "ddim_fds": ["--sampler", "ddim", "--fds"],
}


def parse_variants(text: str) -> list[str]:
    values = [item.strip() for item in text.split(",") if item.strip()]
    if not values:
        raise argparse.ArgumentTypeError("variants cannot be empty")
    bad = [item for item in values if item not in VARIANT_COMMANDS]
    if bad:
        allowed = ",".join(VARIANT_COMMANDS)
        raise argparse.ArgumentTypeError(f"unknown variants {bad}; allowed: {allowed}")
    return values


def scandir_png_count(path: Path) -> int:
    if not path.exists():
        return 0
    total = 0
    stack = [path]
    while stack:
        cur = stack.pop()
        with os.scandir(cur) as entries:
            for entry in entries:
                if entry.is_dir():
                    stack.append(Path(entry.path))
                elif entry.is_file() and entry.name.endswith(".png"):
                    total += 1
    return total


def scandir_file_count(path: Path, name: str) -> int:
    if not path.exists():
        return 0
    total = 0
    stack = [path]
    while stack:
        cur = stack.pop()
        with os.scandir(cur) as entries:
            for entry in entries:
                if entry.is_dir():
                    stack.append(Path(entry.path))
                elif entry.is_file() and entry.name == name:
                    total += 1
    return total


def run_checked(cmd: list[str], env: dict[str, str], cwd: Path) -> dict[str, Any]:
    start = time.time()
    print("[RUN] " + " ".join(cmd), flush=True)
    proc = subprocess.run(cmd, cwd=str(cwd), env=env, text=True)
    elapsed = time.time() - start
    return {"cmd": cmd, "returncode": proc.returncode, "elapsed_sec": elapsed}


def preflight(args: argparse.Namespace) -> dict[str, Any]:
    checks: dict[str, Any] = {}
    checks["repo_root"] = str(REPO_ROOT)
    checks["dit_script_exists"] = DIT_SCRIPT.exists()
    checks["dit_model_exists"] = args.dit_model.exists()
    checks["cuda_visible_devices"] = args.gpu

    if not checks["dit_script_exists"]:
        raise FileNotFoundError(f"Cannot find DiT script: {DIT_SCRIPT}")
    if not checks["dit_model_exists"]:
        raise FileNotFoundError(
            f"Cannot find DiT model: {args.dit_model}. "
            "Pass --dit-model to the local checkpoint directory."
        )

    env = make_env(args)
    import_cmd = [
        sys.executable,
        "-c",
        (
            "import torch, diffusers, numpy, PIL; "
            "import fds.schedulers; "
            "print('torch', torch.__version__); "
            "print('cuda', torch.cuda.is_available())"
        ),
    ]
    result = subprocess.run(
        import_cmd,
        cwd=str(REPO_ROOT),
        env=env,
        text=True,
        capture_output=True,
    )
    checks["import_returncode"] = result.returncode
    checks["import_stdout"] = result.stdout.strip().splitlines()
    checks["import_stderr_tail"] = result.stderr.strip().splitlines()[-8:]
    if result.returncode != 0:
        print(result.stdout)
        print(result.stderr)
        raise RuntimeError("Dependency import check failed")

    if not args.allow_cpu:
        cuda_ok = any(line.strip() == "cuda True" for line in checks["import_stdout"])
        if not cuda_ok:
            raise RuntimeError("CUDA is not available. Use --allow-cpu only for syntax-level debugging.")

    return checks


def make_env(args: argparse.Namespace) -> dict[str, str]:
    env = os.environ.copy()
    env["PYTHONUTF8"] = "1"
    env["CUDA_VISIBLE_DEVICES"] = args.gpu
    old_pythonpath = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = str(REPO_ROOT) if not old_pythonpath else str(REPO_ROOT) + os.pathsep + old_pythonpath
    return env


def variant_save_name(variant: str, nfe: int) -> str:
    if variant == "ode":
        return f"dpmpp_nfe{nfe}"
    if variant == "sde":
        return f"dpmpp_sde_nfe{nfe}"
    if variant == "fds":
        return f"dpmpp_fds_shuffled_lorenz_nfe{nfe}"
    if variant == "ddim_ode":
        return f"ddim_nfe{nfe}"
    if variant == "ddim_fds":
        return f"ddim_fds_shuffled_lorenz_nfe{nfe}"
    raise ValueError(variant)


def build_dit_cmd(args: argparse.Namespace, variant: str) -> list[str]:
    sampler = "dpmpp"
    extra = list(VARIANT_COMMANDS[variant])
    if variant.startswith("ddim"):
        sampler = "ddim"
        extra = [item for item in extra if item not in ["--sampler", "ddim"]]
    cmd = [
        sys.executable,
        str(DIT_SCRIPT),
        "--sampler",
        sampler,
        "--nfe",
        str(args.nfe),
        "--seed_base",
        str(args.seed_base),
        "--num_seeds",
        str(args.num_seeds),
        "--num_classes",
        str(args.num_classes),
        "--images_per_class",
        str(args.images_per_class),
        "--batch_size_gen",
        str(args.batch_size_gen),
        "--dit_model",
        str(args.dit_model),
        "--save_dir",
        str(args.save_dir),
    ]
    cmd.extend(extra)
    if args.show_progress:
        cmd.append("--show_progress")
    if args.run_eval:
        cmd.append("--run_eval")
        cmd.extend(["--ref_npz", str(args.ref_npz)])
        cmd.extend(["--ref_pool_cache", str(args.ref_pool_cache)])
        cmd.extend(["--inception_path", str(args.inception_path)])
        if args.delete_after_eval:
            cmd.append("--delete_after_eval")
    return cmd


def validate_outputs(args: argparse.Namespace, variants: list[str]) -> dict[str, Any]:
    expected_png = 0 if args.run_eval and args.delete_after_eval else args.num_seeds * args.num_classes * args.images_per_class
    report: dict[str, Any] = {}
    for variant in variants:
        out_dir = args.save_dir / variant_save_name(variant, args.nfe)
        png_count = scandir_png_count(out_dir)
        timing_count = scandir_file_count(out_dir, "timing.json")
        ok = timing_count >= args.num_seeds
        if expected_png > 0:
            ok = ok and png_count >= expected_png
        report[variant] = {
            "output_dir": str(out_dir),
            "png_count": png_count,
            "expected_png_at_least": expected_png,
            "timing_json_count": timing_count,
            "passed": ok,
        }
        if not ok:
            raise RuntimeError(f"Smoke output check failed for {variant}: {report[variant]}")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(
        description="One-command FDS release smoke test. Defaults run tiny DiT DPM++ ODE, SDE, and FDS jobs."
    )
    parser.add_argument("--gpu", default="0", help="GPU id exposed to every generation subprocess.")
    parser.add_argument("--save-dir", type=Path, default=DEFAULT_SAVE_DIR)
    parser.add_argument("--dit-model", type=Path, default=DEFAULT_DIT_MODEL)
    parser.add_argument("--variants", type=parse_variants, default=parse_variants("ode,sde,fds"))
    parser.add_argument("--nfe", type=int, default=8)
    parser.add_argument("--seed-base", type=int, default=42)
    parser.add_argument("--num-seeds", type=int, default=1)
    parser.add_argument("--num-classes", type=int, default=2)
    parser.add_argument("--images-per-class", type=int, default=1)
    parser.add_argument("--batch-size-gen", type=int, default=8)
    parser.add_argument("--show-progress", action="store_true")
    parser.add_argument("--allow-cpu", action="store_true", help="Allow CPU fallback for dependency debugging only.")
    parser.add_argument("--run-eval", action="store_true", help="Optional. Requires AFLoPS assets and is not needed for smoke testing.")
    parser.add_argument("--delete-after-eval", action="store_true")
    parser.add_argument("--ref-npz", type=Path, default=Path("/root/autodl-tmp/VIRTUAL_imagenet256_labeled.npz"))
    parser.add_argument("--ref-pool-cache", type=Path, default=Path("/root/autodl-tmp/ref_pool_virtual10k.npy"))
    parser.add_argument("--inception-path", type=Path, default=Path("/root/autodl-tmp/classify_image_graph_def.pb"))
    parser.add_argument("--preflight-only", action="store_true")
    args = parser.parse_args()

    args.save_dir = args.save_dir.resolve()
    args.dit_model = args.dit_model.resolve()
    args.ref_npz = args.ref_npz.resolve()
    args.ref_pool_cache = args.ref_pool_cache.resolve()
    args.inception_path = args.inception_path.resolve()

    if args.run_eval:
        for path in [args.ref_npz, args.ref_pool_cache, args.inception_path]:
            if not path.exists():
                raise FileNotFoundError(f"Evaluation asset does not exist: {path}")

    args.save_dir.mkdir(parents=True, exist_ok=True)
    env = make_env(args)
    summary: dict[str, Any] = {
        "repo_root": str(REPO_ROOT),
        "save_dir": str(args.save_dir),
        "started_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "settings": {
            "gpu": args.gpu,
            "variants": args.variants,
            "nfe": args.nfe,
            "seed_base": args.seed_base,
            "num_seeds": args.num_seeds,
            "num_classes": args.num_classes,
            "images_per_class": args.images_per_class,
            "run_eval": args.run_eval,
        },
    }

    print("[INFO] FDS release smoke test")
    print(f"[INFO] repo_root={REPO_ROOT}")
    print(f"[INFO] save_dir={args.save_dir}")
    summary["preflight"] = preflight(args)

    if not args.preflight_only:
        runs = []
        for variant in args.variants:
            cmd = build_dit_cmd(args, variant)
            result = run_checked(cmd, env, REPO_ROOT)
            runs.append(result)
            if result["returncode"] != 0:
                summary["runs"] = runs
                summary_path = args.save_dir / "smoke_summary.json"
                summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
                raise SystemExit(result["returncode"])
        summary["runs"] = runs
        summary["outputs"] = validate_outputs(args, args.variants)

    summary["finished_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
    summary["passed"] = True
    summary_path = args.save_dir / "smoke_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"[INFO] summary={summary_path}")
    print("SMOKE TEST PASSED")


if __name__ == "__main__":
    main()
