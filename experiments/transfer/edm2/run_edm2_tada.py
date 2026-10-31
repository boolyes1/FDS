from __future__ import annotations

import argparse
import json
import math
import os
import statistics
import subprocess
import sys
from pathlib import Path
from typing import Any


DEFAULT_NFES = (5, 6, 7, 8, 9, 10, 11, 12, 15, 20)
DEFAULT_SEEDS = (42, 43, 44, 45, 46)


def parse_ints(text: str) -> list[int]:
    values = [int(item.strip()) for item in text.split(",") if item.strip()]
    if not values:
        raise ValueError("List cannot be empty")
    return values


def metrics_complete(path: Path) -> bool:
    if not path.exists():
        return False
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return False
    return all(isinstance(data.get(key), (int, float)) and math.isfinite(data[key]) for key in ("fid", "fd_dinov2"))


def png_count(directory: Path) -> int:
    if not directory.exists():
        return 0
    count = 0
    with os.scandir(directory) as entries:
        for entry in entries:
            if entry.is_file() and entry.name.endswith(".png"):
                count += 1
    return count


def delete_pngs(directory: Path) -> int:
    if not directory.exists():
        return 0
    removed = 0
    with os.scandir(directory) as entries:
        for entry in entries:
            if entry.is_file() and entry.name.endswith(".png"):
                Path(entry.path).unlink()
                removed += 1
    return removed


def method_dir(base_dir: Path, nfe: int, seed: int) -> Path:
    return base_dir / "tada" / f"nfe{nfe}" / f"seed{seed}"


def parse_metrics(output: str) -> dict[str, float]:
    metrics: dict[str, float] = {}
    for line in output.splitlines():
        line = line.strip()
        for key in ("fid", "fd_dinov2"):
            if line.startswith(key):
                parts = line.split("=")
                if len(parts) == 2:
                    metrics[key] = float(parts[1].strip())
    return metrics


def run_tada_official(args: argparse.Namespace, nfe: int, seed: int, outdir: Path) -> dict[str, float]:
    tada_root = Path(args.tada_root)
    img_seed_start = seed * 100000
    env = os.environ.copy()
    env["PYTHONUTF8"] = "1"
    env["PYTHONPATH"] = (
        f"{tada_root}:"
        f"{tada_root / 'third_party' / 'edm2'}:"
        f"{tada_root / 'third_party' / 'dpm_solver_v3'}"
    )
    cmd = [
        "torchrun",
        "--standalone",
        "--nproc_per_node=1",
        f"--master_port={args.tada_port}",
        "codebase/edm2/sample.py",
        "gen",
        f"--net={args.net}",
        f"--ref={args.ref}",
        "--diz=tada_ts",
        "--method=tada",
        "--order=2",
        "--nd=2",
        "--k=1",
        "--benchmark=1",
        f"--nfe={nfe}",
        f"--seed={img_seed_start}",
        f"--num={args.num}",
        f"--batch={args.batch}",
        "--metrics=fid,fd_dinov2",
        f"--outdir={outdir}",
    ]
    print("[TADA][CMD] " + " ".join(cmd), flush=True)
    result = subprocess.run(cmd, capture_output=True, text=True, cwd=tada_root, env=env, check=False)
    output = result.stdout + result.stderr
    metrics = parse_metrics(output)
    if result.returncode != 0 or not all(
        isinstance(metrics.get(key), (int, float)) and math.isfinite(metrics[key])
        for key in ("fid", "fd_dinov2")
    ):
        raise RuntimeError(f"TADA official run failed nfe={nfe} seed={seed}, returncode={result.returncode}\n{output[-1600:]}")
    (outdir / "tada_stdout_tail.txt").write_text(output[-4000:], encoding="utf-8")
    return metrics


def run_worker(args: argparse.Namespace) -> None:
    base_dir = Path(args.base_dir)
    nfes = parse_ints(args.nfes)
    seeds = parse_ints(args.seeds)
    sample_py = Path(args.tada_root) / "codebase" / "edm2" / "sample.py"
    if not sample_py.exists():
        raise FileNotFoundError(f"Cannot find official TADA entry point {sample_py}")

    source = sample_py.read_text(encoding="utf-8", errors="replace")
    for token in ("--method", "--benchmark", "--nfe", "--order", "--nd", "--k"):
        if token not in source:
            raise RuntimeError(f"Official TADA entry point is missing expected token {token}")

    print(f"[TADA] official sample.py={sample_py}", flush=True)
    print(f"[TADA] nfes={nfes} seeds={seeds}", flush=True)
    for nfe in nfes:
        for seed in seeds:
            outdir = method_dir(base_dir, nfe, seed)
            outdir.mkdir(parents=True, exist_ok=True)
            metrics_path = outdir / "metrics.json"
            img_seed_start = seed * 100000

            if metrics_complete(metrics_path):
                removed = delete_pngs(outdir) if args.delete_after_eval else 0
                print(f"[TADA][SKIP] nfe={nfe} seed={seed}, metrics exists, removed_png={removed}", flush=True)
                continue

            metrics = run_tada_official(args, nfe, seed, outdir)
            result = {
                "method": "tada",
                "nfe": nfe,
                "seed": seed,
                "img_seed_start": img_seed_start,
                "num": args.num,
                "official_command": "sample.py gen --method=tada --order=2 --diz=tada_ts --nd=2 --k=1 --benchmark=1",
                **metrics,
            }
            metrics_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
            print(f"[TADA][DONE] nfe={nfe} seed={seed} metrics={metrics}", flush=True)

            if args.delete_after_eval:
                removed = delete_pngs(outdir)
                print(f"[TADA][CLEAN] nfe={nfe} seed={seed} removed_png={removed}", flush=True)


def collect_summary(args: argparse.Namespace) -> dict[str, Any]:
    base_dir = Path(args.base_dir)
    nfes = parse_ints(args.nfes)
    seeds = parse_ints(args.seeds)
    rows: list[dict[str, Any]] = []

    for nfe in nfes:
        records = []
        for seed in seeds:
            path = method_dir(base_dir, nfe, seed) / "metrics.json"
            if path.exists():
                records.append(json.loads(path.read_text(encoding="utf-8")))
        fids = [float(r["fid"]) for r in records if r.get("fid") is not None]
        dinos = [float(r["fd_dinov2"]) for r in records if r.get("fd_dinov2") is not None]
        rows.append(
            {
                "nfe": nfe,
                "method": "tada",
                "seeds_requested": seeds,
                "seeds_done": [r.get("seed") for r in records if r.get("fid") is not None],
                "num_per_seed": args.num,
                "fid_mean": statistics.fmean(fids) if fids else None,
                "fid_std_pop": statistics.pstdev(fids) if len(fids) > 1 else 0.0 if fids else None,
                "fd_dinov2_mean": statistics.fmean(dinos) if dinos else None,
                "fd_dinov2_std_pop": statistics.pstdev(dinos) if len(dinos) > 1 else 0.0 if dinos else None,
                "records": records,
            }
        )

    return {
        "experiment": "edm2_tada",
        "created_by": str(Path(__file__).resolve()),
        "tada_root": args.tada_root,
        "base_dir": args.base_dir,
        "net": args.net,
        "ref": args.ref,
        "nfes": nfes,
        "method": "tada",
        "seeds": seeds,
        "rows": rows,
    }


def write_summary(args: argparse.Namespace) -> None:
    summary = collect_summary(args)
    script_dir = Path(__file__).resolve().parent
    base_dir = Path(args.base_dir)
    script_dir.mkdir(parents=True, exist_ok=True)
    base_dir.mkdir(parents=True, exist_ok=True)
    for path in (script_dir / "summary_edm2_tada.json", base_dir / "summary_edm2_tada.json"):
        path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        print(f"[SUMMARY] saved {path}", flush=True)


def run_controller(args: argparse.Namespace) -> None:
    script = Path(__file__).resolve()
    if args.dry_run:
        print(json.dumps({
            "dry_run": True,
            "tada_gpu": args.tada_gpu,
            "nfes": parse_ints(args.nfes),
            "seeds": parse_ints(args.seeds),
            "base_dir": args.base_dir,
        }, indent=2), flush=True)
        return

    log_dir = script.parent / "logs_tada"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / "tada_worker.log"
    cmd = [
        sys.executable,
        str(script),
        "--worker",
        "--tada_root",
        args.tada_root,
        "--base_dir",
        args.base_dir,
        "--net",
        args.net,
        "--ref",
        args.ref,
        "--nfes",
        args.nfes,
        "--seeds",
        args.seeds,
        "--num",
        str(args.num),
        "--batch",
        str(args.batch),
        "--tada_port",
        str(args.tada_port),
    ]
    if args.delete_after_eval:
        cmd.append("--delete_after_eval")
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = args.tada_gpu
    env["PYTHONUTF8"] = "1"
    print(f"[LAUNCH] worker=tada gpu={args.tada_gpu} log={log_path}", flush=True)
    with log_path.open("w", encoding="utf-8") as f:
        ret = subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT, env=env, check=False)
    if ret.returncode != 0:
        raise RuntimeError(f"tada worker failed, log={log_path}")
    (log_dir / "launcher_summary.json").write_text(
        json.dumps([{"worker": "tada", "gpu": args.tada_gpu, "log": str(log_path), "returncode": ret.returncode}], indent=2),
        encoding="utf-8",
    )
    write_summary(args)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tada_root", default="/root/ml-tada")
    parser.add_argument("--base_dir", default="/root/autodl-tmp/edm2_tada")
    parser.add_argument("--net", default="/root/autodl-tmp/edm2-img64-s.pkl")
    parser.add_argument("--ref", default="/root/autodl-tmp/img64-ref.pkl")
    parser.add_argument("--nfes", default=",".join(str(v) for v in DEFAULT_NFES))
    parser.add_argument("--seeds", default=",".join(str(v) for v in DEFAULT_SEEDS))
    parser.add_argument("--num", type=int, default=50000)
    parser.add_argument("--batch", type=int, default=64)
    parser.add_argument("--tada_gpu", default="1")
    parser.add_argument("--tada_port", type=int, default=29611)
    parser.add_argument("--delete_after_eval", action="store_true", default=True)
    parser.add_argument("--dry_run", action="store_true")
    parser.add_argument("--collect_only", action="store_true")
    parser.add_argument("--worker", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.collect_only:
        write_summary(args)
        return
    if args.worker:
        run_worker(args)
        return
    run_controller(args)


if __name__ == "__main__":
    main()
