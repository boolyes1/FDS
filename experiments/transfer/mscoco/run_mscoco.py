"""
MS-COCO transfer experiment with async eval on a dedicated GPU.
- Model loaded once per variant
- After generation completes, eval submitted to GPU5 async queue
- Generation continues immediately without waiting for eval
- PNG deleted only after eval succeeds
"""
import argparse
import fcntl
import glob
import json
import math
import os
import queue
import subprocess
import sys
import threading
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from fds.schedulers import ChaosDPMSolverScheduler

SD_MODEL_PATH = "/root/.cache/huggingface/hub/models--runwayml--stable-diffusion-v1-5/snapshots/451f4fe16113bff5a5d2269ed5ad43b0592e9a14"
PROMPTS_FILE  = "/root/autodl-tmp/coco/val2017_captions_5k.txt"
REF_STATS     = "/root/autodl-tmp/coco/fid_stats_coco5k.npz"
REF_FEATURES  = "/root/autodl-tmp/coco/ref_features_coco5k.npy"
SAVE_DIR      = "/root/autodl-tmp/mscoco_fds"
EVAL_GPU      = "5"  # dedicated eval GPU

FDS_PARAMS = dict(
    tau0=0.3, beta=0.01, p_start=0.3, p_end=0.7, seed=0,
    ratio_cap=0.09, ratio_cap_end=0.045,
    xi_source="shuffled_lorenz", lorenz_sigma=10.0, lorenz_rho=28.0,
    lorenz_beta=8.0/3.0, lorenz_dt=0.005,
)


def load_ref_stats(ref_stats_path):
    print(f"Loading ref stats: {ref_stats_path}")
    data = np.load(ref_stats_path)
    keys = list(data.keys())
    if "mu" in data and "sigma" in data:
        mu, sigma = data["mu"], data["sigma"]
    elif "mean" in data and "cov" in data:
        mu, sigma = data["mean"], data["cov"]
    else:
        raise RuntimeError(f"npz must contain mu/sigma or mean/cov. Found: {keys}")
    data.close()
    print(f"  mu shape: {mu.shape}")
    return mu, sigma


def get_image_files(sample_dir):
    exts = {".jpg", ".jpeg", ".png", ".bmp"}
    files = sorted([
        os.path.join(sample_dir, f)
        for f in os.listdir(sample_dir)
        if Path(f).suffix.lower() in exts
    ])
    print(f"Found {len(files)} images")
    if not files:
        raise RuntimeError(f"No images in sample_dir: {sample_dir}")
    return files


def extract_inception_features(image_files, batch_size=64):
    """Extract InceptionV3 2048-d features. uint8 input matches npz preprocessing."""
    print("Extracting InceptionV3 features...")
    try:
        from torch_fidelity.feature_extractor_inceptionv3 import FeatureExtractorInceptionV3
    except ImportError as exc:
        raise RuntimeError("torch_fidelity is not installed.") from exc

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"  device: {device}")

    feat_extractor = FeatureExtractorInceptionV3(
        name="inception-v3-compat", features_list=["2048"]
    ).to(device).eval()

    from torch.utils.data import Dataset, DataLoader

    class ImageDataset(Dataset):
        def __init__(self, paths):
            self.paths = paths

        def __len__(self):
            return len(self.paths)

        def __getitem__(self, idx):
            try:
                img = Image.open(self.paths[idx]).convert("RGB")
                w, h = img.size
                crop = min(w, h)
                img = img.crop(((w - crop) // 2, (h - crop) // 2,
                                (w + crop) // 2, (h + crop) // 2))
                img = img.resize((299, 299), Image.BILINEAR)
                arr = np.array(img, dtype=np.uint8)
                return torch.from_numpy(arr).permute(2, 0, 1)
            except Exception as exc:
                print(f"  load failed {self.paths[idx]}: {exc}")
                return torch.zeros((3, 299, 299), dtype=torch.uint8)

    loader = DataLoader(ImageDataset(image_files), batch_size=batch_size,
                        shuffle=False, num_workers=4)
    all_features = []
    with torch.no_grad():
        for i, batch in enumerate(loader):
            batch = batch.to(device)
            feats = feat_extractor(batch)
            if isinstance(feats, dict):
                feats = feats["2048"]
            elif isinstance(feats, (tuple, list)):
                feats = feats[0]
            all_features.append(feats.cpu().numpy())
            if (i + 1) % 10 == 0:
                print(f"  {(i + 1) * batch_size}/{len(image_files)}")

    features = np.concatenate(all_features, axis=0)
    print(f"  features shape: {features.shape}")
    return features


def compute_fid(sample_features, ref_mu, ref_sigma):
    print("Computing FID...")
    sample_mu = np.mean(sample_features, axis=0)
    sample_sigma = np.cov(sample_features, rowvar=False)

    eps = 1e-6
    ref_sigma = ref_sigma + eps * np.eye(ref_sigma.shape[0])
    sample_sigma = sample_sigma + eps * np.eye(sample_sigma.shape[0])
    diff = sample_mu - ref_mu

    try:
        from scipy.linalg import sqrtm
        covmean = sqrtm(ref_sigma.dot(sample_sigma))
    except ImportError:
        eigvals, eigvecs = np.linalg.eigh(ref_sigma.dot(sample_sigma))
        eigvals = np.maximum(eigvals, 0)
        covmean = eigvecs.dot(np.diag(np.sqrt(eigvals))).dot(eigvecs.T)

    if np.iscomplexobj(covmean):
        covmean = covmean.real

    fid = float(diff.dot(diff) + np.trace(ref_sigma + sample_sigma - 2 * covmean))
    print(f"  FID: {fid:.4f}")
    return fid


def compute_precision_recall(sample_features, ref_features, k=3):
    """Compute Precision and Recall via kNN."""
    print("Computing Precision/Recall...")
    sample_norm = sample_features / (np.linalg.norm(sample_features, axis=1, keepdims=True) + 1e-8)
    ref_norm = ref_features / (np.linalg.norm(ref_features, axis=1, keepdims=True) + 1e-8)

    chunk_size = 500
    ref_radii = []
    for i in range(0, len(ref_norm), chunk_size):
        chunk = ref_norm[i:i + chunk_size]
        dists = 1.0 - chunk @ ref_norm.T
        dists = np.sort(dists, axis=1)
        ref_radii.append(dists[:, k])
    ref_radii = np.concatenate(ref_radii)

    sample_radii = []
    for i in range(0, len(sample_norm), chunk_size):
        chunk = sample_norm[i:i + chunk_size]
        dists = 1.0 - chunk @ sample_norm.T
        dists = np.sort(dists, axis=1)
        sample_radii.append(dists[:, k])
    sample_radii = np.concatenate(sample_radii)

    precision_hits = 0
    for i in range(0, len(sample_norm), chunk_size):
        chunk = sample_norm[i:i + chunk_size]
        dists = 1.0 - chunk @ ref_norm.T
        in_manifold = (dists <= ref_radii[np.newaxis, :]).any(axis=1)
        precision_hits += in_manifold.sum()
    precision = float(precision_hits) / len(sample_norm)

    recall_hits = 0
    for i in range(0, len(ref_norm), chunk_size):
        chunk = ref_norm[i:i + chunk_size]
        dists = 1.0 - chunk @ sample_norm.T
        in_manifold = (dists <= sample_radii[np.newaxis, :]).any(axis=1)
        recall_hits += in_manifold.sum()
    recall = float(recall_hits) / len(ref_norm)

    print(f"  Precision: {precision:.4f}, Recall: {recall:.4f}")
    return precision, recall


def compute_clip_score(image_files, prompts_file=None):
    print("Computing CLIP score...")
    try:
        import clip
    except ImportError:
        print("Warning: clip not installed, skipping")
        return float("nan")

    if prompts_file is None or not os.path.exists(prompts_file):
        print("Warning: prompts file not found, using generic prompt")
        prompts = ["an image"] * len(image_files)
    else:
        with open(prompts_file, "r") as f:
            prompts = [line.strip() for line in f if line.strip()]
        prompts = prompts[:len(image_files)]
        if len(prompts) < len(image_files):
            prompts += ["an image"] * (len(image_files) - len(prompts))

    device = "cuda" if torch.cuda.is_available() else "cpu"
    try:
        model, preprocess = clip.load("ViT-B/32", device=device)
        model.eval()
    except Exception as exc:
        print(f"Warning: CLIP load failed: {exc}")
        return float("nan")

    text_features_list = []
    with torch.no_grad():
        for i in range(0, len(prompts), 256):
            tokens = clip.tokenize(prompts[i:i + 256], truncate=True).to(device)
            text_features = model.encode_text(tokens)
            text_features = F.normalize(text_features, dim=-1)
            text_features_list.append(text_features.cpu())
    text_features = torch.cat(text_features_list, dim=0)

    image_features_list = []
    with torch.no_grad():
        for i in range(0, len(image_files), 64):
            batch_paths = image_files[i:i + 64]
            imgs = []
            for path in batch_paths:
                try:
                    imgs.append(preprocess(Image.open(path).convert("RGB")))
                except Exception:
                    imgs.append(torch.zeros(3, 224, 224))
            batch_tensor = torch.stack(imgs).to(device)
            image_features = model.encode_image(batch_tensor)
            image_features = F.normalize(image_features, dim=-1)
            image_features_list.append(image_features.cpu())
            if (i // 64 + 1) % 10 == 0:
                print(f"  {i + len(batch_paths)}/{len(image_files)}")
    image_features = torch.cat(image_features_list, dim=0)

    similarity = (image_features * text_features).sum(dim=-1)
    clip_score = float(similarity.mean().item())
    print(f"  CLIP score: {clip_score:.4f}")
    return clip_score


def run_eval_worker(args):
    print("=== COCO Eval ===")
    print(f"ref_stats:  {args.ref_stats_eval}")
    print(f"sample_dir: {args.sample_dir}")
    print(f"save_path:  {args.save_path}")
    os.makedirs(args.save_path, exist_ok=True)

    ref_mu, ref_sigma = load_ref_stats(args.ref_stats_eval)
    sample_images = get_image_files(args.sample_dir)
    sample_features = extract_inception_features(sample_images)
    fid_score = compute_fid(sample_features, ref_mu, ref_sigma)

    if os.path.exists(args.ref_features_eval):
        ref_features = np.load(args.ref_features_eval)
        precision, recall = compute_precision_recall(sample_features, ref_features)
    else:
        print(f"Warning: ref features not found at {args.ref_features_eval}, skipping Precision/Recall")
        precision, recall = float("nan"), float("nan")

    prompts_file = args.prompts_file_eval if os.path.exists(args.prompts_file_eval) else None
    clip_score = compute_clip_score(sample_images, prompts_file)

    results = {
        "fid": fid_score,
        "clip_score": clip_score,
        "precision": precision,
        "recall": recall,
        "num_samples": len(sample_images),
    }
    out_file = os.path.join(args.save_path, "eval_results.json")
    with open(out_file, "w") as f:
        json.dump(results, f, indent=2)

    print("\n=== Results ===")
    print(f"FID:        {fid_score:.4f}")
    print(f"CLIP Score: {clip_score:.4f}")
    print(f"Precision:  {precision:.4f}")
    print(f"Recall:     {recall:.4f}")
    print(f"Saved:      {out_file}")


# ---------------------------------------------------------------------------
# Async eval worker (runs on EVAL_GPU in a background thread)
# ---------------------------------------------------------------------------

class EvalWorker:
    """Single background thread that processes eval jobs one at a time on EVAL_GPU."""

    def __init__(
        self,
        eval_gpu=EVAL_GPU,
        ref_stats=REF_STATS,
        ref_features=REF_FEATURES,
        prompts_file=PROMPTS_FILE,
    ):
        self.eval_gpu = eval_gpu
        self.ref_stats = ref_stats
        self.ref_features = ref_features
        self.prompts_file = prompts_file
        self._q = queue.Queue()
        self._t = threading.Thread(target=self._loop, daemon=True)
        self._t.start()

    def submit(self, exp_dir, eval_out, key, record_callback):
        """Submit an eval job. record_callback(metrics) called on completion."""
        self._q.put((exp_dir, eval_out, key, record_callback))

    def wait_all(self):
        """Block until all submitted jobs are done."""
        self._q.join()

    def _loop(self):
        lock_path = f"/tmp/eval_gpu{self.eval_gpu}.lock"
        while True:
            exp_dir, eval_out, key, callback = self._q.get()
            try:
                print(f"  [EVAL] {key}")
                os.makedirs(eval_out, exist_ok=True)
                env = os.environ.copy()
                env["CUDA_VISIBLE_DEVICES"] = self.eval_gpu
                cmd = [
                    sys.executable, str(Path(__file__).resolve()), "--eval_worker",
                    self.ref_stats, exp_dir,
                    "--save_path", eval_out,
                    "--ref_features_eval", self.ref_features,
                    "--prompts_file_eval", self.prompts_file,
                ]
                # File lock: serialize eval across all processes sharing GPU5
                with open(lock_path, 'w') as lf:
                    fcntl.flock(lf, fcntl.LOCK_EX)
                    subprocess.run(cmd, check=True, env=env)

                # Read metrics
                with open(Path(eval_out) / "eval_results.json") as f:
                    metrics = json.load(f)
                if not metrics_complete(metrics):
                    raise RuntimeError("Incomplete evaluation metrics; PNG files retained.")
                callback(metrics)

                # Delete PNGs only after successful eval
                for f in glob.glob(os.path.join(exp_dir, "*.png")):
                    os.remove(f)

                print(f"  [EVAL DONE] {key}: FID={metrics.get('fid', 'N/A'):.3f}")
            except Exception as e:
                print(f"  [EVAL ERROR] {key}: {e}")
            finally:
                self._q.task_done()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def metrics_complete(metrics):
    return all(
        isinstance(metrics.get(key), (int, float)) and math.isfinite(metrics[key])
        for key in ("fid", "clip_score", "precision", "recall")
    )


def is_done(key, save_dir):
    gpu_id = os.environ.get("CUDA_VISIBLE_DEVICES", "all")
    for path in [
        os.path.join(save_dir, f"results_{gpu_id}.json"),
        os.path.join(save_dir, "results.json"),
    ]:
        if os.path.exists(path):
            try:
                if any(r["key"] == key and metrics_complete(r) for r in json.load(open(path))):
                    return True
            except Exception:
                pass
    return False


_save_lock = threading.Lock()

def save_record(record, save_dir):
    gpu_id = os.environ.get("CUDA_VISIBLE_DEVICES", "all")
    results_path = os.path.join(save_dir, f"results_{gpu_id}.json")
    with _save_lock:
        all_results = json.load(open(results_path)) if os.path.exists(results_path) else []
        all_results = [r for r in all_results if r["key"] != record["key"]]
        all_results.append(record)
        with open(results_path, "w") as f:
            json.dump(all_results, f, indent=2)


# ---------------------------------------------------------------------------
# Main generation loop
# ---------------------------------------------------------------------------

def _cfg_tag(guidance_scale):
    return f"cfg{guidance_scale:g}".replace(".", "p")


def run_variant(variant, nfe_list, seeds, prompts, save_dir, eval_worker, sd_model,
                guidance_scale=7.5, include_cfg_in_key=False):
    from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler
    from fds.schedulers import ChaosDPMSolverScheduler

    variant_key = f"{variant}_{_cfg_tag(guidance_scale)}" if include_cfg_in_key else variant
    pending = [(nfe, seed) for nfe in nfe_list for seed in seeds
               if not is_done(f"{variant_key}_nfe{nfe}_seed{seed}", save_dir)]
    if not pending:
        print(f"[SKIP ALL] {variant_key}")
        return

    print(f"[LOAD] {variant_key} model...")
    device = torch.device("cuda")
    pipe = StableDiffusionPipeline.from_pretrained(
        sd_model, torch_dtype=torch.float16, local_files_only=True,
        safety_checker=None, requires_safety_checker=False
    )
    base_config = dict(pipe.scheduler.config)

    if variant == "ode":
        pipe.scheduler = DPMSolverMultistepScheduler.from_config(base_config)
    elif variant == "sde":
        pipe.scheduler = DPMSolverMultistepScheduler.from_config(
            base_config, algorithm_type="sde-dpmsolver++"
        )
    elif variant == "fds":
        s = ChaosDPMSolverScheduler.from_config(base_config)
        s.set_chaos_params(**FDS_PARAMS)
        pipe.scheduler = s
    else:
        raise ValueError(f"Unknown variant: {variant}")

    pipe = pipe.to(device)
    print(f"[LOADED] {variant_key}")

    for nfe, seed in pending:
        key = f"{variant_key}_nfe{nfe}_seed{seed}"
        if is_done(key, save_dir):
            print(f"[SKIP] {key}")
            continue

        print(f"[GEN] {key}")
        exp_dir = os.path.join(save_dir, key)
        os.makedirs(exp_dir, exist_ok=True)

        existing = len(list(Path(exp_dir).glob("*.png")))
        print(f"  existing PNGs: {existing}/{len(prompts)}")

        t0 = time.time()
        with torch.no_grad():
            for i, prompt in enumerate(prompts):
                out_path = os.path.join(exp_dir, f"{i:05d}.png")
                if os.path.exists(out_path):
                    continue
                gen = torch.Generator(device=device).manual_seed(seed * 100000 + i)
                img = pipe(prompt, num_inference_steps=nfe, generator=gen,
                           guidance_scale=guidance_scale, output_type="pil").images[0]
                img.save(out_path)
                if (i + 1) % 500 == 0:
                    print(f"  generated {i+1}/{len(prompts)}")

        elapsed = time.time() - t0
        print(f"  [GEN DONE] {key} in {elapsed:.0f}s, submitting eval...")

        eval_out = os.path.join(exp_dir, "eval")

        # Capture loop variables for callback closure
        _key, _variant, _nfe, _seed, _elapsed = key, variant, nfe, seed, elapsed

        def make_callback(k, v, n, s, e):
            def callback(metrics):
                record = {"key": k, "variant": v, "nfe": n, "seed": s,
                          "guidance_scale": guidance_scale,
                          "elapsed_sec": e, **metrics}
                save_record(record, save_dir)
            return callback

        eval_worker.submit(exp_dir, eval_out, _key,
                           make_callback(_key, variant, _nfe, _seed, _elapsed))

    del pipe
    torch.cuda.empty_cache()


# ---------------------------------------------------------------------------
# Merge / print
# ---------------------------------------------------------------------------

def merge_results(save_dir):
    all_results = []
    seen_keys = set()
    merged_path = os.path.join(save_dir, "results.json")
    if os.path.exists(merged_path):
        for r in json.load(open(merged_path)):
            if metrics_complete(r) and r["key"] not in seen_keys:
                all_results.append(r)
                seen_keys.add(r["key"])
    for path in glob.glob(os.path.join(save_dir, "results_*.json")):
        for r in json.load(open(path)):
            if metrics_complete(r) and r["key"] not in seen_keys:
                all_results.append(r)
                seen_keys.add(r["key"])
    with open(merged_path, "w") as f:
        json.dump(all_results, f, indent=2)
    return all_results


def print_table(results_path):
    if not os.path.exists(results_path):
        print("No results yet.")
        return
    all_results = json.load(open(results_path))
    grouped = defaultdict(list)
    for r in all_results:
        grouped[(r["variant"], r.get("guidance_scale"), r["nfe"])].append(r)

    print(f"\n{'variant':8s} {'CFG':>5s} {'NFE':>4s} {'FID':>10s} {'CLIP':>8s} {'Prec':>7s} {'Rec':>7s} {'n':>3s}")
    print("-" * 58)
    for (variant, cfg, nfe) in sorted(grouped.keys(), key=lambda x: (x[0], -1.0 if x[1] is None else x[1], x[2])):
        recs = grouped[(variant, cfg, nfe)]
        fids  = [r["fid"]        for r in recs if r.get("fid")        is not None]
        clips = [r["clip_score"] for r in recs if r.get("clip_score") is not None]
        precs = [r["precision"]  for r in recs if r.get("precision")  is not None]
        recs_ = [r["recall"]     for r in recs if r.get("recall")     is not None]
        fid_str  = f"{np.mean(fids):.3f}±{np.std(fids):.3f}" if fids else "N/A"
        clip_str = f"{np.mean(clips):.4f}" if clips else "N/A"
        prec_str = f"{np.mean(precs):.3f}" if precs else "N/A"
        rec_str  = f"{np.mean(recs_):.3f}" if recs_ else "N/A"
        cfg_str = "N/A" if cfg is None else f"{cfg:g}"
        print(f"{variant:8s} {cfg_str:>5s} {nfe:4d} {fid_str:>10s} {clip_str:>8s} {prec_str:>7s} {rec_str:>7s} {len(recs):3d}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval_worker", action="store_true")
    parser.add_argument("ref_stats_eval", nargs="?")
    parser.add_argument("sample_dir", nargs="?")
    parser.add_argument("--save_path")
    parser.add_argument("--ref_features_eval", default=REF_FEATURES)
    parser.add_argument("--prompts_file_eval", default=PROMPTS_FILE)
    parser.add_argument("--variants", nargs="+", default=["ode", "sde", "fds"])
    parser.add_argument("--nfe_list", type=int, nargs="+", default=[5, 8, 10, 20])
    parser.add_argument("--num_seeds", type=int, default=5)
    parser.add_argument("--seed_base", type=int, default=42)
    parser.add_argument("--save_dir", default=SAVE_DIR)
    parser.add_argument("--sd_model", default=SD_MODEL_PATH)
    parser.add_argument("--prompts_file", default=PROMPTS_FILE)
    parser.add_argument("--ref_stats", default=REF_STATS)
    parser.add_argument("--ref_features", default=REF_FEATURES)
    parser.add_argument("--eval_gpu", default=EVAL_GPU,
                        help="GPU id for async eval worker (default: 5)")
    parser.add_argument("--guidance_scale", type=float, default=7.5)
    parser.add_argument("--cfg_sweep", type=float, nargs="+", default=None,
                        help="Run a CFG sweep. Defaults to ode and fds unless --variants is explicitly supplied.")
    parser.add_argument("--cfg_nfe_list", type=int, nargs="+", default=[8, 20])
    parser.add_argument("--cfg_num_seeds", type=int, default=1)
    parser.add_argument("--summary", action="store_true")
    args = parser.parse_args()

    if args.eval_worker:
        if not args.ref_stats_eval or not args.sample_dir or not args.save_path:
            raise ValueError("--eval_worker requires ref_stats, sample_dir, and --save_path.")
        run_eval_worker(args)
        return

    os.makedirs(args.save_dir, exist_ok=True)

    if args.summary:
        merge_results(args.save_dir)
        print_table(os.path.join(args.save_dir, "results.json"))
        return

    with open(args.prompts_file) as f:
        prompts = [line.strip() for line in f if line.strip()]
    print(f"Loaded {len(prompts)} prompts")

    if args.cfg_sweep is not None:
        if "--variants" not in sys.argv:
            args.variants = ["ode", "fds"]
        args.nfe_list = args.cfg_nfe_list
        args.num_seeds = args.cfg_num_seeds

    seeds = [args.seed_base + i for i in range(args.num_seeds)]

    # Single shared eval worker (serialized on EVAL_GPU)
    eval_worker = EvalWorker(
        eval_gpu=args.eval_gpu,
        ref_stats=args.ref_stats,
        ref_features=args.ref_features,
        prompts_file=args.prompts_file,
    )

    if args.cfg_sweep is not None:
        for cfg in args.cfg_sweep:
            for variant in args.variants:
                run_variant(
                    variant, args.nfe_list, seeds, prompts, args.save_dir,
                    eval_worker, args.sd_model, guidance_scale=cfg,
                    include_cfg_in_key=True,
                )
    else:
        for variant in args.variants:
            run_variant(
                variant, args.nfe_list, seeds, prompts, args.save_dir,
                eval_worker, args.sd_model, guidance_scale=args.guidance_scale,
                include_cfg_in_key=False,
            )

    print("All generation done. Waiting for eval queue to drain...")
    eval_worker.wait_all()
    print("All eval done.")

    merge_results(args.save_dir)
    print_table(os.path.join(args.save_dir, "results.json"))


if __name__ == "__main__":
    main()
