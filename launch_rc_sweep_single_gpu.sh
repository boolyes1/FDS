#!/usr/bin/env bash
set -euo pipefail

# Single-GPU r_c sensitivity sweep for Table 6.
# Fixed setting: DPM++, DiT-XL/2, NFE=8, 5k samples, 1 seed.
# Sweep: r_c in {0.01, 0.02, 0.04, 0.06, 0.09, 0.12, 0.15}.

GPU="${GPU:-0}"
SAVE_DIR="${SAVE_DIR:-/root/autodl-tmp/exp_b_ratio_cap_nfe8}"
DIT_MODEL="${DIT_MODEL:-/root/autodl-tmp/01_models/DiT-XL-2-256/DiT-XL-2-256}"
REF_NPZ="${REF_NPZ:-/root/autodl-tmp/VIRTUAL_imagenet256_labeled.npz}"
REF_POOL_CACHE="${REF_POOL_CACHE:-/root/autodl-tmp/ref_pool_virtual10k.npy}"
INCEPTION_PATH="${INCEPTION_PATH:-/root/autodl-tmp/classify_image_graph_def.pb}"

RCS=(0.01 0.02 0.04 0.06 0.09 0.12 0.15)

cd "$(dirname "$0")"
mkdir -p "${SAVE_DIR}"

echo "[INFO] r_c sweep for Table 6"
echo "[INFO] setting=DPM++ DiT-XL/2 NFE=8 5k samples seed=42"
echo "[INFO] GPU=${GPU}"
echo "[INFO] SAVE_DIR=${SAVE_DIR}"

for rc in "${RCS[@]}"; do
  task_key="dpmpp_rc${rc}_nfe8"
  echo "[INFO] Running ${task_key}"
  PYTHONUTF8=1 CUDA_VISIBLE_DEVICES="${GPU}" python -X utf8 \
    experiments/ablations/hyperparam/run_ratio_cap_sensitivity.py \
    --sampler dpmpp \
    --task_key "${task_key}" \
    --save_dir "${SAVE_DIR}" \
    --dit_model "${DIT_MODEL}" \
    --ref_npz "${REF_NPZ}" \
    --ref_pool_cache "${REF_POOL_CACHE}" \
    --inception_path "${INCEPTION_PATH}"
done

echo "[INFO] Done. Results: ${SAVE_DIR}/exp_b_ratio_cap_results.json"
