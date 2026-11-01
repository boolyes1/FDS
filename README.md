# FDS - Fast-Driver Sampler

[**Paper (PDF)**](assets/FDS_paper.pdf) · [**Download PDF**](https://github.com/boolyes1/FDS/raw/refs/heads/main/assets/FDS_paper.pdf)

Official implementation of **A Favorable Regime Between ODE and SDE for Few-Step Diffusion Sampling** (NeurIPS 2026).

**Yihao Bu, Dan Lian, and Zhenguo Gao**  
Mathematics, Shanghai Jiao Tong University

This repository contains the Fast-Driver Sampler (FDS) implementation and the
scripts used to reproduce the paper experiments.

Repository: [https://github.com/boolyes1/FDS](https://github.com/boolyes1/FDS)

## Citation

If you use FDS in your research, please cite our paper.

```bibtex
@inproceedings{bu2026fds,
  title = {A Favorable Regime Between {ODE} and {SDE} for Few-Step Diffusion Sampling},
  author = {Bu, Yihao and Lian, Dan and Gao, Zhenguo},
  booktitle = {Advances in Neural Information Processing Systems},
  year = {2026},
  url = {https://github.com/boolyes1/FDS}
}
```

## Requirements

Install the Python dependencies with

```bash
pip install -r requirements.txt
```

The scripts are designed for Linux GPU servers. The scheduler interface is pinned
to diffusers 0.37.1. CPU scheduler checks use PyTorch 2.5.1. Install a PyTorch
build compatible with your server's CUDA environment. ImageNet evaluation scripts use
the AFLoPS evaluator at `/root/AFLoPS/eval.py`. The model checkpoints,
reference statistics, and COCO prompt files can be supplied through command-line
arguments where the script exposes them. Run any script with `--help` to inspect
the exact defaults.

## Repository Layout

- `fds` contains the scheduler and shuffled-Lorenz driver implementation.
- `experiments/main/dit/run_dit.py` reproduces DiT-XL/2 ImageNet-256 results.
- `experiments/transfer/edm2` reproduces EDM2-img64-s ImageNet-64 results.
- `experiments/transfer/mscoco` reproduces the SD v1.5 MS-COCO transfer test.
- `experiments/ablations` contains the public ablation and control scripts.
- `EXPECTED_RESULTS.md` lists the paper-level values for checking runs.

## Smoke Test

After installing dependencies and placing the DiT checkpoint at the default path,
run a tiny generation test with one GPU.

```bash
PYTHONUTF8=1 python -X utf8 run_smoke_test.py --gpu 0
```

By default, this runs DPM++ ODE, SDE, and FDS at NFE 8 with one seed, two
classes, and one image per class. It checks imports, CUDA visibility, generated
PNG counts, and timing JSON files, then writes `smoke_summary.json` under
`/root/autodl-tmp/fds_smoke_test`. Use `--dit-model` and `--save-dir` to match a
different server layout.

```bash
PYTHONUTF8=1 python -X utf8 run_smoke_test.py --gpu 0 \
  --dit-model /path/to/DiT-XL-2-256 \
  --save-dir /path/to/fds_smoke_test
```


### Ablation Smoke Tests

Ablation smoke tests should skip FID evaluation because tiny sample counts make
the Frechet covariance estimate ill-conditioned. These commands only check model
loading, scheduler construction, generation, and JSON writing.

```bash
PYTHONUTF8=1 CUDA_VISIBLE_DEVICES=0 python -X utf8   experiments/ablations/driver_ablation/run_driver_ablation.py   --num_seeds 1 --num_classes 2 --images_per_class 1   --sources shuffled_lorenz gaussian --skip_eval   --save_dir /root/autodl-tmp/fds_smoke_ablation/driver

PYTHONUTF8=1 CUDA_VISIBLE_DEVICES=0 python -X utf8   experiments/ablations/hyperparam/run_hyperparam_sensitivity.py   --task_key tau0.3_beta0.01_w0.3-0.7   --num_classes 2 --images_per_class 1 --skip_eval   --save_dir /root/autodl-tmp/fds_smoke_ablation/hyperparam

PYTHONUTF8=1 CUDA_VISIBLE_DEVICES=0 python -X utf8   experiments/ablations/hyperparam/run_ratio_cap_sensitivity.py   --task_key dpmpp_rc0.09_nfe8   --num_classes 2 --images_per_class 1 --skip_eval   --save_dir /root/autodl-tmp/fds_smoke_ablation/ratio_cap

PYTHONUTF8=1 CUDA_VISIBLE_DEVICES=0 python -X utf8   experiments/ablations/protocol_controls/run_tiled_gaussian_control.py   --conditions TiledGaussian --n_samples 8 --batch_size 4 --skip_eval   --out_dir /root/autodl-tmp/fds_smoke_ablation/tiled

PYTHONUTF8=1 CUDA_VISIBLE_DEVICES=0 python -X utf8   experiments/ablations/protocol_controls/run_gaussian_Q_L_red.py   --n_samples 8 --batch_size 4 --seeds 42 --skip_eval   --out_dir /root/autodl-tmp/fds_smoke_ablation/gaussian_ql_red
```

## Shared FDS Settings

Unless noted otherwise, FDS uses the shuffled-Lorenz driver with
`tau0 = 0.3`, `beta = 0.01`, gate window `[0.3, 0.7]`, and
`ratio_cap_end = ratio_cap / 2`.

The reported ratio caps are

| Backbone | NFE 5 | NFE 6 | NFE 7 | NFE 8 | NFE 9 | NFE 10 | NFE 11 | NFE 12 | NFE 15 | NFE 20 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| DiT-XL/2, DPM++ | 0.09 | 0.09 | 0.09 | 0.09 | 0.06 | 0.04 | 0.04 | 0.04 | 0.03 | 0.02 |
| DiT-XL/2, DDIM | 0.09 | n.r. | n.r. | 0.09 | n.r. | 0.09 | n.r. | n.r. | n.r. | 0.02 |
| EDM2-img64-s | 0.09 | 0.09 | 0.09 | 0.09 | 0.09 | 0.09 | 0.06 | 0.06 | 0.03 | 0.03 |
| SD v1.5, MS-COCO | 0.09 | n.r. | n.r. | 0.09 | n.r. | 0.09 | n.r. | n.r. | n.r. | 0.09 |

`n.r.` means the budget is not reported for that setting.

## External Dependencies

DiT evaluation requires the external AFLoPS evaluator at `/root/AFLoPS/eval.py`,
its TensorFlow dependencies, the Inception graph, and ImageNet reference assets.
The evaluator is not included in this repository and runs on CPU.

All EDM2 variants require the external TADA repository, including
`third_party/edm2` and `third_party/dpm_solver_v3`. Install its dependencies and
set the repository location before running the EDM2 scripts.

```bash
export FDS_TADA_ROOT=/path/to/ml-tada
```

For `run_edm2_tada.py`, also pass `--tada_root /path/to/ml-tada`.
The required entry point is `codebase/edm2/sample.py`, supporting `--method`,
`--benchmark`, `--nfe`, `--order`, `--nd`, and `--k`.
The original TADA commit identifier is not recorded in this release.

COCO evaluation requires reference statistics, reference features, aligned
prompts, and CLIP weights. Set `--eval_gpu` to a GPU separate from generation
if the default GPU 5 is unavailable. Failed or incomplete evaluations retain PNGs.

## Assets

Prepare the following assets before running the scripts.

```text
DiT-XL/2-256 checkpoint
ImageNet-256 reference npz
ImageNet-256 reference pool
TensorFlow Inception graph used by AFLoPS
EDM2-img64-s checkpoint
ImageNet-64 EDM2 reference file
Stable Diffusion v1.5 checkpoint
MS-COCO val2017 5k prompt file
MS-COCO FID statistics
MS-COCO reference features for Precision and Recall
```

The defaults match our server layout. Override paths with script arguments such
as `--dit_model`, `--ref_npz`, `--ref_pool_cache`, `--inception_path`, `--net`,
`--ref`, `--sd_model`, `--prompts_file`, `--ref_stats`, and `--ref_features`.

## DiT ImageNet-256

Script

```text
experiments/main/dit/run_dit.py
```

Default setting

```text
model: DiT-XL/2-256
guidance scale: 1.5
samples per seed: 50,000
classes: 1,000
images per class: 50
seeds: 42,43,44,45,46
samplers: DPM++ and DDIM
DPM++ methods: ODE, SDE, FDS
DDIM methods: ODE, FDS
```

Commands

```bash
# DPM++ ODE
CUDA_VISIBLE_DEVICES=0 python experiments/main/dit/run_dit.py \
  --sampler dpmpp --nfe 8 --run_eval --delete_after_eval

# DPM++ SDE
CUDA_VISIBLE_DEVICES=0 python experiments/main/dit/run_dit.py \
  --sampler dpmpp --nfe 8 --sde --run_eval --delete_after_eval

# DPM++ FDS
CUDA_VISIBLE_DEVICES=0 python experiments/main/dit/run_dit.py \
  --sampler dpmpp --nfe 8 --fds --run_eval --delete_after_eval

# DDIM FDS
CUDA_VISIBLE_DEVICES=0 python experiments/main/dit/run_dit.py \
  --sampler ddim --nfe 8 --fds --run_eval --delete_after_eval
```

For the full NFE curve, run the reported budgets
`5,6,7,8,9,10,11,12,15,20`.

Main CLI parameters

```text
--sampler: dpmpp or ddim
--nfe: number of function evaluations
--fds: enable FDS
--sde: enable the DPM++ SDE variant, only valid with --sampler dpmpp
--seed_base: first seed, default 42
--num_seeds: number of seeds, default 5
--num_classes: default 1000
--images_per_class: default 50
--n_samples: total images for the FDS P1-style path, default num_classes * images_per_class
--batch_size: batch size for the FDS P1-style path, default 64
--batch_size_gen: default 200, used by non-FDS paths
--save_dir: output directory
--run_eval: run AFLoPS evaluation
--delete_after_eval: delete PNGs after successful eval
```

## EDM2 ImageNet-64

EDM2 uses second-order DPM++. TADA is run by calling its official codebase
through `--tada_root`.

Anchor budgets, ODE, SDE, and FDS

```bash
CUDA_VISIBLE_DEVICES=0 python experiments/transfer/edm2/run_edm2.py --nfe 8
```

Default parameters for `run_edm2.py`

```text
NFE: required argument, reported budgets are 5,6,7,8,9,10,11,12,15,20
methods: ode,sde,fds
samples per seed: 50,000
seeds: 42,43,44,45,46
batch size: 64
base_dir: /root/autodl-tmp/edm2
net: /root/autodl-tmp/edm2-img64-s.pkl
ref: /root/autodl-tmp/img64-ref.pkl
```

Extra ODE, SDE, and FDS budgets

```bash
python experiments/transfer/edm2/run_edm2.py \
  --jobs 5:0.09,6:0.09,7:0.09,8:0.09,9:0.09,10:0.09,11:0.06,12:0.06,15:0.03,20:0.03 \
  --methods ode,sde,fds
```

Default parameters for extra budgets in `run_edm2.py`

```text
NFE and ratio_cap: 5:0.09, 6:0.09, 7:0.09, 8:0.09, 9:0.09, 10:0.09, 11:0.06, 12:0.06, 15:0.03, 20:0.03
methods: ode,sde,fds
seeds: 42,43,44,45,46
samples per seed: 50,000
```

TADA budgets

```bash
python experiments/transfer/edm2/run_edm2_tada.py
```

Default parameters for `run_edm2_tada.py`

```text
NFE: 5,6,7,8,9,10,11,12,15,20
method: tada
seeds: 42,43,44,45,46
samples per seed: 50,000
tada_gpu: 1
tada_root: /root/ml-tada
delete_after_eval: enabled by default
```

Use `--collect_only` to summarize existing EDM2 results without launching new
jobs.

## MS-COCO With SD v1.5

Script

```text
experiments/transfer/mscoco/run_mscoco.py
```

Default setting

```text
dataset: MS-COCO val2017
samples per seed: 5,000 prompts
variants: ode,sde,fds
NFE: 5,8,10,20
seeds: 42,43,44,45,46
FDS ratio_cap: 0.09
eval GPU: 5
```

Command

```bash
CUDA_VISIBLE_DEVICES=0 python experiments/transfer/mscoco/run_mscoco.py \
  --variants ode sde fds --nfe_list 5 8 10 20
```

Summarize after runs finish

```bash
python experiments/transfer/mscoco/run_mscoco.py --summary
```

CFG robustness sweep

```bash
CUDA_VISIBLE_DEVICES=0 python experiments/transfer/mscoco/run_mscoco.py \
  --cfg_sweep 5.0 7.5 10.0 12.5 --cfg_nfe_list 8 20
```

Main CLI parameters

```text
--variants: default ode sde fds
--nfe_list: default 5 8 10 20
--num_seeds: default 5
--seed_base: default 42
--sd_model: Stable Diffusion v1.5 checkpoint path
--prompts_file: COCO val2017 5k prompt file
--ref_stats: COCO FID stats npz
--ref_features: COCO reference features for Precision and Recall
--eval_gpu: GPU used by the async eval worker
--guidance_scale: classifier-free guidance scale, default 7.5
--cfg_sweep: CFG values for the appendix robustness sweep
--cfg_nfe_list: NFE values for --cfg_sweep, default 8 20
--cfg_num_seeds: number of seeds for --cfg_sweep, default 1
--summary: merge JSON files and print a table
```

## UniPC Boundary Case

Script

```text
experiments/main/dit/run_unipc_boundary.py
```

Command

```bash
CUDA_VISIBLE_DEVICES=0 python experiments/main/dit/run_unipc_boundary.py \
  --methods ode post x0 --nfe_list 5 8 --run_eval --delete_after_eval
```

The script covers the appendix UniPC boundary case. `post` adds the FDS kick
after the full predictor-corrector step. `x0` injects into UniPC's converted
x0 prediction.

## Public Ablations

These scripts reproduce the ablations and controls reported in the paper.

| Script | Purpose | Default setting |
|---|---|---|
| `experiments/ablations/driver_ablation/run_driver_ablation.py` | driver comparison | DPM++, DiT-XL/2, NFE 8, 50k samples, five seeds |
| `experiments/ablations/driver_ablation/run_rossler_seq_vs_shuf.py` | Rossler sequential versus shuffled | DPM++, DiT-XL/2, NFE 8, 50k samples, seeds 42 to 46 |
| `experiments/ablations/protocol_controls/run_gaussian_Q_L_red.py` | clipped Gaussian with matched reduced Q | DPM++, DiT-XL/2, NFE 8, 50k samples |
| `experiments/ablations/protocol_controls/run_tiled_gaussian_control.py` | TiledGaussian control | DPM++, DiT-XL/2, NFE 8, 50k samples, seed 42 |
| `experiments/ablations/hyperparam/run_hyperparam_sensitivity.py` | gate-window sensitivity | DPM++, DiT-XL/2, NFE 8, 5k samples, one seed |
| `experiments/ablations/hyperparam/run_ratio_cap_sensitivity.py` | ratio-cap sensitivity | DPM++, DiT-XL/2, NFE 8, 5k samples, one seed |
| `experiments/analysis/run_driver_diagnostics.py` | offline appendix diagnostics | C_chi, driver proxies, q pipeline, remainder proxy, q_N convergence, margins, trajectory diagnostics |

Commands

```bash
python experiments/ablations/driver_ablation/run_driver_ablation.py
python experiments/ablations/driver_ablation/run_rossler_seq_vs_shuf.py
python experiments/ablations/protocol_controls/run_gaussian_Q_L_red.py
python experiments/ablations/protocol_controls/run_tiled_gaussian_control.py --n_samples 50000 --seed 42 --nfe 8
python experiments/ablations/hyperparam/run_hyperparam_sensitivity.py
python experiments/ablations/hyperparam/run_ratio_cap_sensitivity.py
python experiments/analysis/run_driver_diagnostics.py
```

The hyperparameter scripts use gate windows `[0.2,0.8]`, `[0.3,0.7]`, and
`[0.4,0.6]`, and ratio caps `0.01,0.02,0.04,0.06,0.09,0.12,0.15`.

The analysis script writes JSON files only. It does not generate paper figures.
Use `--tasks` to run a subset, for example `--tasks q_pipeline margin_values`.

## Checking Results

Generated metrics are written as JSON files under each script's output
directory. Compare the aggregated values with `EXPECTED_RESULTS.md`. Small
differences are expected across hardware, library versions, and evaluation
backends.
