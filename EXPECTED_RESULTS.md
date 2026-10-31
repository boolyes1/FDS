# Expected Results

This file lists the paper-level results to use when checking a reproduction run.
Small differences are expected across hardware, library versions, and evaluation
backends.

Suggested tolerance for 50k ImageNet runs is about `0.05` FID for deterministic
baselines and about `0.10` FID for FDS or SDE variants. For 5k MS-COCO runs,
use about `0.30` FID.

## ImageNet-256 with DiT-XL/2

Results report `FID / Recall` over 50k samples and five seeds.

| Method | NFE 5 | NFE 8 | NFE 10 | NFE 20 |
|---|---:|---:|---:|---:|
| DPM++ ODE | 16.49 / 0.443 | 4.79 / 0.521 | 4.38 / 0.527 | 3.62 / 0.549 |
| DPM++ SDE | 37.46 / 0.386 | 10.54 / 0.459 | 6.40 / 0.490 | 3.86 / 0.543 |
| DPM++ FDS | 16.15 / 0.450 | 3.95 / 0.553 | 4.00 / 0.542 | 3.46 / 0.557 |
| DDIM ODE | 38.72 / 0.436 | 10.39 / 0.481 | 6.25 / 0.514 | 3.33 / 0.543 |
| DDIM FDS | 38.50 / 0.436 | 9.96 / 0.489 | 5.70 / 0.517 | 3.07 / 0.553 |

## ImageNet-64 with EDM2-img64-s

Results report `FID / DINOv2-FD` over 50k samples and five seeds.

| Method | NFE 5 | NFE 8 | NFE 10 | NFE 20 |
|---|---:|---:|---:|---:|
| DPM++ ODE | 14.23 / 259.2 | 4.55 / 141.3 | 3.14 / 119.6 | 1.86 / 96.9 |
| DPM++ SDE | 35.37 / 443.6 | 17.56 / 267.5 | 12.42 / 206.2 | 4.95 / 104.7 |
| TADA | 12.61 / 245.5 | 4.56 / 140.5 | 3.37 / 120.5 | 2.00 / 96.5 |
| DPM++ FDS | 13.88 / 255.0 | 4.00 / 137.5 | 2.66 / 117.8 | 1.80 / 97.7 |

## Driver Ablation

DPM++, DiT-XL/2, ImageNet-256, NFE 8, 50k samples, five seeds. Values report
`FID`, `Precision`, and `Recall`.

| Driver | FID | Precision | Recall |
|---|---:|---:|---:|
| ODE baseline | 4.79 +/- 0.04 | 0.792 +/- 0.001 | 0.521 +/- 0.005 |
| Sine wave | 4.74 +/- 0.04 | 0.792 +/- 0.001 | 0.521 +/- 0.002 |
| AR(1), rho 0.9 | 5.24 +/- 0.02 | 0.775 +/- 0.001 | 0.530 +/- 0.002 |
| Logistic map | 4.66 +/- 0.06 | 0.791 +/- 0.001 | 0.526 +/- 0.005 |
| Gaussian iid | 4.61 +/- 0.05 | 0.791 +/- 0.001 | 0.530 +/- 0.002 |
| Gaussian Q_L_red | 4.25 +/- 0.02 | 0.788 +/- 0.001 | 0.545 +/- 0.006 |
| Rossler sequential | 5.05 +/- 0.30 | 0.778 +/- 0.012 | 0.531 +/- 0.003 |
| shuffled-Rossler | 4.56 +/- 0.03 | 0.790 +/- 0.001 | 0.531 +/- 0.003 |
| Lorenz | 4.33 +/- 0.03 | 0.767 +/- 0.002 | 0.565 +/- 0.001 |
| FDS, shuffled-Lorenz | 3.95 +/- 0.03 | 0.779 +/- 0.002 | 0.553 +/- 0.003 |

## MS-COCO with Stable Diffusion v1.5

Results report `FID / Precision / Recall` on COCO val2017, 5k samples, five
seeds.

| Method | NFE 5 | NFE 8 | NFE 10 | NFE 20 |
|---|---:|---:|---:|---:|
| ODE | 29.46 / 0.622 / 0.530 | 26.10 / 0.678 / 0.577 | 25.54 / 0.697 / 0.584 | 24.60 / 0.713 / 0.609 |
| SDE | 45.56 / 0.454 / 0.421 | 31.81 / 0.612 / 0.502 | 29.41 / 0.652 / 0.519 | 26.95 / 0.700 / 0.569 |
| FDS | 28.80 / 0.628 / 0.548 | 25.37 / 0.688 / 0.586 | 24.41 / 0.712 / 0.603 | 23.09 / 0.729 / 0.636 |

## Key Controls

| Control | Setting | Expected result |
|---|---|---|
| Gaussian control Q_L | DPM++, NFE 8, 50k, seed 42 | FID 4.237, Precision 0.793, Recall 0.548 |
| TiledGaussian | DPM++, NFE 8, 50k, seed 42 | FID 4.239, Precision 0.788, Recall 0.534 |
| Lorenz reference | DPM++, NFE 8, 50k, seed 42 | FID 4.278, Precision 0.766, Recall 0.564 |

## Hyperparameter Sensitivity

DPM++, DiT-XL/2, NFE 8, 5k samples, one seed.

| Parameter | Value | FID | IS | Precision | Recall |
|---|---|---:|---:|---:|---:|
| Gate window | [0.2, 0.8] | 13.13 | 180.2 | 0.661 | 0.761 |
| Gate window | [0.3, 0.7] | 10.17 | 228.1 | 0.766 | 0.734 |
| Gate window | [0.4, 0.6] | 10.33 | 242.7 | 0.792 | 0.710 |
| r_max | 0.01 | 10.39 | omitted | omitted | omitted |
| r_max | 0.02 | 10.37 | omitted | omitted | omitted |
| r_max | 0.04 | 10.26 | omitted | omitted | omitted |
| r_max | 0.06 | 10.17 | omitted | omitted | omitted |
| r_max | 0.09 | 10.17 | omitted | omitted | omitted |
| r_max | 0.12 | 11.26 | omitted | omitted | omitted |
| r_max | 0.15 | 18.26 | omitted | omitted | omitted |

## Hyperparameters

FDS uses shuffled-Lorenz with gate window `[0.3, 0.7]`, `tau0 = 0.3`,
`beta = 0.01`, and `r_min = r_max / 2`.

| Backbone | 5 | 6 | 7 | 8 | 9 | 10 | 11 | 12 | 15 | 20 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| DiT-XL/2, DPM++ | 0.09 | 0.09 | 0.09 | 0.09 | 0.06 | 0.04 | 0.04 | 0.04 | 0.03 | 0.02 |
| DiT-XL/2, DDIM | 0.09 | n.r. | n.r. | 0.09 | n.r. | 0.09 | n.r. | n.r. | n.r. | 0.02 |
| EDM2-img64-s | 0.09 | 0.09 | 0.09 | 0.09 | 0.09 | 0.09 | 0.06 | 0.06 | 0.03 | 0.03 |
| SD v1.5, MS-COCO | 0.09 | n.r. | n.r. | 0.09 | n.r. | 0.09 | n.r. | n.r. | n.r. | 0.09 |
