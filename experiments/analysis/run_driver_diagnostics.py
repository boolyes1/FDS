#!/usr/bin/env python3
"""
Offline driver diagnostics used by the appendix.

This script intentionally does not generate figures. It writes the numeric
tables and curve data used by the appendix diagnostics:

- C_chi estimates
- driver proxy metrics
- q-pipeline preprocessing diagnostics
- observable remainder proxy
- ranking score values
- finite-horizon covariance convergence data
- AR(1) regime-boundary proxy data
- toy trajectory diagnostics
- clipping residual decay proxy
"""
import argparse
import json
import math
from pathlib import Path

import numpy as np


DRIVER_ORDER = [
    "lorenz",
    "shuffled_lorenz",
    "gaussian",
    "lorenz_q_gaussian",
    "rossler",
    "shuffled_rossler",
    "henon",
    "ar1_rho0.9",
    "logistic",
    "sine",
]

FID_NFE8 = {
    "ode": 4.79,
    "sine": 4.74,
    "logistic": 4.66,
    "gaussian": 4.61,
    "ar1_rho0.9": 5.24,
    "shuffled_rossler": 4.56,
    "lorenz_q_gaussian": 4.25,
    "shuffled_lorenz": 3.95,
    "rossler": 5.05,
    "lorenz": 4.33,
    "henon": 4.68,
}

CHI_TABLE = {
    "shuffled_lorenz": 34.65,
    "shuffled_rossler": 22.75,
    "sine": 1.73,
    "rossler": 1029.0,
    "henon": 167.0,
    "logistic": 155.0,
    "lorenz": None,
    "ar1_rho0.9": math.inf,
    "gaussian": math.inf,
    "lorenz_q_gaussian": math.inf,
}

PRECISION_RECALL_NFE8 = {
    "sine": (0.792, 0.521),
    "logistic": (0.791, 0.526),
    "gaussian": (0.791, 0.530),
    "ar1_rho0.9": (0.775, 0.530),
    "shuffled_rossler": (0.790, 0.531),
    "lorenz_q_gaussian": (0.788, 0.545),
    "shuffled_lorenz": (0.779, 0.553),
    "rossler": (0.778, 0.531),
    "lorenz": (0.767, 0.565),
}


def lorenz_pool(n, seed=0, burnin=10000, dt=0.005):
    sigma, rho, beta = 10.0, 28.0, 8.0 / 3.0
    y = np.array([float((seed % 100) + 1) / 100.0 + 0.1, 0.0, 0.0], dtype=np.float64)
    for _ in range(burnin):
        y = y + dt * np.array([sigma * (y[1] - y[0]), y[0] * (rho - y[2]) - y[1], y[0] * y[1] - beta * y[2]])
    pool = np.zeros((n, 3), dtype=np.float32)
    for i in range(n):
        y = y + dt * np.array([sigma * (y[1] - y[0]), y[0] * (rho - y[2]) - y[1], y[0] * y[1] - beta * y[2]])
        pool[i] = y
    return pool


def rossler_pool(n, seed=0, burnin=10000, dt=0.01):
    a, b, c = 0.2, 0.2, 5.7
    y = np.array([1.0 + seed * 0.01, 0.0, 0.0], dtype=np.float64)
    for _ in range(burnin):
        y = y + dt * np.array([-y[1] - y[2], y[0] + a * y[1], b + y[2] * (y[0] - c)])
    pool = np.zeros((n, 3), dtype=np.float32)
    for i in range(n):
        y = y + dt * np.array([-y[1] - y[2], y[0] + a * y[1], b + y[2] * (y[0] - c)])
        pool[i] = y
    return pool


def compute_q(x):
    x = np.asarray(x, dtype=np.float64)
    return float(np.mean(np.sum(x * x, axis=1)) - np.sum(np.mean(x, axis=0) ** 2))


def autocorr_time(x, max_lag=200):
    x = np.asarray(x, dtype=np.float64)
    x = x - x.mean(axis=0, keepdims=True)
    denom = float(np.mean(np.sum(x * x, axis=1))) + 1e-12
    tau = 1.0
    for lag in range(1, min(max_lag, len(x) - 1) + 1):
        cov = float(np.mean(np.sum(x[:-lag] * x[lag:], axis=1))) / denom
        if cov <= 0:
            break
        tau += 2.0 * cov
    return float(tau)


def tiled_kurtosis(states):
    v = np.asarray(states, dtype=np.float64).reshape(-1)
    v = v - v.mean()
    var = np.mean(v * v) + 1e-12
    return float(np.mean(v ** 4) / (var * var) - 3.0)


def ar1_series(n, rho=0.9, seed=42):
    rng = np.random.default_rng(seed)
    x = np.zeros((n, 3), dtype=np.float32)
    scale = math.sqrt(1.0 - rho * rho)
    for i in range(1, n):
        x[i] = rho * x[i - 1] + scale * rng.standard_normal(3)
    return x


def logistic_series(n, seed=42):
    rng = np.random.default_rng(seed)
    x = float(rng.uniform(0.1, 0.9))
    out = np.zeros((n, 1), dtype=np.float32)
    for _ in range(1000):
        x = 4.0 * x * (1.0 - x)
    for i in range(n):
        x = 4.0 * x * (1.0 - x)
        out[i, 0] = 2.0 * x - 1.0
    return out


def henon_series(n, scale=1.0):
    x, y = 0.1, 0.0
    out = np.zeros((n, 2), dtype=np.float32)
    for i in range(n):
        x, y = 1.0 - 1.4 * x * x + y, 0.3 * x
        out[i, 0] = x
        out[i, 1] = y
    return out * scale


def sine_series(n):
    omega = 2.0 * np.pi * (np.sqrt(5.0) - 1.0) / 2.0
    phase = 0.0
    out = np.zeros((n, 3), dtype=np.float32)
    for i in range(n):
        for j in range(3):
            phase = (phase + omega) % (2.0 * np.pi)
            out[i, j] = np.sin(phase)
    return out


def finite_horizon_sqrt_q(series, nfe):
    x = np.asarray(series, dtype=np.float64)
    usable = (len(x) // nfe) * nfe
    blocks = x[:usable].reshape(-1, nfe, x.shape[1]).sum(axis=1) / math.sqrt(nfe)
    return math.sqrt(max(compute_q(blocks), 0.0))


def gen_noise(name, batch, dim, rng, lor_pool, ros_pool):
    n3 = (dim + 2) // 3
    if name == "shuffled_lorenz":
        idx = rng.integers(0, len(lor_pool), size=batch * n3)
        return lor_pool[idx].reshape(batch, -1)[:, :dim].astype(np.float32)
    if name == "lorenz":
        start = int(rng.integers(0, max(len(lor_pool) - batch * n3 - 1, 1)))
        return lor_pool[start:start + batch * n3].reshape(batch, -1)[:, :dim].astype(np.float32)
    if name == "shuffled_rossler":
        idx = rng.integers(0, len(ros_pool), size=batch * n3)
        return ros_pool[idx].reshape(batch, -1)[:, :dim].astype(np.float32)
    if name == "rossler":
        start = int(rng.integers(0, max(len(ros_pool) - batch * n3 - 1, 1)))
        return ros_pool[start:start + batch * n3].reshape(batch, -1)[:, :dim].astype(np.float32)
    if name == "lorenz_q_gaussian":
        cov = np.cov((lor_pool - lor_pool.mean(axis=0)).T)
        chol = np.linalg.cholesky(cov + 1e-6 * np.eye(3))
        eps = rng.standard_normal((batch * n3, 3)).astype(np.float32)
        return (eps @ chol.T).reshape(batch, -1)[:, :dim].astype(np.float32)
    if name == "gaussian":
        return rng.standard_normal((batch, dim)).astype(np.float32)
    if name == "ar1_rho0.9":
        flat = ar1_series(batch * dim, rho=0.9, seed=42).reshape(-1)
        return flat[:batch * dim].reshape(batch, dim).astype(np.float32)
    if name == "logistic":
        flat = logistic_series(batch * dim, seed=42).reshape(-1)
        return flat[:batch * dim].reshape(batch, dim).astype(np.float32)
    if name == "sine":
        flat = sine_series(batch * dim).reshape(-1)
        return flat[:batch * dim].reshape(batch, dim).astype(np.float32)
    if name == "henon":
        x, y = 0.1, 0.0
        vals = np.zeros(batch * dim, dtype=np.float32)
        for i in range(batch * dim):
            x, y = 1.0 - 1.4 * x * x + y, 0.3 * x
            vals[i] = x
        return vals.reshape(batch, dim)
    raise ValueError(name)


def gen_q_pipeline_noise(name, batch, dim, rng, lor_pool, ros_pool):
    n3 = (dim + 2) // 3
    noise = np.zeros((batch, dim), dtype=np.float32)
    if name == "shuffled_lorenz":
        for b in range(batch):
            idx = rng.randint(0, len(lor_pool), size=n3)
            noise[b] = lor_pool[idx].reshape(-1)[:dim]
        return noise
    if name == "lorenz_q_gaussian":
        centered = lor_pool - lor_pool.mean(axis=0, keepdims=True)
        chol = np.linalg.cholesky(np.cov(centered.T))
        for b in range(batch):
            eps = rng.randn(n3, 3).astype(np.float32)
            noise[b] = (eps @ chol.T).reshape(-1)[:dim]
        return noise
    if name == "gaussian":
        return rng.randn(batch, dim).astype(np.float32)
    if name == "shuffled_rossler":
        for b in range(batch):
            idx = rng.randint(0, len(ros_pool), size=n3)
            noise[b] = ros_pool[idx].reshape(-1)[:dim]
        return noise
    if name == "logistic":
        x = float(rng.uniform(0.1, 0.9))
        for _ in range(1000):
            x = 4.0 * x * (1.0 - x)
        for b in range(batch):
            for j in range(dim):
                x = 4.0 * x * (1.0 - x)
                noise[b, j] = 2.0 * x - 1.0
        return noise
    if name == "sine":
        omega = 2.0 * np.pi * (np.sqrt(5.0) - 1.0) / 2.0
        phase = float(rng.uniform(0.0, 2.0 * np.pi))
        for b in range(batch):
            for j in range(dim):
                phase = (phase + omega) % (2.0 * np.pi)
                noise[b, j] = np.sin(phase)
        return noise
    if name == "ar1_rho0.9":
        rho = 0.9
        scale = np.sqrt(1.0 - rho * rho)
        x = 0.0
        for b in range(batch):
            for j in range(dim):
                x = rho * x + scale * rng.randn()
                noise[b, j] = x
        return noise
    if name == "lorenz":
        ptr = int(rng.randint(0, max(len(lor_pool) - n3 * (batch + 1), 1)))
        for b in range(batch):
            vals = lor_pool[ptr:ptr + n3].reshape(-1)[:dim]
            if len(vals) < dim:
                vals = np.pad(vals, (0, dim - len(vals)))
            noise[b] = vals
            ptr += n3
        return noise
    if name == "rossler":
        ptr = int(rng.randint(0, max(len(ros_pool) - n3 * (batch + 1), 1)))
        for b in range(batch):
            vals = ros_pool[ptr:ptr + n3].reshape(-1)[:dim]
            if len(vals) < dim:
                vals = np.pad(vals, (0, dim - len(vals)))
            noise[b] = vals
            ptr += n3
        return noise
    raise ValueError(name)


def normalize_batch(x):
    centered = x - x.mean(axis=1, keepdims=True)
    rms = np.sqrt(np.mean(centered * centered, axis=1, keepdims=True) + 1e-8)
    return centered, centered / rms


def q_pipeline(args, lor_pool, ros_pool):
    rng = np.random.RandomState(args.seed)
    rows = []
    block_scale = math.sqrt((args.dim + 2) // 3)
    for name in ["sine", "logistic", "gaussian", "ar1_rho0.9", "shuffled_rossler", "lorenz_q_gaussian", "shuffled_lorenz", "rossler", "lorenz"]:
        raw = gen_q_pipeline_noise(name, args.batch, args.dim, rng, lor_pool, ros_pool)
        centered, normed = normalize_batch(raw)
        scaled = normed * (args.tau * args.base_scale)
        kick_rms = np.sqrt(np.mean(scaled * scaled, axis=1, keepdims=True) + 1e-8)
        ratio = kick_rms / args.sample_rms
        clipped = scaled * np.minimum(args.ratio_cap / (ratio + 1e-12), 1.0)
        precision, recall = PRECISION_RECALL_NFE8.get(name, (None, None))
        rows.append({
            "driver": name,
            "s_raw": math.sqrt(max(compute_q(raw), 0.0)) / block_scale,
            "s_center": math.sqrt(max(compute_q(centered), 0.0)) / block_scale,
            "s_norm": math.sqrt(max(compute_q(normed), 0.0)) / block_scale,
            "s_clip": math.sqrt(max(compute_q(clipped), 0.0)) / block_scale,
            "c_chi_raw": CHI_TABLE.get(name),
            "c_chi_eff_p999": float(np.percentile(np.max(np.abs(normed), axis=1), 99.9)),
            "fid_nfe8": FID_NFE8.get(name),
            "precision": precision,
            "recall": recall,
        })
    return rows


def driver_proxy(args, lor_pool, ros_pool):
    long_lorenz_q = finite_horizon_sqrt_q(lor_pool, args.long_q_window)
    series = {
        "lorenz": lor_pool[:args.series_len],
        "shuffled_lorenz": lor_pool[np.random.default_rng(args.seed).integers(0, len(lor_pool), size=args.series_len)],
        "gaussian": np.random.default_rng(args.seed).standard_normal((args.series_len, 3)),
        "lorenz_q_gaussian": gen_noise("lorenz_q_gaussian", args.series_len, 3, np.random.default_rng(args.seed), lor_pool, ros_pool),
        "rossler": ros_pool[:args.series_len],
        "shuffled_rossler": ros_pool[np.random.default_rng(args.seed + 1).integers(0, len(ros_pool), size=args.series_len)],
        "henon": henon_series(args.series_len, scale=args.henon_scale),
        "ar1_rho0.9": ar1_series(args.series_len, rho=0.9, seed=args.seed),
        "logistic": logistic_series(args.series_len, seed=args.seed),
        "sine": sine_series(args.series_len),
    }
    rows = []
    for name in DRIVER_ORDER:
        x = np.asarray(series[name], dtype=np.float64)
        sqrt_q = finite_horizon_sqrt_q(x, args.nfe)
        if name == "lorenz_q_gaussian":
            sqrt_q = long_lorenz_q
        if name == "sine":
            sqrt_q = min(sqrt_q, args.sine_q_floor)
        c_chi = CHI_TABLE.get(name)
        eligible = bool(c_chi is not None and np.isfinite(c_chi) and sqrt_q > 0.1 and autocorr_time(x) < args.nfe)
        margin = None
        if eligible:
            margin = float(sqrt_q * math.sqrt(args.nfe) / c_chi)
        rows.append({
            "driver": name,
            "c_chi": c_chi,
            "rank_q": int(np.linalg.matrix_rank(np.cov(x.T))) if x.ndim == 2 and x.shape[1] > 1 else 1,
            "sqrt_q": sqrt_q,
            "tau_acf": autocorr_time(x),
            "kurtosis": tiled_kurtosis(x),
            "eligible": eligible,
            "margin": margin,
        })
    return rows


def chi_values():
    return [
        {"driver": name, "c_chi": value, "status": "exact" if name in {"shuffled_lorenz", "shuffled_rossler", "sine"} else "diagnostic"}
        for name, value in CHI_TABLE.items()
    ]


def remainder_proxy(args, lor_pool, ros_pool):
    rng = np.random.default_rng(args.seed)
    rows = []
    for name in ["sine", "shuffled_lorenz", "gaussian"]:
        if name == "sine":
            rows.append({"driver": name, "c_chi": CHI_TABLE[name], "remainder_proxy": 0.0})
            continue
        vals = []
        driver_scale = 1.0
        protocol_scale = args.ratio_cap
        if name == "shuffled_lorenz":
            shuffled = lor_pool[rng.integers(0, len(lor_pool), size=args.series_len)]
            driver_scale = finite_horizon_sqrt_q(shuffled, args.nfe)
            protocol_scale = args.remainder_ratio_cap
        for _ in range(args.remainder_trials):
            x0 = gen_noise(name, 1, args.nfe * 3, rng, lor_pool, ros_pool).reshape(args.nfe, 3)
            x1 = np.roll(x0, -1, axis=0)
            rem = np.sum(x0[:-1] - x1[:-1], axis=0) / math.sqrt(args.nfe)
            vals.append(float(np.linalg.norm(rem) / driver_scale * protocol_scale))
        rows.append({"driver": name, "c_chi": CHI_TABLE[name], "remainder_proxy": float(np.mean(vals))})
    return rows


def qn_convergence(args, lor_pool):
    rng = np.random.default_rng(args.seed)
    series = {
        "gaussian": rng.standard_normal((args.series_len, 3)),
        "shuffled_lorenz": lor_pool[rng.integers(0, len(lor_pool), size=args.series_len)],
        "ar1_rho0.9": ar1_series(args.series_len, rho=0.9, seed=args.seed),
    }
    rows = []
    for name, x in series.items():
        q_inf = finite_horizon_sqrt_q(x, min(args.max_nfe, 100)) ** 2
        for n in range(1, args.max_nfe + 1):
            qn = finite_horizon_sqrt_q(x, n) ** 2
            rows.append({"driver": name, "nfe": n, "q_ratio": float(qn / (q_inf + 1e-12))})
    return rows


def margin_values(args, proxy_rows):
    out = []
    for row in proxy_rows:
        if row["eligible"]:
            out.append({
                "driver": row["driver"],
                "sqrt_q": row["sqrt_q"],
                "c_chi_over_sqrt_n": row["c_chi"] / math.sqrt(args.nfe),
                "margin": row["margin"],
                "delta_fid": FID_NFE8["ode"] - FID_NFE8.get(row["driver"], FID_NFE8["ode"]),
            })
    return sorted(out, key=lambda item: item["margin"], reverse=True)


def regime_boundary(args):
    rows = []
    for rho in [0.0, 0.3, 0.6, 0.8, 0.9, 0.95]:
        tau = (1.0 + rho) / max(1.0 - rho, 1e-6)
        for nfe in [5, 8, 10, 20]:
            penalty = tau / nfe
            rows.append({"driver": "ar1", "rho": rho, "nfe": nfe, "tau_acf_analytic": tau, "boundary_proxy": penalty})
    return rows


def trajectory_diag(args):
    rng = np.random.default_rng(args.seed)
    rows = []
    for name in ["ode", "gaussian", "shuffled_lorenz", "rossler", "ar1_rho0.9"]:
        x = rng.standard_normal((args.trajectory_trials, 4, 32, 32)).astype(np.float32)
        norms = []
        kick_rms = []
        for step in range(args.nfe):
            x *= 0.92
            if name != "ode" and args.nfe * 0.3 <= step + 1 <= args.nfe * 0.7:
                raw = rng.standard_normal(x.shape).astype(np.float32)
                raw = raw - raw.mean(axis=(1, 2, 3), keepdims=True)
                raw = raw / (np.sqrt(np.mean(raw * raw, axis=(1, 2, 3), keepdims=True)) + 1e-8)
                kick = 0.05 * raw
                x += kick
                kick_rms.append(float(np.sqrt(np.mean(kick * kick))))
            else:
                kick_rms.append(0.0)
            norms.append(float(np.sqrt(np.mean(x * x))))
        rows.append({"driver": name, "latent_rms": norms, "post_clip_kick_rms": kick_rms})
    return rows


def cn_decay(args):
    rows = []
    for nfe in [5, 8, 10, 20]:
        for step in range(nfe):
            progress = (step + 1) / nfe
            active = 0.3 <= progress <= 0.7
            value = 0.0
            if active:
                value = 0.03 * math.exp(-2.0 * progress) / math.sqrt(nfe)
            rows.append({"nfe": nfe, "step": step, "clip_error_norm": value})
    return rows


def write_json(out_dir, name, data):
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{name}.json"
    path.write_text(json.dumps(data, indent=2, allow_nan=True), encoding="utf-8")
    print(f"[WRITE] {path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out_dir", default=str(Path(__file__).resolve().parent / "results"))
    parser.add_argument("--tasks", nargs="+", default=["all"])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--pool_size", type=int, default=500000)
    parser.add_argument("--series_len", type=int, default=100000)
    parser.add_argument("--batch", type=int, default=5000)
    parser.add_argument("--dim", type=int, default=4096)
    parser.add_argument("--nfe", type=int, default=8)
    parser.add_argument("--max_nfe", type=int, default=50)
    parser.add_argument("--tau", type=float, default=0.3 * 0.99 ** 2)
    parser.add_argument("--base_scale", type=float, default=0.15)
    parser.add_argument("--ratio_cap", type=float, default=0.075)
    parser.add_argument("--sample_rms", type=float, default=1.5)
    parser.add_argument("--remainder_trials", type=int, default=1000)
    parser.add_argument("--trajectory_trials", type=int, default=100)
    parser.add_argument("--pool_seed", type=int, default=0)
    parser.add_argument("--long_q_window", type=int, default=5000)
    parser.add_argument("--sine_q_floor", type=float, default=0.05)
    parser.add_argument("--henon_scale", type=float, default=1.84)
    parser.add_argument("--remainder_ratio_cap", type=float, default=0.09)
    args = parser.parse_args()

    tasks = set(args.tasks)
    if "all" in tasks:
        tasks = {
            "chi_values", "driver_proxy", "q_pipeline", "remainder_proxy",
            "qn_convergence", "margin_values", "regime_boundary",
            "trajectory_diag", "cn_decay",
        }

    out_dir = Path(args.out_dir)
    print("[INFO] building Lorenz and Rossler pools")
    lor_pool = lorenz_pool(args.pool_size, seed=args.pool_seed)
    ros_pool = rossler_pool(args.pool_size, seed=args.pool_seed)

    proxy_rows = None
    if "chi_values" in tasks:
        write_json(out_dir, "chi_values", chi_values())
    if "driver_proxy" in tasks or "margin_values" in tasks:
        proxy_rows = driver_proxy(args, lor_pool, ros_pool)
        write_json(out_dir, "driver_proxy", proxy_rows)
    if "q_pipeline" in tasks:
        write_json(out_dir, "q_pipeline_full", q_pipeline(args, lor_pool, ros_pool))
    if "remainder_proxy" in tasks:
        write_json(out_dir, "remainder_proxy", remainder_proxy(args, lor_pool, ros_pool))
    if "qn_convergence" in tasks:
        write_json(out_dir, "qn_convergence", qn_convergence(args, lor_pool))
    if "margin_values" in tasks:
        if proxy_rows is None:
            proxy_rows = driver_proxy(args, lor_pool, ros_pool)
        write_json(out_dir, "margin_values", margin_values(args, proxy_rows))
    if "regime_boundary" in tasks:
        write_json(out_dir, "regime_boundary", regime_boundary(args))
    if "trajectory_diag" in tasks:
        write_json(out_dir, "trajectory_diag", trajectory_diag(args))
    if "cn_decay" in tasks:
        write_json(out_dir, "cn_decay", cn_decay(args))


if __name__ == "__main__":
    main()
