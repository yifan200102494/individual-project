import sys
import csv
import io
import os
import random
import contextlib
from pathlib import Path

import numpy as np
import pybullet as p
import pybullet_data

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import obstacle
import predictor


# =========================
# Benchmark config
# =========================
DT = 1.0 / 240.0
OBS_EVERY = 24                     # 10 Hz observations in a 240 Hz loop
MEAS_DT = DT * OBS_EVERY
NOISE_STD_M = 0.02                # 2 cm Gaussian noise
SEEDS = [0, 1, 2, 3, 4]

# Use the real DynamicObstacle trace rather than a synthetic sine wave
OBSTACLE_BASE_SPEED = 0.005       # align with your "Fast" benchmark level
TRACE_WINDOW_FRAMES = 1440        # 6 s contiguous trace after entering workspace
BURNIN_MAX_STEPS = 8000           # wait until obstacle enters workspace
OUTPUT_DIR = Path(__file__).resolve().parent / "kf_sensitivity_outputs"

# Moderate perturbations around nominal, not extreme retuning
NOMINAL_Q = np.array([1e-3, 1e-3, 1e-3, 1e-2, 1e-2, 1e-2], dtype=float)
NOMINAL_R = np.array([1e-2, 1e-2, 1e-2], dtype=float)

PRESETS = {
    "raw": None,   # reference only
    "nominal": {
        "q": NOMINAL_Q,
        "r": NOMINAL_R,
    },
    "responsive": {
        "q": NOMINAL_Q * 2.0,
        "r": NOMINAL_R * 0.75,
    },
    "conservative": {
        "q": NOMINAL_Q * 0.5,
        "r": NOMINAL_R * 1.5,
    },
}


# =========================
# Utility
# =========================
def connect_direct():
    cid = p.connect(p.DIRECT)
    if cid < 0:
        raise RuntimeError("Failed to connect to PyBullet in DIRECT mode.")
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    p.setGravity(0, 0, -9.8)
    return cid


def rmse_3d(a, b):
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    return float(np.sqrt(np.mean(np.sum((a - b) ** 2, axis=1))))


def accel_rms_from_positions(positions, dt):
    positions = np.asarray(positions, dtype=float)
    if len(positions) < 3:
        return float("nan")
    vel = np.diff(positions, axis=0) / dt
    acc = np.diff(vel, axis=0) / dt
    acc_mag = np.linalg.norm(acc, axis=1)
    if len(acc_mag) == 0:
        return float("nan")
    return float(np.sqrt(np.mean(acc_mag ** 2)))


def make_predictor(q_diag, r_diag):
    kf = predictor.ObstaclePredictor(history_size=20, prediction_horizon=0.8)
    kf.Q = np.diag(np.asarray(q_diag, dtype=float))
    kf.R = np.diag(np.asarray(r_diag, dtype=float))
    return kf


# =========================
# Real trace collection
# =========================
def collect_real_obstacle_trace(seed, base_speed=OBSTACLE_BASE_SPEED,
                                trace_window_frames=TRACE_WINDOW_FRAMES,
                                burnin_max_steps=BURNIN_MAX_STEPS):
    """
    Collect a contiguous obstacle position trace from the real DynamicObstacle
    model after it first enters the workspace.
    """
    random.seed(seed)
    np.random.seed(seed)

    cid = connect_direct()
    try:
        with contextlib.redirect_stdout(io.StringIO()):
            dyn = obstacle.DynamicObstacle()

        dyn.base_speed = base_speed
        dyn.current_speed = base_speed

        entered = False
        gt = []

        with contextlib.redirect_stdout(io.StringIO()):
            for _ in range(burnin_max_steps):
                dyn.update()
                info = dyn.get_state_info()
                if info.get("in_work_area", False):
                    entered = True
                    gt.append(np.asarray(dyn.get_position(), dtype=float))
                    break

            if not entered:
                raise RuntimeError(
                    f"Seed {seed}: obstacle did not enter workspace within {burnin_max_steps} steps."
                )

            for _ in range(trace_window_frames - 1):
                dyn.update()
                gt.append(np.asarray(dyn.get_position(), dtype=float))

        return np.asarray(gt, dtype=float)

    finally:
        if p.isConnected():
            p.disconnect()


# =========================
# Observation generation
# =========================
def make_noisy_observations(gt, obs_every=OBS_EVERY, noise_std=NOISE_STD_M, seed=0):
    rng = np.random.default_rng(seed + 999)
    obs = [None] * len(gt)
    obs_indices = list(range(0, len(gt), obs_every))

    for idx in obs_indices:
        obs[idx] = gt[idx] + rng.normal(0.0, noise_std, size=3)

    return obs, obs_indices


# =========================
# Evaluation
# =========================
def run_raw_reference(gt, obs, obs_indices):
    gt_updates = []
    raw_updates = []

    for idx in obs_indices:
        gt_updates.append(gt[idx])
        raw_updates.append(obs[idx])

    gt_updates = np.asarray(gt_updates, dtype=float)
    raw_updates = np.asarray(raw_updates, dtype=float)

    return {
        "gt_updates": gt_updates,
        "est_updates": raw_updates,
        "rmse_cm": rmse_3d(gt_updates, raw_updates) * 100.0,
        "acc_rms_m_s2": accel_rms_from_positions(raw_updates, MEAS_DT),
    }


def run_kf_reference(gt, obs, obs_indices, q_diag, r_diag):
    kf = make_predictor(q_diag, r_diag)

    gt_updates = []
    est_updates = []

    for idx in obs_indices:
        measurement = obs[idx]
        kf.update(measurement)

        if not kf.initialized:
            continue

        gt_updates.append(gt[idx])
        est_updates.append(kf.state[:3].copy())

    gt_updates = np.asarray(gt_updates, dtype=float)
    est_updates = np.asarray(est_updates, dtype=float)

    return {
        "gt_updates": gt_updates,
        "est_updates": est_updates,
        "rmse_cm": rmse_3d(gt_updates, est_updates) * 100.0,
        "acc_rms_m_s2": accel_rms_from_positions(est_updates, MEAS_DT),
    }


def evaluate_one_seed(seed):
    gt = collect_real_obstacle_trace(seed)
    obs, obs_indices = make_noisy_observations(gt, seed=seed)

    rows = []

    raw_result = run_raw_reference(gt, obs, obs_indices)
    rows.append({
        "seed": seed,
        "method": "raw",
        "rmse_cm": raw_result["rmse_cm"],
        "acc_rms_m_s2": raw_result["acc_rms_m_s2"],
        "n_updates": len(raw_result["gt_updates"]),
    })

    for method, cfg in PRESETS.items():
        if method == "raw":
            continue

        kf_result = run_kf_reference(
            gt=gt,
            obs=obs,
            obs_indices=obs_indices,
            q_diag=cfg["q"],
            r_diag=cfg["r"],
        )

        rows.append({
            "seed": seed,
            "method": method,
            "rmse_cm": kf_result["rmse_cm"],
            "acc_rms_m_s2": kf_result["acc_rms_m_s2"],
            "n_updates": len(kf_result["gt_updates"]),
        })

    return rows


def summarise_rows(rows):
    methods = ["raw", "nominal", "responsive", "conservative"]
    summary = []

    nominal_rmse = None
    raw_rmse = None

    for method in methods:
        subset = [r for r in rows if r["method"] == method]
        rmse_vals = np.array([r["rmse_cm"] for r in subset], dtype=float)
        acc_vals = np.array([r["acc_rms_m_s2"] for r in subset], dtype=float)
        n_updates = int(np.mean([r["n_updates"] for r in subset]))

        record = {
            "method": method,
            "rmse_mean_cm": float(np.mean(rmse_vals)),
            "rmse_std_cm": float(np.std(rmse_vals)),
            "acc_rms_mean_m_s2": float(np.mean(acc_vals)),
            "acc_rms_std_m_s2": float(np.std(acc_vals)),
            "n_seeds": len(subset),
            "n_updates_per_seed": n_updates,
        }
        summary.append(record)

        if method == "raw":
            raw_rmse = record["rmse_mean_cm"]
        if method == "nominal":
            nominal_rmse = record["rmse_mean_cm"]

    for record in summary:
        if raw_rmse is not None and raw_rmse > 0:
            record["rmse_improvement_vs_raw_pct"] = float(
                (raw_rmse - record["rmse_mean_cm"]) / raw_rmse * 100.0
            )
        else:
            record["rmse_improvement_vs_raw_pct"] = float("nan")

        if nominal_rmse is not None and nominal_rmse > 0:
            record["rmse_change_vs_nominal_pct"] = float(
                (record["rmse_mean_cm"] - nominal_rmse) / nominal_rmse * 100.0
            )
        else:
            record["rmse_change_vs_nominal_pct"] = float("nan")

    return summary


def save_csv(path, rows, fieldnames):
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def try_make_plots(summary_rows, output_dir):
    try:
        import matplotlib.pyplot as plt
    except Exception as e:
        print(f"[Info] matplotlib not available, skipping plots: {e}")
        return

    methods = [r["method"] for r in summary_rows]
    rmse_means = [r["rmse_mean_cm"] for r in summary_rows]
    rmse_stds = [r["rmse_std_cm"] for r in summary_rows]
    acc_means = [r["acc_rms_mean_m_s2"] for r in summary_rows]
    acc_stds = [r["acc_rms_std_m_s2"] for r in summary_rows]

    plt.figure(figsize=(7.5, 4.5))
    x = np.arange(len(methods))
    plt.bar(x, rmse_means, yerr=rmse_stds, capsize=4)
    plt.xticks(x, methods, rotation=15)
    plt.ylabel("RMSE (cm)")
    plt.title("KF sensitivity on real obstacle traces")
    plt.tight_layout()
    plt.savefig(output_dir / "kf_sensitivity_rmse_realtrace.png", dpi=200)
    plt.close()

    plt.figure(figsize=(7.5, 4.5))
    x = np.arange(len(methods))
    plt.bar(x, acc_means, yerr=acc_stds, capsize=4)
    plt.xticks(x, methods, rotation=15)
    plt.ylabel("Acceleration RMS (m/s^2)")
    plt.title("KF smoothness on real obstacle traces")
    plt.tight_layout()
    plt.savefig(output_dir / "kf_sensitivity_acc_rms_realtrace.png", dpi=200)
    plt.close()


def print_summary(summary_rows):
    print("\nKF sensitivity on real obstacle traces")
    print(
        f"Config: base_speed={OBSTACLE_BASE_SPEED:.3f}, "
        f"trace_window={TRACE_WINDOW_FRAMES} frames ({TRACE_WINDOW_FRAMES * DT:.2f} s), "
        f"obs_every={OBS_EVERY} ({1.0 / MEAS_DT:.1f} Hz), "
        f"noise_std={NOISE_STD_M * 100:.1f} cm, seeds={SEEDS}"
    )
    print("-" * 105)
    print(
        f"{'method':16s}"
        f"{'RMSE (cm)':>18s}"
        f"{'vs raw (%)':>14s}"
        f"{'Acc RMS (m/s^2)':>22s}"
        f"{'vs nominal (%)':>18s}"
    )
    print("-" * 105)

    for r in summary_rows:
        rmse_str = f"{r['rmse_mean_cm']:.2f} ± {r['rmse_std_cm']:.2f}"
        acc_str = f"{r['acc_rms_mean_m_s2']:.3f} ± {r['acc_rms_std_m_s2']:.3f}"

        if r["method"] == "raw":
            imp_raw = "0.00"
        else:
            imp_raw = f"{r['rmse_improvement_vs_raw_pct']:.2f}"

        if r["method"] == "nominal":
            delta_nom = "0.00"
        else:
            delta_nom = f"{r['rmse_change_vs_nominal_pct']:+.2f}"

        print(
            f"{r['method']:16s}"
            f"{rmse_str:>18s}"
            f"{imp_raw:>14s}"
            f"{acc_str:>22s}"
            f"{delta_nom:>18s}"
        )

    print("-" * 105)
    print("\nThesis-facing rows (exclude raw if you want only the sensitivity presets):")
    for r in summary_rows:
        if r["method"] == "raw":
            continue
        print(
            f"  {r['method']:12s} | "
            f"RMSE = {r['rmse_mean_cm']:.2f} ± {r['rmse_std_cm']:.2f} cm | "
            f"Acc RMS = {r['acc_rms_mean_m_s2']:.3f} ± {r['acc_rms_std_m_s2']:.3f} m/s^2"
        )


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    per_seed_rows = []
    for seed in SEEDS:
        print(f"Running seed {seed} ...")
        rows = evaluate_one_seed(seed)
        per_seed_rows.extend(rows)

    summary_rows = summarise_rows(per_seed_rows)

    save_csv(
        OUTPUT_DIR / "kf_sensitivity_per_seed_realtrace.csv",
        per_seed_rows,
        fieldnames=["seed", "method", "rmse_cm", "acc_rms_m_s2", "n_updates"],
    )

    save_csv(
        OUTPUT_DIR / "kf_sensitivity_summary_realtrace.csv",
        summary_rows,
        fieldnames=[
            "method",
            "rmse_mean_cm",
            "rmse_std_cm",
            "acc_rms_mean_m_s2",
            "acc_rms_std_m_s2",
            "rmse_improvement_vs_raw_pct",
            "rmse_change_vs_nominal_pct",
            "n_seeds",
            "n_updates_per_seed",
        ],
    )

    try_make_plots(summary_rows, OUTPUT_DIR)
    print_summary(summary_rows)

    print("\nSaved files:")
    print(f"  {OUTPUT_DIR / 'kf_sensitivity_per_seed_realtrace.csv'}")
    print(f"  {OUTPUT_DIR / 'kf_sensitivity_summary_realtrace.csv'}")
    print(f"  {OUTPUT_DIR / 'kf_sensitivity_rmse_realtrace.png'}")
    print(f"  {OUTPUT_DIR / 'kf_sensitivity_acc_rms_realtrace.png'}")


if __name__ == "__main__":
    main()