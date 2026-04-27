#!/usr/bin/env python3
"""Generate a dedicated Figure 6.2.4 plot for narrow-passage generalisation.

This script is intentionally separate from existing plotting scripts.
It reads the summary CSV produced by the narrow-passage benchmark and creates
one publication-style figure tailored to Section 6.2.4.

Default output: a 3-panel figure showing
1) success rate,
2) minimum clearance with ±1σ,
3) execution time with ±1σ.

Usage:
    python plot_section_6_2_4_generalisation.py \
        --csv result/benchmark_report_summary.csv \
        --out-dir result
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


REQUIRED_COLUMNS = [
    "gap_width_m",
    "success_rate_pct",
    "mean_duration_sec",
    "std_duration_sec",
    "mean_min_clearance_m",
    "std_min_clearance_m",
]

OPTIONAL_COLUMNS = [
    "mean_total_collision_frames",
    "mean_final_xy_error_m",
    "std_final_xy_error_m",
    "failure_breakdown",
]


def parse_args() -> argparse.Namespace:
    default_results_dir = Path(__file__).resolve().parent / "result"
    parser = argparse.ArgumentParser(
        description="Create a dedicated Figure 6.2.4 narrow-passage generalisation plot."
    )
    parser.add_argument(
        "--csv",
        type=Path,
        default=default_results_dir / "benchmark_report_summary.csv",
        help="Path to benchmark_report_summary.csv",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=default_results_dir,
        help="Directory for generated figures",
    )
    parser.add_argument(
        "--style",
        choices=["three-panel", "two-panel"],
        default="three-panel",
        help="three-panel includes success rate; two-panel focuses on clearance and duration.",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=300,
        help="Figure DPI for saved image.",
    )
    return parser.parse_args()


def validate_columns(df: pd.DataFrame, required: Iterable[str]) -> None:
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(
            "CSV is missing required columns: " + ", ".join(missing)
        )


def load_summary(csv_path: Path) -> pd.DataFrame:
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV not found: {csv_path}")

    df = pd.read_csv(csv_path)
    validate_columns(df, REQUIRED_COLUMNS)

    # Coerce numeric columns defensively.
    for col in REQUIRED_COLUMNS + [c for c in OPTIONAL_COLUMNS if c in df.columns]:
        if col == "failure_breakdown":
            continue
        df[col] = pd.to_numeric(df[col], errors="coerce")

    if df["gap_width_m"].isna().all():
        raise ValueError("All gap_width_m values are missing or invalid.")

    df = df.sort_values("gap_width_m", ascending=False).reset_index(drop=True)
    return df


def gap_labels(gaps_m: np.ndarray) -> list[str]:
    return [f"{g:.2f}" for g in gaps_m]


def add_value_labels(ax: plt.Axes, xs: np.ndarray, ys: np.ndarray, fmt: str, dy: float) -> None:
    for x, y in zip(xs, ys):
        if np.isnan(y):
            continue
        ax.annotate(
            fmt.format(y),
            xy=(x, y),
            xytext=(0, dy),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=9,
        )


def build_figure(df: pd.DataFrame, style: str) -> tuple[plt.Figure, str]:
    x = np.arange(len(df))
    gaps = df["gap_width_m"].to_numpy(dtype=float)
    labels = gap_labels(gaps)

    success = df["success_rate_pct"].to_numpy(dtype=float)
    duration_mean = df["mean_duration_sec"].to_numpy(dtype=float)
    duration_std = df["std_duration_sec"].to_numpy(dtype=float)
    clearance_cm = df["mean_min_clearance_m"].to_numpy(dtype=float) * 100.0
    clearance_std_cm = df["std_min_clearance_m"].to_numpy(dtype=float) * 100.0

    if style == "two-panel":
        fig, axes = plt.subplots(2, 1, figsize=(8.8, 7.2), sharex=True)
        out_name = "figure_6_2_4_narrow_passage_twopanel.png"
    else:
        fig, axes = plt.subplots(3, 1, figsize=(8.8, 9.6), sharex=True)
        out_name = "figure_6_2_4_narrow_passage_threepanel.png"

    fig.suptitle(
        "Figure 6.2.4: Generalisation Across Narrow-Passage Workspace Configurations",
        fontsize=14,
        fontweight="bold",
        y=0.98,
    )

    row = 0
    if style == "three-panel":
        ax = axes[row]
        ax.plot(x, success, marker="o", linewidth=2)
        ax.set_ylabel("Success rate (%)")
        ax.set_ylim(0, 110)
        ax.grid(True, alpha=0.3)
        add_value_labels(ax, x, success, "{:.0f}%", 6)
        ax.text(
            0.01,
            0.08,
            f"n = {int(df['n_trials'].iloc[0])} per gap width" if "n_trials" in df.columns else "",
            transform=ax.transAxes,
            fontsize=9,
        )
        row += 1

    ax = axes[row]
    ax.errorbar(
        x,
        clearance_cm,
        yerr=clearance_std_cm,
        marker="o",
        linewidth=2,
        capsize=4,
    )
    ax.set_ylabel("Minimum clearance (cm)")
    ax.grid(True, alpha=0.3)
    add_value_labels(ax, x, clearance_cm, "{:.2f}", 6)
    row += 1

    ax = axes[row]
    ax.errorbar(
        x,
        duration_mean,
        yerr=duration_std,
        marker="o",
        linewidth=2,
        capsize=4,
    )
    ax.set_ylabel("Execution time (s)")
    ax.set_xlabel("Gap width (m)")
    ax.grid(True, alpha=0.3)
    add_value_labels(ax, x, duration_mean, "{:.2f}", 6)

    if style == "three-panel" and "mean_total_collision_frames" in df.columns:
    # Add a concise note when collisions are all zero.
        collision_vals = df["mean_total_collision_frames"].to_numpy(dtype=float)
        if np.nanmax(collision_vals) == 0:
            axes[-1].text(
                0.99,
                0.06,
                "No timeout or sustained-collision failures in summary table",
                transform=axes[-1].transAxes,
                ha="right",
                fontsize=9,
            )

    for ax in axes:
        ax.set_xticks(x)
        ax.set_xticklabels(labels)

    fig.text(
        0.5,
        0.015,
        "Gap width decreases from left to right. Error bars show ±1 standard deviation.",
        ha="center",
        fontsize=9,
    )
    fig.tight_layout(rect=[0.03, 0.04, 0.98, 0.96])
    return fig, out_name


def main() -> None:
    args = parse_args()
    df = load_summary(args.csv)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    fig, filename = build_figure(df, args.style)
    out_path = args.out_dir / filename
    fig.savefig(out_path, dpi=args.dpi, bbox_inches="tight")
    plt.close(fig)

    print("Loaded rows:", len(df))
    print("Columns:", ", ".join(df.columns))
    print(f"Saved figure to: {out_path}")


if __name__ == "__main__":
    main()
