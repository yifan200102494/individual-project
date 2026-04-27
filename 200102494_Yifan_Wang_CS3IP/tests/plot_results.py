import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['font.sans-serif'] = ['Arial']
plt.rcParams['axes.unicode_minus'] = False

HERE = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(HERE, "result")
CSV_PATH = os.path.join(OUT_DIR, "benchmark_raw_trials_zero_2.csv")

SPEED_ORDER = [0.001, 0.003, 0.005, 0.007, 0.017]
SPEED_LABELS = {
    0.001: "Slow(0.001)",
    0.003: "Medium(0.003)",
    0.005: "Fast(0.005)",
    0.007: "Extreme(0.007)",
    0.017: "Insane(0.017)",
}


def load_and_aggregate(csv_path):
    df = pd.read_csv(csv_path)

    rows = []
    for sp in SPEED_ORDER:
        sub = df[df["speed"] == sp]
        n = len(sub)
        succ = int(sub["success_at_0"].sum())
        fail = n - succ

        reasons = sub.loc[sub["success_at_0"] == 0, "fail_reason"].fillna("").tolist()
        n_coll = sum(r == "Collision" for r in reasons)
        n_drop = sum(r == "Drop/Miss" for r in reasons)
        n_time = sum(r == "Timeout" for r in reasons)

        ok = sub[sub["success_at_0"] == 1]
        min_dist_mean = float(ok["min_dist"].mean()) if len(ok) else float("nan")
        min_dist_std = float(ok["min_dist"].std(ddof=1)) if len(ok) > 1 else 0.0

        rows.append({
            "Level": SPEED_LABELS[sp],
            "Speed": sp,
            "N": n,
            "Success": succ,
            "Success_Rate": 100.0 * succ / n,
            "Failures": fail,
            "Collision": n_coll,
            "Drop": n_drop,
            "Timeout": n_time,
            "Min_Dist": min_dist_mean,
            "Dist_Sigma": min_dist_std,
        })
    return pd.DataFrame(rows)


def plot_figure_6_2_1(df, out_path):
    """Figure 6.2.1: Success rate vs. speed + failure breakdown."""
    fig, ax = plt.subplots(figsize=(10, 6))

    # Success rate line (left axis)
    ax.plot(df["Level"], df["Success_Rate"], marker='s', color='#2ecc71',
            linewidth=3, markersize=10, label='Task Success Rate (%)', zorder=3)
    ax.set_ylabel('Success Rate (%)', fontsize=12, fontweight='bold', color='#27ae60')
    ax.set_ylim(60, 100)
    ax.set_yticks(np.arange(60, 101, 5))
    ax.tick_params(axis='y', labelcolor='#27ae60')

    for i, val in enumerate(df["Success_Rate"]):
        ax.annotate(f"{val:.1f}%", (df["Level"][i], val),
                    xytext=(0, 10), textcoords='offset points',
                    ha='center', fontweight='bold', color='#27ae60')

    # Total-failure bars (right axis)
    ax_t = ax.twinx()
    x = np.arange(len(df))
    width = 0.5

    ax_t.bar(x, df["Failures"], width,
             color='#e67e22', alpha=0.45, label='Failed Trials',
             edgecolor='none', linewidth=0)

    ax_t.set_ylabel('Failed Trials (out of 30)', fontsize=12, fontweight='bold', color='#e67e22')
    ax_t.set_ylim(0, 16)
    ax_t.set_yticks(np.arange(0, 17, 2))
    ax_t.tick_params(axis='y', labelcolor='#e67e22')

    ax.set_xlabel('Interference Speed Category', fontsize=12)
    

    # Combined legend
    lines, labels = ax.get_legend_handles_labels()
    bars, bar_labels = ax_t.get_legend_handles_labels()
    ax.legend(lines + bars, labels + bar_labels, loc='upper right', framealpha=0.9)

    fig.tight_layout()
    fig.savefig(out_path, dpi=300)
    plt.close(fig)


def plot_figure_6_2_2(df, out_path):
    """Figure 6.2.2: Safety-envelope stability (min distance ± σ)."""
    fig, ax = plt.subplots(figsize=(6.4, 3.6))

    x = np.arange(len(df))
    mean = df["Min_Dist"].values
    sigma = df["Dist_Sigma"].values

    ax.errorbar(
        x, mean, yerr=sigma,
        fmt='-o',
        color='#2c3e50',
        ecolor='#c0392b',
        elinewidth=0.9,
        capsize=3,
        capthick=0.9,
        markersize=4.5,
        markerfacecolor='#2c3e50',
        markeredgecolor='#2c3e50',
        linewidth=1.1,
        zorder=3,
    )

    for i, val in enumerate(mean):
        ax.annotate(
            f"{val:.3f}m",
            (x[i], val),
            xytext=(4, 4), textcoords='offset points',
            ha='left', va='bottom', fontsize=7,
            color='#2c3e50',
        )

    ax.set_xticks(x)
    ax.set_xticklabels(df["Level"], fontsize=8)
    ax.set_xlabel('Interference Speed Category', fontsize=9)
    ax.set_ylabel('Minimum Safe Distance (meters)', fontsize=9)
    ax.tick_params(axis='y', labelsize=8)

    lo = min(mean - sigma) - 0.008
    hi = max(mean + sigma) + 0.010
    ax.set_ylim(lo, hi)
    ax.set_xlim(-0.4, len(df) - 0.6)

    ax.grid(True, linestyle='--', alpha=0.35, linewidth=0.6)
    ax.set_axisbelow(True)
    for side in ('top', 'right'):
        ax.spines[side].set_visible(False)
    ax.spines['left'].set_linewidth(0.7)
    ax.spines['bottom'].set_linewidth(0.7)

    fig.tight_layout()
    fig.savefig(out_path, dpi=300)
    plt.close(fig)


def generate_final_report_charts():
    df = load_and_aggregate(CSV_PATH)
    print("Aggregated per-speed results:")
    print(df.to_string(index=False))

    out_621 = os.path.join(OUT_DIR, "figure_6_2_1_success_vs_speed.png")
    out_622 = os.path.join(OUT_DIR, "figure_6_2_2_stability.png")

    plot_figure_6_2_1(df, out_621)
    plot_figure_6_2_2(df, out_622)

    print(f"\n[saved] {out_621}")
    print(f"[saved] {out_622}")


if __name__ == "__main__":
    generate_final_report_charts()
