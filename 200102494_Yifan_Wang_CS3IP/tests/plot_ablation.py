import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['font.sans-serif'] = ['Arial']
plt.rcParams['axes.unicode_minus'] = False

HERE = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(HERE, "result")
CSV_PATH = os.path.join(OUT_DIR, "ablation_continuous_metrics.csv")

SPEED_ORDER = [0.001, 0.003, 0.005, 0.007, 0.017]
SPEED_LABELS = {
    0.001: "Slow\n(0.001)",
    0.003: "Medium\n(0.003)",
    0.005: "Fast\n(0.005)",
    0.007: "Extreme\n(0.007)",
    0.017: "Insane\n(0.017)",
}

HARD_STOP = 0.15  # d_1 hard-stop threshold (m)


def load_ablation(csv_path):
    df = pd.read_csv(csv_path)
    pro = df[df["Mode"] == "proactive"].set_index("Speed").loc[SPEED_ORDER]
    rea = df[df["Mode"] == "reactive"].set_index("Speed").loc[SPEED_ORDER]
    return pro, rea


def plot_figure_6_2_6(pro, rea, out_path):
    """Figure 6.2.6: Mean minimum safety distance ablation."""
    fig, ax = plt.subplots(figsize=(8.5, 4.2))

    x = np.arange(len(SPEED_ORDER))
    width = 0.35

    pro_mean = pro["Mean_Min_Dist"].values
    pro_std = pro["Std_Min_Dist"].values
    rea_mean = rea["Mean_Min_Dist"].values
    rea_std = rea["Std_Min_Dist"].values

    b1 = ax.bar(x - width / 2, pro_mean, width,
                yerr=pro_std, label='APF + KF (Proactive)',
                color='#27ae60', alpha=0.88,
                edgecolor='#1e8449', linewidth=0.8,
                error_kw={"ecolor": "#1e3a2b", "elinewidth": 0.9, "capsize": 3, "capthick": 0.9},
                zorder=3)
    b2 = ax.bar(x + width / 2, rea_mean, width,
                yerr=rea_std, label='APF only (Reactive)',
                color='#c0392b', alpha=0.85,
                edgecolor='#922b21', linewidth=0.8,
                error_kw={"ecolor": "#4a1612", "elinewidth": 0.9, "capsize": 3, "capthick": 0.9},
                zorder=3)

    # Value labels
    for bars, means in [(b1, pro_mean), (b2, rea_mean)]:
        for bar, val in zip(bars, means):
            ax.annotate(f"{val:.3f}",
                        (bar.get_x() + bar.get_width() / 2, val),
                        xytext=(0, 3), textcoords='offset points',
                        ha='center', va='bottom', fontsize=7.5,
                        color='#2c3e50')

    ax.set_xticks(x)
    ax.set_xticklabels([SPEED_LABELS[s] for s in SPEED_ORDER], fontsize=9)
    ax.set_ylabel('Mean Minimum Safety Distance (m)', fontsize=9)
    ax.tick_params(axis='y', labelsize=8)

    y_lo = min(min(pro_mean - pro_std), min(rea_mean - rea_std)) - 0.01
    y_hi = max(max(pro_mean + pro_std), max(rea_mean + rea_std)) + 0.015
    ax.set_ylim(max(0.0, y_lo), y_hi)

    ax.grid(True, axis='y', linestyle='--', alpha=0.35, linewidth=0.6)
    ax.set_axisbelow(True)
    for side in ('top', 'right'):
        ax.spines[side].set_visible(False)
    ax.spines['left'].set_linewidth(0.7)
    ax.spines['bottom'].set_linewidth(0.7)

    ax.legend(loc='upper right', fontsize=8, framealpha=0.9, edgecolor='#bdc3c7')

    fig.tight_layout()
    fig.savefig(out_path, dpi=300)
    plt.close(fig)


def generate_ablation_chart():
    pro, rea = load_ablation(CSV_PATH)
    print("Proactive (APF+KF):")
    print(pro[["Mean_Min_Dist", "Std_Min_Dist"]].to_string())
    print("\nReactive (APF-only):")
    print(rea[["Mean_Min_Dist", "Std_Min_Dist"]].to_string())

    out_path = os.path.join(OUT_DIR, "figure_6_2_6_ablation_min_dist.png")
    plot_figure_6_2_6(pro, rea, out_path)
    print(f"\n[saved] {out_path}")


if __name__ == "__main__":
    generate_ablation_chart()
