import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("benchmark_report_summary_1.csv")
df = df.sort_values("gap_width_m", ascending=False)

fig, ax1 = plt.subplots(figsize=(10, 6))

ax1.errorbar(
    df["gap_width_m"],
    df["mean_min_clearance_m"],
    yerr=df["std_min_clearance_m"],
    fmt='-o',
    linewidth=2,
    capsize=5
)
ax1.set_xlabel("Gap Width (m)")
ax1.set_ylabel("Mean Minimum Clearance (m)")
ax1.set_title("Figure 6.2.4: Benchmark Results Across Narrow-Passage Workspace Configurations")
ax1.grid(True, linestyle='--', alpha=0.5)

plt.tight_layout()
plt.savefig("figure_6_2_4_narrow_passage.png", dpi=300)
plt.show()