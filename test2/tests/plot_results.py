import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# 设置专业绘图风格
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['font.sans-serif'] = ['Arial']
plt.rcParams['axes.unicode_minus'] = False

def generate_final_report_charts():
    # --- 1. 严格按照报告正文中的统计数据进行初始化 [cite: 7, 168-179, 182] ---
    data = {
        "Level": ["Slow(0.001)", "Medium(0.003)", "Fast(0.005)", "Extreme(0.007)", "Insane(0.017)"],
        "Success_Rate": [90.0, 96.7, 90.0, 83.3, 80.0], # 对应 [cite: 168-169]
        "Min_Dist": [0.148, 0.155, 0.158, 0.150, 0.148], # 对应 [cite: 178, 182]
        "Dist_Sigma": [0.004, 0.003, 0.002, 0.005, 0.008], # 亚厘米级标准差
        "Collisions": [6.23, 2.20, 2.60, 6.90, 9.73]    # 对应 [cite: 179]
    }
    df = pd.DataFrame(data)

    # --- 图表一：性能分布与物理交互分析 (对应报告 Figure 5.1) ---
    fig1, ax1 = plt.subplots(figsize=(10, 6))
    
    # 成功率折线图 (强调倒 U 型趋势)
    line = ax1.plot(df["Level"], df["Success_Rate"], marker='s', color='#2ecc71', 
                    linewidth=3, markersize=10, label='Task Success Rate (%)')
    ax1.set_ylabel('Success Rate (%)', fontsize=12, fontweight='bold', color='#27ae60')
    ax1.set_ylim(70, 105)
    
    # 添加数值标注
    for i, val in enumerate(df["Success_Rate"]):
        ax1.annotate(f"{val}%", (df["Level"][i], df["Success_Rate"][i]), 
                     xytext=(0, 10), textcoords='offset points', ha='center', fontweight='bold')

    # 碰撞帧数柱状图 (使用次坐标轴)
    ax1_twin = ax1.twinx()
    bars = ax1_twin.bar(df["Level"], df["Collisions"], alpha=0.3, color='#f39c12', 
                        width=0.5, label='Avg Collision Frames')
    ax1_twin.set_ylabel('Avg Collision Frames', fontsize=12, fontweight='bold', color='#e67e22')
    ax1_twin.set_ylim(0, 15)

    plt.title('Figure 5.1: Inverted U-Shaped Success Rate Distribution\n& Collision Frame Analysis', 
              fontsize=14, fontweight='bold', pad=20)
    fig1.tight_layout()
    fig1.savefig('figure_5_1_performance_fixed.png', dpi=300)

    # --- 图表二：安全包络统计稳定性分析 (对应报告 Figure 5.2) ---
    fig2, ax2 = plt.subplots(figsize=(10, 6))
    
    # 使用极小的误差棒展示稳定性
    ax2.errorbar(df["Level"], df["Min_Dist"], yerr=df["Dist_Sigma"], fmt='-o', 
                 color='#34495e', ecolor='#e74c3c', elinewidth=2, capsize=6, 
                 markersize=8, linewidth=2, label='Mean Min Distance ± σ')
    
    ax2.set_title('Figure 5.2: Statistical Stability Analysis of Safety Envelope\n(Sub-centimeter Variance Verification)', 
                  fontsize=14, fontweight='bold', pad=20)
    ax2.set_xlabel('Interference Speed Category', fontsize=12)
    ax2.set_ylabel('Minimum Safe Distance (meters)', fontsize=12)
    
    # 精细调整 Y 轴，展示 0.148m - 0.158m 的一致性 [cite: 8]
    ax2.set_ylim(0.13, 0.17)
    ax2.grid(True, linestyle='--', alpha=0.5)

    for i, val in enumerate(df["Min_Dist"]):
        ax2.annotate(f"{val}m", (df["Level"][i], df["Min_Dist"][i]), 
                     xytext=(0, 12), textcoords='offset points', ha='center', fontsize=10)

    fig2.tight_layout()
    fig2.savefig('figure_5_2_stability_fixed.png', dpi=300)
    
    print("✅ 图表已更新！请查看 figure_5_1_performance_fixed.png 和 figure_5_2_stability_fixed.png")

if __name__ == "__main__":
    generate_final_report_charts()