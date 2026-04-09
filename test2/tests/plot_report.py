import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# 读取你的最终黄金数据
df = pd.read_csv('ablation_final_goldilocks.csv')

plt.style.use('seaborn-v0_8-whitegrid')
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

speeds = df['Speed'].unique()
x = np.arange(len(speeds))
width = 0.35

proactive_sr = df[df['Mode']=='proactive']['Success_Rate'].values
reactive_sr = df[df['Mode']=='reactive']['Success_Rate'].values

proactive_accel = df[df['Mode']=='proactive']['Avg_Max_Accel'].values
reactive_accel = df[df['Mode']=='reactive']['Avg_Max_Accel'].values

# --- 上图：成功率对比 ---
rects1 = ax1.bar(x - width/2, proactive_sr, width, label='Proactive (APF + KF)', color='#2ecc71', edgecolor='black', linewidth=1.2)
rects2 = ax1.bar(x + width/2, reactive_sr, width, label='Reactive Baseline (APF Only)', color='#e74c3c', edgecolor='black', linewidth=1.2)

ax1.set_ylabel('Task Success Rate (%)', fontsize=12, fontweight='bold')
ax1.set_title('Figure 5.4: Ablation Study - Performance vs. Kinematic Safety Limits', fontsize=14, fontweight='bold', pad=15)
ax1.set_ylim(0, 115)
ax1.legend(loc='upper right', frameon=True, shadow=True)

for rects in [rects1, rects2]:
    for rect in rects:
        height = rect.get_height()
        ax1.annotate(f'{height:.1f}%', xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 4), textcoords="offset points", ha='center', va='bottom', fontweight='bold', fontsize=10)

# --- 下图：物理加速度对比 ---
rects3 = ax2.bar(x - width/2, proactive_accel, width, label='Proactive (APF + KF)', color='#27ae60', alpha=0.8, edgecolor='black', linewidth=1.2)
rects4 = ax2.bar(x + width/2, reactive_accel, width, label='Reactive Baseline (APF Only)', color='#c0392b', alpha=0.8, edgecolor='black', linewidth=1.2)

ax2.set_ylabel('Avg Max Acceleration (m/s²)', fontsize=12, fontweight='bold')
ax2.set_xlabel('Dynamic Obstacle Speed (m/step)', fontsize=12, fontweight='bold')
ax2.set_xticks(x)
ax2.set_xticklabels(['Slow (0.003)', 'Extreme (0.007)', 'Insane (0.017)'], fontsize=11)
ax2.set_ylim(0, 22)

# 画一条 ISO 红色警戒线
ax2.axhline(y=20.0, color='red', linestyle='--', linewidth=2, label='ISO Safety Fault Limit (20 m/s²)')
ax2.legend(loc='upper left', frameon=True, shadow=True)

for rects in [rects3, rects4]:
    for rect in rects:
        height = rect.get_height()
        ax2.annotate(f'{height:.2f}', xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 4), textcoords="offset points", ha='center', va='bottom', fontweight='bold', fontsize=10)

plt.tight_layout()
plt.savefig('Figure_5_4_Ablation_Study.png', dpi=300)
print("✅ 论文图表已生成：Figure_5_4_Ablation_Study.png")