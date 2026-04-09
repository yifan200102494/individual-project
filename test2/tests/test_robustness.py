import sys
import os
import numpy as np
import matplotlib.pyplot as plt

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from predictor import ObstaclePredictor


def run_robustness_test(save_dir="results", show_plot=True):
    # 固定随机种子，保证结果可复现
    np.random.seed(42)

    # 仿真参数：1 秒，240 Hz
    dt = 1 / 240
    total_steps = 240
    time_axis = np.arange(total_steps) * dt

    # 1) 真实轨迹：障碍物以 0.5 m/s 匀速直线运动
    true_trajectory = np.array([
        [0.5 * (i * dt), 0.2, 0.1] for i in range(total_steps)
    ])

    # 2) 带噪声观测
    noise_std = 0.02  # 2 cm
    noisy_observations = true_trajectory + np.random.normal(
        0, noise_std, true_trajectory.shape
    )

    # 3) KF 滤波
    predictor = ObstaclePredictor()
    filtered_trajectory = []

    for obs in noisy_observations:
        predictor.update(obs)
        filtered_trajectory.append(predictor.state[:3].copy())

    filtered_trajectory = np.array(filtered_trajectory)

    # 4) 计算 RMSE
    error_raw = np.sqrt(np.mean(np.sum((noisy_observations - true_trajectory) ** 2, axis=1)))
    error_filtered = np.sqrt(np.mean(np.sum((filtered_trajectory - true_trajectory) ** 2, axis=1)))
    reduction = (error_raw - error_filtered) / error_raw * 100

    print("\n--- Test results of the robustness of the perception layer ---")
    print(f"Standard deviation of simulated sensor noise: {noise_std * 100:.1f} cm")
    print(f"Root Mean Square Error of the original visual observation (RMSE): {error_raw * 100:.4f} cm")
    print(f"Estimated error after Kalman filtering (RMSE): {error_filtered * 100:.4f} cm")
    print(f"[Conclusion] The error has been reduced: {reduction:.2f}%")

    if reduction > 30:
        print("✅ Test passed: KF successfully smoothed out the visual noise, proving the value of high-frequency prediction.")

    # 创建保存目录
    os.makedirs(save_dir, exist_ok=True)

    # 5) 画图
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

    # 左图：x 方向位置随时间变化
    axes[0].plot(time_axis, true_trajectory[:, 0], label="True x-position")
    axes[0].plot(time_axis, noisy_observations[:, 0], label="Noisy observation", alpha=0.6)
    axes[0].plot(time_axis, filtered_trajectory[:, 0], label="KF filtered", linewidth=2)
    axes[0].set_xlabel("Time (s)")
    axes[0].set_ylabel("X position (m)")
    axes[0].set_title("Trajectory Tracking Under Sensor Noise")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # 右图：RMSE 对比
    axes[1].bar(
        ["Raw observation", "KF filtered"],
        [error_raw * 100, error_filtered * 100]
    )
    axes[1].set_ylabel("RMSE (cm)")
    axes[1].set_title(f"RMSE Before and After Kalman Filtering")
    axes[1].grid(True, axis="y", alpha=0.3)

    plt.tight_layout()

    out_path = os.path.join(save_dir, "robustness_test_plot.png")
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    print(f"Figure saved to: {out_path}")

    if show_plot:
        plt.show()
    else:
        plt.close(fig)

    return {
        "noise_std_cm": noise_std * 100,
        "rmse_raw_cm": error_raw * 100,
        "rmse_filtered_cm": error_filtered * 100,
        "reduction_percent": reduction,
        "figure_path": out_path,
    }


if __name__ == "__main__":
    run_robustness_test()