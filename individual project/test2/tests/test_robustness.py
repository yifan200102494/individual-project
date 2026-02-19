import sys
import os
import numpy as np
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from predictor import ObstaclePredictor

def run_robustness_test():
    # 设定仿真参数：1秒时间，240Hz
    dt = 1/240
    total_steps = 240
    
    # 1. 生成“真实”轨迹：障碍物以 0.5m/s 匀速直线运动
    # 轨迹方程：p(t) = p0 + v*t
    true_trajectory = np.array([[0.5 * (i * dt), 0.2, 0.1] for i in range(total_steps)])
    
    # 2. 模拟“带有噪声”的视觉观测数据
    # 假设视觉传感器有 ±0.02m (2厘米) 的随机跳变噪声
    noise_std = 0.02
    noisy_observations = true_trajectory + np.random.normal(0, noise_std, true_trajectory.shape)
    
    # 3. 初始化你的预测器
    predictor = ObstaclePredictor()
    filtered_trajectory = []
    
    # 模拟实时数据流输入
    for obs in noisy_observations:
        predictor.update(obs)
        filtered_trajectory.append(predictor.state[:3].copy())
    
    filtered_trajectory = np.array(filtered_trajectory)
    
    # 4. 定量计算误差：均方根误差 (RMSE)
    # 计算原始观测值与真实值的偏差
    error_raw = np.sqrt(np.mean(np.sum((noisy_observations - true_trajectory)**2, axis=1)))
    # 计算滤波后估计值与真实值的偏差
    error_filtered = np.sqrt(np.mean(np.sum((filtered_trajectory - true_trajectory)**2, axis=1)))
    
    reduction = (error_raw - error_filtered) / error_raw * 100
    
    print("\n--- Test results of the robustness of the perception layer ---")
    print(f"Standard deviation of simulated sensor noise: {noise_std*100:.1f} cm")
    print(f"Root Mean Square Error of the original visual observation (RMSE): {error_raw*100:.4f} cm")
    print(f"Estimated error after Kalman filtering (RMSE): {error_filtered*100:.4f} cm")
    print(f"【Conclusion】 The error has been reduced.: {reduction:.2f}%")
    
    if reduction > 30:
        print("✅ Test passed: KF successfully smoothed out the visual noise, proving the value of high-frequency prediction.")

if __name__ == "__main__":
    run_robustness_test()