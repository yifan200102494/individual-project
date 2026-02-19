import time
import numpy as np
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from predictor import ObstaclePredictor
from avoidance import VisualAvoidanceSystem

def run_benchmark():
    predictor = ObstaclePredictor()
    avoidance = VisualAvoidanceSystem()
    
    print("Real-time performance benchmark testing")
    
    # 1. 测试预测模块 (Kalman Filter)
    start = time.perf_counter()
    for _ in range(1000):
        predictor.update([0.5, 0.5, 0.5])
        _ = predictor.predict_position(steps_ahead=10)
    end = time.perf_counter()
    kf_time = (end - start) # 总耗时(ms)
    print(f"Average time consumption of the prediction module (KF): {kf_time:.6f} ms")

    # 2. 测试避障决策模块 (Modified APF) [cite: 86]
    # 模拟输入：当前位置，目标位置，障碍物位置
    curr, targ, obs = [0, 0, 0], [0, 0, 1], [0.3, 0.3, 0.3]
    start = time.perf_counter()
    for _ in range(1000):
        _ = avoidance.compute_modified_step(curr, targ, obs)
    end = time.perf_counter()
    apf_time = (end - start)
    print(f"Average time consumption of the decision-making module (APF): {apf_time:.6f} ms")
    
    total = kf_time + apf_time
    print(f"\nTotal time consumption of core computing: {total:.6f} ms")
    print(f"Occupation of the 240Hz cycle budget proportion: {(total/4.16)*100:.2f}%")

if __name__ == "__main__":
    run_benchmark()