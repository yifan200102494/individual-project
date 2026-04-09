import sys
import os
import numpy as np

# 确保 Python 能找到你的项目根目录
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from predictor import ObstaclePredictor 

def test_kalman_physics_logic():
    # 按照文档设定 dt = 1/240s 
    dt = 1/240
    predictor = ObstaclePredictor() 
    
    # 1. 模拟初始化状态：位置 (0,0,0)，速度 (2, 0, 0)
    # 状态向量 x = [px, py, pz, vx, vy, vz] 
    predictor.state = np.array([0.0, 0.0, 0.0, 2.0, 0.0, 0.0])
    predictor.initialized = True # 必须手动设为 True，否则 predict_position 会返回 None
    
    # 2. 调用你代码里的预测函数，预测 1 步之后的位置
    # predict_position(steps_ahead=1) 应该等于 原位置 + 速度 * 1 * dt 
    predicted_pos, confidence = predictor.predict_position(steps_ahead=1)
    
    # 3. 验证结果
    expected_px = 0.0 + 2.0 * dt
    assert np.isclose(predicted_pos[0], expected_px), f"Physical prediction failure: Expectation{expected_px}, In fact {predicted_pos[0]}"
    print(f"\n✅ Physical logic verification passed! Predicted position: {predicted_pos[0]:.6f}")