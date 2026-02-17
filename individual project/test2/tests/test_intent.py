import numpy as np
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from avoidance import VisualAvoidanceSystem

def run_intent_test():
    # 初始化，基础警戒距离 0.45m
    avoidance = VisualAvoidanceSystem(safe_distance=0.45)
    curr, targ, obs = [0, 0, 0], [0, 0, 1], [0.4, 0, 0] # 障碍物在 0.55m 处
    
    print("--- Intent recognition redundancy test ---")
    
    # 1. 测试静态场景
    avoidance.set_obstacle_motion(velocity=[0, 0, 0], is_moving=False, direction='stationary')
    _, status_static = avoidance.compute_modified_step(curr, targ, obs)
    print(f"Scene A (Obstacle stationary): State = {status_static} (except: NORMAL)")

    # 2. 测试“靠近”场景 (速度 0.5m/s)
    # 根据代码逻辑，eff_safe = 0.45 + min(0.5 * 50, 0.15) = 0.60m
    # 此时 0.55m < 0.60m，应触发避障
    avoidance.set_obstacle_motion(velocity=[-0.5, 0, 0], is_moving=True, direction='approaching')
    _, status_approaching = avoidance.compute_modified_step(curr, targ, obs)
    print(f"Scene B (Obstacle Close): State = {status_approaching} (except: AVOIDING/LIFTING)")

    if status_static == "NORMAL" and ("AVOIDING" in status_approaching or "LIFTING" in status_approaching):
        print("✅ Test passed: The system successfully identified the intrusion intention and dynamically expanded the security alert zone.")

if __name__ == "__main__":
    run_intent_test()