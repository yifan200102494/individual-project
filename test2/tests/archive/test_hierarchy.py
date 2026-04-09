import sys
import os
import numpy as np
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from avoidance import VisualAvoidanceSystem

def run_hierarchy_test():
    avoidance = VisualAvoidanceSystem()
    curr, targ = [0, 0, 0.4], [0.5, 0, 0.4] # 机器人试图水平移动
    
    print("--- Hierarchical decision-making logic test ---")
    
    # 1. 场景 A: 低矮障碍物 (Z=0.1)
    obs_low = [0.25, 0, 0.1]
    avoidance.set_obstacle_height_info({"clearance_height": 0.2, "confidence": 0.9})
    _, status_low = avoidance.compute_modified_step(curr, targ, obs_low)
    print(f"Scene A (Low Obstacle): Decision = {status_low} (except: OVERHEAD_CROSS)")

    # 2. 场景 B: 高大障碍物 (Z=0.6)
    obs_high = [0.25, 0, 0.6]
    avoidance.set_obstacle_height_info({"clearance_height": 0.7, "confidence": 0.9})
    _, status_high = avoidance.compute_modified_step(curr, targ, obs_high)
    print(f"Scene B (Tall Obstacle): Decision = {status_high} (except: AVOIDING/LIFTING)")

    if "OVERHEAD" in status_low and "OVERHEAD" not in status_high:
        print("✅ Test passed: The system successfully implemented the spatial strategy switching based on the height of obstacles.")

if __name__ == "__main__":
    run_hierarchy_test()