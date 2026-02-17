import sys
import os
import pytest
# 确保路径正确
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from avoidance import VisualAvoidanceSystem

def test_payload_boundary_expansion():
    # 初始化避障系统，默认警戒距离为 0.45 [cite: 148]
    avoidance = VisualAvoidanceSystem(safe_distance=0.45)
    
    # 1. 模拟抓取了一个大型工件 (水平半径扩展 0.1m) [cite: 154, 155]
    avoidance.set_grabbed_object_bounds(radius_extend=0.1, bottom_extend=0.05)
    
    # 2. 验证计算步骤中是否考虑了扩展量 [cite: 61]
    # 在你的代码中，eff_safe = self.d_th2 + self.grabbed_radius_extend
    curr, targ, obs = [0, 0, 0], [0, 0, 1], [0.5, 0, 0]
    
    # 当障碍物在 0.5m 处时：
    # 无负载：0.5 > 0.45 (安全)
    # 有负载：0.5 < (0.45 + 0.1) (触发避障)
    _, status = avoidance.compute_modified_step(curr, targ, obs)
    
    # 修改断言逻辑：只要状态不是直走（NORMAL/CLEAR_PATH），就说明负载感知起到了作用
    is_avoiding = "LIFTING" in status or "AVOIDING" in status
    assert is_avoiding, f"Load sensing is not in effect. The status should not be on the normal path; it is actually {status}"
    
    print(f"\n✅ Load sensing verification successful: The negative load volume triggered the obstacle avoidance logic. The current state is {status}")