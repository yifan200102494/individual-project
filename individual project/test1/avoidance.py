import numpy as np
import math

class VisualAvoidanceSystem:
    def __init__(self, safe_distance=0.45, stop_distance=0.15):
        # d_th2: 警戒距离 (开始规划避障)
        self.d_th2 = safe_distance 
        # d_th1: 最小安全距离 (以前是急停/后退，现在我们要在这里强制跨越)
        self.d_th1 = stop_distance

    def compute_modified_step(self, current_pos, target_pos, obstacle_pos):
        """
        基于 PDF Equation (2)-(4) 改进版：
        引入切向偏转(Tangential Deflection)解决死锁问题，实现“跨越”而非“后退”。
        """
        curr_arr = np.array(current_pos)
        targ_arr = np.array(target_pos)
        obs_arr = np.array(obstacle_pos)
        
        # 1. 向量基础计算
        v_c_full = targ_arr - curr_arr         # 目标向量 (Robot -> Target)
        dist_to_target = np.linalg.norm(v_c_full)
        
        c_vec = obs_arr - curr_arr             # 碰撞向量 (Robot -> Obstacle)
        dist_to_obs = np.linalg.norm(c_vec)

        # 归一化方向
        if dist_to_target > 0.01:
            v_c_dir = v_c_full / dist_to_target
        else:
            return current_pos, "ARRIVED"
            
        if dist_to_obs > 0.001:
            c_hat = c_vec / dist_to_obs 
        else:
            c_hat = np.array([1, 0, 0])

        # ==========================================
        # 2. 状态豁免与任务逻辑
        # ==========================================
        if dist_to_target < 0.02:
            return current_pos, "ARRIVED"

        # Docking Mode: 极近距离时，无视避障，强制进场 (防止在终点上方不敢下去)
        if dist_to_target < 0.10: 
             step_size = 0.01
             next_pos = curr_arr + v_c_dir * step_size
             return next_pos.tolist(), "DOCKING"

        # Leaving: 如果我们正背对障碍物移动(点积<0)，不需要避障，全速离开
        dot_product = np.dot(v_c_dir, c_hat)
        if dot_product < -0.1: 
            step_size = 0.05 
            next_pos = curr_arr + v_c_dir * step_size
            return next_pos.tolist(), "LEAVING"

        # ==========================================
        # 3. 动态避障核心算法 (改进版)
        # ==========================================
        
        # A. 安全区域 -> 直走
        if dist_to_obs > self.d_th2:
            step_size = 0.05
            next_pos = curr_arr + v_c_dir * step_size
            return next_pos.tolist(), "NORMAL"

        # --- 进入避障计算 ---
        
        # 计算危险系数 (0.0 = 安全, 1.0 = 贴脸)
        # 限制范围，防止除零错误
        risk_level = (self.d_th2 - dist_to_obs) / (self.d_th2 - self.d_th1)
        risk_level = np.clip(risk_level, 0.0, 1.0)
        
        # [PDF Eq 2] 分解速度向量：平行分量(撞墙分量)
        v_parallel_val = np.dot(v_c_full, c_hat)
        v_parallel = v_parallel_val * c_hat
        
        # [PDF Eq 3] 分解速度向量：垂直分量(切线分量，即原本的绕行方向)
        v_perp = v_c_full - v_parallel
        
        # === 核心修改：死锁破坏机制 ===
        
        # 1. 增强垂直分量 (Z轴偏置)
        # 如果垂直分量太小 (说明正对着障碍物，死锁！)，或者单纯为了更稳健
        # 我们强制把垂直分量替换为“向上”的向量
        
        perp_magnitude = np.linalg.norm(v_perp)
        
        # 死锁检测：如果原本的绕行力度很小，说明我们正对着障碍物中心
        is_deadlock = perp_magnitude < 0.1
        
        # 构造“向上跨越”的向量
        up_vector = np.array([0.0, 0.0, 1.0])
        
        if is_deadlock:
            # 死锁状态：完全依靠向上飞
            # 这里的 2.0 是强度，保证它能克服重力和距离
            v_perp = up_vector * 2.0 
        else:
            # 非死锁：在原本的绕行方向上，叠加向上的趋势
            # 随着危险程度增加，Z轴权重越来越大
            v_perp += up_vector * (risk_level * 3.0)

        # 2. 智能斥力 (Forward Momentum Preservation)
        # 之前的问题是：避障斥力把向前的速度抵消了，导致停在原地。
        # 现在的逻辑：如果我已经有了足够的向上速度(v_perp)，我就允许保留一部分向前的速度。
        
        # 基础斥力 (向后推)
        repulsion_force = -1.0 * risk_level * c_hat * 2.0
        
        # 如果我们正在向上抬 (v_perp.z > 0)，我们可以减少水平方向的斥力
        # 这样机器人就会画出一道抛物线：既向上，也略微向前
        if v_perp[2] > 0.5:
             repulsion_force *= 0.5 # 斥力减半，允许“顶风作案”跨过去

        # [PDF Eq 4] 合成最终向量
        v_modified = v_perp + repulsion_force
        
        # 3. 归一化并输出
        if np.linalg.norm(v_modified) > 0:
            move_dir = v_modified / np.linalg.norm(v_modified)
            
            # 动态步长：危险时走慢点，安全时走快点
            step_size = 0.03 if risk_level > 0.5 else 0.05
            
            next_pos = curr_arr + move_dir * step_size
            return next_pos.tolist(), "AVOIDING_CROSS"
        else:
            # 极罕见情况，为了防止卡死，给一个微小的向上扰动
            return (curr_arr + np.array([0,0,0.01])).tolist(), "STUCK_RECOVERY"