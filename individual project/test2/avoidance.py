import numpy as np
import math

class VisualAvoidanceSystem:
    def __init__(self, safe_distance=0.45, stop_distance=0.15):
        # ==========================================
        # 【关键修改 1】 扩大警戒范围
        # d_th2 (safe_distance): 警戒线。原来是 0.25，现在改为 0.45。
        # 含义：当障碍物进入 0.45米 范围内，机器人就开始“绕路”，而不是直走。
        # 这样能给机器人足够的空间在不停车的情况下画出弧线。
        # ==========================================
        self.d_th2 = safe_distance 
        
        # d_th1 (stop_distance): 红线。保持 0.15。
        # 含义：只有距离小于 0.15米 时，才触发 PDF 里的 Strategy 2 (Stop)。
        self.d_th1 = stop_distance

    def compute_modified_step(self, current_pos, target_pos, obstacle_pos):
        """
        基于 PDF Equation (2)-(4) 实现动态路径修改
        """
        curr_arr = np.array(current_pos)
        targ_arr = np.array(target_pos)
        obs_arr = np.array(obstacle_pos)
        
        # 1. 基础向量计算
        # v_c: 机器人原本想要走的向量 (Current Motion Vector)
        v_c_full = targ_arr - curr_arr
        dist_to_target = np.linalg.norm(v_c_full)
        
        # c_vec: 从机器人指向障碍物的向量 (Collision Vector) [cite: 325]
        c_vec = obs_arr - curr_arr
        dist_to_obs = np.linalg.norm(c_vec)

        # 归一化方向
        if dist_to_target > 0:
            v_c_dir = v_c_full / dist_to_target
        else:
            return current_pos, "ARRIVED"
            
        if dist_to_obs > 0:
            c_hat = c_vec / dist_to_obs # 障碍物方向单位向量
        else:
            c_hat = np.array([1, 0, 0])

        # ==========================================
        # 2. 状态豁免判断
        # ==========================================
        
        # A. 到达判断
        if dist_to_target < 0.02:
            return current_pos, "ARRIVED"

        # B. 终点区豁免 (Docking Mode)
        # 只要距离目标很近，就无视障碍物，强制进场
        if dist_to_target < 0.15:
             step_size = 0.02
             next_pos = curr_arr + v_c_dir * step_size
             return next_pos.tolist(), "DOCKING"

        # C. 顺风放行 (Leaving)
        # 如果障碍物在机器人身后 (点积 < 0)，完全不理会
        dot_product = np.dot(v_c_dir, c_hat)
        if dot_product < 0: 
            step_size = 0.05 # 全速前进
            next_pos = curr_arr + v_c_dir * step_size
            return next_pos.tolist(), "LEAVING"

        # ==========================================
        # 3. 核心避障算法 (对应 PDF Section 3.5)
        # ==========================================
        
        # 情况 A: 安全区域 (大于 d_th2) -> 直走
        if dist_to_obs > self.d_th2:
            step_size = 0.05
            next_pos = curr_arr + v_c_dir * step_size
            return next_pos.tolist(), "NORMAL"

        # 情况 B: 极度危险 (小于 d_th1) -> 策略 2: Stop / Retreat [cite: 361]
        if dist_to_obs < self.d_th1:
            # 这里我们做一个微小的后退，而不是死锁
            step_size = 0.01
            retreat_dir = -1.0 * c_hat # 向障碍物反方向退
            next_pos = curr_arr + retreat_dir * step_size
            return next_pos.tolist(), "EMERGENCY_RETREAT"

        # 情况 C: 动态避障区 (d_th1 < dist < d_th2) -> 策略 4: Modify Path [cite: 362]
        # 这是你要的“边走边躲”逻辑
        
        # 计算排斥系数 (0.0 到 1.0)
        # 距离越近，系数越大，斥力越强
        repulsion_factor = (self.d_th2 - dist_to_obs) / (self.d_th2 - self.d_th1)
        
        # [PDF Eq 2] 分解速度向量
        # v_parallel: 平行于障碍物方向的分量 (这就好比是“撞向障碍物的速度”)
        v_parallel_val = np.dot(v_c_full, c_hat)
        v_parallel = v_parallel_val * c_hat
        
        # [PDF Eq 3] 垂直分量 (这是“绕开障碍物的速度”，我们要保留甚至加强它)
        v_perp = v_c_full - v_parallel
        
        # [PDF Eq 4 & Fig 13] 修改平行分量
        # 我们不仅要减小撞向障碍物的速度，还要给它一个反向推力
        # v_a_parallel 是修正后的平行分量
        # 这里的 5.0 是增益系数，让机器人反应更灵敏
        avoidance_force = -1.0 * repulsion_factor * c_hat * 5.0 
        
        # 合成新方向：保留原有的侧向移动 + 施加反向推力
        # 如果 v_perp 本身很小（比如正对着障碍物走），我们需要人为制造侧向力
        if np.linalg.norm(v_perp) < 0.1:
            # 制造一个侧向扰动，优先向上方或侧方绕行
            # 这里简单地利用外积产生一个垂直方向
            up_vec = np.array([0, 0, 1])
            side_vec = np.cross(c_hat, up_vec)
            v_perp += side_vec * 2.0 # 强行侧移
            
        v_modified = v_perp + avoidance_force
        
        # 归一化并应用步长
        if np.linalg.norm(v_modified) > 0:
            move_dir = v_modified / np.linalg.norm(v_modified)
            step_size = 0.03 # 避障时稍微减速，保证平滑
            next_pos = curr_arr + move_dir * step_size
            return next_pos.tolist(), "AVOIDING_ACTIVE"
        else:
            return curr_arr.tolist(), "WAITING"