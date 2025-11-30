import numpy as np
import math

class VisualAvoidanceSystem:
    def __init__(self, safe_distance=0.45, stop_distance=0.15):
        # d_th2: 警戒距离 (开始规划避障)
        self.d_th2 = safe_distance 
        # d_th1: 最小安全距离
        self.d_th1 = stop_distance

    def compute_modified_step(self, current_pos, target_pos, obstacle_pos):
        """
        改进版避障算法 v3：
        - 优先检测是否已经越过障碍物高度
        - 区分"抬起阶段"和"搬运阶段"
        - 搬运阶段时保持高度，水平绕行
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
        # 2. 基础状态检测
        # ==========================================
        if dist_to_target < 0.02:
            return current_pos, "ARRIVED"

        # Docking Mode: 极近距离时，无视避障，强制进场
        if dist_to_target < 0.10: 
             step_size = 0.01
             next_pos = curr_arr + v_c_dir * step_size
             return next_pos.tolist(), "DOCKING"

        # ==========================================
        # 3. 【最高优先级】高度优势检测
        # 如果已经明显高于障碍物，保持高度水平穿越
        # ==========================================
        height_above_obs = curr_arr[2] - obs_arr[2]  # 当前高度 - 障碍物高度
        clearance_threshold = 0.08  # 需要高于障碍物8cm才认为可以安全越过
        
        # 判断移动方向：主要是水平移动还是垂直移动
        vertical_diff = targ_arr[2] - curr_arr[2]  # 目标与当前的Z轴差距
        horizontal_diff = np.linalg.norm(targ_arr[:2] - curr_arr[:2])  # 水平距离
        
        # 【关键】如果已经越过障碍物高度，保持高度水平穿越
        if height_above_obs > clearance_threshold:
            horizontal_dist_to_obs = np.linalg.norm(curr_arr[:2] - obs_arr[:2])
            
            # 构造水平移动向量（保持当前高度）
            move_dir = np.array([v_c_dir[0], v_c_dir[1], 0])
            
            if np.linalg.norm(move_dir) > 0.01:
                move_dir = move_dir / np.linalg.norm(move_dir)
            else:
                # 如果目标正好在正上/下方，直接向目标移动
                return (curr_arr + v_c_dir * 0.05).tolist(), "OVERHEAD_VERTICAL"
            
            # 根据水平距离决定是否需要略微绕行
            if horizontal_dist_to_obs < 0.15:
                # 障碍物很近，需要绕行
                to_obs_h = np.array([obs_arr[0] - curr_arr[0], obs_arr[1] - curr_arr[1]])
                if np.linalg.norm(to_obs_h) > 0.01:
                    to_obs_h = to_obs_h / np.linalg.norm(to_obs_h)
                    # 绕行方向
                    bypass_left = np.array([-to_obs_h[1], to_obs_h[0], 0])
                    bypass_right = np.array([to_obs_h[1], -to_obs_h[0], 0])
                    target_h = np.array([v_c_dir[0], v_c_dir[1], 0])
                    if np.dot(bypass_left[:2], target_h[:2]) > np.dot(bypass_right[:2], target_h[:2]):
                        bypass_dir = bypass_left
                    else:
                        bypass_dir = bypass_right
                    # 混合方向：绕行 + 目标方向
                    move_dir = bypass_dir * 0.5 + move_dir * 0.5
                    move_dir = move_dir / np.linalg.norm(move_dir)
            
            # 只有当目标明显更低且已远离障碍物时，才允许下降
            if vertical_diff < -0.05 and horizontal_dist_to_obs > 0.25:
                move_dir[2] = max(vertical_diff / dist_to_target, -0.2)
                move_dir = move_dir / np.linalg.norm(move_dir)
            
            step_size = 0.05
            next_pos = curr_arr + move_dir * step_size
            return next_pos.tolist(), "OVERHEAD_CROSS"

        # ==========================================
        # 4. 抬起模式检测：目标主要在上方
        # ==========================================
        is_lifting = vertical_diff > 0.05 and vertical_diff > horizontal_diff * 0.5
        
        if is_lifting:
            horizontal_dist_to_obs = np.linalg.norm(curr_arr[:2] - obs_arr[:2])
            # 如果障碍物在水平方向有一定距离，或者我们已经高于障碍物（但还没达到越过阈值）
            if horizontal_dist_to_obs > 0.20 or (height_above_obs > 0.05 and height_above_obs <= clearance_threshold):
                step_size = 0.04
                # 构造一个以向上为主、略微朝目标的移动向量
                lift_dir = np.array([0, 0, 1.0])  # 主要向上
                lift_dir += v_c_dir * 0.3  # 略微朝目标
                lift_dir = lift_dir / np.linalg.norm(lift_dir)
                next_pos = curr_arr + lift_dir * step_size
                return next_pos.tolist(), "LIFTING"

        # ==========================================
        # 5. 搬运模式检测（补充）：高度不够但在安全水平距离外
        # ==========================================
        if height_above_obs > 0 and height_above_obs <= clearance_threshold:
            horizontal_dist_to_obs = np.linalg.norm(curr_arr[:2] - obs_arr[:2])
            # 如果水平距离足够远，可以直接移动（但保持高度）
            if horizontal_dist_to_obs > 0.25:
                move_dir = np.array([v_c_dir[0], v_c_dir[1], 0])
                if np.linalg.norm(move_dir) > 0.01:
                    move_dir = move_dir / np.linalg.norm(move_dir)
                    step_size = 0.05
                    next_pos = curr_arr + move_dir * step_size
                    return next_pos.tolist(), "TRANSPORT_FAR"

        # Leaving: 如果我们正背对障碍物移动(点积<0)，不需要避障，全速离开
        dot_product = np.dot(v_c_dir, c_hat)
        if dot_product < -0.1: 
            step_size = 0.05 
            next_pos = curr_arr + v_c_dir * step_size
            return next_pos.tolist(), "LEAVING"

        # ==========================================
        # 6. 动态避障核心算法
        # ==========================================
        
        # A. 安全区域 -> 直走
        if dist_to_obs > self.d_th2:
            step_size = 0.05
            next_pos = curr_arr + v_c_dir * step_size
            return next_pos.tolist(), "NORMAL"

        # --- 进入避障计算 ---
        
        # 计算危险系数 (0.0 = 安全, 1.0 = 贴脸)
        risk_level = (self.d_th2 - dist_to_obs) / (self.d_th2 - self.d_th1)
        risk_level = np.clip(risk_level, 0.0, 1.0)
        
        # 分解速度向量
        v_parallel_val = np.dot(v_c_full, c_hat)
        v_parallel = v_parallel_val * c_hat
        v_perp = v_c_full - v_parallel
        
        # 死锁检测
        perp_magnitude = np.linalg.norm(v_perp)
        is_deadlock = perp_magnitude < 0.1
        
        # 向上跨越的向量
        up_vector = np.array([0.0, 0.0, 1.0])
        
        # 【关键修改】根据高度动态调整向上的力度
        # 如果已经高于障碍物，完全不需要向上飞
        if height_above_obs > 0.12:
            up_scale = 0.0  # 已经足够高（12cm+），不再向上
        elif height_above_obs > 0.08:
            up_scale = 0.1  # 接近足够高（8-12cm），微弱向上
        elif height_above_obs > 0.04:
            up_scale = 0.3  # 刚刚越过（4-8cm），适度向上
        else:
            up_scale = 1.0  # 还在障碍物下方或刚超过，全力向上
        
        if is_deadlock:
            v_perp = up_vector * 2.0 * up_scale
        else:
            v_perp += up_vector * (risk_level * 2.0 * up_scale)

        # 水平斥力
        c_hat_horizontal = np.array([c_hat[0], c_hat[1], 0])
        c_hat_h_norm = np.linalg.norm(c_hat_horizontal)
        if c_hat_h_norm > 0.01:
            c_hat_horizontal = c_hat_horizontal / c_hat_h_norm
        else:
            c_hat_horizontal = np.array([1, 0, 0])
        
        repulsion_force = -1.0 * risk_level * c_hat_horizontal * 1.5
        
        if v_perp[2] > 0.5:
            repulsion_force *= 0.3

        # 合成最终向量
        v_modified = v_perp + repulsion_force
        
        # 只有在还没越过障碍物时才强制向上
        if vertical_diff > 0.05 and v_modified[2] < 0.3 and height_above_obs < 0:
            v_modified[2] = max(v_modified[2], 0.5)
        
        # 归一化并输出
        if np.linalg.norm(v_modified) > 0:
            move_dir = v_modified / np.linalg.norm(v_modified)
            step_size = 0.03 if risk_level > 0.5 else 0.05
            next_pos = curr_arr + move_dir * step_size
            return next_pos.tolist(), "AVOIDING"
        else:
            return (curr_arr + np.array([0,0,0.01])).tolist(), "STUCK_RECOVERY"
