import numpy as np
import math

class VisualAvoidanceSystem:
    def __init__(self, safe_distance=0.45, stop_distance=0.15):
        # d_th2: 警戒距离 (开始规划避障)
        self.d_th2 = safe_distance 
        # d_th1: 最小安全距离
        self.d_th1 = stop_distance
        
        # 障碍物速度信息（由预测系统提供）
        self.obstacle_velocity = np.array([0.0, 0.0, 0.0])
        self.obstacle_is_moving = False
        self.obstacle_direction = None  # 'approaching', 'leaving', 'stationary'
        
        # ==========================================
        # 【新增】障碍物高度信息（由侧视摄像头提供）
        # ==========================================
        self.obstacle_height_info = {
            "max_height": 0.0,          # 障碍物最高点Z坐标
            "clearance_height": 0.15,   # 建议的安全越过高度（默认15cm）
            "confidence": 0.0           # 测量置信度
        }
        
    def set_obstacle_height_info(self, height_info):
        """
        设置障碍物高度信息（由侧视摄像头调用）
        
        Args:
            height_info: 包含 max_height, clearance_height, confidence 的字典
        """
        if height_info:
            self.obstacle_height_info.update(height_info)
        
    def set_obstacle_motion(self, velocity, is_moving, direction):
        """
        设置障碍物运动信息（由预测系统调用）
        
        Args:
            velocity: 障碍物速度向量 [vx, vy, vz]
            is_moving: 是否在运动
            direction: 运动方向 'approaching'/'leaving'/'stationary'
        """
        self.obstacle_velocity = np.array(velocity) if velocity else np.array([0, 0, 0])
        self.obstacle_is_moving = is_moving
        self.obstacle_direction = direction

    def compute_modified_step(self, current_pos, target_pos, obstacle_pos):
        
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

        # Docking Mode: 接近目标时，无视避障，强制进场
        # 【修复】增加触发距离到 0.18，避免在目标附近抽搐
        if dist_to_target < 0.18: 
             step_size = 0.04
             next_pos = curr_arr + v_c_dir * step_size
             return next_pos.tolist(), "DOCKING"
        
        # 【新增】如果障碍物已经很远（> 0.30），直接走向目标，不再避障
        if dist_to_obs > 0.30:
            step_size = 0.05
            next_pos = curr_arr + v_c_dir * step_size
            return next_pos.tolist(), "CLEAR_PATH"

        # ==========================================
        # 3. 【最高优先级】高度优势检测（使用侧视摄像头数据）
        # 使用实际测量的障碍物最高点来判断是否可以安全越过
        # ==========================================
        # 从侧视摄像头获取的安全越过高度
        clearance_height = self.obstacle_height_info.get("clearance_height", 0.15)
        obstacle_max_height = self.obstacle_height_info.get("max_height", obs_arr[2])
        height_confidence = self.obstacle_height_info.get("confidence", 0.0)
        
        # 如果侧视摄像头测量置信度低，回退到使用障碍物中心点高度
        if height_confidence < 0.3:
            # 低置信度时，使用保守估计（障碍物位置 + 10cm）
            effective_clearance = obs_arr[2] + 0.10
        else:
            effective_clearance = clearance_height
        
        # 当前高度与安全越过高度的差距
        height_above_clearance = curr_arr[2] - effective_clearance
        height_above_obs = curr_arr[2] - obs_arr[2]  # 兼容旧逻辑
        clearance_threshold = 0.02  # 只需高于安全高度2cm即可越过
        
        # 判断移动方向：主要是水平移动还是垂直移动
        vertical_diff = targ_arr[2] - curr_arr[2]  # 目标与当前的Z轴差距
        horizontal_diff = np.linalg.norm(targ_arr[:2] - curr_arr[:2])  # 水平距离
        
        # 【关键】如果已经越过安全高度，保持高度水平穿越
        # 使用侧视摄像头测量的实际安全高度进行判断
        if height_above_clearance > clearance_threshold:
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
            # 【修复】放宽下降条件：只要高于障碍物足够多，就可以开始下降
            if vertical_diff < -0.05:
                # 根据高度优势和水平距离动态决定下降速度
                if horizontal_dist_to_obs > 0.25:
                    # 远离障碍物，可以较快下降
                    descent_rate = max(vertical_diff / dist_to_target, -0.3)
                elif horizontal_dist_to_obs > 0.15 and height_above_obs > 0.12:
                    # 中等距离但高度足够，允许缓慢下降
                    descent_rate = max(vertical_diff / dist_to_target, -0.15)
                elif height_above_obs > 0.15:
                    # 即使水平距离近，只要高度优势足够大（15cm+），也可以微量下降
                    descent_rate = max(vertical_diff / dist_to_target, -0.08)
                else:
                    descent_rate = 0
                
                if descent_rate != 0:
                    move_dir[2] = descent_rate
                    move_dir = move_dir / np.linalg.norm(move_dir)
            
            step_size = 0.05
            next_pos = curr_arr + move_dir * step_size
            return next_pos.tolist(), "OVERHEAD_CROSS"

        # ==========================================
        # 4. 抬起模式检测：需要越过障碍物时自动升高
        # 【改进】使用侧视摄像头测量的安全高度来引导
        # ==========================================
        # 检测是否需要抬升：
        # 1. 当前高度低于安全越过高度
        # 2. 且障碍物在附近
        needs_lifting = height_above_clearance < 0 and dist_to_obs < self.d_th2
        is_lifting = vertical_diff > 0.05 and vertical_diff > horizontal_diff * 0.5
        
        if needs_lifting or is_lifting:
            horizontal_dist_to_obs = np.linalg.norm(curr_arr[:2] - obs_arr[:2])
            
            # 计算需要升高多少
            height_deficit = effective_clearance - curr_arr[2]  # 还差多少高度
            
            if height_deficit > 0 or horizontal_dist_to_obs < 0.25:
                step_size = 0.04
                # 构造一个以向上为主、略微朝目标的移动向量
                # 【关键】升高力度根据高度差距动态调整
                up_weight = min(height_deficit / 0.1, 1.0) if height_deficit > 0 else 0.3
                lift_dir = np.array([0, 0, up_weight])  # 主要向上
                lift_dir += v_c_dir * (1.0 - up_weight * 0.7)  # 根据升高需求调整水平分量
                if np.linalg.norm(lift_dir) > 0.01:
                    lift_dir = lift_dir / np.linalg.norm(lift_dir)
                next_pos = curr_arr + lift_dir * step_size
                return next_pos.tolist(), f"LIFTING_TO_{effective_clearance:.2f}"

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
        # 6. 动态避障核心算法（支持预测性避障）
        # ==========================================
        
        # A. 安全区域判断（考虑障碍物运动方向）
        effective_safe_dist = self.d_th2
        
        # 如果障碍物正在接近，提前进入避障模式
        if self.obstacle_is_moving and self.obstacle_direction == 'approaching':
            # 根据障碍物速度动态增加安全距离
            speed_magnitude = np.linalg.norm(self.obstacle_velocity)
            # 速度越快，安全距离越大（最多增加 30%）
            speed_bonus = min(speed_magnitude * 50, 0.15)  
            effective_safe_dist = self.d_th2 + speed_bonus
        elif self.obstacle_is_moving and self.obstacle_direction == 'leaving':
            # 障碍物远离时，可以略微激进
            effective_safe_dist = self.d_th2 * 0.85
        
        if dist_to_obs > effective_safe_dist:
            step_size = 0.05
            next_pos = curr_arr + v_c_dir * step_size
            return next_pos.tolist(), "NORMAL"

        # --- 进入避障计算 ---
        
        # 计算危险系数 (0.0 = 安全, 1.0 = 贴脸)
        # 使用有效安全距离计算风险
        risk_level = (effective_safe_dist - dist_to_obs) / (effective_safe_dist - self.d_th1)
        risk_level = np.clip(risk_level, 0.0, 1.0)
        
        # 如果障碍物正在接近，增加风险系数
        if self.obstacle_is_moving and self.obstacle_direction == 'approaching':
            risk_level = min(risk_level * 1.3, 1.0)
        
        # 分解速度向量
        v_parallel_val = np.dot(v_c_full, c_hat)
        v_parallel = v_parallel_val * c_hat
        v_perp = v_c_full - v_parallel
        
        # 死锁检测
        perp_magnitude = np.linalg.norm(v_perp)
        is_deadlock = perp_magnitude < 0.1
        
        # 向上跨越的向量
        up_vector = np.array([0.0, 0.0, 1.0])
        
        # 【关键修改】根据实际安全高度动态调整向上的力度
        # 使用侧视摄像头测量的 clearance_height 来精确控制
        if height_above_clearance > 0.05:
            up_scale = 0.0  # 已经高于安全高度5cm+，不再向上
        elif height_above_clearance > 0.02:
            up_scale = 0.1  # 接近安全高度（2-5cm），微弱向上
        elif height_above_clearance > 0:
            up_scale = 0.3  # 刚刚达到安全高度，适度向上
        else:
            # 还没达到安全高度，根据差距决定向上力度
            height_deficit = -height_above_clearance  # 还差多少
            up_scale = min(height_deficit / 0.05, 1.0)  # 差距越大，向上越强
        
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
        
        # 【预测性避障】考虑障碍物运动方向，选择更智能的逃逸方向
        if self.obstacle_is_moving and np.linalg.norm(self.obstacle_velocity) > 0.0003:
            obs_vel_horizontal = np.array([self.obstacle_velocity[0], self.obstacle_velocity[1], 0])
            obs_vel_norm = np.linalg.norm(obs_vel_horizontal)
            
            if obs_vel_norm > 0.0001:
                obs_vel_dir = obs_vel_horizontal / obs_vel_norm
                
                # 计算与障碍物运动方向垂直的两个逃逸方向
                escape_left = np.array([-obs_vel_dir[1], obs_vel_dir[0], 0])
                escape_right = np.array([obs_vel_dir[1], -obs_vel_dir[0], 0])
                
                # 选择更接近目标的逃逸方向
                target_horizontal = np.array([v_c_dir[0], v_c_dir[1], 0])
                if np.dot(escape_left, target_horizontal) > np.dot(escape_right, target_horizontal):
                    preferred_escape = escape_left
                else:
                    preferred_escape = escape_right
                
                # 将逃逸方向混合到斥力中
                escape_weight = min(obs_vel_norm * 200, 0.5)  # 速度越快，逃逸方向权重越大
                repulsion_force = repulsion_force * (1 - escape_weight) + preferred_escape * escape_weight * risk_level * 1.5
        
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
