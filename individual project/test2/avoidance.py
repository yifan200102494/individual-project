import numpy as np

class VisualAvoidanceSystem:
    def __init__(self, safe_distance=0.45, stop_distance=0.15):
        self.d_th2 = safe_distance  # 警戒距离
        self.d_th1 = stop_distance  # 最小安全距离
        
        self.obstacle_velocity = np.array([0.0, 0.0, 0.0])
        self.obstacle_is_moving = False
        self.obstacle_direction = None
        self.obstacle_height_info = {"max_height": 0.0, "clearance_height": 0.15, "confidence": 0.0}
        
    def set_obstacle_height_info(self, info):
        if info: self.obstacle_height_info.update(info)
        
    def set_obstacle_motion(self, velocity, is_moving, direction):
        self.obstacle_velocity = np.array(velocity) if velocity else np.array([0, 0, 0])
        self.obstacle_is_moving, self.obstacle_direction = is_moving, direction

    def _get_clearance_height(self, obs_z):
        """获取有效安全高度"""
        if self.obstacle_height_info.get("confidence", 0) < 0.3:
            return obs_z + 0.10
        return self.obstacle_height_info.get("clearance_height", 0.15)
    
    def _compute_overhead_move(self, curr, targ, obs, v_dir, h_above_obs, vert_diff, dist):
        """计算高空越障移动方向"""
        h_dist = np.linalg.norm(curr[:2] - obs[:2])
        move_dir = np.array([v_dir[0], v_dir[1], 0])
        
        if np.linalg.norm(move_dir) < 0.01:
            return (curr + v_dir * 0.05).tolist(), "OVERHEAD_VERTICAL"
        move_dir = move_dir / np.linalg.norm(move_dir)
        
        # 近距绕行
        if h_dist < 0.15:
            to_obs = obs[:2] - curr[:2]
            if np.linalg.norm(to_obs) > 0.01:
                to_obs = to_obs / np.linalg.norm(to_obs)
                bl, br = np.array([-to_obs[1], to_obs[0], 0]), np.array([to_obs[1], -to_obs[0], 0])
                bypass = bl if np.dot(bl[:2], v_dir[:2]) > np.dot(br[:2], v_dir[:2]) else br
                move_dir = (bypass * 0.5 + move_dir * 0.5)
                move_dir = move_dir / np.linalg.norm(move_dir)
        
        # 下降判断
        if vert_diff < -0.05:
            if h_dist > 0.25: rate = max(vert_diff / dist, -0.3)
            elif h_dist > 0.15 and h_above_obs > 0.12: rate = max(vert_diff / dist, -0.15)
            elif h_above_obs > 0.15: rate = max(vert_diff / dist, -0.08)
            else: rate = 0
            if rate != 0:
                move_dir[2] = rate
                move_dir = move_dir / np.linalg.norm(move_dir)
        
        return (curr + move_dir * 0.05).tolist(), "OVERHEAD_CROSS"
    
    def _compute_escape_direction(self, v_dir, risk):
        
        if not self.obstacle_is_moving or np.linalg.norm(self.obstacle_velocity) < 0.0003:
            return np.array([0, 0, 0])
        
        obs_vel_h = np.array([self.obstacle_velocity[0], self.obstacle_velocity[1], 0])
        v_norm = np.linalg.norm(obs_vel_h)
        if v_norm < 0.0001:
            return np.array([0, 0, 0])
        
        obs_dir = obs_vel_h / v_norm
        el, er = np.array([-obs_dir[1], obs_dir[0], 0]), np.array([obs_dir[1], -obs_dir[0], 0])
        target_h = np.array([v_dir[0], v_dir[1], 0])
        escape = el if np.dot(el, target_h) > np.dot(er, target_h) else er
        return escape * min(v_norm * 200, 0.5) * risk * 1.5

    def compute_modified_step(self, current_pos, target_pos, obstacle_pos):
        curr, targ, obs = np.array(current_pos), np.array(target_pos), np.array(obstacle_pos)
        
        v_full = targ - curr
        dist_targ = np.linalg.norm(v_full)
        c_vec = obs - curr
        dist_obs = np.linalg.norm(c_vec)
        
        if dist_targ < 0.01: return current_pos, "ARRIVED"
        v_dir = v_full / dist_targ
        c_hat = c_vec / dist_obs if dist_obs > 0.001 else np.array([1, 0, 0])
        
        # 基础状态检测
        if dist_targ < 0.02: return current_pos, "ARRIVED"
        if dist_targ < 0.18: return (curr + v_dir * 0.04).tolist(), "DOCKING"
        if dist_obs > 0.30: return (curr + v_dir * 0.05).tolist(), "CLEAR_PATH"
        
        # 高度计算
        eff_clear = self._get_clearance_height(obs[2])
        h_above_clear = curr[2] - eff_clear
        h_above_obs = curr[2] - obs[2]
        vert_diff = targ[2] - curr[2]
        horiz_diff = np.linalg.norm(targ[:2] - curr[:2])
        
        # 高空越障
        if h_above_clear > 0.02:
            return self._compute_overhead_move(curr, targ, obs, v_dir, h_above_obs, vert_diff, dist_targ)
        
        # 抬升模式
        if (h_above_clear < 0 and dist_obs < self.d_th2) or (vert_diff > 0.05 and vert_diff > horiz_diff * 0.5):
            h_dist = np.linalg.norm(curr[:2] - obs[:2])
            deficit = eff_clear - curr[2]
            if deficit > 0 or h_dist < 0.25:
                up_w = min(deficit / 0.1, 1.0) if deficit > 0 else 0.3
                lift = np.array([0, 0, up_w]) + v_dir * (1.0 - up_w * 0.7)
                if np.linalg.norm(lift) > 0.01: lift = lift / np.linalg.norm(lift)
                return (curr + lift * 0.04).tolist(), f"LIFTING_TO_{eff_clear:.2f}"
        
        # 远距运输
        if 0 < h_above_obs <= 0.02 and np.linalg.norm(curr[:2] - obs[:2]) > 0.25:
            m = np.array([v_dir[0], v_dir[1], 0])
            if np.linalg.norm(m) > 0.01:
                return (curr + m / np.linalg.norm(m) * 0.05).tolist(), "TRANSPORT_FAR"
        
        # 离开检测
        if np.dot(v_dir, c_hat) < -0.1:
            return (curr + v_dir * 0.05).tolist(), "LEAVING"
        
        # 动态安全距离
        eff_safe = self.d_th2
        if self.obstacle_is_moving and self.obstacle_direction == 'approaching':
            eff_safe += min(np.linalg.norm(self.obstacle_velocity) * 50, 0.15)
        elif self.obstacle_is_moving and self.obstacle_direction == 'leaving':
            eff_safe *= 0.85
        
        if dist_obs > eff_safe:
            return (curr + v_dir * 0.05).tolist(), "NORMAL"
        
        # 避障计算
        risk = np.clip((eff_safe - dist_obs) / (eff_safe - self.d_th1), 0, 1)
        if self.obstacle_is_moving and self.obstacle_direction == 'approaching':
            risk = min(risk * 1.3, 1.0)
        
        v_perp = v_full - np.dot(v_full, c_hat) * c_hat
        
        # 向上力度
        if h_above_clear > 0.05: up_s = 0.0
        elif h_above_clear > 0.02: up_s = 0.1
        elif h_above_clear > 0: up_s = 0.3
        else: up_s = min(-h_above_clear / 0.05, 1.0)
        
        up = np.array([0.0, 0.0, 1.0])
        if np.linalg.norm(v_perp) < 0.1:
            v_perp = up * 2.0 * up_s
        else:
            v_perp += up * risk * 2.0 * up_s
        
        # 水平斥力
        c_h = np.array([c_hat[0], c_hat[1], 0])
        c_h = c_h / np.linalg.norm(c_h) if np.linalg.norm(c_h) > 0.01 else np.array([1, 0, 0])
        repel = -risk * c_h * 1.5 + self._compute_escape_direction(v_dir, risk)
        if v_perp[2] > 0.5: repel *= 0.3
        
        v_mod = v_perp + repel
        if vert_diff > 0.05 and v_mod[2] < 0.3 and h_above_obs < 0:
            v_mod[2] = max(v_mod[2], 0.5)
        
        if np.linalg.norm(v_mod) > 0:
            move = v_mod / np.linalg.norm(v_mod)
            return (curr + move * (0.03 if risk > 0.5 else 0.05)).tolist(), "AVOIDING"
        return (curr + np.array([0, 0, 0.01])).tolist(), "STUCK_RECOVERY"
