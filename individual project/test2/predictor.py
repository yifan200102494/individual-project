import numpy as np
from collections import deque

class ObstaclePredictor:
    
    def __init__(self, history_size=15, prediction_horizon=0.5):
        self.history_size = history_size
        self.prediction_horizon = prediction_horizon
        self.position_history = deque(maxlen=history_size)
        
        self.state = None
        self.covariance = None
        self.initialized = False
        self.dt = 1.0 / 240.0
        
        self.velocity_estimate = np.zeros(3)
        self.speed_magnitude = 0.0
        self.is_moving = False
        self.movement_direction = None
        
        self._init_kalman_params()
        
    def _init_kalman_params(self):
       
        dt = self.dt
        self.F = np.array([[1,0,0,dt,0,0], [0,1,0,0,dt,0], [0,0,1,0,0,dt],
                           [0,0,0,1,0,0], [0,0,0,0,1,0], [0,0,0,0,0,1]])
        self.H = np.array([[1,0,0,0,0,0], [0,1,0,0,0,0], [0,0,1,0,0,0]])
        self.Q = np.diag([0.001, 0.001, 0.001, 0.01, 0.01, 0.01])
        self.R = np.diag([0.01, 0.01, 0.01])
        
    def update(self, observed_position, timestamp=None):
        
        if observed_position is None:
            return
        obs = np.array(observed_position)
        self.position_history.append(obs.copy())
        
        if not self.initialized:
            self.state = np.concatenate([obs, [0, 0, 0]])
            self.covariance = np.eye(6) * 0.1
            self.initialized = True
            return
        
        # 卡尔曼滤波
        state_pred = self.F @ self.state
        cov_pred = self.F @ self.covariance @ self.F.T + self.Q
        
        y = obs - self.H @ state_pred
        S = self.H @ cov_pred @ self.H.T + self.R
        K = cov_pred @ self.H.T @ np.linalg.inv(S)
        
        self.state = state_pred + K @ y
        self.covariance = (np.eye(6) - K @ self.H) @ cov_pred
        
        # 更新速度
        self.velocity_estimate = self.state[3:6]
        self.speed_magnitude = np.linalg.norm(self.velocity_estimate)
        self.is_moving = self.speed_magnitude > 0.0005
        
        vx = self.velocity_estimate[0]
        self.movement_direction = 'approaching' if vx < -0.0003 else ('leaving' if vx > 0.0003 else 'stationary')
            
    def predict_position(self, steps_ahead=None):
        
        if not self.initialized:
            return None, 0.0
        if steps_ahead is None:
            steps_ahead = int(self.prediction_horizon / self.dt)
        
        pred = self.state[:3] + self.state[3:6] * steps_ahead * self.dt
        
        # 置信度计算
        data_conf = min(len(self.position_history) / self.history_size, 1.0)
        time_decay = np.exp(-steps_ahead * self.dt / self.prediction_horizon)
        
        if len(self.position_history) >= 3:
            vels = np.diff(np.array(list(self.position_history)), axis=0)
            stability = 1.0 / (1.0 + np.sum(np.std(vels, axis=0)) * 100) if len(vels) > 1 else 0.5
        else:
            stability = 0.3
            
        return pred.tolist(), data_conf * time_decay * stability
        
    def get_avoidance_position(self, current_robot_pos, lead_time_factor=1.5):
        
        if not self.initialized:
            return None, {"status": "not_initialized"}
            
        pos = self.state[:3]
        info = {"current_pos": pos.tolist(), "velocity": self.velocity_estimate.tolist(),
                "speed": self.speed_magnitude, "is_moving": self.is_moving, "direction": self.movement_direction}
        
        if not self.is_moving:
            info.update({"status": "stationary", "predicted_pos": pos.tolist()})
            return pos.tolist(), info
        
        # 动态预测步数
        base = int(self.prediction_horizon / self.dt)
        mult = {"approaching": 1.5, "leaving": 0.5}.get(self.movement_direction, 1.0)
        steps = int(base * lead_time_factor * mult)
        info["strategy"] = {"approaching": "aggressive", "leaving": "relaxed"}.get(self.movement_direction, "normal")
        
        pred, conf = self.predict_position(steps)
        info.update({"predicted_pos": pred, "confidence": conf, "status": "predicted"})
        
        alpha = conf if conf < 0.3 else min(conf * 1.2, 0.9)
        eff = [pos[i] * (1 - alpha) + pred[i] * alpha for i in range(3)]
        info.update({"effective_pos": eff, "blend_alpha": alpha})
        
        return eff, info
        
    def should_preemptive_avoid(self, robot_pos, robot_target, safety_margin=0.15):
        
        if not self.initialized or not self.is_moving:
            return False, 0.0, "proceed_normal"
            
        r_pos, r_tgt = np.array(robot_pos), np.array(robot_target)
        r_dir = r_tgt - r_pos
        r_dist = np.linalg.norm(r_dir)
        if r_dist < 0.01:
            return False, 0.0, "at_target"
        r_dir = r_dir / r_dist
        
        max_threat = 0.0
        for t in range(10, 150, 10):
            pred, conf = self.predict_position(t)
            if not pred:
                continue
            r_future = r_pos + r_dir * min(0.05 * t, r_dist)
            dist = np.linalg.norm(r_future - np.array(pred))
            if dist < safety_margin:
                threat = (safety_margin - dist) / safety_margin * conf
                max_threat = max(max_threat, threat)
        
        if max_threat > 0.7: return True, max_threat, "emergency_avoid"
        if max_threat > 0.4: return True, max_threat, "preemptive_avoid"
        if max_threat > 0.2: return True, max_threat, "cautious_proceed"
        return False, max_threat, "proceed_normal"
            
    def get_motion_trend(self):
        
        if not self.initialized:
            return {"status": "no_data", "is_moving": False}
        return {"status": "tracking", "is_moving": self.is_moving, "direction": self.movement_direction,
                "velocity": self.velocity_estimate.tolist(), "speed": self.speed_magnitude,
                "position": self.state[:3].tolist()}
        
    def reset(self):
        
        self.position_history.clear()
        self.state = self.covariance = None
        self.initialized = self.is_moving = False
        self.velocity_estimate = np.zeros(3)
        self.speed_magnitude = 0.0
        self.movement_direction = None