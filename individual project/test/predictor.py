import numpy as np
from collections import deque

class ObstaclePredictor:
    """
    障碍物运动预测器
    通过记录历史轨迹，使用线性预测/卡尔曼滤波预测障碍物未来位置
    """
    
    def __init__(self, history_size=15, prediction_horizon=0.5):
        """
        Args:
            history_size: 保存的历史位置数量
            prediction_horizon: 预测时间窗口（秒），越大预测越远
        """
        self.history_size = history_size
        self.prediction_horizon = prediction_horizon
        
        # 历史轨迹队列 [(timestamp, position), ...]
        self.position_history = deque(maxlen=history_size)
        
        # 卡尔曼滤波器状态
        # 状态向量: [x, y, z, vx, vy, vz] (位置 + 速度)
        self.state = None
        self.covariance = None
        self.initialized = False
        
        # 时间步长 (假设 240Hz 仿真)
        self.dt = 1.0 / 240.0
        
        # 卡尔曼滤波器参数
        self._init_kalman_params()
        
        # 速度缓存（用于判断运动趋势）
        self.velocity_estimate = np.array([0.0, 0.0, 0.0])
        self.speed_magnitude = 0.0
        self.is_moving = False
        self.movement_direction = None  # 'approaching', 'leaving', 'stationary'
        
    def _init_kalman_params(self):
        """初始化卡尔曼滤波器参数"""
        # 状态转移矩阵 (位置 = 位置 + 速度 * dt)
        self.F = np.array([
            [1, 0, 0, self.dt, 0, 0],
            [0, 1, 0, 0, self.dt, 0],
            [0, 0, 1, 0, 0, self.dt],
            [0, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0, 1]
        ])
        
        # 观测矩阵 (只观测位置)
        self.H = np.array([
            [1, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0],
            [0, 0, 1, 0, 0, 0]
        ])
        
        # 过程噪声协方差 (较小，假设障碍物运动平滑)
        q = 0.001
        self.Q = np.array([
            [q, 0, 0, 0, 0, 0],
            [0, q, 0, 0, 0, 0],
            [0, 0, q, 0, 0, 0],
            [0, 0, 0, q*10, 0, 0],
            [0, 0, 0, 0, q*10, 0],
            [0, 0, 0, 0, 0, q*10]
        ])
        
        # 观测噪声协方差 (取决于传感器精度)
        r = 0.01
        self.R = np.array([
            [r, 0, 0],
            [0, r, 0],
            [0, 0, r]
        ])
        
    def update(self, observed_position, timestamp=None):
        """
        更新预测器状态
        
        Args:
            observed_position: 观测到的障碍物位置 [x, y, z]
            timestamp: 时间戳（可选）
        """
        if observed_position is None:
            return
            
        obs = np.array(observed_position)
        
        # 记录历史
        self.position_history.append(obs.copy())
        
        # 初始化卡尔曼滤波器
        if not self.initialized:
            self.state = np.array([obs[0], obs[1], obs[2], 0, 0, 0])
            self.covariance = np.eye(6) * 0.1
            self.initialized = True
            return
        
        # === 卡尔曼滤波预测步骤 ===
        # 预测
        state_pred = self.F @ self.state
        cov_pred = self.F @ self.covariance @ self.F.T + self.Q
        
        # 更新
        y = obs - self.H @ state_pred  # 测量残差
        S = self.H @ cov_pred @ self.H.T + self.R  # 残差协方差
        K = cov_pred @ self.H.T @ np.linalg.inv(S)  # 卡尔曼增益
        
        self.state = state_pred + K @ y
        self.covariance = (np.eye(6) - K @ self.H) @ cov_pred
        
        # 更新速度估计
        self.velocity_estimate = self.state[3:6]
        self.speed_magnitude = np.linalg.norm(self.velocity_estimate)
        self.is_moving = self.speed_magnitude > 0.0005  # 速度阈值
        
        # 判断运动方向（相对于原点/工作区中心）
        self._analyze_movement_direction()
        
    def _analyze_movement_direction(self):
        """分析障碍物运动方向"""
        if not self.is_moving:
            self.movement_direction = 'stationary'
            return
            
        # X轴速度判断（正=远离机械臂，负=接近机械臂）
        vx = self.velocity_estimate[0]
        
        if vx < -0.0003:  # 障碍物正在接近工作区
            self.movement_direction = 'approaching'
        elif vx > 0.0003:  # 障碍物正在远离
            self.movement_direction = 'leaving'
        else:
            self.movement_direction = 'stationary'
            
    def predict_position(self, steps_ahead=None):
        """
        预测障碍物未来位置
        
        Args:
            steps_ahead: 预测多少步之后的位置，None则使用默认prediction_horizon
            
        Returns:
            predicted_position: 预测的未来位置 [x, y, z]
            confidence: 预测置信度 (0-1)
        """
        if not self.initialized:
            return None, 0.0
            
        if steps_ahead is None:
            steps_ahead = int(self.prediction_horizon / self.dt)
        
        # 使用当前状态进行线性外推
        current_pos = self.state[:3]
        velocity = self.state[3:6]
        
        # 预测位置 = 当前位置 + 速度 * 时间
        predicted_pos = current_pos + velocity * steps_ahead * self.dt
        
        # 计算置信度（基于速度稳定性和历史数据量）
        confidence = self._compute_confidence(steps_ahead)
        
        return predicted_pos.tolist(), confidence
        
    def _compute_confidence(self, steps_ahead):
        """计算预测置信度"""
        # 基础置信度取决于历史数据量
        data_confidence = min(len(self.position_history) / self.history_size, 1.0)
        
        # 预测越远置信度越低
        time_decay = np.exp(-steps_ahead * self.dt / self.prediction_horizon)
        
        # 速度越稳定置信度越高
        if len(self.position_history) >= 3:
            positions = np.array(list(self.position_history))
            velocities = np.diff(positions, axis=0)
            if len(velocities) > 1:
                velocity_std = np.std(velocities, axis=0)
                stability = 1.0 / (1.0 + np.sum(velocity_std) * 100)
            else:
                stability = 0.5
        else:
            stability = 0.3
            
        return data_confidence * time_decay * stability
        
    def get_avoidance_position(self, current_robot_pos, lead_time_factor=1.5):
        """
        获取用于避障计算的有效障碍物位置
        这是核心方法：返回一个"预测+膨胀"的虚拟障碍物位置
        
        Args:
            current_robot_pos: 机器人当前位置
            lead_time_factor: 预判时间因子，越大越保守
            
        Returns:
            effective_obstacle_pos: 用于避障的有效位置
            prediction_info: 预测详情字典
        """
        if not self.initialized:
            return None, {"status": "not_initialized"}
            
        current_pos = self.state[:3]
        
        info = {
            "current_pos": current_pos.tolist(),
            "velocity": self.velocity_estimate.tolist(),
            "speed": self.speed_magnitude,
            "is_moving": self.is_moving,
            "direction": self.movement_direction
        }
        
        # 如果障碍物静止，直接返回当前位置
        if not self.is_moving:
            info["status"] = "stationary"
            info["predicted_pos"] = current_pos.tolist()
            return current_pos.tolist(), info
            
        # 动态调整预测时间
        # 障碍物越快，预测越远；障碍物接近时预测更远
        base_steps = int(self.prediction_horizon / self.dt)
        
        if self.movement_direction == 'approaching':
            # 障碍物接近时，增加预测距离（更保守）
            prediction_steps = int(base_steps * lead_time_factor * 1.5)
            info["strategy"] = "aggressive_prediction"
        elif self.movement_direction == 'leaving':
            # 障碍物远离时，减少预测（不需要过度避让）
            prediction_steps = int(base_steps * lead_time_factor * 0.5)
            info["strategy"] = "relaxed_prediction"
        else:
            prediction_steps = int(base_steps * lead_time_factor)
            info["strategy"] = "normal_prediction"
            
        predicted_pos, confidence = self.predict_position(prediction_steps)
        
        info["predicted_pos"] = predicted_pos
        info["confidence"] = confidence
        info["prediction_steps"] = prediction_steps
        info["status"] = "predicted"
        
        # 根据置信度混合当前位置和预测位置
        # 低置信度时更依赖当前位置
        if confidence < 0.3:
            # 置信度太低，用当前位置+小偏移
            alpha = confidence
        else:
            alpha = min(confidence * 1.2, 0.9)  # 最多90%依赖预测
            
        effective_pos = [
            current_pos[0] * (1 - alpha) + predicted_pos[0] * alpha,
            current_pos[1] * (1 - alpha) + predicted_pos[1] * alpha,
            current_pos[2] * (1 - alpha) + predicted_pos[2] * alpha
        ]
        
        info["effective_pos"] = effective_pos
        info["blend_alpha"] = alpha
        
        return effective_pos, info
        
    def should_preemptive_avoid(self, robot_pos, robot_target, safety_margin=0.15):
        """
        判断是否需要提前避障
        
        Args:
            robot_pos: 机器人当前位置
            robot_target: 机器人目标位置  
            safety_margin: 安全裕度
            
        Returns:
            should_avoid: 是否需要提前避障
            threat_level: 威胁等级 (0-1)
            recommendation: 建议动作
        """
        if not self.initialized or not self.is_moving:
            return False, 0.0, "proceed_normal"
            
        robot_pos = np.array(robot_pos)
        robot_target = np.array(robot_target)
        
        # 预测未来轨迹
        future_positions = []
        for t in range(10, 150, 10):  # 预测未来多个时间点
            pred_pos, conf = self.predict_position(t)
            if pred_pos:
                future_positions.append((t, np.array(pred_pos), conf))
                
        if not future_positions:
            return False, 0.0, "proceed_normal"
            
        # 计算机器人预计路径
        robot_direction = robot_target - robot_pos
        robot_dist = np.linalg.norm(robot_direction)
        if robot_dist < 0.01:
            return False, 0.0, "at_target"
        robot_direction = robot_direction / robot_dist
        
        max_threat = 0.0
        critical_time = None
        
        for t, obs_pred, conf in future_positions:
            # 估计机器人在该时间点的位置
            robot_speed = 0.05  # 假设机器人速度约 5cm/step
            robot_future = robot_pos + robot_direction * min(robot_speed * t, robot_dist)
            
            # 计算届时的距离
            dist = np.linalg.norm(robot_future - obs_pred)
            
            # 威胁等级计算
            if dist < safety_margin:
                threat = (safety_margin - dist) / safety_margin * conf
                if threat > max_threat:
                    max_threat = threat
                    critical_time = t
                    
        # 决策
        if max_threat > 0.7:
            return True, max_threat, "emergency_avoid"
        elif max_threat > 0.4:
            return True, max_threat, "preemptive_avoid"
        elif max_threat > 0.2:
            return True, max_threat, "cautious_proceed"
        else:
            return False, max_threat, "proceed_normal"
            
    def get_motion_trend(self):
        """
        获取障碍物运动趋势信息
        
        Returns:
            trend_info: 包含运动趋势的详细信息
        """
        if not self.initialized:
            return {
                "status": "no_data",
                "is_moving": False
            }
            
        return {
            "status": "tracking",
            "is_moving": self.is_moving,
            "direction": self.movement_direction,
            "velocity": self.velocity_estimate.tolist(),
            "speed": self.speed_magnitude,
            "position": self.state[:3].tolist()
        }
        
    def reset(self):
        """重置预测器状态"""
        self.position_history.clear()
        self.state = None
        self.covariance = None
        self.initialized = False
        self.velocity_estimate = np.array([0.0, 0.0, 0.0])
        self.speed_magnitude = 0.0
        self.is_moving = False
        self.movement_direction = None

