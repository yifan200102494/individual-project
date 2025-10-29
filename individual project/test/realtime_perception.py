"""
实时感知模块
实现连续的、自适应的障碍物感知和运动预测
"""

import pybullet as p
import numpy as np
from collections import deque


class AdaptivePerceptionSystem:
    """
    自适应感知系统
    - 连续感知障碍物
    - 跟踪障碍物运动
    - 预测障碍物未来位置
    """
    
    def __init__(self, robot_id, sensor_link_id, history_size=10):
        """
        初始化感知系统
        
        Args:
            robot_id: 机器人ID
            sensor_link_id: 传感器连杆ID
            history_size: 历史记录大小
        """
        self.robot_id = robot_id
        self.sensor_link_id = sensor_link_id
        self.history_size = history_size
        
        # 障碍物跟踪历史
        self.obstacle_history = {}  # {obs_id: deque of positions}
        self.obstacle_velocities = {}  # {obs_id: velocity vector}
        
        # 自适应感知参数
        self.base_ray_range = 1.5
        self.base_grid_size = 5  # 减小网格大小以提高效率
        self.focused_ray_range = 0.8  # 近距离精细扫描
        self.focused_grid_size = 7
        
    def perceive_with_prediction(self, ignore_ids=None, debug=False):
        """
        感知障碍物并预测其未来位置
        
        Args:
            ignore_ids: 要忽略的物体ID列表
            debug: 是否显示调试信息
            
        Returns:
            dict: {
                'current_obstacles': [(obs_id, position, velocity)],
                'predicted_obstacles': [(obs_id, predicted_position, confidence)]
            }
        """
        if ignore_ids is None:
            ignore_ids = set()
        else:
            ignore_ids = set(ignore_ids)
        
        # 1. 快速全局扫描（粗略）
        global_obstacles = self._global_scan(ignore_ids, debug=debug)
        
        # 2. 对检测到的障碍物进行精细扫描
        refined_obstacles = []
        for obs_id, approx_pos in global_obstacles.items():
            if obs_id not in ignore_ids and obs_id > 0:  # 忽略地面和无效ID
                refined_pos = self._focused_scan(obs_id, approx_pos, debug=debug)
                refined_obstacles.append((obs_id, refined_pos))
        
        # 3. 更新历史和计算速度
        current_obstacles = []
        for obs_id, position in refined_obstacles:
            velocity = self._update_obstacle_tracking(obs_id, position)
            current_obstacles.append((obs_id, position, velocity))
        
        # 4. 预测未来位置
        predicted_obstacles = self._predict_future_positions(current_obstacles)
        
        if debug:
            self._visualize_predictions(predicted_obstacles)
        
        return {
            'current_obstacles': current_obstacles,
            'predicted_obstacles': predicted_obstacles
        }
    
    def _global_scan(self, ignore_ids, debug=False):
        """
        快速全局扫描（使用稀疏射线）
        
        Returns:
            dict: {obs_id: approximate_position}
        """
        try:
            link_state = p.getLinkState(self.robot_id, self.sensor_link_id, 
                                        computeForwardKinematics=True)
        except Exception as e:
            print(f"  [感知错误] 无法获取 link state: {e}")
            return {}
        
        sensor_pos_world = np.array(link_state[0])
        sensor_orn_world = np.array(link_state[1])
        sensor_rot_matrix = np.array(p.getMatrixFromQuaternion(sensor_orn_world)).reshape(3, 3)
        
        # 使用较少的方向进行快速扫描
        fast_directions = [
            (2, 1.0),   # 向下
            (0, 1.0),   # 向前
            (1, 1.0),   # 向左
            (1, -1.0),  # 向右
        ]
        
        ray_froms, ray_tos = self._generate_rays(
            sensor_pos_world, sensor_rot_matrix, fast_directions,
            self.base_ray_range, self.base_grid_size, 0.6
        )
        
        results = p.rayTestBatch(ray_froms, ray_tos)
        
        # 收集检测到的障碍物及其大致位置
        obstacle_hits = {}
        for res in results:
            hit_id = res[0]
            if hit_id not in ignore_ids and hit_id > 0:
                hit_pos = np.array(res[3])
                if hit_id not in obstacle_hits:
                    obstacle_hits[hit_id] = []
                obstacle_hits[hit_id].append(hit_pos)
        
        # 计算每个障碍物的平均位置
        obstacle_positions = {}
        for obs_id, positions in obstacle_hits.items():
            obstacle_positions[obs_id] = np.mean(positions, axis=0)
        
        if debug:
            for i, res in enumerate(results):
                hit_id = res[0]
                from_pos = ray_froms[i]
                to_pos = ray_tos[i]
                if hit_id == -1:
                    p.addUserDebugLine(from_pos, to_pos, [0.0, 0.5, 0.0], 
                                     lifeTime=0.1, lineWidth=1)
                elif hit_id not in ignore_ids:
                    hit_pos = res[3]
                    p.addUserDebugLine(from_pos, hit_pos, [1.0, 0.0, 0.0], 
                                     lifeTime=0.1, lineWidth=2)
        
        return obstacle_positions
    
    def _focused_scan(self, obs_id, approx_pos, debug=False):
        """
        对特定障碍物进行精细扫描
        
        Args:
            obs_id: 障碍物ID
            approx_pos: 大致位置
            
        Returns:
            np.array: 精确位置
        """
        try:
            # 使用AABB获取精确位置
            aabb_min, aabb_max = p.getAABB(obs_id)
            center = np.array([
                (aabb_min[0] + aabb_max[0]) / 2,
                (aabb_min[1] + aabb_max[1]) / 2,
                (aabb_min[2] + aabb_max[2]) / 2
            ])
            return center
        except:
            return approx_pos
    
    def _update_obstacle_tracking(self, obs_id, position):
        """
        更新障碍物跟踪历史并计算速度
        
        Args:
            obs_id: 障碍物ID
            position: 当前位置
            
        Returns:
            np.array: 速度向量
        """
        if obs_id not in self.obstacle_history:
            self.obstacle_history[obs_id] = deque(maxlen=self.history_size)
            self.obstacle_velocities[obs_id] = np.array([0.0, 0.0, 0.0])
        
        # 添加当前位置到历史
        self.obstacle_history[obs_id].append(position.copy())
        
        # 计算速度（如果有足够的历史数据）
        if len(self.obstacle_history[obs_id]) >= 2:
            recent_positions = list(self.obstacle_history[obs_id])
            
            # 使用最近的几个位置计算平均速度
            velocities = []
            for i in range(1, min(5, len(recent_positions))):
                dt = 0.1  # 假设每次更新间隔约0.1秒
                velocity = (recent_positions[-1] - recent_positions[-i]) / (dt * i)
                velocities.append(velocity)
            
            if velocities:
                avg_velocity = np.mean(velocities, axis=0)
                # 平滑速度估计
                self.obstacle_velocities[obs_id] = 0.7 * self.obstacle_velocities[obs_id] + 0.3 * avg_velocity
        
        return self.obstacle_velocities[obs_id]
    
    def _predict_future_positions(self, current_obstacles, prediction_horizon=0.5):
        """
        预测障碍物未来位置
        
        Args:
            current_obstacles: [(obs_id, position, velocity)]
            prediction_horizon: 预测时间范围（秒）
            
        Returns:
            list: [(obs_id, predicted_position, confidence)]
        """
        predicted = []
        
        for obs_id, position, velocity in current_obstacles:
            # 线性预测
            predicted_pos = position + velocity * prediction_horizon
            
            # 计算置信度（基于速度的稳定性）
            if obs_id in self.obstacle_history and len(self.obstacle_history[obs_id]) >= 3:
                # 速度变化小 -> 高置信度
                velocity_magnitude = np.linalg.norm(velocity)
                confidence = min(1.0, velocity_magnitude / 0.5) if velocity_magnitude > 0.01 else 0.0
            else:
                confidence = 0.0
            
            predicted.append((obs_id, predicted_pos, confidence))
        
        return predicted
    
    def _visualize_predictions(self, predicted_obstacles):
        """可视化预测结果"""
        for obs_id, pred_pos, confidence in predicted_obstacles:
            if confidence > 0.3:
                # 用黄色显示预测位置
                p.addUserDebugLine(
                    pred_pos - [0, 0, 0.1], 
                    pred_pos + [0, 0, 0.1],
                    [1.0, 1.0, 0.0], 
                    lifeTime=0.2, 
                    lineWidth=3
                )
    
    def _generate_rays(self, sensor_pos, rot_matrix, directions, 
                      ray_range, grid_size, fov_width):
        """生成射线（简化版）"""
        ray_froms_world = []
        ray_tos_world = []
        
        grid_coords = np.linspace(-fov_width, fov_width, grid_size)
        start_offset = 0.01
        
        for sensor_dir in directions:
            axis_idx, direction = sensor_dir
            grid_axis_1 = (axis_idx + 1) % 3
            grid_axis_2 = (axis_idx + 2) % 3
            
            for u_grid in grid_coords:
                for v_grid in grid_coords:
                    ray_from_local = np.array([0.0, 0.0, 0.0])
                    ray_from_local[axis_idx] = direction * start_offset
                    
                    ray_to_local = np.array([0.0, 0.0, 0.0])
                    ray_to_local[axis_idx] = direction * ray_range
                    ray_to_local[grid_axis_1] = u_grid
                    ray_to_local[grid_axis_2] = v_grid
                    
                    ray_from_world = sensor_pos + rot_matrix.dot(ray_from_local)
                    ray_to_world = sensor_pos + rot_matrix.dot(ray_to_local)
                    
                    ray_froms_world.append(ray_from_world)
                    ray_tos_world.append(ray_to_world)
        
        return ray_froms_world, ray_tos_world
    
    def get_dynamic_safety_margin(self, obs_id):
        """
        根据障碍物速度动态调整安全边距
        
        Args:
            obs_id: 障碍物ID
            
        Returns:
            float: 安全边距
        """
        base_margin = 0.15
        
        if obs_id in self.obstacle_velocities:
            velocity_magnitude = np.linalg.norm(self.obstacle_velocities[obs_id])
            # 速度越快，需要越大的安全边距
            dynamic_margin = base_margin + velocity_magnitude * 0.3
            return min(dynamic_margin, 0.4)  # 上限0.4米
        
        return base_margin

