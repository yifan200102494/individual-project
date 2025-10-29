"""
增量式路径规划器
实现滚动窗口式的局部路径规划（类似MPC）
"""

import pybullet as p
import numpy as np
from collections import deque

from constants import ROBOT_END_EFFECTOR_LINK_ID, DEFAULT_NULL_SPACE_PARAMS
from collision_detection import is_path_colliding


class IncrementalPlanner:
    """
    增量式规划器
    - 不规划整条路径，而是规划短期的局部路径
    - 持续更新规划以响应动态障碍物
    - 使用滚动窗口策略
    """
    
    def __init__(self, robot_id, planning_horizon=0.3, replan_rate=5):
        """
        初始化增量式规划器
        
        Args:
            robot_id: 机器人ID
            planning_horizon: 规划时间范围（秒）
            replan_rate: 重新规划的频率（每N个仿真步）
        """
        self.robot_id = robot_id
        self.planning_horizon = planning_horizon  # 短期规划范围
        self.replan_rate = replan_rate
        self.step_counter = 0
        
        # 当前规划的路径
        self.current_plan = deque()  # 路径点队列
        self.current_goal = None
        self.current_goal_orn = None
        
    def compute_local_waypoint(self, current_pos, goal_pos, perceived_obstacles, 
                               step_size=0.05):
        """
        计算下一个局部路径点（改进的势场法）
        
        特点：
        - 近距离时排斥力指数增强
        - 考虑障碍物运动方向
        - 防止局部最小值
        
        Args:
            current_pos: 当前位置
            goal_pos: 最终目标位置
            perceived_obstacles: 感知到的障碍物信息
            step_size: 步长
            
        Returns:
            np.array: 下一个路径点
        """
        current = np.array(current_pos)
        goal = np.array(goal_pos)
        
        # 1. 计算到目标的吸引力
        to_goal = goal - current
        dist_to_goal = np.linalg.norm(to_goal)
        
        if dist_to_goal < step_size:
            return goal
        
        # 归一化方向
        direction_to_goal = to_goal / dist_to_goal if dist_to_goal > 0 else np.array([0, 0, 0])
        
        # 吸引力系数（距离越近，吸引力越弱，避免冲向障碍物）
        k_att = min(1.0, dist_to_goal / 0.3)
        attractive_force = direction_to_goal * k_att
        
        # 2. 计算障碍物的排斥力（改进版）
        repulsive_force = np.array([0.0, 0.0, 0.0])
        total_danger_level = 0.0
        
        for obs_id, obs_pos, obs_velocity in perceived_obstacles:
            obs_pos = np.array(obs_pos)
            obs_velocity = np.array(obs_velocity)
            
            # 预测障碍物短期内的位置
            predicted_obs_pos = obs_pos + obs_velocity * self.planning_horizon
            
            # 计算到障碍物的距离
            to_obs = current - predicted_obs_pos
            dist_to_obs = np.linalg.norm(to_obs)
            
            if dist_to_obs < 0.01:
                # 极近距离，使用随机方向强力排斥
                random_escape = np.random.randn(3)
                random_escape[2] = abs(random_escape[2])  # 向上优先
                repulsive_force += 10.0 * (random_escape / np.linalg.norm(random_escape))
                total_danger_level += 10.0
                continue
            
            # 安全距离（根据障碍物速度动态调整）
            velocity_mag = np.linalg.norm(obs_velocity)
            safety_distance = 0.25 + velocity_mag * 0.25
            
            if dist_to_obs < safety_distance:
                # === 改进的排斥力计算 ===
                
                # 1. 基础排斥力（指数衰减）
                normalized_dist = dist_to_obs / safety_distance  # 0到1
                base_strength = 3.0 * np.exp(-normalized_dist * 3.0)  # 指数增强
                
                # 2. 考虑障碍物是否在接近
                to_obs_norm = to_obs / dist_to_obs
                approaching_factor = -np.dot(obs_velocity, to_obs_norm)  # >0表示接近
                if approaching_factor > 0:
                    base_strength *= (1.0 + approaching_factor * 2.0)  # 加倍排斥
                
                # 3. 距离越近，排斥力越强（平方反比）
                distance_factor = 1.0 / (dist_to_obs ** 2 + 0.01)
                
                # 4. 总排斥力
                repulsion = base_strength * distance_factor * to_obs_norm
                
                # 5. 垂直分量增强（鼓励向上逃离）
                if to_obs_norm[2] < 0:  # 障碍物在上方
                    repulsion[2] *= 0.5  # 减弱向下
                else:  # 障碍物在下方或同高度
                    repulsion[2] += 0.3  # 增强向上
                
                repulsive_force += repulsion
                total_danger_level += base_strength
        
        # 3. 动态调整吸引力和排斥力的权重
        if total_danger_level > 2.0:
            # 高危险情况：排斥力主导
            w_att = 0.2
            w_rep = 1.0
        elif total_danger_level > 1.0:
            # 中等危险：平衡
            w_att = 0.5
            w_rep = 0.8
        else:
            # 低危险：吸引力主导
            w_att = 1.0
            w_rep = 0.5
        
        # 4. 组合吸引力和排斥力
        combined_direction = w_att * attractive_force + w_rep * repulsive_force
        
        # 归一化
        combined_norm = np.linalg.norm(combined_direction)
        if combined_norm > 0.001:
            combined_direction = combined_direction / combined_norm
        else:
            # 局部最小值：添加随机扰动
            random_dir = np.random.randn(3)
            random_dir[2] = abs(random_dir[2])  # 向上
            combined_direction = random_dir / np.linalg.norm(random_dir)
        
        # 5. 计算下一个路径点
        # 在高危情况下，使用更大的步长快速逃离
        effective_step_size = step_size * (1.0 + total_danger_level * 0.5)
        effective_step_size = min(effective_step_size, 0.15)  # 上限
        
        next_waypoint = current + combined_direction * effective_step_size
        
        # 6. 工作空间限制
        next_waypoint = self._apply_workspace_limits(next_waypoint)
        
        return next_waypoint
    
    def plan_local_path(self, current_pos, goal_pos, goal_orn, perceived_obstacles,
                       num_waypoints=5):
        """
        规划局部路径（未来几个路径点）
        
        Args:
            current_pos: 当前位置
            goal_pos: 目标位置
            goal_orn: 目标方向
            perceived_obstacles: 感知到的障碍物
            num_waypoints: 要规划的路径点数量
            
        Returns:
            list: 局部路径点列表
        """
        local_path = []
        current = np.array(current_pos)
        
        for i in range(num_waypoints):
            next_wp = self.compute_local_waypoint(
                current, goal_pos, perceived_obstacles, step_size=0.05
            )
            local_path.append(next_wp)
            current = next_wp
            
            # 如果已经接近目标，停止规划
            if np.linalg.norm(current - np.array(goal_pos)) < 0.05:
                break
        
        return local_path
    
    def should_replan(self):
        """判断是否需要重新规划"""
        self.step_counter += 1
        return self.step_counter % self.replan_rate == 0
    
    def _apply_workspace_limits(self, position):
        """应用工作空间限制"""
        # 简单的边界限制
        pos = position.copy()
        pos[0] = np.clip(pos[0], -0.2, 1.0)  # X范围
        pos[1] = np.clip(pos[1], -0.8, 0.8)  # Y范围
        pos[2] = np.clip(pos[2], 0.05, 1.2)  # Z范围
        return pos
    
    def validate_local_path(self, workspace_path, goal_orn, obstacle_ids, 
                           current_gripper_pos):
        """
        验证局部路径是否安全
        
        Args:
            workspace_path: 工作空间路径
            goal_orn: 目标方向
            obstacle_ids: 障碍物ID列表
            current_gripper_pos: 当前夹爪位置
            
        Returns:
            (is_valid, joint_path): 是否有效及关节空间路径
        """
        if not workspace_path:
            return False, []
        
        current_joint_pos = np.asarray([p.getJointState(self.robot_id, i)[0] for i in range(7)])
        joint_path = []
        last_joint_pos = current_joint_pos.copy()
        ik_params = DEFAULT_NULL_SPACE_PARAMS.copy()
        
        for i, wp_pos in enumerate(workspace_path):
            try:
                ik_params["restPoses"] = list(last_joint_pos)
                wp_joints = p.calculateInverseKinematics(
                    self.robot_id, ROBOT_END_EFFECTOR_LINK_ID, 
                    wp_pos, goal_orn, **ik_params
                )[:7]
                
                # 检查碰撞
                if is_path_colliding(self.robot_id, last_joint_pos, wp_joints, 
                                    obstacle_ids, current_gripper_pos, current_gripper_pos):
                    return False, []
                
                joint_path.append(wp_joints)
                last_joint_pos = wp_joints
                
            except Exception as e:
                return False, []
        
        return True, joint_path


class ReactivePlanner:
    """
    反应式规划器
    用于紧急避障和快速响应
    """
    
    def __init__(self, robot_id):
        self.robot_id = robot_id
    
    def compute_emergency_avoidance(self, current_pos, dangerous_obstacles):
        """
        计算紧急避障方向
        
        Args:
            current_pos: 当前位置
            dangerous_obstacles: 危险障碍物列表 [(obs_id, pos, velocity)]
            
        Returns:
            np.array: 避障方向
        """
        if not dangerous_obstacles:
            return np.array([0, 0, 0])
        
        current = np.array(current_pos)
        escape_direction = np.array([0.0, 0.0, 0.0])
        
        for obs_id, obs_pos, obs_velocity in dangerous_obstacles:
            # 远离障碍物
            to_safety = current - np.array(obs_pos)
            dist = np.linalg.norm(to_safety)
            
            if dist > 0.01:
                # 考虑障碍物的运动方向
                obs_vel = np.array(obs_velocity)
                
                # 如果障碍物在靠近，加强逃离力度
                if np.dot(obs_vel, -to_safety) > 0:  # 障碍物正在接近
                    strength = 2.0 / (dist + 0.01)
                else:
                    strength = 1.0 / (dist + 0.01)
                
                escape_direction += strength * (to_safety / dist)
        
        # 归一化
        if np.linalg.norm(escape_direction) > 0:
            escape_direction = escape_direction / np.linalg.norm(escape_direction)
        
        return escape_direction
    
    def check_if_dangerous(self, current_pos, obstacles, danger_threshold=0.10):
        """
        检查是否有危险的障碍物（更保守的检测）
        
        Args:
            current_pos: 当前位置
            obstacles: 障碍物列表
            danger_threshold: 危险距离阈值（降低到0.10米，只在极近时触发）
            
        Returns:
            list: 危险障碍物列表
        """
        dangerous = []
        current = np.array(current_pos)
        
        for obs_id, obs_pos, obs_velocity in obstacles:
            obs_pos_array = np.array(obs_pos)
            dist = np.linalg.norm(current - obs_pos_array)
            velocity_mag = np.linalg.norm(obs_velocity)
            
            # 考虑距离和速度（动态阈值）
            # 基础阈值降低，但高速障碍物仍然提前检测
            dynamic_threshold = danger_threshold + velocity_mag * 0.15
            
            # 额外考虑：障碍物是否在接近
            obs_velocity_array = np.array(obs_velocity)
            to_robot = current - obs_pos_array
            if dist > 0.01:
                to_robot_norm = to_robot / dist
                approaching = -np.dot(obs_velocity_array, to_robot_norm)
                
                # 如果障碍物快速接近，提前警告
                if approaching > 0.1:
                    dynamic_threshold += approaching * 0.1
            
            if dist < dynamic_threshold:
                dangerous.append((obs_id, obs_pos, obs_velocity))
        
        return dangerous

