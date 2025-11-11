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
                               step_size=0.08):
        """
        计算下一个局部路径点（改进的势场法）
        
        特点：
        - 近距离时排斥力指数增强
        - 考虑障碍物运动方向
        - 防止局部最小值
        - 更大的步长以保证连贯性
        
        Args:
            current_pos: 当前位置
            goal_pos: 最终目标位置
            perceived_obstacles: 感知到的障碍物信息
            step_size: 步长（增加到0.08以提高连贯性）
            
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
                
                # 4. 计算基础排斥力
                repulsion = base_strength * distance_factor * to_obs_norm
                
                # 🔥 5. 智能调整排斥力方向：避免往前（X正方向）绕行
                # 如果障碍物在前方（X坐标大于当前位置），增强Y和Z方向的排斥分量
                if obs_pos[0] > current[0]:  # 障碍物在前面
                    # 检查是否试图往前绕行（排斥力X分量为正）
                    if repulsion[0] > 0:
                        # 将部分X方向的排斥力转移到Y和Z方向
                        x_component = repulsion[0]
                        
                        # 减弱X方向的排斥（因为前面可能是工作空间边界或更多障碍物）
                        repulsion[0] *= 0.3
                        
                        # 增强Y方向的排斥（侧向绕行）
                        # 选择远离障碍物的Y方向
                        y_direction = 1.0 if current[1] > obs_pos[1] else -1.0
                        repulsion[1] += x_component * 1.5 * y_direction
                        
                        # 增强Z方向的排斥（向上绕行，最优先）
                        repulsion[2] += x_component * 2.0  # 向上分量最强
                        
                        if np.random.random() < 0.1:  # 10%的概率打印调试信息
                            print(f"  [路径规划] 检测到障碍物在前方，引导从侧面和上方绕行")
                
                # 总排斥力
                repulsion_original = repulsion.copy()
                
                # 6. 垂直分量智能调整（考虑目标位置和障碍物位置）
                goal_z = goal[2]
                current_z = current[2]
                
                # 检查是否已经在目标正上方（准备下降）
                horizontal_to_goal = np.linalg.norm(current[:2] - goal[:2])
                is_above_target = horizontal_to_goal < 0.15 and current_z > goal_z
                
                if to_obs_norm[2] < 0:  # 障碍物在上方
                    repulsion[2] *= 0.5  # 减弱向下排斥
                else:  # 障碍物在下方或同高度
                    # 只有在真正需要下降且已对齐时才减弱向上排斥
                    if goal_z < current_z - 0.05 and is_above_target:
                        # 已在目标正上方，需要下降：减弱向上推力
                        repulsion[2] *= 0.3
                    elif goal_z < current_z - 0.02 and horizontal_to_goal < 0.20:
                        # 接近目标上方，轻微减弱
                        repulsion[2] *= 0.7
                    elif obs_pos[0] > current[0]:
                        # 障碍物在前方：强化向上绕行（已在前面增强过，这里保持）
                        pass  # 保持已增强的向上分量
                    else:
                        # 正常情况：适度向上推力
                        repulsion[2] *= 1.2
                
                repulsive_force += repulsion
                total_danger_level += base_strength
        
        # 3. 动态调整吸引力和排斥力的权重
        # 特殊情况：只在满足以下所有条件时才强制向上：
        # 1. 当前位置过低
        # 2. 目标不在低位
        # 3. 不是正在向下移动
        goal_is_low = goal[2] < 0.25  # 目标是否在低位（降低阈值到25cm，更宽松）
        is_descending = goal[2] < current[2] - 0.03  # 是否正在下降（目标比当前低3cm以上，更敏感）
        
        # 额外判断：如果水平距离已经很近，也认为在下降
        horizontal_dist_to_goal = np.linalg.norm(current[:2] - goal[:2])
        is_above_goal = horizontal_dist_to_goal < 0.15 and current[2] > goal[2]
        
        if current[2] < 0.10 and not goal_is_low and not is_descending and not is_above_goal:
            # 只有在异常低位(<10cm)、目标高、且不是下降、不在目标上方时才强制向上
            repulsive_force[2] += 5.0  # 强力向上推
            total_danger_level += 3.0
            print(f"  [!] 检测到位置异常过低 (Z={current[2]:.3f}m)，目标在高位，强制向上")
        
        # 特殊判断：如果目标是最终放置位置（低位），增强吸引力
        is_final_placement = (goal[2] < 0.20)  # 目标在很低的位置（<20cm）
        
        if is_final_placement:
            # 最终放置：大幅增强吸引力，允许靠近目标
            if total_danger_level > 3.0:
                # 极高危险：仍然排斥力主导
                w_att = 0.3
                w_rep = 1.0
            elif total_danger_level > 1.5:
                # 中等危险：吸引力主导
                w_att = 1.2
                w_rep = 0.5
            else:
                # 低危险：强吸引力
                w_att = 1.5
                w_rep = 0.3
        else:
            # 正常情况
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
        # 平滑调整步长，避免突变
        if total_danger_level > 1.5:
            # 高危情况：适度增加步长
            effective_step_size = step_size * 1.3
        else:
            # 正常情况：使用固定步长保持连贯性
            effective_step_size = step_size
        
        effective_step_size = min(effective_step_size, 0.12)  # 降低上限，避免过大跳跃
        
        next_waypoint = current + combined_direction * effective_step_size
        
        # 6. 工作空间限制
        next_waypoint = self._apply_workspace_limits(next_waypoint)
        
        return next_waypoint
    
    def plan_local_path(self, current_pos, goal_pos, goal_orn, perceived_obstacles,
                       num_waypoints=8):
        """
        规划局部路径（未来几个路径点）- 增加路径点数量以提高平滑度
        
        Args:
            current_pos: 当前位置
            goal_pos: 目标位置
            goal_orn: 目标方向
            perceived_obstacles: 感知到的障碍物
            num_waypoints: 要规划的路径点数量（增加到8个）
            
        Returns:
            list: 局部路径点列表（经过平滑处理）
        """
        raw_path = []
        current = np.array(current_pos)
        
        for i in range(num_waypoints):
            next_wp = self.compute_local_waypoint(
                current, goal_pos, perceived_obstacles, step_size=0.08
            )
            raw_path.append(next_wp)
            current = next_wp
            
            # 如果已经接近目标，停止规划
            if np.linalg.norm(current - np.array(goal_pos)) < 0.05:
                break
        
        # 对路径进行平滑处理
        smoothed_path = self._smooth_path_advanced(raw_path)
        
        return smoothed_path
    
    def _smooth_path(self, path, alpha=0.3):
        """
        对路径进行平滑处理（移动平均滤波）
        
        Args:
            path: 原始路径点列表
            alpha: 平滑系数（0-1之间，越大越平滑但偏离越多）
            
        Returns:
            平滑后的路径
        """
        if len(path) <= 2:
            return path
        
        smoothed = [path[0]]  # 保持起点不变
        
        for i in range(1, len(path) - 1):
            # 对中间点进行加权平均
            prev_point = np.array(smoothed[-1])
            curr_point = np.array(path[i])
            next_point = np.array(path[i + 1])
            
            # 三点平均
            smoothed_point = (1 - alpha) * curr_point + alpha * (prev_point + next_point) / 2
            smoothed.append(smoothed_point)
        
        smoothed.append(path[-1])  # 保持终点不变
        
        return smoothed
    
    def _smooth_path_advanced(self, path, num_points=None):
        """
        高级路径平滑（使用Catmull-Rom样条插值，纯numpy实现）
        
        Args:
            path: 原始路径点列表
            num_points: 插值后的点数（None表示使用原始点数）
            
        Returns:
            平滑后的路径
        """
        if len(path) <= 2:
            return path
        
        try:
            # 将路径点转换为numpy数组
            path_array = np.array(path)
            n_points = len(path_array)
            
            # 如果点数太少，使用简单平滑
            if n_points < 4:
                return self._smooth_path(path, alpha=0.3)
            
            # 使用Catmull-Rom样条插值
            smoothed_path = [path_array[0]]  # 保留起点
            
            # 对每对相邻点进行插值
            segments_per_interval = 3  # 每段插值3个点
            
            for i in range(n_points - 1):
                # 获取控制点（需要前后各一个点）
                p0 = path_array[max(0, i - 1)]
                p1 = path_array[i]
                p2 = path_array[i + 1]
                p3 = path_array[min(n_points - 1, i + 2)]
                
                # Catmull-Rom样条插值
                for j in range(segments_per_interval):
                    t = (j + 1) / (segments_per_interval + 1)
                    t2 = t * t
                    t3 = t2 * t
                    
                    # Catmull-Rom基函数
                    point = 0.5 * (
                        (2 * p1) +
                        (-p0 + p2) * t +
                        (2*p0 - 5*p1 + 4*p2 - p3) * t2 +
                        (-p0 + 3*p1 - 3*p2 + p3) * t3
                    )
                    
                    smoothed_path.append(point)
                
                # 添加当前段的终点
                if i < n_points - 2:
                    smoothed_path.append(path_array[i + 1])
            
            # 确保终点被包含
            smoothed_path.append(path_array[-1])
            
            # 如果指定了点数，重新采样
            if num_points is not None and num_points != len(smoothed_path):
                # 简单的线性重采样
                indices = np.linspace(0, len(smoothed_path) - 1, num_points)
                resampled = []
                for idx in indices:
                    lower = int(np.floor(idx))
                    upper = int(np.ceil(idx))
                    if lower == upper:
                        resampled.append(smoothed_path[lower])
                    else:
                        alpha = idx - lower
                        interpolated = (1 - alpha) * smoothed_path[lower] + alpha * smoothed_path[upper]
                        resampled.append(interpolated)
                return resampled
            
            return smoothed_path
            
        except Exception as e:
            # 如果高级平滑失败，回退到简单平滑
            return self._smooth_path(path, alpha=0.3)
    
    def should_replan(self):
        """判断是否需要重新规划"""
        self.step_counter += 1
        return self.step_counter % self.replan_rate == 0
    
    def _apply_workspace_limits(self, position):
        """应用工作空间限制"""
        from constants import WORKSPACE_LIMITS
        
        # 使用统一的工作空间限制
        pos = position.copy()
        pos[0] = np.clip(pos[0], WORKSPACE_LIMITS["X_MIN"], WORKSPACE_LIMITS["X_MAX"])
        pos[1] = np.clip(pos[1], WORKSPACE_LIMITS["Y_MIN"], WORKSPACE_LIMITS["Y_MAX"])
        pos[2] = np.clip(pos[2], WORKSPACE_LIMITS["Z_MIN"], WORKSPACE_LIMITS["Z_MAX"])
        return pos
    
    def validate_local_path(self, workspace_path, goal_orn, obstacle_ids, 
                           current_gripper_pos, collision_check_steps=5):
        """
        验证局部路径是否安全
        
        Args:
            workspace_path: 工作空间路径
            goal_orn: 目标方向
            obstacle_ids: 障碍物ID列表
            current_gripper_pos: 当前夹爪位置
            collision_check_steps: 碰撞检测插值步数（越小越宽松）
            
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
                
                # 检查碰撞（使用较少的插值步数）
                if is_path_colliding(self.robot_id, last_joint_pos, wp_joints, 
                                    obstacle_ids, current_gripper_pos, current_gripper_pos,
                                    num_steps=collision_check_steps):
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

