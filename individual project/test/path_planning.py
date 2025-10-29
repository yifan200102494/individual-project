"""
路径规划模块
包含势场法(PFM)、弧形路径、路径验证等功能
"""

import pybullet as p
import numpy as np
import random

from constants import DEFAULT_NULL_SPACE_PARAMS, ROBOT_END_EFFECTOR_LINK_ID
from collision_detection import is_path_colliding


# ============================================================
# 势场法 (PFM)
# ============================================================

def calc_attractive_force(current_pos, goal_pos, k_att=1.0):
    """计算吸引力"""
    dist_vec = np.array(goal_pos) - np.array(current_pos)
    dist = np.linalg.norm(dist_vec)
    if dist < 1e-6:
        return np.array([0.0, 0.0, 0.0])
    return k_att * (dist_vec / dist)


def calc_anisotropic_repulsive_force(current_pos, obs_center, obs_aabb_min, obs_aabb_max,
                                     k_rep=0.5, rho_0=0.35, k_aniso_xy=2.0, k_aniso_z=0.5):
    """计算各向异性排斥力"""
    dist_vec = np.array(current_pos) - obs_center
    scaling_factors = np.array([k_aniso_xy, k_aniso_xy, k_aniso_z])
    scaled_dist_vec = dist_vec * scaling_factors
    rho_scaled = np.linalg.norm(scaled_dist_vec)
    
    if rho_scaled > rho_0:
        return np.array([0.0, 0.0, 0.0])
    if rho_scaled < 1e-6:
        return (np.random.rand(3) - 0.5) * 2.0 * k_rep
    
    grad_rho_scaled = scaled_dist_vec / rho_scaled
    magnitude = k_rep * (1.0 / rho_scaled - 1.0 / rho_0) * (1.0 / (rho_scaled**2))
    return magnitude * grad_rho_scaled


def prepare_obstacles_info(obstacle_ids):
    """准备障碍物信息"""
    obstacles_info = []
    for obs_id in obstacle_ids:
        aabb_min, aabb_max = p.getAABB(obs_id)
        obs_center = np.array([
            (aabb_min[0] + aabb_max[0]) / 2,
            (aabb_min[1] + aabb_max[1]) / 2,
            (aabb_min[2] + aabb_max[2]) / 2
        ])
        aabb_diag = np.linalg.norm(np.array(aabb_max) - np.array(aabb_min))
        obstacles_info.append({
            "id": obs_id,
            "center": obs_center,
            "aabb_min": aabb_min,
            "aabb_max": aabb_max,
            "diag": aabb_diag
        })
    return obstacles_info


def plan_path_with_pfm(start_pos, goal_pos, obstacle_ids,
                       step_size=0.02, max_steps=300, goal_threshold=0.05,
                       k_att=1.0, k_rep=1.2, randomize=False):
    """
    使用势场法规划路径
    
    Args:
        start_pos: 起始位置
        goal_pos: 目标位置
        obstacle_ids: 障碍物ID列表
        step_size: 步长
        max_steps: 最大步数
        goal_threshold: 到达目标的阈值
        k_att: 吸引力系数
        k_rep: 排斥力系数
        randomize: 是否随机化参数
    
    Returns:
        list: 路径点列表，失败返回None
    """
    print(f"  >> PFM: 启动势场法路径规划器 (step={step_size:.3f}, k_att={k_att:.2f})...")
    
    obstacles_info = prepare_obstacles_info(obstacle_ids)
    
    if randomize:
        k_rep = k_rep + np.random.uniform(-0.2, 0.2)
    
    rho_0_base = 0.35
    k_aniso_xy = 2.0
    k_aniso_z = 0.5
    path = [np.array(start_pos)]
    current_pos = np.array(start_pos)
    
    last_positions = []
    escape_attempts = 0
    max_escape_attempts = 3

    for i in range(max_steps):
        f_att = calc_attractive_force(current_pos, goal_pos, k_att=k_att)
        f_rep_total = np.array([0.0, 0.0, 0.0])
        
        for obs in obstacles_info:
            rho_0 = (obs["diag"] / 2.0) + rho_0_base
            f_rep_obs = calc_anisotropic_repulsive_force(
                current_pos, obs["center"], obs["aabb_min"], obs["aabb_max"],
                k_rep=k_rep, rho_0=rho_0, k_aniso_xy=k_aniso_xy, k_aniso_z=k_aniso_z
            )
            f_rep_total += f_rep_obs
        
        f_total = f_att + f_rep_total
        
        # 检测局部最小值
        if np.linalg.norm(f_total) < 0.001:
            if escape_attempts < max_escape_attempts:
                print(f"  ⚠️ PFM: 检测到局部最小值，尝试逃逸 ({escape_attempts+1}/{max_escape_attempts})...")
                escape_direction = np.random.randn(3)
                if not randomize:
                    escape_direction[2] = abs(escape_direction[2])
                escape_direction = escape_direction / np.linalg.norm(escape_direction)
                escape_distance = 0.1 if not randomize else 0.15 + np.random.uniform(0, 0.1)
                current_pos = current_pos + escape_direction * escape_distance
                escape_attempts += 1
                continue
            else:
                print(f"  ❌ PFM: 规划失败，在第 {i} 步陷入局部最小值。")
                return None
        
        # 检测是否卡住
        last_positions.append(current_pos.copy())
        if len(last_positions) > 10:
            last_positions.pop(0)
            if np.std([np.linalg.norm(pos - current_pos) for pos in last_positions[-5:]]) < 0.001:
                if escape_attempts < max_escape_attempts:
                    print(f"  ⚠️ PFM: 检测到卡住，尝试侧向逃逸...")
                    to_goal = goal_pos - current_pos
                    to_goal[2] = 0
                    if np.linalg.norm(to_goal) > 0:
                        perpendicular = np.array([-to_goal[1], to_goal[0], 0])
                        perpendicular = perpendicular / np.linalg.norm(perpendicular)
                        perpendicular *= np.random.choice([-1, 1])
                        current_pos = current_pos + perpendicular * 0.15
                        escape_attempts += 1
                        continue
        
        current_pos = current_pos + step_size * (f_total / np.linalg.norm(f_total))
        path.append(current_pos)
        
        if np.linalg.norm(current_pos - np.array(goal_pos)) < goal_threshold:
            path.append(np.array(goal_pos))
            print(f"  ✅ PFM: 成功生成路径，共 {len(path)} 个路径点。")
            return path
    
    print(f"  ❌ PFM: 规划失败，超过最大步数 {max_steps}。")
    return None


# ============================================================
# 弧形路径生成
# ============================================================

def generate_arc_path(start_pos, goal_pos, obstacle_ids, num_points=10):
    """
    生成弧形路径，避开障碍物
    
    Args:
        start_pos: 起始位置
        goal_pos: 目标位置
        obstacle_ids: 障碍物ID列表
        num_points: 路径点数量
    
    Returns:
        list: 路径点列表，失败返回None
    """
    print("  >> 生成弧形路径...")
    
    start = np.array(start_pos)
    goal = np.array(goal_pos)
    mid_point = (start + goal) / 2
    
    arc_direction = np.random.choice(['left', 'right', 'up'])
    
    if arc_direction == 'up':
        arc_height = 0.2 + np.random.uniform(0, 0.15)
        mid_point[2] += arc_height
    else:
        direction = goal - start
        perpendicular = np.array([-direction[1], direction[0], 0]) if arc_direction == 'left' else np.array([direction[1], -direction[0], 0])
        if np.linalg.norm(perpendicular) > 0:
            perpendicular = perpendicular / np.linalg.norm(perpendicular)
            arc_offset = 0.15 + np.random.uniform(0, 0.1)
            mid_point += perpendicular * arc_offset
            mid_point[2] += 0.1
    
    # 生成贝塞尔曲线路径
    path = []
    for i in range(num_points + 1):
        t = i / num_points
        point = (1 - t)**2 * start + 2 * (1 - t) * t * mid_point + t**2 * goal
        path.append(point)
    
    # 检查碰撞
    for obs_id in obstacle_ids:
        aabb_min, aabb_max = p.getAABB(obs_id)
        for point in path:
            if (aabb_min[0] - 0.05 <= point[0] <= aabb_max[0] + 0.05 and
                aabb_min[1] - 0.05 <= point[1] <= aabb_max[1] + 0.05 and
                aabb_min[2] - 0.05 <= point[2] <= aabb_max[2] + 0.05):
                print("  >> 弧形路径与障碍物碰撞，放弃此路径")
                return None
    
    print(f"  ✅ 成功生成{arc_direction}向弧形路径，共 {len(path)} 个路径点")
    return path


# ============================================================
# 路径验证
# ============================================================

def validate_workspace_path(workspace_path, robot_id, goal_orn, obstacle_ids,
                           current_gripper_pos, sampling_step=3):
    """
    验证工作空间路径在关节空间中是否可行
    
    Args:
        workspace_path: 工作空间路径点列表
        robot_id: 机器人ID
        goal_orn: 目标方向
        obstacle_ids: 障碍物ID列表
        current_gripper_pos: 当前夹爪位置
        sampling_step: 采样步长
    
    Returns:
        (is_valid, joint_path): 是否有效，以及关节空间路径
    """
    if workspace_path is None:
        return False, []
    
    print("  >> 验证工作空间路径...")
    current_joint_pos = np.asarray([p.getJointState(robot_id, i)[0] for i in range(7)])
    last_joint_pos = current_joint_pos.copy()
    ik_params = DEFAULT_NULL_SPACE_PARAMS.copy()
    joint_path = []
    
    for i, wp_pos in enumerate(workspace_path):
        try:
            if i % sampling_step != 0 and i != (len(workspace_path) - 1):
                continue

            ik_params["restPoses"] = list(last_joint_pos)
            wp_joints = p.calculateInverseKinematics(
                robot_id, ROBOT_END_EFFECTOR_LINK_ID, wp_pos, goal_orn, **ik_params
            )[:7]
            
            if is_path_colliding(robot_id, last_joint_pos, wp_joints, obstacle_ids,
                                 current_gripper_pos, current_gripper_pos):
                print(f"  ❌ 路径在 C-Space 中发现碰撞 (段 {i})。")
                return False, []
            
            joint_path.append(wp_joints)
            last_joint_pos = wp_joints
        except Exception as e:
            print(f"  ❌ 路径点 {i} IK 求解失败。")
            return False, []
    
    print("  ✅ 路径在 C-Space 中验证安全")
    return True, joint_path


# ============================================================
# 路径历史记录
# ============================================================

PATH_HISTORY = []
MAX_PATH_HISTORY = 10


def add_path_to_history(path_points):
    """添加路径到历史记录"""
    global PATH_HISTORY
    if len(path_points) > 2:
        key_points = [path_points[0], path_points[len(path_points)//2], path_points[-1]]
        path_feature = hash(tuple(map(tuple, key_points)))
        PATH_HISTORY.append(path_feature)
        if len(PATH_HISTORY) > MAX_PATH_HISTORY:
            PATH_HISTORY.pop(0)


def is_path_similar_to_history(path_points):
    """检查路径是否与历史路径相似"""
    if len(path_points) <= 2:
        return False
    key_points = [path_points[0], path_points[len(path_points)//2], path_points[-1]]
    path_feature = hash(tuple(map(tuple, key_points)))
    return path_feature in PATH_HISTORY


# ============================================================
# 绕行策略生成
# ============================================================

def generate_detour_strategies(current_pos, goal_pos, obstacle_ids):
    """
    生成多种绕行策略
    
    Args:
        current_pos: 当前位置
        goal_pos: 目标位置
        obstacle_ids: 障碍物ID列表
    
    Returns:
        list: 策略列表，每个策略是路径点列表
    """
    strategies = []
    
    # 计算障碍物信息
    max_obstacle_z = 0.3
    obstacle_centers = []
    for obs_id in obstacle_ids:
        try:
            aabb_min, aabb_max = p.getAABB(obs_id)
            max_obstacle_z = max(max_obstacle_z, aabb_max[2])
            obs_center = [
                (aabb_min[0] + aabb_max[0])/2,
                (aabb_min[1] + aabb_max[1])/2,
                (aabb_min[2] + aabb_max[2])/2
            ]
            obstacle_centers.append(obs_center)
        except Exception:
            pass

    z_safe_cruise = max(max_obstacle_z + 0.2, current_pos[2] + 0.1, goal_pos[2] + 0.1, 0.6)
    
    # 策略1：直接上升-巡航-下降
    strategies.append([
        np.array([current_pos[0], current_pos[1], z_safe_cruise]),
        np.array([goal_pos[0], goal_pos[1], z_safe_cruise]),
        np.array(goal_pos)
    ])
    
    # 策略2：侧向绕行
    if obstacle_centers:
        obs_center = obstacle_centers[0]
        direction_to_goal = np.array([goal_pos[0] - current_pos[0], goal_pos[1] - current_pos[1], 0])
        if np.linalg.norm(direction_to_goal) > 0:
            direction_to_goal = direction_to_goal / np.linalg.norm(direction_to_goal)
            perpendicular = np.array([-direction_to_goal[1], direction_to_goal[0], 0])
            
            to_obs = np.array([obs_center[0] - current_pos[0], obs_center[1] - current_pos[1], 0])
            if np.dot(perpendicular, to_obs) > 0:
                perpendicular = -perpendicular
            
            side_offset = 0.3
            mid_point = np.array([
                obs_center[0] + perpendicular[0] * side_offset,
                obs_center[1] + perpendicular[1] * side_offset,
                max(current_pos[2], goal_pos[2]) + 0.1
            ])
            
            strategies.append([
                np.array([current_pos[0], current_pos[1], current_pos[2] + 0.1]),
                mid_point,
                np.array([goal_pos[0], goal_pos[1], goal_pos[2] + 0.1]),
                np.array(goal_pos)
            ])
    
    return strategies

