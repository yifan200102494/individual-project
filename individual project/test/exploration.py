"""
随机探索模块
提供多种探索策略以逃离局部最优
"""

import pybullet as p
import numpy as np

from constants import DEFAULT_NULL_SPACE_PARAMS, ROBOT_END_EFFECTOR_LINK_ID, WORKSPACE_LIMITS
from collision_detection import is_path_colliding, is_state_colliding


def perform_random_exploration(robot_id, obstacle_ids, **kwargs):
    """
    执行大范围、长距离的随机探索移动（优化版：减少候选点，提高速度）
    
    Args:
        robot_id: 机器人ID
        obstacle_ids: 障碍物ID列表
        **kwargs: 其他参数
    
    Returns:
        bool: 是否成功执行了随机移动
    """
    print("  >> 开始大范围随机探索移动...")
    
    current_state = p.getLinkState(robot_id, ROBOT_END_EFFECTOR_LINK_ID, computeForwardKinematics=True)
    current_pos = np.array(current_state[0])
    current_orn = current_state[1]
    
    # 生成所有探索候选点（优化后总数减少到约16个）
    exploration_candidates = []
    
    print("    >> 探索策略 1: 大范围工作空间采样...")
    exploration_candidates.extend(generate_workspace_exploration_targets(current_pos, obstacle_ids))
    
    if obstacle_ids:
        print("    >> 探索策略 2: 远离障碍物...")
        exploration_candidates.extend(generate_obstacle_avoidance_targets(current_pos, obstacle_ids))
    
    print("    >> 探索策略 3: 多层级高度探索...")
    exploration_candidates.extend(generate_height_level_targets(current_pos))
    
    print("    >> 探索策略 4: 尝试安全撤退位置...")
    exploration_candidates.extend(generate_safe_retreat_targets())
    
    print("    >> 探索策略 5: 螺旋式探索...")
    exploration_candidates.extend(generate_spiral_targets(current_pos))
    
    print(f"  >> 生成了 {len(exploration_candidates)} 个探索目标点（已优化）")
    
    # 尝试每个候选目标（早期成功则立即返回）
    for idx, target_pos in enumerate(exploration_candidates):
        if try_move_to_exploration_target(robot_id, target_pos, current_orn, obstacle_ids, idx, **kwargs):
            return True
    
    # 尝试关节空间随机移动
    if try_joint_space_exploration(robot_id, obstacle_ids, **kwargs):
        return True
    
    print("  ❌ 随机探索移动全部失败")
    return False


# ============================================================
# 探索目标生成策略
# ============================================================

def generate_workspace_exploration_targets(current_pos, obstacle_ids):
    """生成工作空间探索目标（优化版：减少候选点，增加探索距离）"""
    X_MIN = WORKSPACE_LIMITS["X_MIN"]
    X_MAX = WORKSPACE_LIMITS["X_MAX"]
    Y_MIN = WORKSPACE_LIMITS["Y_MIN"]
    Y_MAX = WORKSPACE_LIMITS["Y_MAX"]
    Z_MIN = WORKSPACE_LIMITS["Z_MIN"]
    Z_MAX = WORKSPACE_LIMITS["Z_MAX"]
    
    targets = []
    
    # 减少到5个点，但增加探索距离
    for _ in range(5):
        random_target = np.array([
            np.random.uniform(X_MIN, X_MAX),
            np.random.uniform(Y_MIN, Y_MAX),
            np.random.uniform(Z_MIN, Z_MAX)
        ])
        
        # 80%概率生成远离当前位置的点（增加概率和距离）
        if np.random.random() > 0.2:
            offset_direction = random_target - current_pos
            offset_norm = np.linalg.norm(offset_direction)
            if offset_norm > 0 and offset_norm < 0.4:
                # 增加探索距离到0.4-0.7米
                offset_direction = offset_direction / offset_norm * np.random.uniform(0.4, 0.7)
                random_target = current_pos + offset_direction
                random_target[0] = np.clip(random_target[0], X_MIN, X_MAX)
                random_target[1] = np.clip(random_target[1], Y_MIN, Y_MAX)
                random_target[2] = np.clip(random_target[2], Z_MIN, Z_MAX)
        targets.append(random_target)
    
    return targets


def generate_obstacle_avoidance_targets(current_pos, obstacle_ids):
    """生成远离障碍物的探索目标（优化版：减少到2个点）"""
    X_MIN = WORKSPACE_LIMITS["X_MIN"]
    X_MAX = WORKSPACE_LIMITS["X_MAX"]
    Y_MIN = WORKSPACE_LIMITS["Y_MIN"]
    Y_MAX = WORKSPACE_LIMITS["Y_MAX"]
    Z_MIN = WORKSPACE_LIMITS["Z_MIN"]
    Z_MAX = WORKSPACE_LIMITS["Z_MAX"]
    
    targets = []
    obstacle_centers = []
    
    for obs_id in obstacle_ids:
        try:
            aabb_min, aabb_max = p.getAABB(obs_id)
            obs_center = np.array([
                (aabb_min[0] + aabb_max[0]) / 2,
                (aabb_min[1] + aabb_max[1]) / 2,
                (aabb_min[2] + aabb_max[2]) / 2
            ])
            obstacle_centers.append(obs_center)
        except:
            pass
    
    if obstacle_centers:
        avg_obstacle_pos = np.mean(obstacle_centers, axis=0)
        escape_direction = current_pos - avg_obstacle_pos
        if np.linalg.norm(escape_direction[:2]) > 0:
            escape_direction = escape_direction / np.linalg.norm(escape_direction)
            
            # 只生成2个远离点（中等和远距离）
            for dist in [0.5, 0.8]:
                escape_target = current_pos + escape_direction * dist
                escape_target[2] = current_pos[2] + np.random.uniform(-0.1, 0.4)
                
                escape_target[0] = np.clip(escape_target[0], X_MIN, X_MAX)
                escape_target[1] = np.clip(escape_target[1], Y_MIN, Y_MAX)
                escape_target[2] = np.clip(escape_target[2], Z_MIN, Z_MAX)
                
                targets.append(escape_target)
    
    return targets


def generate_height_level_targets(current_pos):
    """生成多层级高度探索目标（优化版：减少到2个点）"""
    X_MIN = WORKSPACE_LIMITS["X_MIN"]
    X_MAX = WORKSPACE_LIMITS["X_MAX"]
    Y_MIN = WORKSPACE_LIMITS["Y_MIN"]
    Y_MAX = WORKSPACE_LIMITS["Y_MAX"]
    Z_MIN = WORKSPACE_LIMITS["Z_MIN"]
    Z_MAX = WORKSPACE_LIMITS["Z_MAX"]
    
    targets = []
    
    # 只尝试最高点和中等高度
    for z_level in [Z_MAX, Z_MAX * 0.65]:
        high_target = current_pos.copy()
        high_target[2] = z_level
        high_target[0] += np.random.uniform(-0.3, 0.3)
        high_target[1] += np.random.uniform(-0.3, 0.3)
        high_target[0] = np.clip(high_target[0], X_MIN, X_MAX)
        high_target[1] = np.clip(high_target[1], Y_MIN, Y_MAX)
        targets.append(high_target)
    
    return targets


def generate_safe_retreat_targets():
    """生成安全撤退位置（优化版：减少到3个点）"""
    return [
        np.array([0.4, 0.0, 0.6]),
        np.array([0.4, 0.35, 0.6]),
        np.array([0.4, -0.35, 0.6]),
    ]


def generate_spiral_targets(current_pos):
    """生成螺旋式探索目标（优化版：减少到4个点）"""
    X_MIN = WORKSPACE_LIMITS["X_MIN"]
    X_MAX = WORKSPACE_LIMITS["X_MAX"]
    Y_MIN = WORKSPACE_LIMITS["Y_MIN"]
    Y_MAX = WORKSPACE_LIMITS["Y_MAX"]
    Z_MIN = WORKSPACE_LIMITS["Z_MIN"]
    Z_MAX = WORKSPACE_LIMITS["Z_MAX"]
    
    targets = []
    num_spiral_points = 4  # 减少到4个方向
    
    for i in range(num_spiral_points):
        angle = (2 * np.pi * i) / num_spiral_points
        # 只使用一个较大的半径
        radius = 0.5
        spiral_target = current_pos.copy()
        spiral_target[0] += radius * np.cos(angle)
        spiral_target[1] += radius * np.sin(angle)
        spiral_target[2] += np.random.uniform(0.1, 0.3)  # 倾向于向上
        spiral_target[0] = np.clip(spiral_target[0], X_MIN, X_MAX)
        spiral_target[1] = np.clip(spiral_target[1], Y_MIN, Y_MAX)
        spiral_target[2] = np.clip(spiral_target[2], Z_MIN, Z_MAX)
        targets.append(spiral_target)
    
    return targets


# ============================================================
# 探索执行
# ============================================================

def try_move_to_exploration_target(robot_id, target_pos, current_orn, obstacle_ids, idx, **kwargs):
    """尝试移动到探索目标（优化版：增加速度）"""
    from motion_control import move_to_joints
    
    print(f"  >> 尝试探索目标 {idx+1}: {target_pos}")
    
    try:
        target_joints = p.calculateInverseKinematics(
            robot_id, ROBOT_END_EFFECTOR_LINK_ID,
            target_pos, current_orn,
            **DEFAULT_NULL_SPACE_PARAMS
        )[:7]
        
        current_joints = np.asarray([p.getJointState(robot_id, i)[0] for i in range(7)])
        current_gripper = [p.getJointState(robot_id, 9)[0], p.getJointState(robot_id, 10)[0]]
        
        if not is_path_colliding(robot_id, current_joints, target_joints,
                               obstacle_ids, current_gripper, current_gripper):
            print(f"    ✓ 目标 {idx+1} 路径安全，执行移动...")
            # 提高探索速度到3.0
            success = move_to_joints(robot_id, target_joints, max_velocity=3.0, **kwargs)
            
            if success:
                print(f"  ✅ 随机探索成功移动到新位置!")
                return True
            else:
                print(f"    ✗ 执行移动失败")
        else:
            print(f"    ✗ 目标 {idx+1} 路径会碰撞")
    
    except Exception as e:
        print(f"    ✗ 目标 {idx+1} IK求解失败: {e}")
    
    return False


def try_joint_space_exploration(robot_id, obstacle_ids, **kwargs):
    """尝试关节空间随机移动（优化版：减少到3次尝试，增加速度）"""
    from motion_control import move_to_joints
    
    print("  >> 所有探索目标都失败，尝试关节空间移动...")
    current_joints = np.asarray([p.getJointState(robot_id, i)[0] for i in range(7)])
    
    # 减少尝试次数从5次到3次
    for attempt in range(3):
        # 增加幅度，让每次移动更大
        amplitude = 0.6 + (attempt * 0.2)
        joint_offset = np.random.uniform(-amplitude, amplitude, size=7)
        
        if attempt < 1:
            joint_offset[0] *= 0.6
            joint_offset[-2:] *= 0.5
        else:
            joint_offset[0] *= 0.8
            joint_offset[-2:] *= 0.7
        
        target_joints = current_joints + joint_offset
        
        # 检查关节限制
        for i in range(7):
            joint_info = p.getJointInfo(robot_id, i)
            lower_limit = joint_info[8]
            upper_limit = joint_info[9]
            target_joints[i] = np.clip(target_joints[i], lower_limit, upper_limit)
        
        current_gripper = [p.getJointState(robot_id, 9)[0], p.getJointState(robot_id, 10)[0]]
        if not is_state_colliding(robot_id, target_joints, obstacle_ids, current_gripper):
            print(f"    >> 尝试关节微调 {attempt+1}/3...")
            # 提高速度到2.0
            success = move_to_joints(robot_id, target_joints, max_velocity=2.0, timeout=4, **kwargs)
            if success:
                print(f"  ✅ 通过关节微调成功改变位置!")
                return True
    
    return False

