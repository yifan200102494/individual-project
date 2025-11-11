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
    
    目的：通过大幅度移动改变机械臂与障碍物的相对位置，
         使得重新规划时可能找到新的可行路径
    
    Args:
        robot_id: 机器人ID
        obstacle_ids: 障碍物ID列表
        **kwargs: 其他参数
    
    Returns:
        bool: 是否成功执行了随机移动
    """
    print("\n" + "="*60)
    print("  🔍 开始大范围3D随机探索（包含XYZ三轴）...")
    print("="*60)
    
    current_state = p.getLinkState(robot_id, ROBOT_END_EFFECTOR_LINK_ID, computeForwardKinematics=True)
    current_pos = np.array(current_state[0])
    current_orn = current_state[1]
    
    print(f"  >> 当前位置: X={current_pos[0]:.3f}, Y={current_pos[1]:.3f}, Z={current_pos[2]:.3f}")
    
    # 生成所有探索候选点（优化后总数减少到约16个）
    exploration_candidates = []
    
    # 🔥 策略1：优先尝试抬高机械臂（最有效的避障方式）
    print("    >> 探索策略 1: ⬆️  优先抬高机械臂（最有效）...")
    exploration_candidates.extend(generate_lift_first_targets(current_pos))
    
    print("    >> 探索策略 2: 多层级高度探索...")
    exploration_candidates.extend(generate_height_level_targets(current_pos))
    
    print("    >> 探索策略 3: 大范围工作空间采样...")
    exploration_candidates.extend(generate_workspace_exploration_targets(current_pos, obstacle_ids))
    
    if obstacle_ids:
        print("    >> 探索策略 4: 远离障碍物...")
        exploration_candidates.extend(generate_obstacle_avoidance_targets(current_pos, obstacle_ids))
    
    print("    >> 探索策略 5: 尝试安全撤退位置...")
    exploration_candidates.extend(generate_safe_retreat_targets())
    
    print("    >> 探索策略 6: 螺旋式探索...")
    exploration_candidates.extend(generate_spiral_targets(current_pos))
    
    print(f"  >> 生成了 {len(exploration_candidates)} 个3D探索目标点")
    print(f"     (包含多个高度层级，Z轴变化范围: 0.2-0.8米)\n")
    
    # 尝试每个候选目标（早期成功则立即返回）
    successful_move = False
    for idx, target_pos in enumerate(exploration_candidates):
        if try_move_to_exploration_target(robot_id, target_pos, current_orn, obstacle_ids, idx, **kwargs):
            # 计算移动距离
            new_state = p.getLinkState(robot_id, ROBOT_END_EFFECTOR_LINK_ID, computeForwardKinematics=True)
            new_pos = np.array(new_state[0])
            distance_moved = np.linalg.norm(new_pos - current_pos)
            print(f"\n  ✅ 探索移动成功！")
            print(f"     移动距离: {distance_moved:.3f}m")
            print(f"     新位置: X={new_pos[0]:.3f}, Y={new_pos[1]:.3f}, Z={new_pos[2]:.3f}")
            print(f"     Z轴变化: {new_pos[2] - current_pos[2]:+.3f}m")
            print("="*60 + "\n")
            return True
    
    # 尝试关节空间随机移动
    if try_joint_space_exploration(robot_id, obstacle_ids, **kwargs):
        new_state = p.getLinkState(robot_id, ROBOT_END_EFFECTOR_LINK_ID, computeForwardKinematics=True)
        new_pos = np.array(new_state[0])
        distance_moved = np.linalg.norm(new_pos - current_pos)
        print(f"\n  ✅ 关节空间探索成功！")
        print(f"     移动距离: {distance_moved:.3f}m")
        print(f"     新位置: X={new_pos[0]:.3f}, Y={new_pos[1]:.3f}, Z={new_pos[2]:.3f}")
        print("="*60 + "\n")
        return True
    
    print("\n  ❌ 所有探索策略均未成功")
    print("="*60 + "\n")
    return False


# ============================================================
# 探索目标生成策略
# ============================================================

def generate_lift_first_targets(current_pos):
    """
    🔥 优先抬高机械臂策略（新增，最优先）
    
    这是最直接有效的避障策略：先把机械臂抬高，可以：
    1. 避开大部分低位障碍物
    2. 从高处重新规划路径更容易
    3. 简单有效，成功率高
    
    生成多个不同高度的抬高目标点
    """
    X_MIN = WORKSPACE_LIMITS["X_MIN"]
    X_MAX = WORKSPACE_LIMITS["X_MAX"]
    Y_MIN = WORKSPACE_LIMITS["Y_MIN"]
    Y_MAX = WORKSPACE_LIMITS["Y_MAX"]
    Z_MIN = WORKSPACE_LIMITS["Z_MIN"]
    Z_MAX = WORKSPACE_LIMITS["Z_MAX"]
    
    targets = []
    
    # 策略1：直接向上抬高（保持XY不变）
    for height_offset in [0.3, 0.4, 0.5]:  # 抬高30cm、40cm、50cm
        lift_target = current_pos.copy()
        lift_target[2] = min(current_pos[2] + height_offset, Z_MAX)
        targets.append(lift_target)
    
    # 策略2：抬到最高位置（XY略微调整）
    for xy_offset in [(0, 0), (0.1, 0), (-0.1, 0), (0, 0.1), (0, -0.1)]:
        high_target = current_pos.copy()
        high_target[0] = np.clip(current_pos[0] + xy_offset[0], X_MIN, X_MAX)
        high_target[1] = np.clip(current_pos[1] + xy_offset[1], Y_MIN, Y_MAX)
        high_target[2] = Z_MAX  # 直接到最高点
        targets.append(high_target)
    
    # 策略3：抬高+向后撤（抬高同时往X正方向移动，远离工作台）
    for i in range(2):
        retreat_target = current_pos.copy()
        retreat_target[0] = np.clip(current_pos[0] + 0.2, X_MIN, X_MAX)  # 向后退20cm
        retreat_target[2] = Z_MAX * 0.85  # 抬到较高位置（85%高度）
        targets.append(retreat_target)
    
    return targets


def generate_workspace_exploration_targets(current_pos, obstacle_ids):
    """生成工作空间探索目标（优化版：减少候选点，增加探索距离，增强Z轴探索）"""
    X_MIN = WORKSPACE_LIMITS["X_MIN"]
    X_MAX = WORKSPACE_LIMITS["X_MAX"]
    Y_MIN = WORKSPACE_LIMITS["Y_MIN"]
    Y_MAX = WORKSPACE_LIMITS["Y_MAX"]
    Z_MIN = WORKSPACE_LIMITS["Z_MIN"]
    Z_MAX = WORKSPACE_LIMITS["Z_MAX"]
    
    targets = []
    
    # 减少到5个点，但增加探索距离，增强Z轴探索
    for i in range(5):
        random_target = np.array([
            np.random.uniform(X_MIN, X_MAX),
            np.random.uniform(Y_MIN, Y_MAX),
            np.random.uniform(Z_MIN, Z_MAX)
        ])
        
        # 80%概率生成远离当前位置的点（增加概率和距离，包括Z轴）
        if np.random.random() > 0.2:
            offset_direction = random_target - current_pos
            offset_norm = np.linalg.norm(offset_direction)
            if offset_norm > 0 and offset_norm < 0.4:
                # 增加探索距离到0.4-0.7米（3D空间）
                offset_direction = offset_direction / offset_norm * np.random.uniform(0.4, 0.7)
                
                # 特别加强Z轴的探索：50%的情况额外增加Z轴偏移
                if i % 2 == 0:
                    offset_direction[2] += np.random.uniform(0.2, 0.5)  # 额外向上探索
                
                random_target = current_pos + offset_direction
                random_target[0] = np.clip(random_target[0], X_MIN, X_MAX)
                random_target[1] = np.clip(random_target[1], Y_MIN, Y_MAX)
                random_target[2] = np.clip(random_target[2], Z_MIN, Z_MAX)
        targets.append(random_target)
    
    return targets


def generate_obstacle_avoidance_targets(current_pos, obstacle_ids):
    """生成远离障碍物的探索目标（优化版：减少到2个点，增强Z轴探索）"""
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
        # 计算3D逃离方向（包含Z轴）
        if np.linalg.norm(escape_direction) > 0:
            escape_direction = escape_direction / np.linalg.norm(escape_direction)
            
            # 只生成2个远离点（中等和远距离），增强Z轴变化
            for i, dist in enumerate([0.5, 0.8]):
                escape_target = current_pos + escape_direction * dist
                # 大幅增加Z轴的探索范围：从-0.1~0.4改为0.1~0.6
                # 第一个点向上较少，第二个点向上更多
                z_offset = np.random.uniform(0.1, 0.4) if i == 0 else np.random.uniform(0.3, 0.6)
                escape_target[2] = current_pos[2] + z_offset
                
                escape_target[0] = np.clip(escape_target[0], X_MIN, X_MAX)
                escape_target[1] = np.clip(escape_target[1], Y_MIN, Y_MAX)
                escape_target[2] = np.clip(escape_target[2], Z_MIN, Z_MAX)
                
                targets.append(escape_target)
    
    return targets


def generate_height_level_targets(current_pos):
    """生成多层级高度探索目标（优化版：增加到4个不同高度层级）"""
    X_MIN = WORKSPACE_LIMITS["X_MIN"]
    X_MAX = WORKSPACE_LIMITS["X_MAX"]
    Y_MIN = WORKSPACE_LIMITS["Y_MIN"]
    Y_MAX = WORKSPACE_LIMITS["Y_MAX"]
    Z_MIN = WORKSPACE_LIMITS["Z_MIN"]
    Z_MAX = WORKSPACE_LIMITS["Z_MAX"]
    
    targets = []
    
    # 增加更多高度层级的探索：最高点、较高、中等、较低
    for z_level in [Z_MAX, Z_MAX * 0.75, Z_MAX * 0.5, Z_MAX * 0.35]:
        high_target = current_pos.copy()
        high_target[2] = z_level
        # 增加XY方向的探索范围
        high_target[0] += np.random.uniform(-0.4, 0.4)
        high_target[1] += np.random.uniform(-0.4, 0.4)
        high_target[0] = np.clip(high_target[0], X_MIN, X_MAX)
        high_target[1] = np.clip(high_target[1], Y_MIN, Y_MAX)
        targets.append(high_target)
    
    return targets


def generate_safe_retreat_targets():
    """生成安全撤退位置（优化版：增加不同高度的撤退点）"""
    return [
        np.array([0.4, 0.0, 0.7]),    # 高位中央
        np.array([0.4, 0.35, 0.6]),   # 中高位右侧
        np.array([0.4, -0.35, 0.6]),  # 中高位左侧
        np.array([0.4, 0.0, 0.45]),   # 中低位中央
        np.array([0.3, 0.3, 0.8]),    # 更高位对角
    ]


def generate_spiral_targets(current_pos):
    """生成螺旋式探索目标（优化版：增加Z轴变化的3D螺旋）"""
    X_MIN = WORKSPACE_LIMITS["X_MIN"]
    X_MAX = WORKSPACE_LIMITS["X_MAX"]
    Y_MIN = WORKSPACE_LIMITS["Y_MIN"]
    Y_MAX = WORKSPACE_LIMITS["Y_MAX"]
    Z_MIN = WORKSPACE_LIMITS["Z_MIN"]
    Z_MAX = WORKSPACE_LIMITS["Z_MAX"]
    
    targets = []
    num_spiral_points = 6  # 增加到6个方向，形成更完整的3D螺旋
    
    for i in range(num_spiral_points):
        angle = (2 * np.pi * i) / num_spiral_points
        # 使用中等半径
        radius = 0.5
        spiral_target = current_pos.copy()
        spiral_target[0] += radius * np.cos(angle)
        spiral_target[1] += radius * np.sin(angle)
        # 大幅增加Z轴变化范围，形成真正的3D螺旋：从0.1-0.3改为0.2-0.6
        # 随着角度增加，高度也逐渐增加
        z_increment = 0.2 + (i / num_spiral_points) * 0.4  # 0.2到0.6的渐进变化
        spiral_target[2] += z_increment
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
    
    print(f"  >> 尝试目标 {idx+1}: [X={target_pos[0]:.2f}, Y={target_pos[1]:.2f}, Z={target_pos[2]:.2f}]", end=" ")
    
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
            print("✓ 路径安全，移动中...")
            # 提高探索速度到3.0
            success = move_to_joints(robot_id, target_joints, max_velocity=3.0, **kwargs)
            
            if success:
                return True
            else:
                print(f"       ✗ 移动执行失败")
        else:
            print("✗ 路径碰撞")
    
    except Exception as e:
        print(f"✗ IK失败")
    
    return False


def try_joint_space_exploration(robot_id, obstacle_ids, **kwargs):
    """尝试关节空间随机移动（优化版：减少到3次尝试，增加速度）"""
    from motion_control import move_to_joints
    
    print("\n  >> 尝试关节空间随机探索...")
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
            print(f"     尝试 {attempt+1}/3: 安全，移动中...")
            # 提高速度到2.0
            success = move_to_joints(robot_id, target_joints, max_velocity=2.0, timeout=4, **kwargs)
            if success:
                return True
            else:
                print(f"     尝试 {attempt+1}/3: 移动失败")
        else:
            print(f"     尝试 {attempt+1}/3: 关节配置碰撞")
    
    return False

