"""
运动控制模块
包含仿真、关节运动、夹爪控制等功能
"""

import pybullet as p
import time
import numpy as np
import random

from constants import DELTA_T, ROBOT_END_EFFECTOR_LINK_ID, PROXIMITY_FAILSAFE_DISTANCE, WORKSPACE_LIMITS
from collision_detection import is_state_colliding
from path_planning import calc_anisotropic_repulsive_force, prepare_obstacles_info


# 全局仿真步数计数器
_GLOBAL_SIM_STEP_COUNTER = 0


# ============================================================
# 辅助函数：计算逃离方向
# ============================================================

def _compute_escape_direction(robot_id, obstacle_ids):
    """
    计算机械臂应该逃离的方向（基于PFM斥力）
    
    Args:
        robot_id: 机器人ID
        obstacle_ids: 障碍物ID列表
    
    Returns:
        np.array: 逃离方向的单位向量，如果无法计算则返回None
    """
    try:
        # 获取当前末端执行器位置
        ee_state = p.getLinkState(robot_id, ROBOT_END_EFFECTOR_LINK_ID, computeForwardKinematics=True)
        current_pos = np.array(ee_state[0])
        
        # 准备障碍物信息
        obstacles_info = prepare_obstacles_info(obstacle_ids)
        
        # 计算总斥力
        f_rep_total = np.array([0.0, 0.0, 0.0])
        k_rep = 2.0  # 更强的斥力系数
        
        for obs in obstacles_info:
            rho_0 = (obs["diag"] / 2.0) + 0.35
            f_rep_obs = calc_anisotropic_repulsive_force(
                current_pos, obs["center"], obs["aabb_min"], obs["aabb_max"],
                k_rep=k_rep, rho_0=rho_0, k_aniso_xy=2.0, k_aniso_z=1.0  # 增加z方向的斥力
            )
            f_rep_total += f_rep_obs
        
        # 如果斥力太小，添加一个随机的向上逃离方向
        if np.linalg.norm(f_rep_total) < 0.01:
            # 随机方向 + 向上偏好
            random_dir = np.random.randn(3)
            random_dir[2] = abs(random_dir[2]) + 0.5  # 偏向向上
            return random_dir / np.linalg.norm(random_dir)
        
        # 归一化斥力方向
        escape_direction = f_rep_total / np.linalg.norm(f_rep_total)
        
        return escape_direction
    except Exception as e:
        print(f"  [!] 计算逃离方向失败: {e}")
        return None


def _move_to_safety(robot_id, obstacle_ids, escape_distance=0.15, max_steps=20, sim_kwargs=None):
    """
    将机械臂移动到安全位置（远离障碍物）
    
    Args:
        robot_id: 机器人ID
        obstacle_ids: 障碍物ID列表
        escape_distance: 逃离距离（米）
        max_steps: 最大步数
        sim_kwargs: 仿真参数
    
    Returns:
        bool: 是否成功移动到安全位置
    """
    if sim_kwargs is None:
        sim_kwargs = {"slow_down": True}
    
    print(f"  >> [PFM逃离] 计算斥力方向并主动后退...")
    
    # 计算逃离方向
    escape_direction = _compute_escape_direction(robot_id, obstacle_ids)
    
    if escape_direction is None:
        print(f"  [!] 无法计算逃离方向，尝试随机向上移动")
        escape_direction = np.array([0, 0, 1.0])
    
    # 获取当前末端执行器位置和方向
    ee_state = p.getLinkState(robot_id, ROBOT_END_EFFECTOR_LINK_ID, computeForwardKinematics=True)
    current_pos = np.array(ee_state[0])
    current_orn = ee_state[1]
    
    # 计算目标位置
    target_pos = current_pos + escape_direction * escape_distance
    
    # 应用工作空间限制
    target_pos[0] = np.clip(target_pos[0], WORKSPACE_LIMITS["X_MIN"], WORKSPACE_LIMITS["X_MAX"])
    target_pos[1] = np.clip(target_pos[1], WORKSPACE_LIMITS["Y_MIN"], WORKSPACE_LIMITS["Y_MAX"])
    target_pos[2] = np.clip(target_pos[2], WORKSPACE_LIMITS["Z_MIN"], WORKSPACE_LIMITS["Z_MAX"])
    
    print(f"  >> 逃离方向: {escape_direction}, 目标位置: {target_pos}")
    
    try:
        # 计算目标关节位置
        target_joints = p.calculateInverseKinematics(
            robot_id, ROBOT_END_EFFECTOR_LINK_ID,
            target_pos, current_orn,
            maxNumIterations=100, residualThreshold=0.001
        )[:7]
        
        # 执行移动
        for step in range(max_steps):
            # 设置关节目标
            for joint_id in range(7):
                p.setJointMotorControl2(
                    robot_id, joint_id,
                    controlMode=p.POSITION_CONTROL,
                    targetPosition=target_joints[joint_id],
                    maxVelocity=1.5,  # 较快的速度
                    force=150
                )
            
            simulate(steps=1, **sim_kwargs)
            
            # 检查是否到达目标
            current_joint_pos = np.asarray([p.getJointState(robot_id, i)[0] for i in range(7)])
            if np.allclose(current_joint_pos, target_joints, atol=0.05):
                print(f"  ✅ [PFM逃离] 成功移动到安全位置")
                return True
        
        print(f"  ⚠️ [PFM逃离] 部分完成，但未完全到达目标位置")
        return True  # 即使没完全到达，也认为成功（至少移动了一段距离）
        
    except Exception as e:
        print(f"  [!] 移动到安全位置失败: {e}")
        return False


# ============================================================
# 仿真控制
# ============================================================

def simulate(steps=None, seconds=None, slow_down=True,
             interferer_id=None, interferer_joints=None, interferer_update_rate=120):
    """
    执行仿真步进
    
    Args:
        steps: 执行的步数
        seconds: 执行的秒数
        slow_down: 是否实时减速
        interferer_id: 干扰物体ID
        interferer_joints: 干扰关节列表
        interferer_update_rate: 干扰更新频率
    """
    global _GLOBAL_SIM_STEP_COUNTER
    
    seconds_passed = 0.0
    steps_this_call = 0
    start_time = time.time()
    
    while True:
        p.stepSimulation()
        
        _GLOBAL_SIM_STEP_COUNTER += 1
        steps_this_call += 1
        
        # 干扰物体运动
        if interferer_id is not None and interferer_joints is not None:
            if _GLOBAL_SIM_STEP_COUNTER % interferer_update_rate == 0:
                joint_to_move = random.choice(interferer_joints)
                joint_info = p.getJointInfo(interferer_id, joint_to_move)
                joint_min = joint_info[8]
                joint_max = joint_info[9]
                target_pos = random.uniform(joint_min, joint_max)
                p.setJointMotorControl2(
                    bodyUniqueId=interferer_id,
                    jointIndex=joint_to_move,
                    controlMode=p.POSITION_CONTROL,
                    targetPosition=target_pos,
                    maxVelocity=1.5,
                    force=100
                )

        seconds_passed += DELTA_T
        if slow_down:
            time_elapsed = time.time() - start_time
            wait_time = seconds_passed - time_elapsed
            time.sleep(max(wait_time, 0))
        
        if steps is not None and steps_this_call >= steps:
            break
        if seconds is not None and seconds_passed >= seconds:
            break


# ============================================================
# 关节运动控制
# ============================================================

def move_to_joints(robot_id, target_joint_pos, max_velocity=2.0, timeout=5, **kwargs):
    """
    移动到目标关节位置，包含碰撞检测和平滑速度控制
    
    Args:
        robot_id: 机器人ID
        target_joint_pos: 目标关节位置
        max_velocity: 最大速度（提高到2.0以增加连贯性）
        timeout: 超时时间（秒）
        **kwargs: 其他参数
    
    Returns:
        bool: 是否成功到达目标
    """
    
    # 提取仿真参数
    sim_kwargs = {
        "interferer_id": kwargs.get("interferer_id"),
        "interferer_joints": kwargs.get("interferer_joints"),
        "interferer_update_rate": kwargs.get("interferer_update_rate", 120),
        "slow_down": kwargs.get("slow_down", True)
    }
    
    interferer_id = kwargs.get("interferer_id")
    obstacle_ids = kwargs.get("obstacle_ids", [])
    
    target_joint_pos = np.asarray(target_joint_pos)
    num_arm_joints = len(target_joint_pos)
    
    counter = 0
    collision_count = 0  # 碰撞计数器
    max_collision_retries = 3  # 最大重试次数
    
    while True:
        # Failsafe 1: 基于感知的碰撞检测
        if obstacle_ids:
            current_joint_pos_check = np.asarray([p.getJointState(robot_id, i)[0] for i in range(num_arm_joints)])
            current_gripper_pos_check = [p.getJointState(robot_id, 9)[0], p.getJointState(robot_id, 10)[0]]
            
            if is_state_colliding(robot_id, current_joint_pos_check, obstacle_ids, current_gripper_pos_check):
                print("  [!!] 执行时碰撞检测 (基于感知)")
                print("  [!!] 启动PFM逃离模式，主动远离障碍物...")
                
                # 首先停止当前运动
                for joint_id in range(num_arm_joints):
                    p.setJointMotorControl2(
                        robot_id, joint_id, controlMode=p.VELOCITY_CONTROL,
                        targetVelocity=0, force=200
                    )
                simulate(steps=2, **sim_kwargs)
                
                # 调用PFM逃离函数
                escape_success = _move_to_safety(robot_id, obstacle_ids, escape_distance=0.2, 
                                                 max_steps=30, sim_kwargs=sim_kwargs)
                
                if escape_success:
                    collision_count += 1
                    if collision_count >= max_collision_retries:
                        print(f"  [!!] 碰撞次数过多({collision_count})，放弃当前目标")
                        return False
                    # 逃离成功后继续尝试
                    print(f"  >> 逃离成功，继续尝试到达目标 (重试 {collision_count}/{max_collision_retries})")
                    simulate(steps=5, **sim_kwargs)
                    continue
                else:
                    print(f"  [!!] 无法逃离，放弃运动")
                    return False
        
        # Failsafe 2: 基于真实距离的近距离保护
        if interferer_id is not None:
            closest_points = p.getClosestPoints(robot_id, interferer_id, PROXIMITY_FAILSAFE_DISTANCE)
            
            if closest_points:
                print(f"  [!!] 执行时近距离保护 (< {PROXIMITY_FAILSAFE_DISTANCE}m)")
                print("  [!!] 启动PFM逃离模式，主动远离干扰物...")
                
                # 首先停止当前运动
                for joint_id in range(num_arm_joints):
                    p.setJointMotorControl2(
                        robot_id, joint_id, controlMode=p.VELOCITY_CONTROL,
                        targetVelocity=0, force=200
                    )
                simulate(steps=2, **sim_kwargs)
                
                # 调用PFM逃离函数（使用interferer_id作为障碍物）
                escape_success = _move_to_safety(robot_id, [interferer_id], escape_distance=0.2,
                                                 max_steps=30, sim_kwargs=sim_kwargs)
                
                if escape_success:
                    collision_count += 1
                    if collision_count >= max_collision_retries:
                        print(f"  [!!] 近距离保护触发次数过多({collision_count})，放弃当前目标")
                        return False
                    # 逃离成功后继续尝试
                    print(f"  >> 逃离成功，继续尝试到达目标 (重试 {collision_count}/{max_collision_retries})")
                    simulate(steps=5, **sim_kwargs)
                    continue
                else:
                    print(f"  [!!] 无法逃离，放弃运动")
                    return False
        
        # 动态调整速度：计算当前距离，距离近时减速
        current_joint_pos = np.asarray([p.getJointState(robot_id, i)[0] for i in range(num_arm_joints)])
        joint_distance = np.linalg.norm(current_joint_pos - target_joint_pos)
        
        # 平滑减速曲线
        if joint_distance < 0.3:
            adaptive_velocity = max_velocity * max(0.4, joint_distance / 0.3)
        else:
            adaptive_velocity = max_velocity
        
        # 设置电机目标（使用自适应速度）
        for joint_id in range(num_arm_joints):
            p.setJointMotorControl2(
                robot_id, joint_id, controlMode=p.POSITION_CONTROL,
                targetPosition=target_joint_pos[joint_id],
                maxVelocity=adaptive_velocity, force=120
            )

        # 检查是否到达目标
        if np.allclose(current_joint_pos, target_joint_pos, atol=0.01):
            return True

        # 步进仿真和超时
        simulate(steps=1, **sim_kwargs)
        
        counter += 1
        if counter > timeout / DELTA_T:
            print('WARNING: timeout while moving to joint position.')
            return False
    
    return True


def move_to_pose(robot_id, target_ee_pos, target_ee_orientation=None, **kwargs):
    """
    移动到目标末端执行器位姿
    
    Args:
        robot_id: 机器人ID
        target_ee_pos: 目标末端执行器位置
        target_ee_orientation: 目标末端执行器方向（可选）
        **kwargs: 其他参数
    
    Returns:
        bool: 是否成功到达目标
    """
    if target_ee_orientation is None:
        joint_pos_all = p.calculateInverseKinematics(
            robot_id, ROBOT_END_EFFECTOR_LINK_ID, targetPosition=target_ee_pos,
            maxNumIterations=100, residualThreshold=0.001)
    else:
        joint_pos_all = p.calculateInverseKinematics(
            robot_id, ROBOT_END_EFFECTOR_LINK_ID,
            targetPosition=target_ee_pos, targetOrientation=target_ee_orientation,
            maxNumIterations=100, residualThreshold=0.001)
    joint_pos_arm = list(joint_pos_all[0:7])
    
    return move_to_joints(robot_id, joint_pos_arm, **kwargs)


# ============================================================
# 夹爪控制
# ============================================================

def gripper_open(robot_id, **kwargs):
    """
    打开夹爪
    
    Args:
        robot_id: 机器人ID
        **kwargs: 仿真参数
    """
    p.setJointMotorControl2(robot_id, 9, controlMode=p.POSITION_CONTROL, targetPosition=0.04, force=100)
    p.setJointMotorControl2(robot_id, 10, controlMode=p.POSITION_CONTROL, targetPosition=0.04, force=100)
    simulate(seconds=1.0, **kwargs)


def gripper_close(robot_id, **kwargs):
    """
    闭合夹爪
    
    Args:
        robot_id: 机器人ID
        **kwargs: 仿真参数
    """
    p.setJointMotorControl2(robot_id, 9, controlMode=p.VELOCITY_CONTROL, targetVelocity=-0.05, force=100)
    for _ in range(int(0.5 / DELTA_T)):
        simulate(steps=1, **kwargs)
        finger_pos = p.getJointState(robot_id, 9)[0]
        p.setJointMotorControl2(robot_id, 10, controlMode=p.POSITION_CONTROL, targetPosition=finger_pos, force=100)

