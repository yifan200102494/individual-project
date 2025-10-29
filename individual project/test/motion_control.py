"""
运动控制模块
包含仿真、关节运动、夹爪控制等功能
"""

import pybullet as p
import time
import numpy as np
import random

from constants import DELTA_T, ROBOT_END_EFFECTOR_LINK_ID, PROXIMITY_FAILSAFE_DISTANCE
from collision_detection import is_state_colliding


# 全局仿真步数计数器
_GLOBAL_SIM_STEP_COUNTER = 0


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

def move_to_joints(robot_id, target_joint_pos, max_velocity=1, timeout=5, **kwargs):
    """
    移动到目标关节位置，包含碰撞检测
    
    Args:
        robot_id: 机器人ID
        target_joint_pos: 目标关节位置
        max_velocity: 最大速度
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
    while True:
        # Failsafe 1: 基于感知的碰撞检测
        if obstacle_ids:
            current_joint_pos_check = np.asarray([p.getJointState(robot_id, i)[0] for i in range(num_arm_joints)])
            current_gripper_pos_check = [p.getJointState(robot_id, 9)[0], p.getJointState(robot_id, 10)[0]]
            
            if is_state_colliding(robot_id, current_joint_pos_check, obstacle_ids, current_gripper_pos_check):
                print("  [!!] 执行时碰撞检测 (基于感知)")
                print("  [!!] 立即停止机器人...")
                
                for joint_id in range(num_arm_joints):
                    p.setJointMotorControl2(
                        robot_id, joint_id, controlMode=p.VELOCITY_CONTROL,
                        targetVelocity=0, force=200
                    )
                simulate(steps=1, **sim_kwargs)
                return False
        
        # Failsafe 2: 基于真实距离的近距离保护
        if interferer_id is not None:
            closest_points = p.getClosestPoints(robot_id, interferer_id, PROXIMITY_FAILSAFE_DISTANCE)
            
            if closest_points:
                print(f"  [!!] 执行时近距离保护 (< {PROXIMITY_FAILSAFE_DISTANCE}m)")
                print("  [!!] 立即停止机器人...")
                
                for joint_id in range(num_arm_joints):
                    p.setJointMotorControl2(
                        robot_id, joint_id, controlMode=p.VELOCITY_CONTROL,
                        targetVelocity=0, force=200
                    )
                simulate(steps=1, **sim_kwargs)
                return False
        
        # 设置电机目标
        for joint_id in range(num_arm_joints):
            p.setJointMotorControl2(
                robot_id, joint_id, controlMode=p.POSITION_CONTROL,
                targetPosition=target_joint_pos[joint_id],
                maxVelocity=max_velocity, force=100
            )

        # 检查是否到达目标
        current_joint_pos = np.asarray([p.getJointState(robot_id, i)[0] for i in range(num_arm_joints)])
        
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

