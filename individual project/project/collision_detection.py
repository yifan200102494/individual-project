"""
碰撞检测模块
提供状态碰撞检测和路径碰撞检测功能
"""

import pybullet as p
import numpy as np


def is_state_colliding(robot_id, joint_pos, obstacle_ids, gripper_pos):
    """
    检查给定状态是否碰撞
    
    Args:
        robot_id: 机器人ID
        joint_pos: 关节位置
        obstacle_ids: 障碍物ID列表
        gripper_pos: 夹爪位置 [左指, 右指]
    
    Returns:
        bool: 是否发生碰撞
    """
    state_id = p.saveState()
    
    # 设置机器人到指定状态
    for i in range(len(joint_pos)):
        p.resetJointState(robot_id, i, joint_pos[i])
    p.resetJointState(robot_id, 9, gripper_pos[0])
    p.resetJointState(robot_id, 10, gripper_pos[1])
    
    p.performCollisionDetection()
    
    # 检查与每个障碍物的碰撞
    is_colliding = False
    for obstacle_id in obstacle_ids:
        contacts = p.getContactPoints(bodyA=robot_id, bodyB=obstacle_id)
        if len(contacts) > 0:
            is_colliding = True
            break
    
    # 恢复原始状态
    p.restoreState(state_id)
    p.removeState(state_id)
    
    return is_colliding


def is_path_colliding(robot_id, start_joints, end_joints, obstacle_ids,
                      start_gripper_pos, end_gripper_pos, num_steps=25):
    """
    检查关节空间路径是否碰撞
    
    Args:
        robot_id: 机器人ID
        start_joints: 起始关节位置
        end_joints: 结束关节位置
        obstacle_ids: 障碍物ID列表
        start_gripper_pos: 起始夹爪位置
        end_gripper_pos: 结束夹爪位置
        num_steps: 插值步数
    
    Returns:
        bool: 路径是否碰撞
    """
    start_joints = np.asarray(start_joints)
    end_joints = np.asarray(end_joints)
    start_gripper_pos = np.asarray(start_gripper_pos)
    end_gripper_pos = np.asarray(end_gripper_pos)

    # 关闭渲染以提高性能
    p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 0)
    
    path_is_colliding = False
    for i in range(num_steps + 1):
        alpha = i / num_steps
        interpolated_joints = (1 - alpha) * start_joints + alpha * end_joints
        interpolated_gripper = (1 - alpha) * start_gripper_pos + alpha * end_gripper_pos
        
        if is_state_colliding(robot_id, interpolated_joints, obstacle_ids, interpolated_gripper):
            path_is_colliding = True
            break
    
    # 恢复渲染
    p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 1)
    
    return path_is_colliding

