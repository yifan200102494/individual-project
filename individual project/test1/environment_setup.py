"""
环境设置模块
负责初始化PyBullet环境，加载所有物体（机械臂、托盘、小方块）
"""

import pybullet as p
import pybullet_data
import numpy as np
from obstacle_controller import ObstacleArmController
from config import (
    ROBOT_BASE_POSITION, ROBOT_BASE_ORIENTATION,
    OBJECT_POSITION, OBJECT_ORIENTATION,
    TRAY_POSITION, TRAY_ORIENTATION,
    OBSTACLE_ARM_POSITION, OBSTACLE_ARM_YAW, OBSTACLE_ARM_SCALE,
    FORBIDDEN_ZONE_CENTER, FORBIDDEN_ZONE_RADIUS,
    CAMERA_DISTANCE, CAMERA_YAW, CAMERA_PITCH, CAMERA_TARGET,
    OBSTACLE_ARM_COLOR, GRAVITY
)


def setup_environment():
    """
    初始化PyBullet环境，加载所有物体
    
    Returns:
        tuple: (robotId, objectId, trayId, obstacleArmId, obstacle_controller)
    """
    # 连接到PyBullet GUI
    p.connect(p.GUI)
    
    # 配置可视化器
    p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)
    p.resetDebugVisualizerCamera(
        cameraDistance=CAMERA_DISTANCE,
        cameraYaw=CAMERA_YAW,
        cameraPitch=CAMERA_PITCH,
        cameraTargetPosition=CAMERA_TARGET
    )
    
    # 设置重力
    p.setGravity(*GRAVITY)
    
    # 添加PyBullet数据路径（包含URDF文件）
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    
    # 加载地面
    p.loadURDF('plane.urdf')
    
    # 加载机械臂 (Franka Panda)
    robotId = p.loadURDF(
        'franka_panda/panda.urdf',
        basePosition=ROBOT_BASE_POSITION,
        baseOrientation=ROBOT_BASE_ORIENTATION,
        useFixedBase=True
    )
    
    # 加载小方块（要被抓取的物体）
    objectId = p.loadURDF(
        'cube_small.urdf',
        basePosition=OBJECT_POSITION,
        baseOrientation=OBJECT_ORIENTATION
    )
    
    # 加载托盘（目标区域）
    trayId = p.loadURDF(
        'tray/traybox.urdf',
        basePosition=TRAY_POSITION,
        baseOrientation=TRAY_ORIENTATION
    )
    
    # ========== 加载障碍臂 ==========
    # 加载障碍臂 (Kuka) - 缩放到与Franka Panda相似的大小
    obstacleArmId = p.loadURDF(
        "kuka_iiwa/model.urdf",
        basePosition=OBSTACLE_ARM_POSITION,
        baseOrientation=p.getQuaternionFromEuler([0, 0, OBSTACLE_ARM_YAW]),
        useFixedBase=True,
        globalScaling=OBSTACLE_ARM_SCALE
    )
    
    # 给障碍臂上色（红色，表示障碍物）
    num_joints = p.getNumJoints(obstacleArmId)
    for j in range(-1, num_joints):  # -1 是基座
        p.changeVisualShape(obstacleArmId, j, rgbaColor=OBSTACLE_ARM_COLOR)
    
    # 设置障碍臂的初始姿态
    for i in range(num_joints):
        joint_info = p.getJointInfo(obstacleArmId, i)
        if joint_info[2] == p.JOINT_REVOLUTE:  # 如果是可旋转关节
            # 设置到关节范围的中间位置
            lower_limit = joint_info[8]
            upper_limit = joint_info[9]
            mid_position = (lower_limit + upper_limit) / 2.0
            p.resetJointState(obstacleArmId, i, mid_position)
    # ========================================
    
    # 设置机械臂初始姿态（Home位置）
    home_joint_positions = [0.0, -0.785, 0.0, -2.356, 0.0, 1.57, 0.785]
    for i in range(7):  # Panda有7个手臂关节
        p.resetJointState(robotId, i, home_joint_positions[i])
    
    # 设置夹爪初始状态（打开）
    gripper_indices = [9, 10]
    gripper_open_position = 0.04
    for i in gripper_indices:
        p.resetJointState(robotId, i, gripper_open_position)
    
    print("=" * 60)
    print("环境设置完成")
    print("=" * 60)
    print(f"机械臂ID: {robotId}")
    print(f"小方块ID: {objectId}")
    print(f"托盘ID: {trayId}")
    print(f"障碍臂ID: {obstacleArmId}")  # 恢复
    print("=" * 60)
    
    # ========== 创建障碍臂控制器 ==========
    # 创建障碍臂控制器，避让主机械臂的工作空间
    obstacle_controller = ObstacleArmController(
        obstacle_arm_id=obstacleArmId,
        main_robot_id=robotId,
        forbidden_zone_center=FORBIDDEN_ZONE_CENTER,
        forbidden_zone_radius=FORBIDDEN_ZONE_RADIUS
    )
    print("=" * 60)
    # =======================================
    
    # 返回所有需要的对象
    return robotId, objectId, trayId, obstacleArmId, obstacle_controller