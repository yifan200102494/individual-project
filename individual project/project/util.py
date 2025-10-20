import time
import pybullet as p
import numpy as np

# --- 常量: 与原始 util.py 和 test.py 完全一致 ---
ROBOT_HOME_CONFIG = [0.0, -0.7854, 0.0, -2.3562, 0.0, 1.5708, 0.7854]
ROBOT_END_EFFECTOR_LINK_ID = 8 # 正确的IK计算link
DELTA_T = 1./240

# --- 【重要修正】从您上传的文件中完整复制的 simulate 函数 ---
def simulate(steps=None, seconds=None, slow_down=True):
    """
    包装了pybullet的stepSimulation函数，允许更多控制。
    """
    seconds_passed = 0.0
    steps_passed = 0
    start_time = time.time()

    while True:
        p.stepSimulation()
        steps_passed += 1
        seconds_passed += DELTA_T

        if slow_down:
            time_elapsed = time.time() - start_time
            wait_time = seconds_passed - time_elapsed
            time.sleep(max(wait_time, 0))
        if steps is not None and steps_passed >= steps:
            break
        if seconds is not None and seconds_passed >= seconds:
            break

def get_arm_joint_pos(robot_id):
    """获取机械臂7个关节的当前位置"""
    return [p.getJointState(robot_id, i)[0] for i in range(7)]

def move_to_joints(robot_id, target_joint_pos, max_velocity=1, timeout=5):
    """移动机器人到指定关节位置 (移植自 util.py)"""
    target_joint_pos = np.asarray(target_joint_pos)
    
    # 设置电机控制
    for joint_id in range(len(target_joint_pos)):
        p.setJointMotorControl2(
            robot_id, joint_id, p.POSITION_CONTROL,
            targetPosition=target_joint_pos[joint_id],
            maxVelocity=max_velocity, force=100
        )

    # 等待直到到达目标或超时
    counter = 0
    while not np.allclose(get_arm_joint_pos(robot_id), target_joint_pos, atol=0.01):
        simulate(steps=1)
        counter += 1
        if counter > timeout / DELTA_T:
            print('WARNING: move_to_joints 超时')
            break

def gripper_open(robot_id):
    """打开夹爪 (移植自 util.py)"""
    p.setJointMotorControl2(robot_id, 9, p.POSITION_CONTROL, targetPosition=0.04, force=100)
    p.setJointMotorControl2(robot_id, 10, p.POSITION_CONTROL, targetPosition=0.04, force=100)
    simulate(seconds=1)

def gripper_close(robot_id):
    """关闭夹爪 (移植自 util.py 的速度控制版本)"""
    p.setJointMotorControl2(robot_id, 9, p.VELOCITY_CONTROL, targetVelocity=-0.05, force=100)
    
    for _ in range(int(0.5 / DELTA_T)):
        simulate(steps=1)
        finger_pos = p.getJointState(robot_id, 9)[0]
        p.setJointMotorControl2(robot_id, 10, p.POSITION_CONTROL, targetPosition=finger_pos, force=100)

def move_to_pose(robot_id, target_ee_pos, target_ee_orientation):
    """移动末端执行器到指定位姿 (移植自 test.py)"""
    joint_pos_all = p.calculateInverseKinematics(
        robot_id, ROBOT_END_EFFECTOR_LINK_ID,
        targetPosition=target_ee_pos,
        targetOrientation=target_ee_orientation,
        maxNumIterations=100,
        residualThreshold=0.001
    )
    move_to_joints(robot_id, list(joint_pos_all[0:7]))

