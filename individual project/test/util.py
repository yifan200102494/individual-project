import pybullet as p
import time
import numpy as np

# --- 常量 ---
JOINT_TYPES = ["REVOLUTE", "PRISMATIC", "SPHERICAL", "PLANAR", "FIXED"]
ROBOT_HOME_CONFIG = [0.0, -0.7854, 0.0, -2.3562, 0.0, 1.5708, 0.7854]
ROBOT_END_EFFECTOR_LINK_ID = 8
DELTA_T = 1./240

# --- 【最终版】可靠的碰撞检测函数 ---
def is_state_colliding(robot_id, joint_pos, obstacle_id):
    """
    使用 save/restore State 机制来可靠地检查碰撞，避免副作用。
    """
    # 保存当前世界的完整状态
    state_id = p.saveState()
    
    # 将机器人“传送”到要检查的姿态
    for i in range(len(joint_pos)):
        p.resetJointState(robot_id, i, joint_pos[i])
        
    # 必须执行一步仿真来更新物理引擎中各个连杆的最终位置
    # 因为 restoreState 会撤销这一步的所有影响，所以这是安全的
    p.stepSimulation() 
    
    # 检查机器人与障碍物之间的接触点
    is_colliding = False
    contacts = p.getContactPoints(bodyA=robot_id, bodyB=obstacle_id)
    if len(contacts) > 0:
        is_colliding = True
        
    # 恢复到原始世界状态
    p.restoreState(state_id)
    p.removeState(state_id) # 清理内存
    
    return is_colliding

# --- 【新增】动态路径规划与执行函数 ---
def plan_and_execute_motion(robot_id, goal_pos, goal_orn, obstacle_id):
    """
    规划并执行从当前位置到目标位置的运动，会主动规避障碍物。
    """
    print(f"--- 正在规划前往 {goal_pos} 的路径 ---")
    
    # 1. 检查直接路径
    print("  1. 检查直接路径...")
    target_joints = p.calculateInverseKinematics(robot_id, ROBOT_END_EFFECTOR_LINK_ID, goal_pos, goal_orn)
    
    if not is_state_colliding(robot_id, target_joints[:7], obstacle_id):
        print("  >> 直接路径安全，正在执行...")
        move_to_joints(robot_id, target_joints[:7])
        return True # 成功
    
    # 2. 如果直接路径被阻挡，则尝试规划备用路径
    print("  >> 直接路径被阻挡，正在规划备用路径...")
    current_pos, _, _, _ = p.getLinkState(robot_id, ROBOT_END_EFFECTOR_LINK_ID, computeForwardKinematics=True)
    safe_height = 0.45 # 定义一个比障碍物更高的安全高度

    # 备用路径点1: 从当前位置垂直向上
    waypoint1 = [current_pos[0], current_pos[1], safe_height]
    print(f"     2a. 检查备用路径点1: {waypoint1}")
    waypoint1_joints = p.calculateInverseKinematics(robot_id, ROBOT_END_EFFECTOR_LINK_ID, waypoint1, goal_orn)
    if is_state_colliding(robot_id, waypoint1_joints[:7], obstacle_id):
        print("  >> 备用路径规划失败：向上移动时发生碰撞。")
        return False # 失败

    # 备用路径点2: 在安全高度水平移动到目标点上方
    waypoint2 = [goal_pos[0], goal_pos[1], safe_height]
    print(f"     2b. 检查备用路径点2: {waypoint2}")
    waypoint2_joints = p.calculateInverseKinematics(robot_id, ROBOT_END_EFFECTOR_LINK_ID, waypoint2, goal_orn)
    if is_state_colliding(robot_id, waypoint2_joints[:7], obstacle_id):
        print("  >> 备用路径规划失败：水平移动时发生碰撞。")
        return False # 失败
        
    # 备用路径的最后一步（从waypoint2到最终目标点）是安全的，因为我们在最开始已经检查过最终目标点了
    
    print("  >> 备用路径安全，正在执行...")
    move_to_joints(robot_id, waypoint1_joints[:7])
    move_to_joints(robot_id, waypoint2_joints[:7])
    move_to_joints(robot_id, target_joints[:7])
    
    return True # 成功


# --- 核心函数 ---
# ... (之前的函数保持不变) ...
def simulate(steps=None, seconds=None, slow_down=True):
    """
    Wraps pybullet's stepSimulation function.
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

def move_to_joints(robot_id, target_joint_pos, max_velocity=1, timeout=5):
    """
    Moves the robot to a given joint position.
    """
    target_joint_pos = np.asarray(target_joint_pos)
    
    # set control
    for joint_id in range(len(target_joint_pos)):
        p.setJointMotorControl2(
            robot_id,
            joint_id,
            controlMode=p.POSITION_CONTROL,
            targetPosition=target_joint_pos[joint_id],
            maxVelocity=max_velocity,
            force=100
        )

    # loop and check
    counter = 0
    while True:
        current_joint_pos = np.asarray([p.getJointState(robot_id, i)[0] for i in range(7)])
        if np.allclose(current_joint_pos, target_joint_pos, atol=0.005):
            break
        
        simulate(steps=1)
        counter += 1
        if counter > timeout / DELTA_T:
            print('WARNING: timeout while moving to joint position.')
            break

def gripper_open(robot_id):
    """
    Opens the gripper of the robot.
    """
    p.setJointMotorControl2(robot_id, 9, controlMode=p.POSITION_CONTROL, targetPosition=0.04, force=100)
    p.setJointMotorControl2(robot_id, 10, controlMode=p.POSITION_CONTROL, targetPosition=0.04, force=100)
    simulate(seconds=1.0)

def gripper_close(robot_id):
    """
    Closes the gripper of the robot.
    """
    p.setJointMotorControl2(robot_id, 9, controlMode=p.VELOCITY_CONTROL, targetVelocity=-0.05, force=100)
    
    # --- 【重要改动】将等待时间缩短为原来的三分之一 (1.5s -> 0.5s) ---
    for _ in range(int(0.5 / DELTA_T)):
        simulate(steps=1)
        finger_pos = p.getJointState(robot_id, 9)[0]
        p.setJointMotorControl2(robot_id, 10, controlMode=p.POSITION_CONTROL, targetPosition=finger_pos, force=100)

def move_to_pose(robot_id, target_ee_pos, target_ee_orientation=None):
    """
    Moves the robot to a given end-effector pose.
    """
    if target_ee_orientation is None:
        joint_pos_all = p.calculateInverseKinematics(
            robot_id,
            ROBOT_END_EFFECTOR_LINK_ID,
            targetPosition=target_ee_pos,
            maxNumIterations=100,
            residualThreshold=0.001
        )
    else:
        joint_pos_all = p.calculateInverseKinematics(
            robot_id,
            ROBOT_END_EFFECTOR_LINK_ID,
            targetPosition=target_ee_pos,
            targetOrientation=target_ee_orientation,
            maxNumIterations=100,
            residualThreshold=0.001
        )

    joint_pos_arm = list(joint_pos_all[0:7])
    move_to_joints(robot_id, joint_pos_arm)



