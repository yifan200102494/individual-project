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
# --- 【新增】路径碰撞检测函数 ---
def is_path_colliding(robot_id, start_joints, end_joints, obstacle_id, num_steps=10):
    """
    检查从 start_joints 到 end_joints 的整个路径是否与障碍物碰撞。
    """
    start_joints = np.asarray(start_joints)
    end_joints = np.asarray(end_joints)
    
    for i in range(num_steps + 1):
        alpha = i / num_steps
        # 在关节空间中进行线性插值
        interpolated_joints = (1 - alpha) * start_joints + alpha * end_joints
        
        # 检查插值点的状态
        if is_state_colliding(robot_id, interpolated_joints, obstacle_id):
            # 只要路径上有一个点碰撞，就返回 True
            return True
            
    # 走完了所有插值点都没碰撞
    return False
# --- 【新增】备用路径搜索器 ---
def find_safe_avoidance_path(robot_id, start_joints, start_pos, goal_joints, goal_pos, goal_orn, obstacle_id):
    """
    通过搜索不同的 safe_x 坐标，尝试找到一条安全的5段式规避路径。
    这是机器人的“自主计算”逻辑。
    """
    print("     2a. 正在查询障碍物的AABB(包围盒)...")
    aabb_min, aabb_max = p.getAABB(obstacle_id)
    obstacle_top_z = aabb_max[2]
    safe_height = obstacle_top_z + 0.05 # Z轴安全高度
    print(f"     >> 障碍物顶部在 z={obstacle_top_z:.3f}。 动态规划安全高度为: z={safe_height:.3f}")

    # 定义要搜索的安全X坐标 "候选项"
    # 从 0.0 (基座) 开始向两侧搜索，越靠前的路径越短
    safe_x_candidates = [0.0, 0.2, -0.2, 0.3, -0.3]
    
    print(f"     2b. 正在搜索 {len(safe_x_candidates)} 条备用X-Y路径...")

    for safe_x in safe_x_candidates:
        print(f"         - 正在测试 safe_x = {safe_x}...")
        
        # --- 1. 计算航点 ---
        # Waypoint 1: 水平侧移 (在当前高度，移动到安全X坐标)
        waypoint1 = [safe_x, start_pos[1], start_pos[2]]
        # Waypoint 2: 垂直向上 (在安全X，升到安全高度)
        waypoint2 = [safe_x, start_pos[1], safe_height]
        # Waypoint 3: 水平前移 (在安全X和安全高度，移动到目标的Y坐标)
        waypoint3 = [safe_x, goal_pos[1], safe_height]
        # Waypoint 4: 水平对齐 (在安全高度，移动回目标的X坐标)
        waypoint4 = [goal_pos[0], goal_pos[1], safe_height]

        # --- 2. 计算IK (逆运动学) ---
        try:
            waypoint1_joints = p.calculateInverseKinematics(robot_id, ROBOT_END_EFFECTOR_LINK_ID, waypoint1, goal_orn)[:7]
            waypoint2_joints = p.calculateInverseKinematics(robot_id, ROBOT_END_EFFECTOR_LINK_ID, waypoint2, goal_orn)[:7]
            waypoint3_joints = p.calculateInverseKinematics(robot_id, ROBOT_END_EFFECTOR_LINK_ID, waypoint3, goal_orn)[:7]
            waypoint4_joints = p.calculateInverseKinematics(robot_id, ROBOT_END_EFFECTOR_LINK_ID, waypoint4, goal_orn)[:7]
        except Exception as e:
            # 如果这个 safe_x 导致IK无解 (比如机械臂够不到)，则放弃它
            print(f"         - 测试失败: IK无解。")
            continue # 尝试下一个 safe_x

        # --- 3. 检查5段路径的碰撞 ---
        path_segments = [
            (start_joints, waypoint1_joints, "(1) 侧移"),
            (waypoint1_joints, waypoint2_joints, "(2) 向上"),
            (waypoint2_joints, waypoint3_joints, "(3) 向前"),
            (waypoint3_joints, waypoint4_joints, "(4) 对齐"),
            (waypoint4_joints, goal_joints, "(5) 下降")
        ]
        
        path_is_safe = True
        for (js_start, js_end, name) in path_segments:
            if is_path_colliding(robot_id, js_start, js_end, obstacle_id):
                print(f"         - 测试失败: 路径 {name} 在 safe_x = {safe_x} 时碰撞。")
                path_is_safe = False
                break # 停止检查此 safe_x, 尝试下一个
        
        # --- 4. 如果5段都安全，我们就找到了！ ---
        if path_is_safe:
            print(f"     >> 搜索成功！找到安全路径，使用 safe_x = {safe_x}")
            # 返回一个包含所有中间关节目标的列表
            return [waypoint1_joints, waypoint2_joints, waypoint3_joints, waypoint4_joints, goal_joints]

    # --- 5. 如果所有候选项都失败了 ---
    print("     >> 搜索失败: 尝试了所有 safe_x 候选项，均无法找到安全路径。")
    return None
# --- 【新增】动态路径规划与执行函数 ---
# --- 【最终版】动态路径规划与执行函数 ---
# --- 【最终版】动态路径规划与执行函数 ---
# --- 【最终版 v3】动态路径规划与执行函数 ---
# --- 【最终版 v4】动态路径规划与执行函数 ---
def plan_and_execute_motion(robot_id, goal_pos, goal_orn, obstacle_id):
    """
    规划并执行从当前位置到目标位置的运动，会主动规避障碍物。
    【已更新】使用 "find_safe_avoidance_path" 搜索器来动态寻找规避路径。
    """
    print(f"--- 正在规划前往 {goal_pos} 的路径 ---")
    
    # 0. 获取当前状态
    current_joint_pos = np.asarray([p.getJointState(robot_id, i)[0] for i in range(7)])
    current_pos, *_ = p.getLinkState(robot_id, ROBOT_END_EFFECTOR_LINK_ID, computeForwardKinematics=True)
    
    # 1. 检查直接路径
    print("  1. 检查直接路径...")
    try:
        # 预先计算好目标点的关节，后面搜索器也需要用
        target_joints = p.calculateInverseKinematics(robot_id, ROBOT_END_EFFECTOR_LINK_ID, goal_pos, goal_orn)[:7]
    except Exception as e:
        print(f"  >> 致命错误: 目标位置 {goal_pos} IK无解，任务中止。 {e}")
        return False
        
    if not is_path_colliding(robot_id, current_joint_pos, target_joints, obstacle_id):
        print("  >> 直接路径安全，正在执行...")
        move_to_joints(robot_id, target_joints)
        return True # 成功
    
    # 2. 如果直接路径被阻挡，则启动备用路径搜索器
    print("  >> 直接路径被阻挡，正在启动备用路径搜索器...")
    
    planned_path_joints_list = find_safe_avoidance_path(
        robot_id,
        current_joint_pos,
        current_pos,
        target_joints, # 传入已计算好的目标关节
        goal_pos,
        goal_orn,
        obstacle_id
    )
    
    # 3. 检查搜索结果
    if planned_path_joints_list is None:
        # 搜索器没找到路径
        # 失败信息已在 find_safe_avoidance_path 中打印
        return False 
    
    # 4. 如果搜索成功，则按顺序执行路径
    print("  >> 备用路径搜索成功，正在执行...")
    for joints_target in planned_path_joints_list:
        move_to_joints(robot_id, joints_target)
    
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



