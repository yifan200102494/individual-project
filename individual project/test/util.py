import pybullet as p
import time
import numpy as np

# --- 常量 ---
JOINT_TYPES = ["REVOLUTE", "PRISMATIC", "SPHERICAL", "PLANAR", "FIXED"]
ROBOT_HOME_CONFIG = [0.0, -0.7854, 0.0, -2.3562, 0.0, 1.5708, 0.7854]
ROBOT_END_EFFECTOR_LINK_ID = 8
DELTA_T = 1./240

# ============================================================
# ✅ 碰撞检测模块
# ============================================================
def is_state_colliding(robot_id, joint_pos, obstacle_id, gripper_pos):
    """检测给定关节状态是否与障碍物碰撞。"""
    state_id = p.saveState()
    for i in range(len(joint_pos)):
        p.resetJointState(robot_id, i, joint_pos[i])
    p.resetJointState(robot_id, 9, gripper_pos[0])
    p.resetJointState(robot_id, 10, gripper_pos[1])
    p.performCollisionDetection()
    contacts = p.getContactPoints(bodyA=robot_id, bodyB=obstacle_id)
    is_colliding = len(contacts) > 0
    p.restoreState(state_id)
    p.removeState(state_id)
    return is_colliding


def is_path_colliding(robot_id, start_joints, end_joints, obstacle_id,
                      start_gripper_pos, end_gripper_pos, num_steps=50):
    """检测从 start_joints 到 end_joints 的路径是否与障碍物碰撞。"""
    start_joints = np.asarray(start_joints)
    end_joints = np.asarray(end_joints)
    start_gripper_pos = np.asarray(start_gripper_pos)
    end_gripper_pos = np.asarray(end_gripper_pos)

    for i in range(num_steps + 1):
        alpha = i / num_steps
        interpolated_joints = (1 - alpha) * start_joints + alpha * end_joints
        interpolated_gripper = (1 - alpha) * start_gripper_pos + alpha * end_gripper_pos
        if is_state_colliding(robot_id, interpolated_joints, obstacle_id, interpolated_gripper):
            return True
    return False


# ============================================================
# ✅ 自动避障路径规划模块
# ============================================================
def plan_and_execute_motion(robot_id, goal_pos, goal_orn, obstacle_id, target_joints_override=None):
    """带自动避障功能的路径规划与执行。"""
    print(f"--- 正在规划前往 {goal_pos} 的路径 ---")

    num_arm_joints = 7
    home_rest_poses = list(ROBOT_HOME_CONFIG)
    default_null_space_params = {
        "lowerLimits": [-np.pi*2]*num_arm_joints,
        "upperLimits": [np.pi*2]*num_arm_joints,
        "jointRanges": [np.pi*4]*num_arm_joints,
        "restPoses": home_rest_poses
    }

    # 当前关节状态
    current_joint_pos = np.asarray([p.getJointState(robot_id, i)[0] for i in range(7)])
    current_gripper_pos = [p.getJointState(robot_id, 9)[0], p.getJointState(robot_id, 10)[0]]
    current_pos, *_ = p.getLinkState(robot_id, ROBOT_END_EFFECTOR_LINK_ID, computeForwardKinematics=True)

    # ======================================================
    # ✅ 1️⃣ 自动避障 IK 模式
    # ======================================================
    if target_joints_override is not None:
        print("  >> 使用了 'target_joints_override'，启用自动避障 IK 模式。")

        # 取当前末端姿态
        ee_state = p.getLinkState(robot_id, ROBOT_END_EFFECTOR_LINK_ID, computeForwardKinematics=True)
        current_ee_pos = np.array(ee_state[0])
        goal_pos = np.array(goal_pos)

        # 获取障碍物的包围盒
        aabb_min, aabb_max = p.getAABB(obstacle_id)
        obs_center = np.array([(aabb_min[0]+aabb_max[0])/2,
                               (aabb_min[1]+aabb_max[1])/2,
                               (aabb_min[2]+aabb_max[2])/2])
        obs_half_size = (np.array(aabb_max) - np.array(aabb_min)) / 2

        # 判断是否XY平面重叠
        overlap_x = (aabb_min[0] < goal_pos[0] < aabb_max[0])
        overlap_y = (aabb_min[1] < goal_pos[1] < aabb_max[1])

        if overlap_x and overlap_y:
            print("  ⚠️ 检测到目标与障碍物XY重叠区域，自动规划上抬避障路径。")

            safe_height = aabb_max[2] + 0.15  # 高于障碍物顶部15cm
            mid_pos = np.array([goal_pos[0], goal_pos[1], safe_height])
            side_offset = obs_half_size[0] + 0.15

            # 左右侧候选绕行点
            side_candidates = [
                np.array([obs_center[0] - side_offset, obs_center[1], safe_height]),
                np.array([obs_center[0] + side_offset, obs_center[1], safe_height])
            ]

            for candidate in side_candidates:
                try:
                    wp1 = current_ee_pos.copy()
                    wp2 = candidate
                    wp3 = mid_pos
                    waypoints = [wp1, wp2, wp3, goal_pos]
                    print(f"  >> 尝试自动避障路径: {waypoints}")
                    path_ok = True
                    prev_j = np.asarray([p.getJointState(robot_id, i)[0] for i in range(7)])
                    for wp in waypoints:
                        j_wp = p.calculateInverseKinematics(robot_id, ROBOT_END_EFFECTOR_LINK_ID, wp, goal_orn)[:7]
                        if is_path_colliding(robot_id, prev_j, j_wp, obstacle_id, [0.04,0.04], [0.04,0.04]):
                            path_ok = False
                            break
                        prev_j = j_wp
                    if path_ok:
                        print("  ✅ 自动避障路径安全，执行中...")
                        for wp in waypoints:
                            j_wp = p.calculateInverseKinematics(robot_id, ROBOT_END_EFFECTOR_LINK_ID, wp, goal_orn)[:7]
                            move_to_joints(robot_id, j_wp)
                        return True
                except Exception:
                    continue
            print("  ❌ 所有自动绕行路径失败，将尝试默认路径。")

        # 若无重叠或绕行失败，使用默认 home joints
        target_joints = target_joints_override
    else:
        # 普通情况使用IK求解
        target_joints = p.calculateInverseKinematics(
            robot_id, ROBOT_END_EFFECTOR_LINK_ID, goal_pos, goal_orn, **default_null_space_params
        )[:7]

    # ======================================================
    # ✅ 2️⃣ 检查直接路径
    # ======================================================
    if not is_path_colliding(robot_id, current_joint_pos, target_joints, obstacle_id,
                             current_gripper_pos, current_gripper_pos):
        print("  >> 直接路径安全，正在执行...")
        move_to_joints(robot_id, target_joints)
        return True

    # ======================================================
    # ✅ 3️⃣ 动态备用路径（绕障）
    # ======================================================
    print("  >> 直接路径被阻挡，启动动态备用路径规划...")
    aabb_min, aabb_max = p.getAABB(obstacle_id)
    obstacle_top_z = aabb_max[2]
    safe_height = obstacle_top_z + 0.10
    obstacle_min_x, obstacle_max_x = aabb_min[0], aabb_max[0]
    X_MARGIN = 0.10
    safe_x_candidates = [obstacle_min_x - X_MARGIN, obstacle_max_x + X_MARGIN, 0.0]

    for safe_x in safe_x_candidates:
        try:
            waypoint1_pos = [current_pos[0], current_pos[1], safe_height]
            waypoint2_pos = [safe_x, goal_pos[1], safe_height]
            waypoint3_pos = [goal_pos[0], goal_pos[1], safe_height]

            waypoint1_joints = p.calculateInverseKinematics(robot_id, ROBOT_END_EFFECTOR_LINK_ID, waypoint1_pos, goal_orn)[:7]
            waypoint2_joints = p.calculateInverseKinematics(robot_id, ROBOT_END_EFFECTOR_LINK_ID, waypoint2_pos, goal_orn)[:7]
            waypoint3_joints = p.calculateInverseKinematics(robot_id, ROBOT_END_EFFECTOR_LINK_ID, waypoint3_pos, goal_orn)[:7]

            path_segments = [
                (current_joint_pos, waypoint1_joints),
                (waypoint1_joints, waypoint2_joints),
                (waypoint2_joints, waypoint3_joints),
                (waypoint3_joints, target_joints)
            ]

            safe = True
            for (js_start, js_end) in path_segments:
                if is_path_colliding(robot_id, js_start, js_end, obstacle_id,
                                     current_gripper_pos, current_gripper_pos):
                    safe = False
                    break

            if safe:
                print(f"  ✅ 找到安全绕行路径 (safe_x={safe_x:.2f})，执行中...")
                for js in [waypoint1_joints, waypoint2_joints, waypoint3_joints, target_joints]:
                    move_to_joints(robot_id, js)
                return True
        except Exception:
            continue

    print("  ❌ 所有路径尝试失败，未能找到安全路线。")
    return False


# ============================================================
# ✅ 运动与夹爪控制
# ============================================================
def simulate(steps=None, seconds=None, slow_down=True):
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
    target_joint_pos = np.asarray(target_joint_pos)
    for joint_id in range(len(target_joint_pos)):
        p.setJointMotorControl2(
            robot_id,
            joint_id,
            controlMode=p.POSITION_CONTROL,
            targetPosition=target_joint_pos[joint_id],
            maxVelocity=max_velocity,
            force=100
        )

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
    p.setJointMotorControl2(robot_id, 9, controlMode=p.POSITION_CONTROL, targetPosition=0.04, force=100)
    p.setJointMotorControl2(robot_id, 10, controlMode=p.POSITION_CONTROL, targetPosition=0.04, force=100)
    simulate(seconds=1.0)


def gripper_close(robot_id):
    p.setJointMotorControl2(robot_id, 9, controlMode=p.VELOCITY_CONTROL, targetVelocity=-0.05, force=100)
    for _ in range(int(0.5 / DELTA_T)):
        simulate(steps=1)
        finger_pos = p.getJointState(robot_id, 9)[0]
        p.setJointMotorControl2(robot_id, 10, controlMode=p.POSITION_CONTROL, targetPosition=finger_pos, force=100)


def move_to_pose(robot_id, target_ee_pos, target_ee_orientation=None):
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
    move_to_joints(robot_id, joint_pos_arm)
