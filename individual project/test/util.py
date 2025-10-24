# util.py (已修复全局计数器)

import pybullet as p
import time
import numpy as np
import random 

# --- 常量 ---
JOINT_TYPES = ["REVOLUTE", "PRISMATIC", "SPHERICAL", "PLANAR", "FIXED"]
ROBOT_HOME_CONFIG = [0.0, -0.7854, 0.0, -2.3562, 0.0, 1.5708, 0.7854]
ROBOT_END_EFFECTOR_LINK_ID = 8
DELTA_T = 1./240

# --- 【新增】全局仿真步数计数器 ---
_GLOBAL_SIM_STEP_COUNTER = 0
# --- 修改结束 ---


# ============================================================
# ✅ 碰撞检测模块 (保持不变)
# ============================================================
def is_state_colliding(robot_id, joint_pos, obstacle_ids, gripper_pos):
    """检测给定关节状态是否与 *一组* 障碍物碰撞。"""
    # ... (此函数代码保持不变) ...
    state_id = p.saveState()
    for i in range(len(joint_pos)):
        p.resetJointState(robot_id, i, joint_pos[i])
    p.resetJointState(robot_id, 9, gripper_pos[0])
    p.resetJointState(robot_id, 10, gripper_pos[1])
    
    p.performCollisionDetection()
    
    is_colliding = False
    for obstacle_id in obstacle_ids:
        contacts = p.getContactPoints(bodyA=robot_id, bodyB=obstacle_id)
        if len(contacts) > 0:
            is_colliding = True
            break
            
    p.restoreState(state_id)
    p.removeState(state_id)
    return is_colliding


def is_path_colliding(robot_id, start_joints, end_joints, obstacle_ids,
                      start_gripper_pos, end_gripper_pos, num_steps=50):
    """检测从 start_joints 到 end_joints 的路径是否与 *一组* 障碍物碰撞。"""
    # ... (此函数代码保持不变) ...
    start_joints = np.asarray(start_joints)
    end_joints = np.asarray(end_joints)
    start_gripper_pos = np.asarray(start_gripper_pos)
    end_gripper_pos = np.asarray(end_gripper_pos)

    for i in range(num_steps + 1):
        alpha = i / num_steps
        interpolated_joints = (1 - alpha) * start_joints + alpha * end_joints
        interpolated_gripper = (1 - alpha) * start_gripper_pos + alpha * end_gripper_pos
        if is_state_colliding(robot_id, interpolated_joints, obstacle_ids, interpolated_gripper):
            return True
    return False

# ============================================================
# ⭐ PFM (势场法) 模块 (保持不变)
# ============================================================
def calc_attractive_force(current_pos, goal_pos, k_att=1.0):
    # ... (此函数代码保持不变) ...
    dist_vec = np.array(goal_pos) - np.array(current_pos)
    dist = np.linalg.norm(dist_vec)
    if dist < 1e-6: return np.array([0.0, 0.0, 0.0])
    return k_att * (dist_vec / dist)

def calc_anisotropic_repulsive_force(current_pos, obs_center, obs_aabb_min, obs_aabb_max,
                                     k_rep=0.5, rho_0=0.35, k_aniso_xy=2.0, k_aniso_z=0.5):
    # ... (此函数代码保持不变) ...
    dist_vec = np.array(current_pos) - obs_center
    scaling_factors = np.array([k_aniso_xy, k_aniso_xy, k_aniso_z])
    scaled_dist_vec = dist_vec * scaling_factors
    rho_scaled = np.linalg.norm(scaled_dist_vec)
    if rho_scaled > rho_0: return np.array([0.0, 0.0, 0.0])
    if rho_scaled < 1e-6: return (np.random.rand(3) - 0.5) * 2.0 * k_rep
    grad_rho_scaled = scaled_dist_vec / rho_scaled
    magnitude = k_rep * (1.0 / rho_scaled - 1.0 / rho_0) * (1.0 / (rho_scaled**2))
    return magnitude * grad_rho_scaled

def plan_path_with_pfm(start_pos, goal_pos, obstacle_ids,
                       step_size=0.05, max_steps=3000, goal_threshold=0.05):
    # ... (此函数代码保持不变) ...
    print("  >> PFM: 启动势场法路径规划器...")
    obstacles_info = []
    for obs_id in obstacle_ids:
        aabb_min, aabb_max = p.getAABB(obs_id)
        obs_center = np.array([(aabb_min[0] + aabb_max[0]) / 2, (aabb_min[1] + aabb_max[1]) / 2, (aabb_min[2] + aabb_max[2]) / 2])
        aabb_diag = np.linalg.norm(np.array(aabb_max) - np.array(aabb_min))
        obstacles_info.append({"id": obs_id, "center": obs_center, "aabb_min": aabb_min, "aabb_max": aabb_max, "diag": aabb_diag})
    
    rho_0_base = 0.35; k_rep = 1.0; k_att = 1.0; step_size = 0.02
    k_aniso_xy = 2.0; k_aniso_z = 0.5
    path = [np.array(start_pos)]
    current_pos = np.array(start_pos)

    for i in range(max_steps):
        f_att = calc_attractive_force(current_pos, goal_pos, k_att=k_att)
        f_rep_total = np.array([0.0, 0.0, 0.0])
        for obs in obstacles_info:
            rho_0 = (obs["diag"] / 2.0) + rho_0_base 
            f_rep_obs = calc_anisotropic_repulsive_force(current_pos, obs["center"], obs["aabb_min"], obs["aabb_max"], k_rep=k_rep, rho_0=rho_0, k_aniso_xy=k_aniso_xy, k_aniso_z=k_aniso_z)
            f_rep_total += f_rep_obs
            
        f_total = f_att + f_rep_total
        if np.linalg.norm(f_total) < 0.001:
            print(f"  ❌ PFM: 规划失败，在第 {i} 步陷入局部最小值。")
            return None
        current_pos = current_pos + step_size * (f_total / np.linalg.norm(f_total))
        path.append(current_pos)
        if np.linalg.norm(current_pos - np.array(goal_pos)) < goal_threshold:
            path.append(np.array(goal_pos)) 
            print(f"  ✅ PFM: 成功生成路径，共 {len(path)} 个路径点。")
            return path
    print(f"  ❌ PFM: 规划失败，超过最大步数 {max_steps}。")
    return None

# ============================================================
# ✅ 自动避障路径规划模块 (保持不变)
# ============================================================
def plan_and_execute_motion(robot_id, goal_pos, goal_orn, obstacle_ids, target_joints_override=None, **kwargs): 
    # ... (此函数代码保持不变) ...
    """带自动避障功能的路径规划与执行。"""
    print(f"--- 正在规划前往 {goal_pos} 的路径 (避开 {len(obstacle_ids)} 个障碍物) ---")

    num_arm_joints = 7
    home_rest_poses = list(ROBOT_HOME_CONFIG)
    default_null_space_params = {
        "lowerLimits": [-np.pi*2]*num_arm_joints, "upperLimits": [np.pi*2]*num_arm_joints,
        "jointRanges": [np.pi*4]*num_arm_joints, "restPoses": home_rest_poses
    }
    current_joint_pos = np.asarray([p.getJointState(robot_id, i)[0] for i in range(7)])
    current_gripper_pos = [p.getJointState(robot_id, 9)[0], p.getJointState(robot_id, 10)[0]]
    current_pos, *_ = p.getLinkState(robot_id, ROBOT_END_EFFECTOR_LINK_ID, computeForwardKinematics=True)

    # 1. 自动避障 IK 模式
    if target_joints_override is not None:
        print("  >> 使用了 'target_joints_override'，启用自动避障 IK 模式。")
        ee_state = p.getLinkState(robot_id, ROBOT_END_EFFECTOR_LINK_ID, computeForwardKinematics=True)
        current_ee_pos = np.array(ee_state[0]); goal_pos = np.array(goal_pos)
        aabb_min, aabb_max = p.getAABB(obstacle_ids[0]) 
        obs_center = np.array([(aabb_min[0]+aabb_max[0])/2, (aabb_min[1]+aabb_max[1])/2, (aabb_min[2]+aabb_max[2])/2])
        obs_half_size = (np.array(aabb_max) - np.array(aabb_min)) / 2
        overlap_x = (aabb_min[0] < goal_pos[0] < aabb_max[0])
        overlap_y = (aabb_min[1] < goal_pos[1] < aabb_max[1])
        if overlap_x and overlap_y:
            print("  ⚠️ 检测到目标与障碍物XY重叠区域，自动规划上抬避障路径。")
            safe_height = aabb_max[2] + 0.15
            mid_pos = np.array([goal_pos[0], goal_pos[1], safe_height])
            side_offset = obs_half_size[0] + 0.15
            side_candidates = [
                np.array([obs_center[0] - side_offset, obs_center[1], safe_height]),
                np.array([obs_center[0] + side_offset, obs_center[1], safe_height])
            ]
            for candidate in side_candidates:
                try:
                    wp1 = current_ee_pos.copy(); wp2 = candidate; wp3 = mid_pos
                    waypoints = [wp1, wp2, wp3, goal_pos]
                    print(f"  >> 尝试自动避障路径: {waypoints}")
                    path_ok = True
                    prev_j = np.asarray([p.getJointState(robot_id, i)[0] for i in range(7)])
                    for wp in waypoints:
                        j_wp = p.calculateInverseKinematics(robot_id, ROBOT_END_EFFECTOR_LINK_ID, wp, goal_orn)[:7]
                        if is_path_colliding(robot_id, prev_j, j_wp, obstacle_ids, [0.04,0.04], [0.04,0.04]):
                            path_ok = False; break
                        prev_j = j_wp
                    if path_ok:
                        print("  ✅ 自动避障路径安全，执行中...")
                        for wp in waypoints:
                            j_wp = p.calculateInverseKinematics(robot_id, ROBOT_END_EFFECTOR_LINK_ID, wp, goal_orn)[:7]
                            move_to_joints(robot_id, j_wp, **kwargs) # 传递 kwargs
                        return True
                except Exception: continue
            print("  ❌ 所有自动绕行路径失败，将尝试默认路径。")
        target_joints = target_joints_override
    else:
        target_joints = p.calculateInverseKinematics(
            robot_id, ROBOT_END_EFFECTOR_LINK_ID, goal_pos, goal_orn, **default_null_space_params
        )[:7]

    # 2. 检查直接路径
    if not is_path_colliding(robot_id, current_joint_pos, target_joints, obstacle_ids,
                             current_gripper_pos, current_gripper_pos):
        print("  >> 直接路径安全，正在执行...")
        move_to_joints(robot_id, target_joints, **kwargs) # 传递 kwargs
        return True

    # 3. 使用 PFM 规划器
    print("  >> 直接路径被阻挡，启动 PFM 路径规划器...")
    workspace_path = plan_path_with_pfm(
        start_pos=current_pos, goal_pos=goal_pos, obstacle_ids=obstacle_ids
    )
    if workspace_path is None:
        print("  ❌ PFM 规划器未能找到路径。")
        return False

    joint_space_path = []; last_joint_pos = current_joint_pos 
    ik_params = default_null_space_params.copy()
    for i, wp_pos in enumerate(workspace_path):
        try:
            ik_params["restPoses"] = list(last_joint_pos) 
            wp_joints = p.calculateInverseKinematics(
                robot_id, ROBOT_END_EFFECTOR_LINK_ID, wp_pos, goal_orn, **ik_params
            )[:7]
            if is_path_colliding(robot_id, last_joint_pos, wp_joints, obstacle_ids,
                                 current_gripper_pos, current_gripper_pos):
                print(f"  ❌ PFM 路径在 C-Space 中发现碰撞 (段 {i})。")
                return False
            joint_space_path.append(wp_joints)
            last_joint_pos = wp_joints 
        except Exception as e:
            print(f"  ❌ PFM 路径点 {i} ({wp_pos}) IK 求解失败。")
            return False

    print(f"  ✅ PFM 路径在 C-Space 中验证安全，执行中...")
    for joint_target in joint_space_path:
        move_to_joints(robot_id, joint_target, max_velocity=1.5, **kwargs) # 传递 kwargs
        
    move_to_joints(robot_id, target_joints, max_velocity=1.0, **kwargs) # 传递 kwargs
    return True

# ============================================================
# ✅ (重大修改) 运动与夹爪控制
# ============================================================

# 【重大修改】使用全局计数器
def simulate(steps=None, seconds=None, slow_down=True, 
             interferer_id=None, interferer_joints=None, interferer_update_rate=120):
    """
    步进仿真，【新增】可选地驱动一个干扰臂。
    【修复】使用全局计数器 _GLOBAL_SIM_STEP_COUNTER 来确保干扰臂逻辑被触发。
    """
    global _GLOBAL_SIM_STEP_COUNTER # <--- 【修改】
    
    seconds_passed = 0.0
    steps_this_call = 0 # <--- 【修改】使用一个局部计数器来控制此函数的退出
    start_time = time.time()
    
    while True:
        p.stepSimulation()
        
        _GLOBAL_SIM_STEP_COUNTER += 1 # <--- 【修改】总是增加全局计数器
        steps_this_call += 1          # <--- 【修改】增加此函数调用的计数器
        
        # --- 【新增】干扰臂随机运动逻辑 ---
        if interferer_id is not None and interferer_joints is not None:
            
            # 【修改】使用全局计数器来检查是否更新
            if _GLOBAL_SIM_STEP_COUNTER % interferer_update_rate == 0:
                print(f"[干扰臂在第 {_GLOBAL_SIM_STEP_COUNTER} 步移动...]") # <-- 添加日志
                
                # 随机选择一个关节
                joint_to_move = random.choice(interferer_joints)
                
                # 获取该关节的限制
                joint_info = p.getJointInfo(interferer_id, joint_to_move)
                joint_min = joint_info[8]
                joint_max = joint_info[9]
                
                # 随机选择一个目标位置
                target_pos = random.uniform(joint_min, joint_max)
                
                # 设置电机控制
                p.setJointMotorControl2(
                    bodyUniqueId=interferer_id,
                    jointIndex=joint_to_move,
                    controlMode=p.POSITION_CONTROL,
                    targetPosition=target_pos,
                    maxVelocity=1.5, # 给一个合理的速度
                    force=100
                )
        # --- 【新增】逻辑结束 ---

        seconds_passed += DELTA_T
        if slow_down:
            time_elapsed = time.time() - start_time
            wait_time = seconds_passed - time_elapsed
            time.sleep(max(wait_time, 0))
        
        # 【修改】使用局部计数器 'steps_this_call' 来决定是否退出
        if steps is not None and steps_this_call >= steps:
            break
        if seconds is not None and seconds_passed >= seconds:
            break

# 【修改】添加 **kwargs
def move_to_joints(robot_id, target_joint_pos, max_velocity=1, timeout=5, **kwargs):
    target_joint_pos = np.asarray(target_joint_pos)
    for joint_id in range(len(target_joint_pos)):
        p.setJointMotorControl2(
            robot_id, joint_id, controlMode=p.POSITION_CONTROL,
            targetPosition=target_joint_pos[joint_id],
            maxVelocity=max_velocity, force=100
        )
    counter = 0
    while True:
        current_joint_pos = np.asarray([p.getJointState(robot_id, i)[0] for i in range(7)])
        if np.allclose(current_joint_pos, target_joint_pos, atol=0.005):
            break
        
        simulate(steps=1, **kwargs) # 【修改】传递 kwargs
        
        counter += 1
        if counter > timeout / DELTA_T:
            print('WARNING: timeout while moving to joint position.')
            break

# 【修改】添加 **kwargs
def gripper_open(robot_id, **kwargs):
    p.setJointMotorControl2(robot_id, 9, controlMode=p.POSITION_CONTROL, targetPosition=0.04, force=100)
    p.setJointMotorControl2(robot_id, 10, controlMode=p.POSITION_CONTROL, targetPosition=0.04, force=100)
    simulate(seconds=1.0, **kwargs) # 【修改】传递 kwargs

# 【修改】添加 **kwargs
def gripper_close(robot_id, **kwargs):
    p.setJointMotorControl2(robot_id, 9, controlMode=p.VELOCITY_CONTROL, targetVelocity=-0.05, force=100)
    for _ in range(int(0.5 / DELTA_T)):
        simulate(steps=1, **kwargs) # 【修改】传递 kwargs
        finger_pos = p.getJointState(robot_id, 9)[0]
        p.setJointMotorControl2(robot_id, 10, controlMode=p.POSITION_CONTROL, targetPosition=finger_pos, force=100)

# 【修改】添加 **kwargs
def move_to_pose(robot_id, target_ee_pos, target_ee_orientation=None, **kwargs):
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
    move_to_joints(robot_id, joint_pos_arm, **kwargs) # 【修改】传递 kwargs