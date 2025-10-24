# util.py (最终完整版 - 已修复IK约束 和 性能优化)

import pybullet as p
import time
import numpy as np
import random 

# --- 常量 ---
JOINT_TYPES = ["REVOLUTE", "PRISMATIC", "SPHERICAL", "PLANAR", "FIXED"] #
ROBOT_HOME_CONFIG = [0.0, -0.7854, 0.0, -2.3562, 0.0, 1.5708, 0.7854] #
ROBOT_END_EFFECTOR_LINK_ID = 8 #
DELTA_T = 1./240 #

# 【新增】定义全局 IK 约束，防止机械臂"折断"
_NUM_ARM_JOINTS = 7
_DEFAULT_NULL_SPACE_PARAMS = {
    "lowerLimits": [-np.pi*2]*_NUM_ARM_JOINTS, 
    "upperLimits": [np.pi*2]*_NUM_ARM_JOINTS,
    "jointRanges": [np.pi*4]*_NUM_ARM_JOINTS, 
    "restPoses": list(ROBOT_HOME_CONFIG)
}
# --- 新增结束 ---

# --- 全局仿真步数计数器 ---
_GLOBAL_SIM_STEP_COUNTER = 0 #

# ============================================================
# ✅ 碰撞检测模块 (修复闪烁)
# ============================================================
def is_state_colliding(robot_id, joint_pos, obstacle_ids, gripper_pos): #
    """
    检查一个特定的关节配置是否与障碍物碰撞。
    【修改】移除此处的渲染开关，以防止在循环中被调用时导致窗口闪烁。
    """
    
    state_id = p.saveState() #
    for i in range(len(joint_pos)):
        p.resetJointState(robot_id, i, joint_pos[i]) #
    p.resetJointState(robot_id, 9, gripper_pos[0]) #
    p.resetJointState(robot_id, 10, gripper_pos[1]) #
    
    p.performCollisionDetection() #
    
    is_colliding = False
    for obstacle_id in obstacle_ids:
        contacts = p.getContactPoints(bodyA=robot_id, bodyB=obstacle_id) #
        if len(contacts) > 0:
            is_colliding = True
            break
            
    p.restoreState(state_id) #
    p.removeState(state_id) #
    
    return is_colliding


def is_path_colliding(robot_id, start_joints, end_joints, obstacle_ids,
                      start_gripper_pos, end_gripper_pos, num_steps=25): #
    """
    检查一条路径（一系列插值点）是否碰撞。
    【新增】在此处（循环外）关闭/打开渲染，以防止闪烁。
    """
    start_joints = np.asarray(start_joints) #
    end_joints = np.asarray(end_joints) #
    start_gripper_pos = np.asarray(start_gripper_pos) #
    end_gripper_pos = np.asarray(end_gripper_pos) #

    # 【新增】在检查 *之前* 关闭渲染
    p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 0) #
    
    path_is_colliding = False
    for i in range(num_steps + 1): #
        alpha = i / num_steps #
        interpolated_joints = (1 - alpha) * start_joints + alpha * end_joints #
        interpolated_gripper = (1 - alpha) * start_gripper_pos + alpha * end_gripper_pos #
        if is_state_colliding(robot_id, interpolated_joints, obstacle_ids, interpolated_gripper): #
            path_is_colliding = True #
            break # 发现碰撞，立即停止 #
            
    # 【新增】在所有检查 *之后* 重新打开渲染
    p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 1) #
            
    return path_is_colliding #

# ============================================================
# ⭐ PFM (势场法) 模块 (保持不变)
# ============================================================
def calc_attractive_force(current_pos, goal_pos, k_att=1.0): #
    # (此函数代码保持不变)
    dist_vec = np.array(goal_pos) - np.array(current_pos) #
    dist = np.linalg.norm(dist_vec) #
    if dist < 1e-6: return np.array([0.0, 0.0, 0.0]) #
    return k_att * (dist_vec / dist) #

def calc_anisotropic_repulsive_force(current_pos, obs_center, obs_aabb_min, obs_aabb_max,
                                     k_rep=0.5, rho_0=0.35, k_aniso_xy=2.0, k_aniso_z=0.5): #
    # (此函数代码保持不变)
    dist_vec = np.array(current_pos) - obs_center #
    scaling_factors = np.array([k_aniso_xy, k_aniso_xy, k_aniso_z]) #
    scaled_dist_vec = dist_vec * scaling_factors #
    rho_scaled = np.linalg.norm(scaled_dist_vec) #
    if rho_scaled > rho_0: return np.array([0.0, 0.0, 0.0]) #
    if rho_scaled < 1e-6: return (np.random.rand(3) - 0.5) * 2.0 * k_rep #
    grad_rho_scaled = scaled_dist_vec / rho_scaled #
    magnitude = k_rep * (1.0 / rho_scaled - 1.0 / rho_0) * (1.0 / (rho_scaled**2)) #
    return magnitude * grad_rho_scaled #

def plan_path_with_pfm(start_pos, goal_pos, obstacle_ids,
                       step_size=0.05, max_steps=300, goal_threshold=0.05): #
    # (此函数代码保持不变)
    print("  >> PFM: 启动势场法路径规划器...") #
    obstacles_info = [] #
    for obs_id in obstacle_ids:
        aabb_min, aabb_max = p.getAABB(obs_id) #
        obs_center = np.array([(aabb_min[0] + aabb_max[0]) / 2, (aabb_min[1] + aabb_max[1]) / 2, (aabb_min[2] + aabb_max[2]) / 2]) #
        aabb_diag = np.linalg.norm(np.array(aabb_max) - np.array(aabb_min)) #
        obstacles_info.append({"id": obs_id, "center": obs_center, "aabb_min": aabb_min, "aabb_max": aabb_max, "diag": aabb_diag}) #
    
    rho_0_base = 0.35; k_rep = 1.0; k_att = 1.0; step_size = 0.02 #
    k_aniso_xy = 2.0; k_aniso_z = 0.5 #
    path = [np.array(start_pos)] #
    current_pos = np.array(start_pos) #

    for i in range(max_steps): #
        f_att = calc_attractive_force(current_pos, goal_pos, k_att=k_att) #
        f_rep_total = np.array([0.0, 0.0, 0.0]) #
        for obs in obstacles_info:
            rho_0 = (obs["diag"] / 2.0) + rho_0_base #
            f_rep_obs = calc_anisotropic_repulsive_force(current_pos, obs["center"], obs["aabb_min"], obs["aabb_max"], k_rep=k_rep, rho_0=rho_0, k_aniso_xy=k_aniso_xy, k_aniso_z=k_aniso_z) #
            f_rep_total += f_rep_obs #
            
        f_total = f_att + f_rep_total #
        if np.linalg.norm(f_total) < 0.001:
            print(f"  ❌ PFM: 规划失败，在第 {i} 步陷入局部最小值。") #
            return None #
        current_pos = current_pos + step_size * (f_total / np.linalg.norm(f_total)) #
        path.append(current_pos) #
        if np.linalg.norm(current_pos - np.array(goal_pos)) < goal_threshold: #
            path.append(np.array(goal_pos))  #
            print(f"  ✅ PFM: 成功生成路径，共 {len(path)} 个路径点。") #
            return path #
    print(f"  ❌ PFM: 规划失败，超过最大步数 {max_steps}。") #
    return None #

# ============================================================
# ✅ (重大修改) 自动避障路径规划模块
# ============================================================
def plan_and_execute_motion(robot_id, goal_pos, goal_orn, obstacle_ids, target_joints_override=None, **kwargs): #
    """带自动避障功能的路径规划与执行。"""
    print(f"--- 正在规划前往 {goal_pos} 的路径 (避开 {len(obstacle_ids)} 个障碍物) ---") #

    # 【新增】创建一个 execution_kwargs 字典，用于将 obstacle_ids 传递给 move_to_joints
    execution_kwargs = kwargs.copy() #
    execution_kwargs["obstacle_ids"] = obstacle_ids #
    # --- 新增结束 ---

    num_arm_joints = 7 #
    
    # 【修改】从全局常量初始化
    default_null_space_params = _DEFAULT_NULL_SPACE_PARAMS.copy()
    default_null_space_params["restPoses"] = list(ROBOT_HOME_CONFIG)
    # --- 修改结束 ---
    
    current_joint_pos = np.asarray([p.getJointState(robot_id, i)[0] for i in range(7)]) #
    current_gripper_pos = [p.getJointState(robot_id, 9)[0], p.getJointState(robot_id, 10)[0]] #
    current_pos, *_ = p.getLinkState(robot_id, ROBOT_END_EFFECTOR_LINK_ID, computeForwardKinematics=True) #

    # 1. 自动避障 IK 模式
    if target_joints_override is not None: #
        print("  >> 使用了 'target_joints_override'，启用自动避障 IK 模式。") #
        ee_state = p.getLinkState(robot_id, ROBOT_END_EFFECTOR_LINK_ID, computeForwardKinematics=True) #
        current_ee_pos = np.array(ee_state[0]); goal_pos = np.array(goal_pos) #
        
        # 自动避障逻辑只应针对干扰臂
        interferer_id = kwargs.get("interferer_id")
        # 【修复】检查 obstacle_ids[0] 是否存在
        if interferer_id in obstacle_ids and obstacle_ids:
            aabb_min, aabb_max = p.getAABB(obstacle_ids[0]) #
            obs_center = np.array([(aabb_min[0]+aabb_max[0])/2, (aabb_min[1]+aabb_max[1])/2, (aabb_min[2]+aabb_max[2])/2]) #
            obs_half_size = (np.array(aabb_max) - np.array(aabb_min)) / 2 #
            overlap_x = (aabb_min[0] < goal_pos[0] < aabb_max[0]) #
            overlap_y = (aabb_min[1] < goal_pos[1] < aabb_max[1]) #
            
            if overlap_x and overlap_y: #
                print("  ⚠️ 检测到目标与障碍物XY重叠区域，自动规划上抬避障路径。") #
                safe_height = aabb_max[2] + 0.15 #
                mid_pos = np.array([goal_pos[0], goal_pos[1], safe_height]) #
                side_offset = obs_half_size[0] + 0.15 #
                side_candidates = [ #
                    np.array([obs_center[0] - side_offset, obs_center[1], safe_height]), #
                    np.array([obs_center[0] + side_offset, obs_center[1], safe_height]) #
                ]
                
                # #######################################################
                # ##### 开始修复：为自动避障添加IK约束 #####
                # #######################################################
                for candidate in side_candidates: #
                    try: #
                        wp1 = current_ee_pos.copy(); wp2 = candidate; wp3 = mid_pos #
                        waypoints = [wp1, wp2, wp3, goal_pos] #
                        print(f"  >> 尝试自动避障路径: {waypoints}") #
                        path_ok = True #
                        prev_j = np.asarray([p.getJointState(robot_id, i)[0] for i in range(7)]) #
                        
                        # --- 【!! 修复 !!】 ---
                        # 1. 为IK求解器准备约束参数
                        ik_params_auto = default_null_space_params.copy() 
                        
                        # 2. 存储规划好的关节路径
                        joint_waypoints = [] 
                        
                        for wp in waypoints: #
                            # 3. 使用 'restPoses' 来确保路径平滑
                            ik_params_auto["restPoses"] = list(prev_j) 
                            
                            # 4. 调用带约束的IK
                            j_wp = p.calculateInverseKinematics(robot_id, ROBOT_END_EFFECTOR_LINK_ID, 
                                                                wp, goal_orn, 
                                                                **ik_params_auto)[:7] # <-- 已修复
                            
                            if is_path_colliding(robot_id, prev_j, j_wp, obstacle_ids, [0.04,0.04], [0.04,0.04]): #
                                path_ok = False; break #
                            
                            prev_j = j_wp #
                            joint_waypoints.append(j_wp) # 5. 存储有效的路径点
                        
                        # --- 修复结束 ---

                        if path_ok: #
                            print("  ✅ 自动避障路径安全，执行中...") #
                            
                            # 【!! 修复 !!】执行已验证过的 'joint_waypoints'
                            # 而不是重新计算一次IK
                            for j_wp_target in joint_waypoints: 
                                success = move_to_joints(robot_id, j_wp_target, **execution_kwargs)  #
                                if not success: #
                                    print("  ❌ 自动避障路径在 *执行* 期间失败。") #
                                    return False #
                            return True #
                    except Exception: continue #
                # #######################################################
                # ##### 修复结束 #####
                # #######################################################
                
                print("  ❌ 所有自动绕行路径失败，将尝试默认路径。") #
                
        target_joints = target_joints_override #
    else:
        target_joints = p.calculateInverseKinematics( #
            robot_id, ROBOT_END_EFFECTOR_LINK_ID, goal_pos, goal_orn, **default_null_space_params
        )[:7]

    # 2. 检查直接路径
    if not is_path_colliding(robot_id, current_joint_pos, target_joints, obstacle_ids,
                             current_gripper_pos, current_gripper_pos): #
        print("  >> 直接路径安全，正在执行...") #
        success = move_to_joints(robot_id, target_joints, **execution_kwargs)  #
        return success  #

    # 3. 使用 PFM 规划器
    print("  >> 直接路径被阻挡，启动 PFM 路径规划器...") #
    workspace_path = plan_path_with_pfm( #
        start_pos=current_pos, goal_pos=goal_pos, obstacle_ids=obstacle_ids
    )
    if workspace_path is None: #
        print("  ❌ PFM 规划器未能找到路径。") #
        return False #

    joint_space_path = []; last_joint_pos = current_joint_pos  #
    ik_params = default_null_space_params.copy() #
    for i, wp_pos in enumerate(workspace_path): #
        try: #
            if i % 3 != 0 and i != (len(workspace_path) - 1): # (但最后一个点必须检查)
                continue

            ik_params["restPoses"] = list(last_joint_pos)  #
            wp_joints = p.calculateInverseKinematics( #
                robot_id, ROBOT_END_EFFECTOR_LINK_ID, wp_pos, goal_orn, **ik_params
            )[:7]
            if is_path_colliding(robot_id, last_joint_pos, wp_joints, obstacle_ids,
                                 current_gripper_pos, current_gripper_pos): #
                print(f"  ❌ PFM 路径在 C-Space 中发现碰撞 (段 {i})。") #
                return False #
            joint_space_path.append(wp_joints) #
            last_joint_pos = wp_joints  #
        except Exception as e: #
            print(f"  ❌ PFM 路径点 {i} ({wp_pos}) IK 求解失败。") #
            return False #

    print(f"  ✅ PFM 路径在 C-Space 中验证安全，执行中...") #
    for joint_target in joint_space_path: #
        success = move_to_joints(robot_id, joint_target, max_velocity=1.5, **execution_kwargs)  #
        if not success: #
            print("  ❌ PFM 路径在 *执行* 期间失败。") #
            return False  #
        
    success_final = move_to_joints(robot_id, target_joints, max_velocity=1.0, **execution_kwargs)  #
    return success_final #

# ============================================================
# ✅ (重大修改) 运动与夹爪控制
# ============================================================

def simulate(steps=None, seconds=None, slow_down=True, 
             interferer_id=None, interferer_joints=None, interferer_update_rate=120): #
    """ (保持不变) 步进仿真并移动干扰臂 """
    global _GLOBAL_SIM_STEP_COUNTER  #
    
    seconds_passed = 0.0 #
    steps_this_call = 0  #
    start_time = time.time() #
    
    while True: #
        p.stepSimulation() #
        
        _GLOBAL_SIM_STEP_COUNTER += 1  #
        steps_this_call += 1           #
        
        if interferer_id is not None and interferer_joints is not None: #
            if _GLOBAL_SIM_STEP_COUNTER % interferer_update_rate == 0: #
                joint_to_move = random.choice(interferer_joints) #
                joint_info = p.getJointInfo(interferer_id, joint_to_move) #
                joint_min = joint_info[8] #
                joint_max = joint_info[9] #
                target_pos = random.uniform(joint_min, joint_max) #
                p.setJointMotorControl2( #
                    bodyUniqueId=interferer_id,
                    jointIndex=joint_to_move,
                    controlMode=p.POSITION_CONTROL,
                    targetPosition=target_pos,
                    maxVelocity=1.5, 
                    force=100
                )

        seconds_passed += DELTA_T #
        if slow_down: #
            time_elapsed = time.time() - start_time #
            wait_time = seconds_passed - time_elapsed #
            time.sleep(max(wait_time, 0)) #
        
        if steps is not None and steps_this_call >= steps: #
            break #
        if seconds is not None and seconds_passed >= seconds: #
            break #


# ==========================================================
# --- 【!! 性能优化 !!】 ---
# 此函数已被替换为优化后的版本
# ==========================================================
def move_to_joints(robot_id, target_joint_pos, max_velocity=1, timeout=5, **kwargs): #
    """
    移动到目标关节位置。
    【升级】在执行期间进行实时避障。
    1. (硬碰撞): 如果 is_state_colliding() 为 True，立即停止并返回 False (触发重规划)。
    2. (警告区): 如果与干扰臂距离 < THRESHOLD，则忽略 target_joint_pos，
       转而计算一个“被推开”的临时目标并向其移动。
    """
    
    # --- 1. 提取参数 ---
    sim_kwargs = { #
        "interferer_id": kwargs.get("interferer_id"), #
        "interferer_joints": kwargs.get("interferer_joints"), #
        "interferer_update_rate": kwargs.get("interferer_update_rate", 120), #
        "slow_down": kwargs.get("slow_down", True)  #
    }
    
    # --- 本地反应式避障参数 ---
    interferer_id = kwargs.get("interferer_id") #
    REACTIVE_DISTANCE_THRESHOLD = 0.06 # “警告区”距离 (米) #
    REACTIVE_NUDGE_STRENGTH = 0.1 # “推开”的力度 (米) #

    # ==========================================================
    # --- 【!! 性能优化 !!】 ---
    # 定义反应式IK的更新频率 (每N步更新一次)
    # 240Hz / 10 = 24Hz。这足以应对实时避障，同时极大降低CPU负载。
    REACTIVE_IK_UPDATE_RATE_STEPS = 10 
    
    # _last_reactive_ik_step 用于跟踪上次IK计算的时间
    _last_reactive_ik_step = -REACTIVE_IK_UPDATE_RATE_STEPS - 1 
    
    # _current_effective_target 用于“暂存”计算出的目标
    # 在非IK计算的步骤中，机器人继续朝这个暂存的目标移动
    _current_effective_target = np.asarray(target_joint_pos).copy()
    # --- 优化结束 ---
    # ==========================================================

    target_joint_pos = np.asarray(target_joint_pos) #
    num_arm_joints = len(target_joint_pos) #
        
    counter = 0 #
    while True: #
        
        # --- 2. 硬碰撞检测 (Failsafe) ---
        obstacle_ids = kwargs.get("obstacle_ids", []) #
        if obstacle_ids:  #
            current_joint_pos_check = np.asarray([p.getJointState(robot_id, i)[0] for i in range(num_arm_joints)]) #
            current_gripper_pos_check = [p.getJointState(robot_id, 9)[0], p.getJointState(robot_id, 10)[0]] #
            
            if is_state_colliding(robot_id, current_joint_pos_check, obstacle_ids, current_gripper_pos_check): #
                print("  [!!] EXECUTION-TIME COLLISION DETECTED! (执行时碰撞！)") #
                print("  [!!] 立即停止机器人... (将触发重规划)") #
                
                for joint_id in range(num_arm_joints): #
                    p.setJointMotorControl2( #
                        robot_id, joint_id, controlMode=p.VELOCITY_CONTROL,
                        targetVelocity=0, force=200 
                    )
                simulate(steps=1, **sim_kwargs) # 应用停止命令 #
                return False # 报告失败 #
        # --- 硬碰撞检测结束 ---

        # --- 3. 【优化】反应式避障 (Warning Zone) ---
        is_in_warning_zone = False #

        if interferer_id is not None: #
            closest_points = p.getClosestPoints(robot_id, interferer_id, REACTIVE_DISTANCE_THRESHOLD) #
            
            if closest_points: #
                is_in_warning_zone = True #
                
                # 【优化】检查是否到了该更新IK的时间
                current_step = _GLOBAL_SIM_STEP_COUNTER #
                if (current_step - _last_reactive_ik_step) >= REACTIVE_IK_UPDATE_RATE_STEPS: #
                    _last_reactive_ik_step = current_step # 重置计时器
                    
                    # --- 只有在需要时才执行以下昂贵计算 ---
                    repulsive_vec = np.zeros(3) #
                    num_points = 0 #
                    for point in closest_points: #
                        repulsive_vec += np.asarray(point[7]) #
                        num_points += 1 #
                    
                    if num_points > 0: #
                        repulsive_vec /= num_points #
                        
                        ee_state = p.getLinkState(robot_id, ROBOT_END_EFFECTOR_LINK_ID, computeForwardKinematics=True) #
                        current_ee_pos = np.asarray(ee_state[0]) #
                        current_ee_orn = np.asarray(ee_state[1]) #
                        
                        nudged_ee_pos = current_ee_pos + repulsive_vec * REACTIVE_NUDGE_STRENGTH #
                        
                        try: #
                            current_joint_pos_for_ik = [p.getJointState(robot_id, i)[0] for i in range(num_arm_joints)]
                            ik_params = _DEFAULT_NULL_SPACE_PARAMS.copy()
                            ik_params["restPoses"] = list(current_joint_pos_for_ik)

                            nudged_joint_pos = p.calculateInverseKinematics( #
                                robot_id, ROBOT_END_EFFECTOR_LINK_ID, 
                                nudged_ee_pos, current_ee_orn,
                                **ik_params 
                            )[:num_arm_joints]
                            
                            # 【优化】更新暂存的目标
                            _current_effective_target = np.asarray(nudged_joint_pos) #
                        except Exception: #
                            # 如果IK失败，暂时恢复原始目标
                             _current_effective_target = target_joint_pos #
                
                # --- 如果还没到更新时间，代码会跳过昂贵的IK，
                # --- 机械臂会继续朝上一次计算出的 _current_effective_target 移动

            else:
                # 【优化】如果离开了警告区，立刻重置回原始目标
                is_in_warning_zone = False #
                _current_effective_target = target_joint_pos #
                                
        # --- 反应式避障结束 ---

        # --- 4. 【修改】在循环中持续设置电机目标 ---
        # 【优化】始终使用 _current_effective_target 作为目标
        for joint_id in range(num_arm_joints): #
            p.setJointMotorControl2( #
                robot_id, joint_id, controlMode=p.POSITION_CONTROL,
                targetPosition=_current_effective_target[joint_id], # <-- 使用暂存的目标
                maxVelocity=max_velocity, force=100
            )

        # --- 5. 检查是否到达 *原始* 目标 ---
        current_joint_pos = np.asarray([p.getJointState(robot_id, i)[0] for i in range(num_arm_joints)]) #
        
        # 只有在 *不在警告区* 且 *到达原始目标* 时，才算成功
        if np.allclose(current_joint_pos, target_joint_pos, atol=0.01): #
            if not is_in_warning_zone:  #
                return True # 报告成功 #

        # --- 6. 步进仿真和超时 (as before) ---
        simulate(steps=1, **sim_kwargs)  #
        
        counter += 1 #
        if counter > timeout / DELTA_T: #
            print('WARNING: timeout while moving to joint position.') #
            return False # 超时也算失败 #
            
    return True 
# --- 优化函数结束 ---


def gripper_open(robot_id, **kwargs): #
    p.setJointMotorControl2(robot_id, 9, controlMode=p.POSITION_CONTROL, targetPosition=0.04, force=100) #
    p.setJointMotorControl2(robot_id, 10, controlMode=p.POSITION_CONTROL, targetPosition=0.04, force=100) #
    simulate(seconds=1.0, **kwargs)  #

def gripper_close(robot_id, **kwargs): #
    p.setJointMotorControl2(robot_id, 9, controlMode=p.VELOCITY_CONTROL, targetVelocity=-0.05, force=100) #
    for _ in range(int(0.5 / DELTA_T)): #
        simulate(steps=1, **kwargs)  #
        finger_pos = p.getJointState(robot_id, 9)[0] #
        p.setJointMotorControl2(robot_id, 10, controlMode=p.POSITION_CONTROL, targetPosition=finger_pos, force=100) #

def move_to_pose(robot_id, target_ee_pos, target_ee_orientation=None, **kwargs): #
    if target_ee_orientation is None: #
        joint_pos_all = p.calculateInverseKinematics( #
            robot_id, ROBOT_END_EFFECTOR_LINK_ID, targetPosition=target_ee_pos,
            maxNumIterations=100, residualThreshold=0.001)
    else:
        joint_pos_all = p.calculateInverseKinematics( #
            robot_id, ROBOT_END_EFFECTOR_LINK_ID,
            targetPosition=target_ee_pos, targetOrientation=target_ee_orientation,
            maxNumIterations=100, residualThreshold=0.001)
    joint_pos_arm = list(joint_pos_all[0:7]) #
    
    return move_to_joints(robot_id, joint_pos_arm, **kwargs) #