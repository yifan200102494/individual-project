# util.py (已修复 TypeError + 射线可见性 + 传感器“失明”BUG + 五向传感器)

import pybullet as p
import time
import numpy as np
import random 

# --- 常量 ---
JOINT_TYPES = ["REVOLUTE", "PRISMATIC", "SPHERICAL", "PLANAR", "FIXED"]
ROBOT_HOME_CONFIG = [0.0, -0.7854, 0.0, -2.3562, 0.0, 1.5708, 0.7854]
ROBOT_END_EFFECTOR_LINK_ID = 8 # 传感器将安装在这里
DELTA_T = 1./240 

_NUM_ARM_JOINTS = 7
_DEFAULT_NULL_SPACE_PARAMS = {
    "lowerLimits": [-np.pi*2]*_NUM_ARM_JOINTS, 
    "upperLimits": [np.pi*2]*_NUM_ARM_JOINTS,
    "jointRanges": [np.pi*4]*_NUM_ARM_JOINTS, 
    "restPoses": list(ROBOT_HOME_CONFIG)
}

# --- 全局仿真步数计数器 ---
_GLOBAL_SIM_STEP_COUNTER = 0

# ============================================================
# 碰撞检测模块 (保持不变)
# ============================================================
def is_state_colliding(robot_id, joint_pos, obstacle_ids, gripper_pos): 
    """(此函数代码保持不变)"""
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
                      start_gripper_pos, end_gripper_pos, num_steps=25):
    """(此函数代码保持不变)"""
    start_joints = np.asarray(start_joints)
    end_joints = np.asarray(end_joints)
    start_gripper_pos = np.asarray(start_gripper_pos)
    end_gripper_pos = np.asarray(end_gripper_pos)

    p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 0)
    
    path_is_colliding = False
    for i in range(num_steps + 1):
        alpha = i / num_steps
        interpolated_joints = (1 - alpha) * start_joints + alpha * end_joints
        interpolated_gripper = (1 - alpha) * start_gripper_pos + alpha * end_gripper_pos
        if is_state_colliding(robot_id, interpolated_joints, obstacle_ids, interpolated_gripper):
            path_is_colliding = True
            break
            
    p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 1)
            
    return path_is_colliding 

# ============================================================
# 🤖 感知模块 (已升级为五向传感器)
# ============================================================
def perceive_obstacles_with_rays(robot_id, sensor_link_id, 
                                 ray_range=1.5, grid_size=7, fov_width=0.8, 
                                 debug=False):
    """(此函数代码已修复并升级)"""
    
    # 1. 获取传感器的“外参”
    try:
        link_state = p.getLinkState(robot_id, sensor_link_id, computeForwardKinematics=True)
    except Exception as e:
        print(f"  [感知错误] 无法获取 link state for {robot_id}, {sensor_link_id}: {e}")
        return set()
        
    sensor_pos_world = np.array(link_state[0])
    sensor_orn_world = np.array(link_state[1])
    
    # 2. 计算传感器的旋转矩阵
    sensor_rot_matrix = np.array(p.getMatrixFromQuaternion(sensor_orn_world)).reshape(3, 3)

    ray_froms_world = []
    ray_tos_world = []
    
    # 定义网格坐标 (用于构建锥体)
    grid_coords_1 = np.linspace(-fov_width, fov_width, grid_size)
    grid_coords_2 = np.linspace(-fov_width, fov_width, grid_size)

    # --- 【新】定义八个感知方向 (本地坐标系) ---
    # (axis_idx, direction)
    # axis_idx: 0=X, 1=Y, 2=Z
    # direction: +1.0 (正向) or -1.0 (反向)
    
    # Franka URDF: +X (向前), +Y (向左), +Z (向下/工具方向)
    # 增加对角线方向以减少盲区
    sensor_directions = [
        (2, 1.0),  # 向下 (+Z) - 用于抓取
        (0, 1.0),  # 向前 (+X) - 用于平移
        (0, -1.0), # 向后 (-X)
        (1, 1.0),  # 向左 (+Y) - 躲避侧面
        (1, -1.0), # 向右 (-Y) - 躲避侧面
        (2, -1.0), # 向上 (-Z) - 检测上方障碍
        # 对角线方向 - 减少盲区
        ('diagonal', [1.0, 1.0, 0.0]),   # 前左对角
        ('diagonal', [1.0, -1.0, 0.0]),  # 前右对角
        ('diagonal', [-1.0, 1.0, 0.0]),  # 后左对角
        ('diagonal', [-1.0, -1.0, 0.0])  # 后右对角
    ]
    
    start_offset = 0.01 # 射线起点的微小偏移 (防止击中自己)
    
    # --- 【新】为每个方向生成射线锥体 ---
    for sensor_dir in sensor_directions:
        
        if sensor_dir[0] == 'diagonal':
            # 处理对角线方向
            direction_vec = np.array(sensor_dir[1])
            direction_vec = direction_vec / np.linalg.norm(direction_vec)  # 归一化
            
            for u_grid in grid_coords_1:
                for v_grid in grid_coords_2:
                    # 射线起点
                    ray_from_local = direction_vec * start_offset
                    
                    # 射线终点 - 沿对角线方向展开，带有小的扩散
                    ray_to_local = direction_vec * ray_range
                    # 添加垂直于主方向的扩散
                    perpendicular_1 = np.array([-direction_vec[1], direction_vec[0], 0])
                    perpendicular_2 = np.array([0, 0, 1])
                    ray_to_local += perpendicular_1 * u_grid * 0.5
                    ray_to_local += perpendicular_2 * v_grid * 0.5
                    
                    # 变换到世界坐标系
                    ray_from_world = sensor_pos_world + sensor_rot_matrix.dot(ray_from_local)
                    ray_to_world = sensor_pos_world + sensor_rot_matrix.dot(ray_to_local)
                    
                    ray_froms_world.append(ray_from_world)
                    ray_tos_world.append(ray_to_world)
        else:
            # 处理轴向方向（原有逻辑）
            axis_idx, direction = sensor_dir
            
            # 确定用于构建网格的另外两个轴
            # (例如: 如果主轴是 Z(2), 网格轴就是 X(0) 和 Y(1))
            grid_axis_1 = (axis_idx + 1) % 3
            grid_axis_2 = (axis_idx + 2) % 3

            for u_grid in grid_coords_1:
                for v_grid in grid_coords_2:
                    
                    # 射线起点 (从中心偏移一点)
                    ray_from_local = np.array([0.0, 0.0, 0.0])
                    ray_from_local[axis_idx] = direction * start_offset
                    
                    # 射线终点 (在网格上展开)
                    ray_to_local = np.array([0.0, 0.0, 0.0])
                    ray_to_local[axis_idx] = direction * ray_range # 沿主轴
                    ray_to_local[grid_axis_1] = u_grid             # 网格 U
                    ray_to_local[grid_axis_2] = v_grid             # 网格 V

                    # 4. 将射线坐标变换到"世界坐标系"
                    ray_from_world = sensor_pos_world + sensor_rot_matrix.dot(ray_from_local)
                    ray_to_world = sensor_pos_world + sensor_rot_matrix.dot(ray_to_local)
                    
                    ray_froms_world.append(ray_from_world)
                    ray_tos_world.append(ray_to_world)

    # 5. 执行批量射线检测
    results = p.rayTestBatch(ray_froms_world, ray_tos_world)
    
    # 6. 处理结果，收集被击中的物体ID
    perceived_object_ids = set()
    
    if debug:
        p.removeAllUserDebugItems() # 清除上一帧的调试线

    for i, res in enumerate(results):
        hit_id = res[0]
        perceived_object_ids.add(hit_id)
        
        if debug:
            hit_pos = res[3]
            from_pos = ray_froms_world[i]
            to_pos = ray_tos_world[i]
            
            if hit_id == -1:
                # =============================================================
                # 【【【 *** 修复射线“不可见” BUG *** 】】】
                #
                # 错误代码: lifeTime=0.1 (一闪而过)
                # 正确代码: lifeTime=0 (永久显示, 直到下次被清除)
                p.addUserDebugLine(from_pos, to_pos, [0.0, 1.0, 0.0], lifeTime=0)
            else:
                # 【【【 *** 修复射线“不可见” BUG *** 】】】
                p.addUserDebugLine(from_pos, hit_pos, [1.0, 0.0, 0.0], lifeTime=0)
                # =============================================================

    return perceived_object_ids
# --- 修复结束 ---
# ============================================================


# ============================================================
# PFM (势场法) 模块 (保持不变)
# ============================================================
def calc_attractive_force(current_pos, goal_pos, k_att=1.0):
    """(此函数代码保持不变)"""
    dist_vec = np.array(goal_pos) - np.array(current_pos)
    dist = np.linalg.norm(dist_vec)
    if dist < 1e-6: return np.array([0.0, 0.0, 0.0])
    return k_att * (dist_vec / dist)

def calc_anisotropic_repulsive_force(current_pos, obs_center, obs_aabb_min, obs_aabb_max,
                                     k_rep=0.5, rho_0=0.35, k_aniso_xy=2.0, k_aniso_z=0.5):
    """(此函数代码保持不变)"""
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
                       step_size=0.05, max_steps=300, goal_threshold=0.05):
    """改进的PFM路径规划器，包含局部最小值逃逸机制"""
    print("  >> PFM: 启动势场法路径规划器...")
    obstacles_info = []
    for obs_id in obstacle_ids:
        aabb_min, aabb_max = p.getAABB(obs_id)
        obs_center = np.array([(aabb_min[0] + aabb_max[0]) / 2, (aabb_min[1] + aabb_max[1]) / 2, (aabb_min[2] + aabb_max[2]) / 2])
        aabb_diag = np.linalg.norm(np.array(aabb_max) - np.array(aabb_min))
        obstacles_info.append({"id": obs_id, "center": obs_center, "aabb_min": aabb_min, "aabb_max": aabb_max, "diag": aabb_diag})
    
    rho_0_base = 0.35; k_rep = 1.2; k_att = 1.0; step_size = 0.02
    k_aniso_xy = 2.0; k_aniso_z = 0.5
    path = [np.array(start_pos)]
    current_pos = np.array(start_pos)
    
    # 局部最小值检测和逃逸
    stuck_counter = 0
    last_positions = []
    escape_attempts = 0
    max_escape_attempts = 3

    for i in range(max_steps):
        f_att = calc_attractive_force(current_pos, goal_pos, k_att=k_att)
        f_rep_total = np.array([0.0, 0.0, 0.0])
        for obs in obstacles_info:
            rho_0 = (obs["diag"] / 2.0) + rho_0_base
            f_rep_obs = calc_anisotropic_repulsive_force(current_pos, obs["center"], obs["aabb_min"], obs["aabb_max"], 
                                                        k_rep=k_rep, rho_0=rho_0, k_aniso_xy=k_aniso_xy, k_aniso_z=k_aniso_z)
            f_rep_total += f_rep_obs
            
        f_total = f_att + f_rep_total
        
        # 检测局部最小值
        if np.linalg.norm(f_total) < 0.001:
            if escape_attempts < max_escape_attempts:
                print(f"  ⚠️ PFM: 检测到局部最小值，尝试逃逸 (尝试 {escape_attempts+1}/{max_escape_attempts})...")
                # 添加随机扰动来逃逸
                escape_direction = np.random.randn(3)
                escape_direction[2] = abs(escape_direction[2])  # 倾向于向上逃逸
                escape_direction = escape_direction / np.linalg.norm(escape_direction)
                current_pos = current_pos + escape_direction * 0.1
                escape_attempts += 1
                continue
            else:
                print(f"  ❌ PFM: 规划失败，在第 {i} 步陷入局部最小值。")
                return None
        
        # 检测是否卡住（位置几乎不变）
        last_positions.append(current_pos.copy())
        if len(last_positions) > 10:
            last_positions.pop(0)
            if np.std([np.linalg.norm(pos - current_pos) for pos in last_positions[-5:]]) < 0.001:
                if escape_attempts < max_escape_attempts:
                    print(f"  ⚠️ PFM: 检测到卡住，尝试侧向逃逸...")
                    # 计算垂直于目标方向的逃逸方向
                    to_goal = goal_pos - current_pos
                    to_goal[2] = 0  # 只在XY平面上
                    if np.linalg.norm(to_goal) > 0:
                        perpendicular = np.array([-to_goal[1], to_goal[0], 0])
                        perpendicular = perpendicular / np.linalg.norm(perpendicular)
                        # 随机选择左或右
                        perpendicular *= np.random.choice([-1, 1])
                        current_pos = current_pos + perpendicular * 0.15
                        escape_attempts += 1
                        continue
        
        current_pos = current_pos + step_size * (f_total / np.linalg.norm(f_total))
        path.append(current_pos)
        
        if np.linalg.norm(current_pos - np.array(goal_pos)) < goal_threshold:
            path.append(np.array(goal_pos)) 
            print(f"  ✅ PFM: 成功生成路径，共 {len(path)} 个路径点。")
            return path
            
    print(f"  ❌ PFM: 规划失败，超过最大步数 {max_steps}。")
    return None

def plan_path_with_pfm_randomized(start_pos, goal_pos, obstacle_ids,
                                  step_size=0.05, max_steps=300, goal_threshold=0.05, k_att=1.0):
    """带随机参数的PFM路径规划器"""
    print(f"  >> PFM(随机化): 启动势场法路径规划器 (step_size={step_size:.3f}, k_att={k_att:.2f})...")
    obstacles_info = []
    for obs_id in obstacle_ids:
        aabb_min, aabb_max = p.getAABB(obs_id)
        obs_center = np.array([(aabb_min[0] + aabb_max[0]) / 2, (aabb_min[1] + aabb_max[1]) / 2, (aabb_min[2] + aabb_max[2]) / 2])
        aabb_diag = np.linalg.norm(np.array(aabb_max) - np.array(aabb_min))
        obstacles_info.append({"id": obs_id, "center": obs_center, "aabb_min": aabb_min, "aabb_max": aabb_max, "diag": aabb_diag})
    
    rho_0_base = 0.35; k_rep = 1.2 + np.random.uniform(-0.2, 0.2)  # 也随机化斥力系数
    k_aniso_xy = 2.0; k_aniso_z = 0.5
    path = [np.array(start_pos)]
    current_pos = np.array(start_pos)
    
    # 局部最小值检测和逃逸
    stuck_counter = 0
    last_positions = []
    escape_attempts = 0
    max_escape_attempts = 3

    for i in range(max_steps):
        f_att = calc_attractive_force(current_pos, goal_pos, k_att=k_att)
        f_rep_total = np.array([0.0, 0.0, 0.0])
        for obs in obstacles_info:
            rho_0 = (obs["diag"] / 2.0) + rho_0_base
            f_rep_obs = calc_anisotropic_repulsive_force(current_pos, obs["center"], obs["aabb_min"], obs["aabb_max"], 
                                                        k_rep=k_rep, rho_0=rho_0, k_aniso_xy=k_aniso_xy, k_aniso_z=k_aniso_z)
            f_rep_total += f_rep_obs
            
        f_total = f_att + f_rep_total
        
        # 检测局部最小值
        if np.linalg.norm(f_total) < 0.001:
            if escape_attempts < max_escape_attempts:
                # 使用更大的随机扰动
                escape_direction = np.random.randn(3)
                escape_direction = escape_direction / np.linalg.norm(escape_direction)
                current_pos = current_pos + escape_direction * (0.15 + np.random.uniform(0, 0.1))
                escape_attempts += 1
                continue
            else:
                return None
        
        current_pos = current_pos + step_size * (f_total / np.linalg.norm(f_total))
        path.append(current_pos)
        
        if np.linalg.norm(current_pos - np.array(goal_pos)) < goal_threshold:
            path.append(np.array(goal_pos))
            print(f"  ✅ PFM(随机化): 成功生成路径，共 {len(path)} 个路径点。")
            return path
            
    return None

def generate_arc_path(start_pos, goal_pos, obstacle_ids, num_points=10):
    """生成弧形路径，避开障碍物"""
    print("  >> 生成弧形路径...")
    
    start = np.array(start_pos)
    goal = np.array(goal_pos)
    
    # 计算中点和弧的高度
    mid_point = (start + goal) / 2
    
    # 随机选择弧的方向（左、右、上）
    arc_direction = np.random.choice(['left', 'right', 'up'])
    
    if arc_direction == 'up':
        # 向上的弧
        arc_height = 0.2 + np.random.uniform(0, 0.15)
        mid_point[2] += arc_height
    elif arc_direction == 'left':
        # 向左的弧
        direction = goal - start
        perpendicular = np.array([-direction[1], direction[0], 0])
        if np.linalg.norm(perpendicular) > 0:
            perpendicular = perpendicular / np.linalg.norm(perpendicular)
            arc_offset = 0.15 + np.random.uniform(0, 0.1)
            mid_point += perpendicular * arc_offset
            mid_point[2] += 0.1  # 稍微抬高
    else:  # right
        # 向右的弧
        direction = goal - start
        perpendicular = np.array([direction[1], -direction[0], 0])
        if np.linalg.norm(perpendicular) > 0:
            perpendicular = perpendicular / np.linalg.norm(perpendicular)
            arc_offset = 0.15 + np.random.uniform(0, 0.1)
            mid_point += perpendicular * arc_offset
            mid_point[2] += 0.1  # 稍微抬高
    
    # 生成弧形路径点
    path = []
    for i in range(num_points + 1):
        t = i / num_points
        # 使用二次贝塞尔曲线
        point = (1 - t)**2 * start + 2 * (1 - t) * t * mid_point + t**2 * goal
        path.append(point)
    
    # 检查路径是否与障碍物碰撞
    for obs_id in obstacle_ids:
        aabb_min, aabb_max = p.getAABB(obs_id)
        for point in path:
            # 简单的AABB碰撞检测
            if (aabb_min[0] - 0.05 <= point[0] <= aabb_max[0] + 0.05 and
                aabb_min[1] - 0.05 <= point[1] <= aabb_max[1] + 0.05 and
                aabb_min[2] - 0.05 <= point[2] <= aabb_max[2] + 0.05):
                print("  >> 弧形路径与障碍物碰撞，放弃此路径")
                return None
    
    print(f"  ✅ 成功生成{arc_direction}向弧形路径，共 {len(path)} 个路径点")
    return path

# ============================================================
# 路径历史记录（用于避免重复路径）
# ============================================================
PATH_HISTORY = []  # 存储历史路径的特征
MAX_PATH_HISTORY = 10  # 最多记录10条历史路径

def add_path_to_history(path_points):
    """添加路径到历史记录"""
    global PATH_HISTORY
    if len(path_points) > 2:
        # 计算路径特征（使用关键点的哈希）
        key_points = [path_points[0], path_points[len(path_points)//2], path_points[-1]]
        path_feature = hash(tuple(map(tuple, key_points)))
        PATH_HISTORY.append(path_feature)
        if len(PATH_HISTORY) > MAX_PATH_HISTORY:
            PATH_HISTORY.pop(0)

def is_path_similar_to_history(path_points):
    """检查路径是否与历史路径相似"""
    if len(path_points) <= 2:
        return False
    key_points = [path_points[0], path_points[len(path_points)//2], path_points[-1]]
    path_feature = hash(tuple(map(tuple, key_points)))
    return path_feature in PATH_HISTORY

# ============================================================
# 自动避障路径规划模块 (增强版，支持路径多样性)
# ============================================================
def plan_and_execute_motion(robot_id, goal_pos, goal_orn, obstacle_ids, target_joints_override=None, **kwargs):
    """
    增强版路径规划，支持路径多样性和历史记录
    """
    
    print(f"--- 正在规划前往 {goal_pos} 的路径 (避开 {len(obstacle_ids)} 个感知到的障碍物) ---")

    execution_kwargs = kwargs.copy()
    execution_kwargs["obstacle_ids"] = obstacle_ids

    num_arm_joints = 7
    
    default_null_space_params = _DEFAULT_NULL_SPACE_PARAMS.copy()
    default_null_space_params["restPoses"] = list(ROBOT_HOME_CONFIG)
    
    current_joint_pos = np.asarray([p.getJointState(robot_id, i)[0] for i in range(7)])
    current_gripper_pos = [p.getJointState(robot_id, 9)[0], p.getJointState(robot_id, 10)[0]]
    current_pos, *_ = p.getLinkState(robot_id, ROBOT_END_EFFECTOR_LINK_ID, computeForwardKinematics=True)
    
    # 随机选择路径规划策略（增加多样性）
    use_alternative_strategy = np.random.random() > 0.5  # 50%概率使用替代策略

    # 1. 自动避障 IK 模式 (此部分逻辑保持不变)
    if target_joints_override is not None:
        print("  >> 使用了 'target_joints_override'，启用自动避障 IK 模式。")
        ee_state = p.getLinkState(robot_id, ROBOT_END_EFFECTOR_LINK_ID, computeForwardKinematics=True)
        current_ee_pos = np.array(ee_state[0]); goal_pos = np.array(goal_pos)
        
        interferer_id = kwargs.get("interferer_id")
        if interferer_id in obstacle_ids and obstacle_ids:
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
                        
                        ik_params_auto = default_null_space_params.copy()
                        joint_waypoints = []
                        
                        for wp in waypoints:
                            ik_params_auto["restPoses"] = list(prev_j)
                            
                            j_wp = p.calculateInverseKinematics(robot_id, ROBOT_END_EFFECTOR_LINK_ID,
                                                                wp, goal_orn,
                                                                **ik_params_auto)[:7]
                            
                            if is_path_colliding(robot_id, prev_j, j_wp, obstacle_ids, [0.04,0.04], [0.04,0.04]):
                                path_ok = False; break
                            
                            prev_j = j_wp
                            joint_waypoints.append(j_wp)
                        
                        if path_ok:
                            print("  ✅ 自动避障路径安全，执行中...")
                            
                            for j_wp_target in joint_waypoints:
                                success = move_to_joints(robot_id, j_wp_target, **execution_kwargs) 
                                if not success:
                                    print("  ❌ 自动避障路径在 *执行* 期间失败。")
                                    return False
                            return True
                    except Exception: continue
                
                print("  ❌ 所有自动绕行路径失败，将尝试默认路径。")
                
        target_joints = target_joints_override
    else:
        target_joints = p.calculateInverseKinematics(
            robot_id, ROBOT_END_EFFECTOR_LINK_ID, goal_pos, goal_orn, **default_null_space_params
        )[:7]

    # 2. 检查直接路径 (此部分逻辑保持不变)
    if not is_path_colliding(robot_id, current_joint_pos, target_joints, obstacle_ids,
                             current_gripper_pos, current_gripper_pos):
        print("  >> 直接路径安全，正在执行...")
        success = move_to_joints(robot_id, target_joints, **execution_kwargs) 
        return success 

    # 3. 使用 PFM 规划器 (Plan A)
    print("  >> 直接路径被阻挡，启动 PFM 路径规划器...")
    workspace_path = plan_path_with_pfm(
        start_pos=current_pos, goal_pos=goal_pos, obstacle_ids=obstacle_ids
    )
    
    # =======================================================================
    # === 🟢 【【【 逻辑修复 】】】 🟢 ===
    # =======================================================================
    
    # --- 3.5. 验证 PFM 路径 (如果找到了) ---
    pfm_path_is_valid = False
    pfm_joint_path = []

    if workspace_path is not None:
        print("  >> PFM 找到了工作空间路径，正在验证 C-Space...")
        last_joint_pos = current_joint_pos.copy()
        ik_params = default_null_space_params.copy()
        pfm_path_is_valid = True # 假设有效，直到被证明无效
        
        for i, wp_pos in enumerate(workspace_path):
            try:
                # (减少采样点以加快验证)
                if i % 3 != 0 and i != (len(workspace_path) - 1):
                    continue

                ik_params["restPoses"] = list(last_joint_pos) 
                wp_joints = p.calculateInverseKinematics(
                    robot_id, ROBOT_END_EFFECTOR_LINK_ID, wp_pos, goal_orn, **ik_params
                )[:7]
                
                if is_path_colliding(robot_id, last_joint_pos, wp_joints, obstacle_ids,
                                     current_gripper_pos, current_gripper_pos):
                    print(f"  ❌ PFM 路径在 C-Space 中发现碰撞 (段 {i})。")
                    pfm_path_is_valid = False # 标记为无效
                    break
                    
                pfm_joint_path.append(wp_joints)
                last_joint_pos = wp_joints 
            except Exception as e:
                print(f"  ❌ PFM 路径点 {i} ({wp_pos}) IK 求解失败。")
                pfm_path_is_valid = False # 标记为无效
                break
    
    # --- 4. 决策：执行 Plan A (如果有效) 或 切换到 Plan B ---

    # === 4a. 如果 PFM 路径有效，则执行 Plan A ===
    if pfm_path_is_valid:
        print(f"  ✅ PFM 路径在 C-Space 中验证安全，执行中...")
        for joint_target in pfm_joint_path:
            success = move_to_joints(robot_id, joint_target, max_velocity=1.5, **execution_kwargs) 
            if not success:
                print("  ❌ PFM 路径在 *执行* 期间失败。")
                return False 
        
        # 移动到最终的精确关节目标
        success_final = move_to_joints(robot_id, target_joints, max_velocity=1.0, **execution_kwargs) 
        return success_final

    # === 4b. 如果 PFM 路径无效 (或 PFM 本身失败)，则触发 Plan B ===
    else:
        if workspace_path is None:
            print("  ❌ PFM 规划器未能找到路径 (可能陷入局部最小值)。")
        else:
            print("  ❌ PFM 路径 C-Space 验证失败。")

        if not obstacle_ids:
            print("  >> PFM/C-Space 失败，且未感知到障碍物。规划终止。")
            return False 

        print(f"  >> 启动 Plan B：尝试自动生成'绕行'路径...")
        
        # 分析所有障碍物，找到最高点
        max_obstacle_z = 0.3
        obstacle_centers = []
        for obs_id in obstacle_ids:
            try:
                aabb_min, aabb_max = p.getAABB(obs_id)
                max_obstacle_z = max(max_obstacle_z, aabb_max[2])
                obs_center = [(aabb_min[0] + aabb_max[0])/2, 
                             (aabb_min[1] + aabb_max[1])/2,
                             (aabb_min[2] + aabb_max[2])/2]
                obstacle_centers.append(obs_center)
            except Exception:
                pass

        # 计算安全高度
        z_safe_cruise = max(max_obstacle_z + 0.2, current_pos[2] + 0.1, goal_pos[2] + 0.1)
        z_safe_cruise = max(z_safe_cruise, 0.6)
        
        # 尝试多种绕行策略
        detour_strategies = []
        
        # 策略1：直接上升-巡航-下降
        detour_strategies.append([
            np.array([current_pos[0], current_pos[1], z_safe_cruise]),
            np.array([goal_pos[0], goal_pos[1], z_safe_cruise]),
            np.array(goal_pos)
        ])
        
        # 策略2：侧向绕行（如果有障碍物中心信息）
        if obstacle_centers:
            # 计算绕过障碍物的侧向路径
            obs_center = obstacle_centers[0]
            direction_to_goal = np.array([goal_pos[0] - current_pos[0], 
                                         goal_pos[1] - current_pos[1], 0])
            if np.linalg.norm(direction_to_goal) > 0:
                direction_to_goal = direction_to_goal / np.linalg.norm(direction_to_goal)
                perpendicular = np.array([-direction_to_goal[1], direction_to_goal[0], 0])
                
                # 判断应该从哪一侧绕过
                to_obs = np.array([obs_center[0] - current_pos[0], 
                                  obs_center[1] - current_pos[1], 0])
                if np.dot(perpendicular, to_obs) > 0:
                    perpendicular = -perpendicular
                
                # 生成侧向绕行路径
                side_offset = 0.3
                mid_point = np.array([
                    obs_center[0] + perpendicular[0] * side_offset,
                    obs_center[1] + perpendicular[1] * side_offset,
                    max(current_pos[2], goal_pos[2]) + 0.1
                ])
                
                detour_strategies.append([
                    np.array([current_pos[0], current_pos[1], current_pos[2] + 0.1]),
                    mid_point,
                    np.array([goal_pos[0], goal_pos[1], goal_pos[2] + 0.1]),
                    np.array(goal_pos)
                ])
        
        # 尝试每个策略
        for strategy_idx, detour_waypoints in enumerate(detour_strategies):
            print(f"  >> Plan B 策略 {strategy_idx + 1}: 尝试规划...")
            
            joint_space_path = []
            last_joint_pos = current_joint_pos.copy()
            ik_params = default_null_space_params.copy()
            path_ok = True

            for i, wp in enumerate(detour_waypoints):
                ik_params["restPoses"] = list(last_joint_pos)
                try:
                    wp_joints = p.calculateInverseKinematics(
                        robot_id, ROBOT_END_EFFECTOR_LINK_ID, wp, goal_orn, **ik_params
                    )[:7]
                    
                    if is_path_colliding(robot_id, last_joint_pos, wp_joints, obstacle_ids,
                                         current_gripper_pos, current_gripper_pos):
                        print(f"    ❌ 策略 {strategy_idx + 1} 在 C-Space 中发现碰撞 (段 {i+1})。")
                        path_ok = False
                        break
                        
                    joint_space_path.append(wp_joints)
                    last_joint_pos = wp_joints
                    
                except Exception as e:
                    print(f"    ❌ 策略 {strategy_idx + 1} 路径点 {i+1} IK 求解失败。")
                    path_ok = False
                    break
            
            if path_ok:
                print(f"  ✅ Plan B 策略 {strategy_idx + 1} 验证安全，执行中...")
                for joint_target in joint_space_path:
                    success = move_to_joints(robot_id, joint_target, max_velocity=1.5, **execution_kwargs) 
                    if not success:
                        print(f"    ❌ 策略 {strategy_idx + 1} 在执行期间失败。")
                        break
                else:
                    # 所有动作都成功执行
                    print(f"  ✅ Plan B 策略 {strategy_idx + 1} 执行成功！")
                    return True
            else:
                print(f"    ⚠️ 策略 {strategy_idx + 1} 规划失败，尝试下一个策略...")
        
        # 所有策略都失败了，尝试随机探索
        print("  ❌ Plan B 所有策略都失败了。")
        print("  >> 最后尝试：执行随机探索移动...")
        
        # =============================================================
        # === 🔴 【【【 *** BUG 修复 (TypeError) *** 】】】 🔴 ===
        # =============================================================
        
        # 'execution_kwargs' 已经包含了 'obstacle_ids'，
        # 而 'obstacle_ids' 也被作为位置参数传递，导致了 TypeError。
        # 我们需要复制一份 kwargs，并从中移除冲突的键。
        
        exploration_kwargs = execution_kwargs.copy()
        exploration_kwargs.pop('obstacle_ids', None) # 移除冲突的键

        exploration_success = perform_random_exploration(
            robot_id, 
            obstacle_ids, 
            **exploration_kwargs  # <--- 使用清理过的 kwargs
        )
        # =============================================================
        # === 修复结束 ===
        # =============================================================

        
        if exploration_success:
            print("  >> 随机探索成功，从新位置重新尝试到达目标...")
            # 从新位置重新进行完整的路径规划（而不是只尝试直接路径）
            current_joint_pos = np.asarray([p.getJointState(robot_id, i)[0] for i in range(7)])
            current_gripper_pos = [p.getJointState(robot_id, 9)[0], p.getJointState(robot_id, 10)[0]]
            current_pos, *_ = p.getLinkState(robot_id, ROBOT_END_EFFECTOR_LINK_ID, computeForwardKinematics=True)
            
            # 尝试多种路径规划策略
            strategies = []
            
            # 策略1：直接路径
            if not is_path_colliding(robot_id, current_joint_pos, target_joints, obstacle_ids,
                                     current_gripper_pos, current_gripper_pos):
                strategies.append(("direct", target_joints, None))
            
            # 策略2：重新使用PFM规划器（带随机参数）
            print("  >> 从新位置使用PFM重新规划...")
            pfm_step_size = 0.02 + np.random.uniform(-0.01, 0.02)  # 随机化步长
            pfm_k_att = 1.0 + np.random.uniform(-0.3, 0.3)  # 随机化吸引力系数
            
            # 修改PFM调用，使用随机参数
            workspace_path_new = plan_path_with_pfm_randomized(
                start_pos=current_pos, goal_pos=goal_pos, obstacle_ids=obstacle_ids,
                step_size=pfm_step_size, k_att=pfm_k_att
            )
            
            if workspace_path_new is not None:
                # 验证PFM路径
                pfm_valid = True
                pfm_joint_waypoints = []
                prev_j = current_joint_pos.copy()
                
                for wp in workspace_path_new[::3]:  # 采样路径点
                    try:
                        j_wp = p.calculateInverseKinematics(
                            robot_id, ROBOT_END_EFFECTOR_LINK_ID,
                            wp, goal_orn, **default_null_space_params
                        )[:7]
                        
                        if is_path_colliding(robot_id, prev_j, j_wp, obstacle_ids,
                                           current_gripper_pos, current_gripper_pos):
                            pfm_valid = False
                            break
                        
                        prev_j = j_wp
                        pfm_joint_waypoints.append(j_wp)
                    except:
                        pfm_valid = False
                        break
                
                if pfm_valid and not is_path_similar_to_history(workspace_path_new):
                    strategies.append(("pfm_new", pfm_joint_waypoints, workspace_path_new))
            
            # 策略3：生成弧形路径
            arc_path = generate_arc_path(current_pos, goal_pos, obstacle_ids)
            if arc_path is not None:
                arc_valid = True
                arc_joint_waypoints = []
                prev_j = current_joint_pos.copy()
                
                for wp in arc_path:
                    try:
                        j_wp = p.calculateInverseKinematics(
                            robot_id, ROBOT_END_EFFECTOR_LINK_ID,
                            wp, goal_orn, **default_null_space_params
                        )[:7]
                        
                        if is_path_colliding(robot_id, prev_j, j_wp, obstacle_ids,
                                           current_gripper_pos, current_gripper_pos):
                            arc_valid = False
                            break
                        
                        prev_j = j_wp
                        arc_joint_waypoints.append(j_wp)
                    except:
                        arc_valid = False
                        break
                
                if arc_valid and not is_path_similar_to_history(arc_path):
                    strategies.append(("arc", arc_joint_waypoints, arc_path))
            
            # 随机选择一个可行策略
            if strategies:
                strategy_name, joint_path, workspace_path = strategies[np.random.randint(len(strategies))]
                print(f"  >> 选择策略: {strategy_name}")
                
                if strategy_name == "direct":
                    success = move_to_joints(robot_id, joint_path, **execution_kwargs)
                    if success:
                        print("  ✅ 通过随机探索+直接路径找到了新路径！")
                        return True
                else:
                    # 执行路径点序列
                    for j_wp in joint_path:
                        success = move_to_joints(robot_id, j_wp, **execution_kwargs)
                        if not success:
                            print(f"  ❌ {strategy_name}策略执行失败")
                            return False
                    
                    # 记录成功的路径到历史
                    if workspace_path is not None:
                        add_path_to_history(workspace_path)
                    
                    print(f"  ✅ 通过随机探索+{strategy_name}策略找到了新路径！")
                    return True
            else:
                print("  >> 新位置没有找到可行路径，等待重试...")
        
        return False

# ============================================================
# 随机探索模块 (新增)
# ============================================================

def perform_random_exploration(robot_id, obstacle_ids, **kwargs):
    """
    【【修改】】
    执行大范围、长距离的随机探索移动，尝试找到更好的位置来规划路径。
    
    Args:
        robot_id: 机器人ID
        obstacle_ids: 当前感知到的障碍物ID列表
        **kwargs: 其他参数（包括仿真参数）
    
    Returns:
        bool: 是否成功执行了随机移动
    """
    print("  >> 开始【大范围】随机探索移动...")
    
    # 获取当前末端执行器位置
    current_state = p.getLinkState(robot_id, ROBOT_END_EFFECTOR_LINK_ID, computeForwardKinematics=True)
    current_pos = np.array(current_state[0])
    current_orn = current_state[1]
    
    # 【【新】】 定义一个安全的大范围工作空间
    # 扩大探索范围，让机械臂能够探索更大的空间
    X_MIN, X_MAX = 0.1, 0.8  # 扩大X轴范围
    Y_MIN, Y_MAX = -0.6, 0.6  # 扩大Y轴范围
    Z_MIN, Z_MAX = 0.15, 0.8  # 扩大Z轴范围，允许更高和更低的位置
    
    # 生成多个随机目标点
    exploration_candidates = []
    
    # -----------------------------------------------------------------
    # 策略 1 (已修改): 在整个工作空间内进行大范围采样
    print(f"    >> 探索策略 1: 大范围工作空间采样 (X:[{X_MIN},{X_MAX}], Y:[{Y_MIN},{Y_MAX}], Z:[{Z_MIN},{Z_MAX}])...")
    # 增加采样点数量，从5个增加到10个，覆盖更多空间
    for _ in range(10): 
        # 使用更激进的随机策略，偏向于远离当前位置
        random_target = np.array([
            np.random.uniform(X_MIN, X_MAX),
            np.random.uniform(Y_MIN, Y_MAX),
            np.random.uniform(Z_MIN, Z_MAX)
        ])
        # 有50%概率生成远离当前位置的点
        if np.random.random() > 0.5:
            # 计算与当前位置的偏移，确保探索点远离当前位置
            offset_direction = random_target - current_pos
            offset_norm = np.linalg.norm(offset_direction)
            if offset_norm > 0 and offset_norm < 0.3:  # 如果太近，推远一些
                offset_direction = offset_direction / offset_norm * np.random.uniform(0.3, 0.5)
                random_target = current_pos + offset_direction
                # 确保仍在工作空间内
                random_target[0] = np.clip(random_target[0], X_MIN, X_MAX)
                random_target[1] = np.clip(random_target[1], Y_MIN, Y_MAX)
                random_target[2] = np.clip(random_target[2], Z_MIN, Z_MAX)
        exploration_candidates.append(random_target)
    # -----------------------------------------------------------------

    
    # -----------------------------------------------------------------
    # 策略 2 (已增强): 远离障碍物 (更远的距离)
    if obstacle_ids:
        # 计算所有障碍物的平均位置
        obstacle_centers = []
        for obs_id in obstacle_ids:
            try:
                aabb_min, aabb_max = p.getAABB(obs_id)
                obs_center = np.array([
                    (aabb_min[0] + aabb_max[0]) / 2,
                    (aabb_min[1] + aabb_max[1]) / 2,
                    (aabb_min[2] + aabb_max[2]) / 2
                ])
                obstacle_centers.append(obs_center)
            except:
                pass
        
        if obstacle_centers:
            avg_obstacle_pos = np.mean(obstacle_centers, axis=0)
            # 计算远离障碍物的方向
            escape_direction = current_pos - avg_obstacle_pos
            if np.linalg.norm(escape_direction[:2]) > 0:  # 主要考虑XY平面
                escape_direction = escape_direction / np.linalg.norm(escape_direction)
                
                # 【【修改】】 大幅增加探索距离
                print("    >> 探索策略 2: 远离障碍物 (超长距离)...")
                # 探索更远的距离，从障碍物逃离得更远
                for dist in [0.3, 0.45, 0.6, 0.75]: # 增加到4个距离级别，最远达0.75米
                    escape_target = current_pos + escape_direction * dist
                    # Z轴也进行更大范围的随机调整
                    escape_target[2] = current_pos[2] + np.random.uniform(-0.2, 0.3)  # 允许上下大幅移动
                    
                    # 确保在工作空间内
                    escape_target[0] = np.clip(escape_target[0], X_MIN, X_MAX)
                    escape_target[1] = np.clip(escape_target[1], Y_MIN, Y_MAX)
                    escape_target[2] = np.clip(escape_target[2], Z_MIN, Z_MAX)
                    
                    exploration_candidates.append(escape_target)
    # -----------------------------------------------------------------


    # -----------------------------------------------------------------
    # 策略 3 (已增强): 多个高度层级探索
    print("    >> 探索策略 3: 多层级高度探索...")
    # 不只是最高点，尝试多个不同高度
    for z_level in [Z_MAX, Z_MAX * 0.8, Z_MAX * 0.6, Z_MIN * 1.5]:
        high_target = current_pos.copy()
        high_target[2] = z_level
        # 在XY平面也进行随机偏移
        high_target[0] += np.random.uniform(-0.2, 0.2)
        high_target[1] += np.random.uniform(-0.2, 0.2)
        # 确保在工作空间内
        high_target[0] = np.clip(high_target[0], X_MIN, X_MAX)
        high_target[1] = np.clip(high_target[1], Y_MIN, Y_MAX)
        exploration_candidates.append(high_target)
    # -----------------------------------------------------------------

    # -----------------------------------------------------------------
    # 【【新】】 策略 4: 多个安全撤退点
    print("    >> 探索策略 4: 尝试多个安全撤退位置...")
    # 不只是Home位置，增加多个预定义的安全位置
    safe_positions = [
        np.array([0.3, 0.0, 0.5]),   # Home位置
        np.array([0.4, 0.3, 0.6]),   # 右上方
        np.array([0.4, -0.3, 0.6]),  # 左上方
        np.array([0.5, 0.0, 0.4]),   # 前方中等高度
        np.array([0.25, 0.4, 0.5]),  # 右侧
        np.array([0.25, -0.4, 0.5]), # 左侧
    ]
    for safe_pos in safe_positions:
        exploration_candidates.append(safe_pos)
    # -----------------------------------------------------------------
    
    # -----------------------------------------------------------------
    # 【【新增】】 策略 5: 螺旋式探索
    print("    >> 探索策略 5: 螺旋式大范围探索...")
    # 在当前位置周围进行螺旋式探索
    num_spiral_points = 6
    for i in range(num_spiral_points):
        angle = (2 * np.pi * i) / num_spiral_points
        # 螺旋半径递增
        for radius in [0.2, 0.35, 0.5]:
            spiral_target = current_pos.copy()
            spiral_target[0] += radius * np.cos(angle)
            spiral_target[1] += radius * np.sin(angle)
            # Z轴也进行变化
            spiral_target[2] += np.random.uniform(-0.15, 0.25)
            # 确保在工作空间内
            spiral_target[0] = np.clip(spiral_target[0], X_MIN, X_MAX)
            spiral_target[1] = np.clip(spiral_target[1], Y_MIN, Y_MAX)
            spiral_target[2] = np.clip(spiral_target[2], Z_MIN, Z_MAX)
            exploration_candidates.append(spiral_target)
    # -----------------------------------------------------------------

    # 尝试每个候选目标
    print(f"  >> (大范围) 生成了 {len(exploration_candidates)} 个探索目标点")
    
    # (这部分检查逻辑保持不变)
    for idx, target_pos in enumerate(exploration_candidates):
        print(f"  >> 尝试探索目标 {idx+1}/{len(exploration_candidates)}: {target_pos}")
        
        # 计算到目标位置的IK
        try:
            target_joints = p.calculateInverseKinematics(
                robot_id, ROBOT_END_EFFECTOR_LINK_ID,
                target_pos, current_orn,
                **_DEFAULT_NULL_SPACE_PARAMS
            )[:7]
            
            # 获取当前关节位置
            current_joints = np.asarray([p.getJointState(robot_id, i)[0] for i in range(7)])
            current_gripper = [p.getJointState(robot_id, 9)[0], p.getJointState(robot_id, 10)[0]]
            
            # 检查路径是否会碰撞 (核心安全保证)
            if not is_path_colliding(robot_id, current_joints, target_joints, 
                                   obstacle_ids, current_gripper, current_gripper):
                # 执行移动
                print(f"    ✓ 目标 {idx+1} 路径安全，执行移动...")
                # 增加移动速度，让探索动作更明显
                success = move_to_joints(robot_id, target_joints, max_velocity=2.0, **kwargs)
                
                if success:
                    print(f"  ✅ 随机探索成功移动到新位置!")
                    return True
                else:
                    print(f"    ✗ 执行移动失败")
            else:
                print(f"    ✗ 目标 {idx+1} 路径会碰撞")
                
        except Exception as e:
            print(f"    ✗ 目标 {idx+1} IK求解失败: {e}")
            continue
    
    # -----------------------------------------------------------------
    # 如果所有探索都失败，尝试最后的手段：(超大幅度的)随机关节运动
    print("  >> 所有探索目标都失败，尝试(超大幅度的)关节运动...")
    current_joints = np.asarray([p.getJointState(robot_id, i)[0] for i in range(7)])
    
    # 增加尝试次数，从3次增加到5次
    for attempt in range(5):
        # 【【修改】】 大幅增加关节空间的探索幅度
        # 使用递增的探索幅度，越后面的尝试越激进
        amplitude = 0.4 + (attempt * 0.15)  # 从0.4递增到1.15弧度（约23度到66度）
        joint_offset = np.random.uniform(-amplitude, amplitude, size=7) 
        
        # 根据尝试次数调整不同关节的移动策略
        if attempt < 2:
            # 前两次尝试，保守一些
            joint_offset[0] *= 0.5
            joint_offset[-2:] *= 0.4
        else:
            # 后面的尝试，更激进
            joint_offset[0] *= 0.7  # 基座关节也允许更大移动
            joint_offset[-2:] *= 0.6  # 末端关节也增加移动幅度
        
        target_joints = current_joints + joint_offset
        
        # 检查关节限制
        for i in range(7):
            joint_info = p.getJointInfo(robot_id, i)
            lower_limit = joint_info[8]
            upper_limit = joint_info[9]
            target_joints[i] = np.clip(target_joints[i], lower_limit, upper_limit)
        
        # 检查是否会碰撞
        current_gripper = [p.getJointState(robot_id, 9)[0], p.getJointState(robot_id, 10)[0]]
        if not is_state_colliding(robot_id, target_joints, obstacle_ids, current_gripper):
            print(f"    >> 尝试关节微调 {attempt+1}/3...")
            # 增加关节移动速度，让摆动更明显
            success = move_to_joints(robot_id, target_joints, max_velocity=1.0, timeout=5, **kwargs)
            if success:
                print(f"  ✅ 通过关节微调成功改变位置!")
                return True
    # -----------------------------------------------------------------
    
    print("  ❌ 随机探索移动全部失败")
    return False

# ============================================================
# 运动与夹爪控制 (保持不变)
# ============================================================

def simulate(steps=None, seconds=None, slow_down=True, 
             interferer_id=None, interferer_joints=None, interferer_update_rate=120):
    """(此函数代码保持不变)"""
    global _GLOBAL_SIM_STEP_COUNTER 
    
    seconds_passed = 0.0
    steps_this_call = 0 
    start_time = time.time()
    
    while True:
        p.stepSimulation()
        
        _GLOBAL_SIM_STEP_COUNTER += 1 
        steps_this_call += 1          
        
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


def move_to_joints(robot_id, target_joint_pos, max_velocity=1, timeout=5, **kwargs):
    """
    (此函数已更新，移除了导致超时的“反应式避障”区域，
     替换为一个“快速失败”的近距离安全Failsafe)
    """
    
    # --- 1. 提取参数 ---
    sim_kwargs = {
        "interferer_id": kwargs.get("interferer_id"),
        "interferer_joints": kwargs.get("interferer_joints"),
        "interferer_update_rate": kwargs.get("interferer_update_rate", 120),
        "slow_down": kwargs.get("slow_down", True) 
    }
    
    interferer_id = kwargs.get("interferer_id")
    obstacle_ids = kwargs.get("obstacle_ids", [])
    
    # 【【新】】 Failsafe 距离：如果传感器没看到，但实际距离小于此值，则立即失败
    PROXIMITY_FAILSAFE_DISTANCE = 0.03 # 3 厘米
    
    target_joint_pos = np.asarray(target_joint_pos)
    num_arm_joints = len(target_joint_pos)
        
    counter = 0
    while True:
        
        # --- 2. 基于“感知”的硬碰撞检测 (Failsafe 1) ---
        # (此检查依赖于 *成功* 的感知)
        if obstacle_ids: 
            current_joint_pos_check = np.asarray([p.getJointState(robot_id, i)[0] for i in range(num_arm_joints)])
            current_gripper_pos_check = [p.getJointState(robot_id, 9)[0], p.getJointState(robot_id, 10)[0]]
            
            if is_state_colliding(robot_id, current_joint_pos_check, obstacle_ids, current_gripper_pos_check):
                print("  [!!] EXECUTION-TIME COLLISION DETECTED! (Failsafe 1: 基于感知)")
                print("  [!!] 立即停止机器人... (将触发重规划)")
                
                for joint_id in range(num_arm_joints):
                    p.setJointMotorControl2(
                        robot_id, joint_id, controlMode=p.VELOCITY_CONTROL,
                        targetVelocity=0, force=200 
                    )
                simulate(steps=1, **sim_kwargs)
                return False # <--- 快速失败
        
        # --- 3. 基于“Groud-Truth”的近距离Failsafe (Failsafe 2) ---
        # (此检查用于在 *感知失败* 时提供保护，防止超时)
        if interferer_id is not None:
            closest_points = p.getClosestPoints(robot_id, interferer_id, PROXIMITY_FAILSAFE_DISTANCE)
            
            if closest_points:
                print(f"  [!!] EXECUTION-TIME PROXIMITY FAILSAFE! (Failsafe 2: 实际距离 < {PROXIMITY_FAILSAFE_DISTANCE}m)")
                print("  [!!] 立即停止机器人... (将触发重规划)")
                
                for joint_id in range(num_arm_joints):
                    p.setJointMotorControl2(
                        robot_id, joint_id, controlMode=p.VELOCITY_CONTROL,
                        targetVelocity=0, force=200 
                    )
                simulate(steps=1, **sim_kwargs)
                return False # <--- 快速失败 (!!)
        
        # --- (已移除) ---
        # 旧的 "Warning Zone" 和 "Nudge" 逻辑已被移除，
        # 因为它会导致在 'is_in_warning_zone' 为 True 时无法成功返回，
        # 从而导致 5 秒钟的超时。
        
        # --- 4. 持续设置电机目标 ---
        # 始终瞄准原始目标
        for joint_id in range(num_arm_joints):
            p.setJointMotorControl2(
                robot_id, joint_id, controlMode=p.POSITION_CONTROL,
                targetPosition=target_joint_pos[joint_id], 
                maxVelocity=max_velocity, force=100
            )

        # --- 5. 检查是否到达 *原始* 目标 ---
        current_joint_pos = np.asarray([p.getJointState(robot_id, i)[0] for i in range(num_arm_joints)])
        
        # (移除了 'and not is_in_warning_zone')
        if np.allclose(current_joint_pos, target_joint_pos, atol=0.01):
            return True

        # --- 6. 步进仿真和超时 ---
        simulate(steps=1, **sim_kwargs) 
        
        counter += 1
        if counter > timeout / DELTA_T:
            print('WARNING: timeout while moving to joint position.')
            return False
            
    return True

def gripper_open(robot_id, **kwargs):
    """(此函数代码保持不变)"""
    p.setJointMotorControl2(robot_id, 9, controlMode=p.POSITION_CONTROL, targetPosition=0.04, force=100)
    p.setJointMotorControl2(robot_id, 10, controlMode=p.POSITION_CONTROL, targetPosition=0.04, force=100)
    simulate(seconds=1.0, **kwargs) 

def gripper_close(robot_id, **kwargs):
    """(此函数代码保持不变)"""
    p.setJointMotorControl2(robot_id, 9, controlMode=p.VELOCITY_CONTROL, targetVelocity=-0.05, force=100)
    for _ in range(int(0.5 / DELTA_T)):
        simulate(steps=1, **kwargs) 
        finger_pos = p.getJointState(robot_id, 9)[0]
        p.setJointMotorControl2(robot_id, 10, controlMode=p.POSITION_CONTROL, targetPosition=finger_pos, force=100)

def move_to_pose(robot_id, target_ee_pos, target_ee_orientation=None, **kwargs):
    """(此函数代码保持不变)"""
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