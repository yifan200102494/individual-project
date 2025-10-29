"""
高层规划器模块
整合各个模块，提供统一的路径规划和执行接口
"""

import pybullet as p
import numpy as np

from constants import DEFAULT_NULL_SPACE_PARAMS, ROBOT_HOME_CONFIG, ROBOT_END_EFFECTOR_LINK_ID
from collision_detection import is_path_colliding
from path_planning import (
    plan_path_with_pfm,
    generate_arc_path,
    validate_workspace_path,
    generate_detour_strategies,
    add_path_to_history,
    is_path_similar_to_history
)
from exploration import perform_random_exploration
from motion_control import move_to_joints


def plan_and_execute_motion(robot_id, goal_pos, goal_orn, obstacle_ids, target_joints_override=None, **kwargs):
    """
    增强版路径规划，支持路径多样性和历史记录
    
    Args:
        robot_id: 机器人ID
        goal_pos: 目标位置
        goal_orn: 目标方向
        obstacle_ids: 障碍物ID列表
        target_joints_override: 覆盖的目标关节位置
        **kwargs: 其他参数
    
    Returns:
        bool: 是否成功到达目标
    """
    
    print(f"--- 正在规划前往 {goal_pos} 的路径 (避开 {len(obstacle_ids)} 个感知到的障碍物) ---")

    execution_kwargs = kwargs.copy()
    execution_kwargs["obstacle_ids"] = obstacle_ids

    default_null_space_params = DEFAULT_NULL_SPACE_PARAMS.copy()
    default_null_space_params["restPoses"] = list(ROBOT_HOME_CONFIG)
    
    current_joint_pos = np.asarray([p.getJointState(robot_id, i)[0] for i in range(7)])
    current_gripper_pos = [p.getJointState(robot_id, 9)[0], p.getJointState(robot_id, 10)[0]]
    current_pos, *_ = p.getLinkState(robot_id, ROBOT_END_EFFECTOR_LINK_ID, computeForwardKinematics=True)
    
    # 1. 自动避障 IK 模式
    if target_joints_override is not None:
        result = try_auto_avoidance_ik_mode(
            robot_id, goal_pos, goal_orn, obstacle_ids, 
            current_joint_pos, current_gripper_pos, 
            default_null_space_params, execution_kwargs, **kwargs
        )
        if result is not None:
            return result
        target_joints = target_joints_override
    else:
        target_joints = p.calculateInverseKinematics(
            robot_id, ROBOT_END_EFFECTOR_LINK_ID, goal_pos, goal_orn, **default_null_space_params
        )[:7]

    # 2. 检查直接路径
    if not is_path_colliding(robot_id, current_joint_pos, target_joints, obstacle_ids,
                             current_gripper_pos, current_gripper_pos):
        print("  >> 直接路径安全，正在执行...")
        success = move_to_joints(robot_id, target_joints, **execution_kwargs)
        return success

    # 3. 使用 PFM 规划器
    print("  >> 直接路径被阻挡，启动 PFM 路径规划器...")
    workspace_path = plan_path_with_pfm(
        start_pos=current_pos, goal_pos=goal_pos, obstacle_ids=obstacle_ids
    )
    
    # 验证 PFM 路径
    pfm_path_is_valid, pfm_joint_path = validate_workspace_path(
        workspace_path, robot_id, goal_orn, obstacle_ids, current_gripper_pos
    )

    # 4. 执行 PFM 路径或切换到 Plan B
    if pfm_path_is_valid:
        print(f"  ✅ PFM 路径验证通过，执行中...")
        for joint_target in pfm_joint_path:
            success = move_to_joints(robot_id, joint_target, max_velocity=1.5, **execution_kwargs)
            if not success:
                print("  ❌ PFM 路径在执行期间失败。")
                return False
        
        success_final = move_to_joints(robot_id, target_joints, max_velocity=1.0, **execution_kwargs)
        return success_final
    else:
        print("  ❌ PFM 路径验证失败。")

        if not obstacle_ids:
            print("  >> PFM失败，且未感知到障碍物。规划终止。")
            return False

        print(f"  >> 启动 Plan B：尝试自动生成绕行路径...")
        
        # 生成并尝试绕行策略
        strategies = generate_detour_strategies(current_pos, goal_pos, obstacle_ids)
        if try_detour_strategies(robot_id, goal_orn, strategies, obstacle_ids,
                                current_gripper_pos, execution_kwargs):
            return True
        
        # Plan B 失败，尝试随机探索
        print("  ❌ Plan B 所有策略都失败了。")
        print("  >> 最后尝试：执行随机探索移动...")
        
        exploration_kwargs = execution_kwargs.copy()
        exploration_kwargs.pop('obstacle_ids', None)

        exploration_success = perform_random_exploration(
            robot_id, obstacle_ids, **exploration_kwargs
        )
        
        if exploration_success:
            print("  >> 随机探索成功，从新位置重新尝试到达目标...")
            return retry_from_new_position(robot_id, goal_pos, goal_orn, target_joints,
                                          obstacle_ids, execution_kwargs)
        
        return False


# ============================================================
# 辅助函数
# ============================================================

def try_auto_avoidance_ik_mode(robot_id, goal_pos, goal_orn, obstacle_ids,
                               current_joint_pos, current_gripper_pos,
                               default_null_space_params, execution_kwargs, **kwargs):
    """尝试自动避障 IK 模式"""
    print("  >> 使用了 'target_joints_override'，启用自动避障 IK 模式。")
    ee_state = p.getLinkState(robot_id, ROBOT_END_EFFECTOR_LINK_ID, computeForwardKinematics=True)
    current_ee_pos = np.array(ee_state[0])
    goal_pos = np.array(goal_pos)
    
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
                    waypoints = [current_ee_pos.copy(), candidate, mid_pos, goal_pos]
                    print(f"  >> 尝试自动避障路径: {waypoints}")
                    path_ok = True
                    prev_j = np.asarray([p.getJointState(robot_id, i)[0] for i in range(7)])
                    
                    ik_params_auto = default_null_space_params.copy()
                    joint_waypoints = []
                    
                    for wp in waypoints:
                        ik_params_auto["restPoses"] = list(prev_j)
                        j_wp = p.calculateInverseKinematics(robot_id, ROBOT_END_EFFECTOR_LINK_ID,
                                                            wp, goal_orn, **ik_params_auto)[:7]
                        
                        if is_path_colliding(robot_id, prev_j, j_wp, obstacle_ids, [0.04,0.04], [0.04,0.04]):
                            path_ok = False
                            break
                        
                        prev_j = j_wp
                        joint_waypoints.append(j_wp)
                    
                    if path_ok:
                        print("  ✅ 自动避障路径安全，执行中...")
                        for j_wp_target in joint_waypoints:
                            success = move_to_joints(robot_id, j_wp_target, **execution_kwargs)
                            if not success:
                                print("  ❌ 自动避障路径在执行期间失败。")
                                return False
                        return True
                except Exception:
                    continue
            
            print("  ❌ 所有自动绕行路径失败，将尝试默认路径。")
    
    return None


def try_detour_strategies(robot_id, goal_orn, strategies, obstacle_ids, current_gripper_pos, execution_kwargs):
    """尝试执行绕行策略"""
    current_joint_pos = np.asarray([p.getJointState(robot_id, i)[0] for i in range(7)])
    
    for strategy_idx, detour_waypoints in enumerate(strategies):
        print(f"  >> Plan B 策略 {strategy_idx + 1}: 尝试规划...")
        
        joint_space_path = []
        last_joint_pos = current_joint_pos.copy()
        ik_params = DEFAULT_NULL_SPACE_PARAMS.copy()
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
                print(f"  ✅ Plan B 策略 {strategy_idx + 1} 执行成功！")
                return True
        else:
            print(f"    ⚠️ 策略 {strategy_idx + 1} 规划失败，尝试下一个策略...")
    
    return False


def retry_from_new_position(robot_id, goal_pos, goal_orn, target_joints,
                           obstacle_ids, execution_kwargs):
    """从新位置重新尝试路径规划"""
    current_joint_pos = np.asarray([p.getJointState(robot_id, i)[0] for i in range(7)])
    current_gripper_pos = [p.getJointState(robot_id, 9)[0], p.getJointState(robot_id, 10)[0]]
    current_pos, *_ = p.getLinkState(robot_id, ROBOT_END_EFFECTOR_LINK_ID, computeForwardKinematics=True)
    
    strategies = []
    
    # 策略1：直接路径
    if not is_path_colliding(robot_id, current_joint_pos, target_joints, obstacle_ids,
                             current_gripper_pos, current_gripper_pos):
        strategies.append(("direct", [target_joints], None))
    
    # 策略2：随机化PFM
    print("  >> 从新位置使用随机化PFM重新规划...")
    pfm_step_size = 0.02 + np.random.uniform(-0.01, 0.02)
    pfm_k_att = 1.0 + np.random.uniform(-0.3, 0.3)
    
    workspace_path_new = plan_path_with_pfm(
        start_pos=current_pos, goal_pos=goal_pos, obstacle_ids=obstacle_ids,
        step_size=pfm_step_size, k_att=pfm_k_att, randomize=True
    )
    
    pfm_valid, pfm_joint_waypoints = validate_workspace_path(
        workspace_path_new, robot_id, goal_orn, obstacle_ids, current_gripper_pos
    )
    
    if pfm_valid and not is_path_similar_to_history(workspace_path_new):
        strategies.append(("pfm_new", pfm_joint_waypoints, workspace_path_new))
    
    # 策略3：弧形路径
    arc_path = generate_arc_path(current_pos, goal_pos, obstacle_ids)
    arc_valid, arc_joint_waypoints = validate_workspace_path(
        arc_path, robot_id, goal_orn, obstacle_ids, current_gripper_pos
    )
    
    if arc_valid and not is_path_similar_to_history(arc_path):
        strategies.append(("arc", arc_joint_waypoints, arc_path))
    
    # 随机选择策略并执行
    if strategies:
        strategy_name, joint_path, workspace_path = strategies[np.random.randint(len(strategies))]
        print(f"  >> 选择策略: {strategy_name}")
        
        for j_wp in joint_path:
            success = move_to_joints(robot_id, j_wp, **execution_kwargs)
            if not success:
                print(f"  ❌ {strategy_name}策略执行失败")
                return False
        
        if workspace_path is not None:
            add_path_to_history(workspace_path)
        
        print(f"  ✅ 通过随机探索+{strategy_name}策略找到了新路径！")
        return True
    else:
        print("  >> 新位置没有找到可行路径，等待重试...")
        return False

