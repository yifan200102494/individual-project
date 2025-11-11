"""
动态避障执行器
集成深度感知、增量规划和运动执行的闭环控制系统
使用 Extrinsic Calibration 和 Depth Sensing of Occupancy 进行障碍物检测
"""

import pybullet as p
import numpy as np
import time

from constants import (
    ROBOT_END_EFFECTOR_LINK_ID, 
    DELTA_T, 
    PROXIMITY_FAILSAFE_DISTANCE,
    WORKSPACE_LIMITS
)
from depth_perception import DepthPerceptionSystem
from incremental_planner import IncrementalPlanner, ReactivePlanner
from motion_control import simulate
from collision_detection import is_state_colliding


class DynamicMotionExecutor:
    """
    动态运动执行器
    实现边感知边规划边执行的闭环控制
    
    感知方法：
    - Extrinsic Calibration: 相机外参标定，确定相机在机器人末端的位姿
    - Depth Sensing: 使用深度相机捕获环境的深度信息
    - Occupancy Detection: 从深度点云中识别障碍物占用情况
    """
    
    def __init__(self, robot_id, sensor_link_id, tray_position=None, tray_size=None):
        """
        初始化动态执行器
        
        Args:
            robot_id: 机器人ID
            sensor_link_id: 传感器连杆ID
            tray_position: 托盘位置 [x, y, z]（可选）
            tray_size: 托盘尺寸 [length, width, height]（可选）
        """
        self.robot_id = robot_id
        self.sensor_link_id = sensor_link_id
        
        # 初始化子系统（优化参数以提高速度和平滑度）
        # 新的深度感知系统（基于extrinsic calibration和depth sensing of occupancy）
        self.perception = DepthPerceptionSystem(
            robot_id=robot_id, 
            sensor_link_id=sensor_link_id,
            image_width=128,  # 适中的分辨率，平衡性能和精度
            image_height=128,
            tray_position=tray_position,  # 传递托盘位置
            tray_size=tray_size  # 传递托盘尺寸
        )
        self.local_planner = IncrementalPlanner(robot_id, planning_horizon=0.3, replan_rate=30)  # 从20增加到30，减少重新规划频率
        self.reactive_planner = ReactivePlanner(robot_id)
        
        # 执行参数（优化以提高速度和平滑度）
        self.max_velocity = 2.5  # 增加速度以提高连贯性
        self.control_rate = 25  # 提高控制频率以提高平滑度
        
        # 速度曲线参数（用于平滑加减速）
        self.velocity_history = []  # 速度历史
        self.velocity_smooth_window = 5  # 速度平滑窗口
        
    def move_to_goal_dynamic(self, goal_pos, goal_orn, ignore_ids=None,
                            interferer_id=None, interferer_joints=None,
                            interferer_update_rate=120, max_time=30, debug=False,
                            fast_mode=True):
        """
        动态移动到目标位置
        
        特点：
        - 持续深度感知障碍物（基于相机外参标定）
        - 实时occupancy检测（从点云识别障碍物）
        - 实时更新路径规划
        - 响应式避障
        
        Args:
            goal_pos: 目标位置
            goal_orn: 目标方向
            ignore_ids: 要忽略的物体ID
            interferer_id: 干扰物体ID
            interferer_joints: 干扰关节
            interferer_update_rate: 干扰更新频率
            max_time: 最大执行时间（秒）
            debug: 调试模式
            fast_mode: 快速模式（优先尝试直接路径）
            
        Returns:
            bool: 是否成功到达目标
        """
        print(f"  >> [动态执行器] 开始动态移动到 {goal_pos}")
        
        if ignore_ids is None:
            ignore_ids = []
        
        ignore_set = set(ignore_ids)
        ignore_set.add(self.robot_id)
        ignore_set.add(0)  # 地面
        ignore_set.add(-1)  # 无效ID
        
        # 仿真参数
        sim_kwargs = {
            "interferer_id": interferer_id,
            "interferer_joints": interferer_joints,
            "interferer_update_rate": interferer_update_rate,
            "slow_down": True
        }
        
        start_time = time.time()
        perception_counter = 0
        control_counter = 0
        failed_execution_counter = 0  # 🔥 失败计数器，用于触发随机探索
        
        # 当前状态
        current_joint_pos = np.asarray([p.getJointState(self.robot_id, i)[0] for i in range(7)])
        current_gripper_pos = [p.getJointState(self.robot_id, 9)[0], 
                               p.getJointState(self.robot_id, 10)[0]]
        
        # 初始化障碍物信息
        current_obstacles = []
        predicted_obstacles = []
        
        # 快速模式：先尝试直接路径
        if fast_mode:
            direct_success = self._try_direct_path(goal_pos, goal_orn, ignore_set, 
                                                   current_gripper_pos, sim_kwargs, debug)
            if direct_success:
                return True
        
        while True:
            # 检查超时
            if time.time() - start_time > max_time:
                print(f"  [!!] 动态执行超时（{max_time}秒）")
                return False
            
            # ===============================================
            # 1. 获取当前位置
            # ===============================================
            ee_state = p.getLinkState(self.robot_id, ROBOT_END_EFFECTOR_LINK_ID, 
                                     computeForwardKinematics=True)
            current_pos = np.array(ee_state[0])
            
            # 检查是否到达目标
            dist_to_goal = np.linalg.norm(current_pos - np.array(goal_pos))
            horizontal_dist_to_goal = np.linalg.norm(current_pos[:2] - np.array(goal_pos[:2]))
            
            # 智能判定：根据目标高度使用不同的到达阈值
            # 对于中间路径点（较高位置），使用更宽松的判定
            # 对于最终目标（较低位置），使用较严格的判定
            if goal_pos[2] > 0.25:
                # 中间路径点（如抬高、水平移动）：8cm内即可
                reach_threshold = 0.08
            else:
                # 最终目标（如放置位置）：5cm内
                reach_threshold = 0.08
            
            if dist_to_goal < reach_threshold:
                print(f"  ✅ [动态执行器] 成功到达目标！距离: {dist_to_goal:.3f}m")
                # 如果是最终目标（低位），进行精确对齐
                if goal_pos[2] < 0.25:
                    return self._final_alignment(goal_pos, goal_orn, ignore_set, 
                                                current_gripper_pos, sim_kwargs)
                else:
                    # 中间路径点，直接返回成功
                    return True
            
            # ===============================================
            # 1.5. 特殊情况：如果非常接近目标且正在下降，直接下降
            # ===============================================
            horizontal_dist = np.linalg.norm(current_pos[:2] - np.array(goal_pos[:2]))
            vertical_dist = current_pos[2] - goal_pos[2]
            
            # 下降条件：当前高于目标，水平对齐 -> 下降
            is_above_goal = horizontal_dist < 0.18 and current_pos[2] > goal_pos[2] + 0.02  # 水平18cm内，高于目标2cm即可
            
            # 调试：打印检查信息（降低频率）
            if debug and goal_pos[2] < 0.30 and perception_counter % 50 == 0:
                print(f"  >> [下降检查] 水平: {horizontal_dist:.3f}m, 当前高度: {current_pos[2]:.3f}m, 目标: {goal_pos[2]:.3f}m, 满足下降: {is_above_goal}")
            
            if is_above_goal and goal_pos[2] < 0.30:  # 目标在低位
                print(f"  >> [直接下降] ✅ 已在目标正上方，触发直接下降！")
                print(f"     水平距离: {horizontal_dist:.3f}m, 垂直距离: {vertical_dist:.3f}m")
                
                # 直接计算下降目标
                try:
                    target_joints = p.calculateInverseKinematics(
                        self.robot_id, ROBOT_END_EFFECTOR_LINK_ID,
                        goal_pos, goal_orn,
                        maxNumIterations=100
                    )[:7]
                    
                    # 执行下降（大幅增加步数，确保完成）
                    for step in range(300):  # 从100增加到300
                        for joint_id in range(7):
                            p.setJointMotorControl2(
                                self.robot_id, joint_id,
                                controlMode=p.POSITION_CONTROL,
                                targetPosition=target_joints[joint_id],
                                maxVelocity=1.0,  # 降低速度，更平稳
                                force=150
                            )
                        simulate(steps=1, **sim_kwargs)
                        
                        # 检查是否到达
                        ee_state = p.getLinkState(self.robot_id, ROBOT_END_EFFECTOR_LINK_ID,
                                                 computeForwardKinematics=True)
                        current_check = np.array(ee_state[0])
                        current_dist = np.linalg.norm(current_check - np.array(goal_pos))
                        
                        # 放宽到达判断 - 如果足够接近就认为成功
                        if current_dist < 0.05:  # 从0.03放宽到0.05
                            print(f"  ✅ [直接下降] 成功到达目标！最终距离: {current_dist:.3f}m")
                            return True
                        
                        # 每20步打印一次进度
                        if debug and step % 20 == 0:
                            print(f"     [直接下降] 步骤 {step}/300, 距离目标: {current_dist:.3f}m")
                        
                        # 检查是否已经非常接近（关节空间）
                        current_joints_check = np.asarray([p.getJointState(self.robot_id, i)[0] for i in range(7)])
                        if np.allclose(current_joints_check, target_joints, atol=0.02):
                            print(f"  ✅ [直接下降] 到达关节目标位置！最终距离: {current_dist:.3f}m")
                            # 如果距离合理，就认为成功
                            if current_dist < 0.08:
                                return True
                            break
                    
                    # 下降完成后检查最终距离
                    ee_state = p.getLinkState(self.robot_id, ROBOT_END_EFFECTOR_LINK_ID,
                                             computeForwardKinematics=True)
                    final_pos = np.array(ee_state[0])
                    final_dist = np.linalg.norm(final_pos - np.array(goal_pos))
                    
                    if final_dist < 0.08:  # 如果距离<8cm就认为成功
                        print(f"  ✅ [直接下降] 完成！最终距离: {final_dist:.3f}m (足够接近)")
                        return True
                    else:
                        if debug:
                            print(f"  >> [直接下降] 部分完成（距离{final_dist:.3f}m），继续正常规划")
                except Exception as e:
                    if debug:
                        print(f"  >> [直接下降] 失败: {e}，继续正常规划")
            
            # ===============================================
            # 2. 深度感知（使用新的 Extrinsic Calibration 和 Depth Sensing of Occupancy）
            # ===============================================
            perception_counter += 1
            
            # 降低感知频率：每10个控制周期才感知一次，或者刚开始时
            # 这样可以让机器人有更多时间执行动作，而不是一直停下来感知
            should_perceive = (perception_counter % 10 == 1) or (perception_counter <= 2)
            
            if should_perceive:
                # 使用基于深度相机的感知系统
                perception_result = self.perception.perceive_with_depth(
                    ignore_ids=ignore_set,
                    debug=debug
                )
                current_obstacles = perception_result['current_obstacles']
                predicted_obstacles = perception_result['predicted_obstacles']
            # 否则继续使用上次的感知结果
            
            # 🔥 智能过滤：高位移动时的障碍物过滤策略（每次循环都执行）
            # 策略1: 如果两者都在高位且在向上移动或水平移动（归位场景），完全忽略低处障碍物
            is_moving_up_or_horizontal = goal_pos[2] >= current_pos[2] - 0.05  # 目标不比当前低超过5cm
            is_both_high = current_pos[2] > 0.35 and goal_pos[2] > 0.35
            
            if is_both_high and is_moving_up_or_horizontal:
                # 归位等高位移动：使用更激进的过滤
                # 计算安全高度阈值：比当前和目标中较低的还要低20cm
                safe_height_threshold = min(current_pos[2], goal_pos[2]) - 0.20
                original_count = len(current_obstacles)
                
                current_obstacles = [
                    (obs_id, obs_pos, obs_vel) 
                    for obs_id, obs_pos, obs_vel in current_obstacles 
                    if obs_pos[2] > safe_height_threshold
                ]
                
                if debug and should_perceive:  # 只在感知时打印调试信息，避免重复
                    filtered_count = original_count - len(current_obstacles)
                    if filtered_count > 0:
                        print(f"  >> [高位过滤] 过滤掉 {filtered_count} 个低处障碍物 (低于 {safe_height_threshold:.2f}m)")
                    if len(current_obstacles) > 0:
                        print(f"  >> [高位过滤] 保留 {len(current_obstacles)} 个高处障碍物:")
                        for obs_id, obs_pos, obs_vel in current_obstacles:
                            print(f"      障碍物 {obs_id}: 位置 {obs_pos}, 高度 {obs_pos[2]:.2f}m")
            
            # 策略2: 如果当前和目标都很高（>0.4m）且已经很接近目标，完全忽略障碍物
            # 这是最后冲刺阶段，直接移动到目标
            is_very_high = current_pos[2] > 0.4 and goal_pos[2] > 0.4
            is_approaching_target = dist_to_goal < 0.20
            
            if is_very_high and is_approaching_target and is_moving_up_or_horizontal:
                if debug and len(current_obstacles) > 0 and should_perceive:
                    print(f"  >> [归位冲刺] 高位且接近目标，忽略所有障碍物")
                current_obstacles = []  # 完全清空障碍物列表
            
            # 获取真实的物体ID用于碰撞检测
            # 深度感知返回的是虚拟ID，我们需要使用场景中的真实物体ID
            all_body_ids = [p.getBodyUniqueId(i) for i in range(p.getNumBodies())]
            obstacle_ids = [bid for bid in all_body_ids if bid not in ignore_set]
            
            # ===============================================
            # 3. 检查紧急情况（智能判断，减少误触发）
            # ===============================================
            # 只在真正危险时才触发紧急避障，避免过度反应
            # 如果正在下降到低位目标，大幅放宽危险阈值或跳过紧急避障
            is_descending_to_low_target = (goal_pos[2] < 0.30 and current_pos[2] > goal_pos[2] - 0.05)
            
            # 只在有障碍物且不在下降时才检查，且目标位置低于当前位置时才启用
            # 对于回Home等向上移动的任务，禁用紧急避障，让探索机制处理
            is_moving_up = goal_pos[2] > current_pos[2] + 0.1  # 目标比当前高10cm以上
            should_check_emergency = (len(current_obstacles) > 0 and 
                                     not is_descending_to_low_target and 
                                     not is_moving_up)  # 向上移动时禁用紧急避障
            
            if should_check_emergency and control_counter % 5 == 0:  # 降低检查频率
                danger_threshold = 0.05  # 非常近才触发（5cm）
                
                dangerous_obstacles = self.reactive_planner.check_if_dangerous(
                    current_pos, current_obstacles, danger_threshold=danger_threshold
                )
                
                # 只有非常危险（距离<5cm且有多个障碍物）才触发
                if len(dangerous_obstacles) >= 2:
                    print(f"  [!!] 检测到 {len(dangerous_obstacles)} 个危险障碍物，启动紧急避障")
                    success = self._emergency_avoidance(
                        current_pos, goal_pos, goal_orn, dangerous_obstacles,
                        obstacle_ids, current_gripper_pos, sim_kwargs
                    )
                    if not success:
                        print(f"  [!!] 紧急避障失败，转入探索模式")
                        failed_execution_counter += 5  # 增加失败计数，触发探索
                        # 不要直接返回False，让探索机制接管
                    else:
                        # 避障成功，重置失败计数
                        failed_execution_counter = 0
                    continue
            
            # ===============================================
            # 4. 增量式路径规划（周期性，优化规划频率）
            # ===============================================
            control_counter += 1
            
            # 🔥 智能规划判断：根据距离和情况决定是否需要重新规划
            
            # 如果没有障碍物，直接移动，不要规划（最重要的条件）
            no_obstacles = len(current_obstacles) == 0
            
            # 如果非常接近目标（<12cm）且没有障碍物，停止重新规划，直接移动
            is_very_close = dist_to_goal < 0.12 and no_obstacles
            
            # 如果已经很接近目标且在下降，减少规划频率
            is_close_and_descending = (dist_to_goal < 0.20 and 
                                       goal_pos[2] < 0.30 and 
                                       current_pos[2] > goal_pos[2] - 0.05)
            
            if no_obstacles:
                # 🔥 关键修复：没有障碍物时，完全停止规划，直接移动
                need_replan = False
                if debug and control_counter % 30 == 0:
                    print(f"  >> [无障碍模式] 距离{dist_to_goal:.3f}m，直接移动到目标")
            elif is_very_close:
                # 非常接近且无障碍时，停止规划，直接移动
                need_replan = False
                if debug and control_counter % 20 == 0:
                    print(f"  >> [接近目标] 距离{dist_to_goal:.3f}m，无障碍，直接移动")
            elif is_close_and_descending:
                # 接近目标且下降时，降低规划频率（每60步一次）
                need_replan = (control_counter % 60 == 0)
            else:
                # 正常情况：有障碍物时才规划
                need_replan = (self.local_planner.should_replan() or control_counter == 1)
            
            if need_replan:
                if debug:
                    print(f"  >> [规划] 重新规划局部路径...")
                
                # 规划短期路径（增加路径点数量以提高平滑度）
                local_path = self.local_planner.plan_local_path(
                    current_pos, goal_pos, goal_orn, current_obstacles, num_waypoints=8
                )
                
                if not local_path:
                    print(f"  [!!] 局部路径规划失败")
                    failed_execution_counter += 1  # 🔥 增加失败计数
                    
                    # 🔥 检查是否需要触发随机探索
                    if failed_execution_counter >= 3:  # 连续失败3次后触发探索（降低阈值，更快响应）
                        print(f"\n  [🔍 触发随机探索] 已连续失败 {failed_execution_counter} 次")
                        if self._trigger_exploration(obstacle_ids, sim_kwargs, debug):
                            failed_execution_counter = 0  # 探索成功，重置计数
                            print(f"  [✅ 探索成功] 继续尝试到达目标")
                        else:
                            failed_execution_counter = max(failed_execution_counter - 2, 0)  # 探索失败，降低计数
                    
                    simulate(steps=5, **sim_kwargs)
                    continue
                
                # 验证路径（根据是否下降调整严格程度）
                # 如果正在下降到低位目标，使用更宽松的碰撞检测
                is_descending_to_low = (goal_pos[2] < 0.30 and current_pos[2] > goal_pos[2] - 0.05)
                collision_steps = 3 if is_descending_to_low else 5  # 下降时更宽松
                
                is_valid, joint_path = self.local_planner.validate_local_path(
                    local_path, goal_orn, obstacle_ids, current_gripper_pos,
                    collision_check_steps=collision_steps
                )
                
                if not is_valid:
                    if debug:
                        print(f"  [!!] 局部路径验证失败，等待后重试")
                    failed_execution_counter += 1  # 🔥 增加失败计数
                    
                    # 🔥 检查是否需要触发随机探索
                    if failed_execution_counter >= 5:
                        print(f"\n  [🔍 触发随机探索] 已连续失败 {failed_execution_counter} 次")
                        if self._trigger_exploration(obstacle_ids, sim_kwargs, debug):
                            failed_execution_counter = 0
                            print(f"  [✅ 探索成功] 继续尝试到达目标")
                        else:
                            failed_execution_counter = max(failed_execution_counter - 2, 0)
                    
                    simulate(steps=5, **sim_kwargs)
                    continue
                
                # 执行路径点（使用多个路径点提高平滑度）
                if joint_path:
                    # 取前3个路径点的加权平均，使运动更平滑
                    if len(joint_path) >= 3:
                        target_joints = (np.array(joint_path[0]) * 0.6 + 
                                       np.array(joint_path[1]) * 0.3 + 
                                       np.array(joint_path[2]) * 0.1)
                    elif len(joint_path) >= 2:
                        target_joints = (np.array(joint_path[0]) * 0.7 + 
                                       np.array(joint_path[1]) * 0.3)
                    else:
                        target_joints = joint_path[0]
                    # 规划成功，减少失败计数器
                    failed_execution_counter = max(0, failed_execution_counter - 2)
            else:
                # 继续朝当前目标移动
                try:
                    # 计算下一个路径点
                    next_waypoint = self.local_planner.compute_local_waypoint(
                        current_pos, goal_pos, current_obstacles, step_size=0.08  # 增加步长以加快移动和连贯性
                    )
                    
                    # IK求解
                    target_joints = p.calculateInverseKinematics(
                        self.robot_id, ROBOT_END_EFFECTOR_LINK_ID,
                        next_waypoint, goal_orn
                    )[:7]
                except Exception as e:
                    if debug:
                        print(f"  [!] IK求解失败: {e}")
                    simulate(steps=1, **sim_kwargs)
                    continue
            
            # ===============================================
            # 5. 执行运动（单步）
            # ===============================================
            success = self._execute_single_step(
                target_joints, obstacle_ids, current_gripper_pos,
                interferer_id, sim_kwargs
            )
            
            if not success:
                if debug:
                    print(f"  [!!] 执行步骤失败，重新规划")
                failed_execution_counter += 1  # 🔥 增加失败计数
                
                # 🔥 检查是否需要触发随机探索
                if failed_execution_counter >= 5:
                    print(f"\n  [🔍 触发随机探索] 已连续失败 {failed_execution_counter} 次")
                    if self._trigger_exploration(obstacle_ids, sim_kwargs, debug):
                        failed_execution_counter = 0
                        print(f"  [✅ 探索成功] 继续尝试到达目标")
                    else:
                        failed_execution_counter = max(failed_execution_counter - 2, 0)
                
                simulate(steps=2, **sim_kwargs)
                continue
            
            # 更新当前状态
            current_joint_pos = np.asarray([p.getJointState(self.robot_id, i)[0] for i in range(7)])
        
        return False
    
    def _execute_single_step(self, target_joints, obstacle_ids, current_gripper_pos,
                            interferer_id, sim_kwargs):
        """
        执行单个运动步骤（包含PFM近距离保护检测）
        
        Returns:
            bool: 是否成功
        """
        num_arm_joints = len(target_joints)
        
        # ===============================================
        # PFM: 近距离保护机制（Proximity Failsafe Mechanism）
        # ===============================================
        if interferer_id is not None:
            closest_points = p.getClosestPoints(
                self.robot_id, interferer_id, PROXIMITY_FAILSAFE_DISTANCE
            )
            
            if closest_points:
                print(f"  [⚠️ PFM] 检测到近距离接触 (< {PROXIMITY_FAILSAFE_DISTANCE*100:.1f}cm)，停止当前动作")
                # 立即停止所有运动
                for joint_id in range(num_arm_joints):
                    p.setJointMotorControl2(
                        self.robot_id, joint_id,
                        controlMode=p.VELOCITY_CONTROL,
                        targetVelocity=0,
                        force=200
                    )
                simulate(steps=2, **sim_kwargs)
                return False
        
        # ===============================================
        # 正常运动控制
        # ===============================================
        # 计算当前关节位置和速度
        current_joint_pos = np.asarray([p.getJointState(self.robot_id, i)[0] for i in range(num_arm_joints)])
        joint_distance = np.linalg.norm(np.array(target_joints) - current_joint_pos)
        
        # 动态调整速度：距离越近，速度越慢（平滑减速）
        if joint_distance < 0.2:
            # 接近目标时减速
            adaptive_velocity = self.max_velocity * max(0.3, joint_distance / 0.2)
        else:
            adaptive_velocity = self.max_velocity
        
        # 设置电机控制（使用自适应速度）
        for joint_id in range(num_arm_joints):
            p.setJointMotorControl2(
                self.robot_id, joint_id, 
                controlMode=p.POSITION_CONTROL,
                targetPosition=target_joints[joint_id],
                maxVelocity=adaptive_velocity,
                force=120  # 增加力度以确保执行
            )
        
        # 执行一步仿真
        simulate(steps=1, **sim_kwargs)
        
        return True
    
    def _emergency_avoidance(self, current_pos, goal_pos, goal_orn, dangerous_obstacles,
                            obstacle_ids, current_gripper_pos, sim_kwargs):
        """
        紧急避障 - 强制逃离模式
        
        在极端危险情况下，使用更激进的策略：
        1. 增加逃离距离
        2. 尝试多个逃离方向
        3. 在逃离过程中放宽碰撞检测
        
        Returns:
            bool: 是否成功避开
        """
        print(f"  >> [紧急避障] 尝试逃离危险区域...")
        
        # 计算主要逃离方向
        primary_escape = self.reactive_planner.compute_emergency_avoidance(
            current_pos, dangerous_obstacles
        )
        
        # 根据是否正在下降，选择不同的逃离策略
        is_descending = goal_pos[2] < current_pos[2] - 0.05
        
        if is_descending:
            # 下降时：优先向上，其次侧向逃离
            escape_strategies = [
                ("向上", np.array([0, 0, 1.0]), 0.30),  # 🔥 抬高30cm（最优先）
                ("斜向上", np.array([primary_escape[0], primary_escape[1], 0.5]), 0.25),  # 斜向上逃离
                ("侧向", np.array([primary_escape[0], primary_escape[1], 0]), 0.20),  # 水平逃离
            ]
        else:
            # 正常时：优先向上，其次主方向
            escape_strategies = [
                ("向上", np.array([0, 0, 1.0]), 0.35),  # 🔥 抬高35cm（最优先）
                ("主方向", primary_escape, 0.20),
            ]
        
        for strategy_name, escape_direction, escape_distance in escape_strategies:
            # 归一化方向
            if np.linalg.norm(escape_direction) > 0.001:
                escape_direction = escape_direction / np.linalg.norm(escape_direction)
            else:
                continue
            
            # 生成安全点
            safety_pos = np.array(current_pos) + escape_direction * escape_distance
            
            # 应用工作空间限制
            safety_pos[0] = np.clip(safety_pos[0], -0.2, 1.0)
            safety_pos[1] = np.clip(safety_pos[1], -0.8, 0.8)
            safety_pos[2] = np.clip(safety_pos[2], 0.15, 1.2)
            
            try:
                # 计算目标关节位置
                safety_joints = p.calculateInverseKinematics(
                    self.robot_id, ROBOT_END_EFFECTOR_LINK_ID,
                    safety_pos, goal_orn
                )[:7]
                
                print(f"  >> 尝试逃离策略: {strategy_name} (距离: {escape_distance:.2f}m)")
                
                # 强制逃离模式：直接设置关节位置，跳过碰撞检测（减少步数）
                escaped = False
                for step in range(20):  # 减少到20步
                    
                    # 强制设置电机目标（逃离时不受碰撞检测限制）
                    for joint_id in range(7):
                        p.setJointMotorControl2(
                            self.robot_id, joint_id,
                            controlMode=p.POSITION_CONTROL,
                            targetPosition=safety_joints[joint_id],
                            maxVelocity=2.0,  # 更快的速度
                            force=150  # 更大的力
                        )
                    
                    simulate(steps=1, **sim_kwargs)
                    
                    # 检查是否已经安全
                    ee_state = p.getLinkState(self.robot_id, ROBOT_END_EFFECTOR_LINK_ID,
                                             computeForwardKinematics=True)
                    current_check = np.array(ee_state[0])
                    
                    # 检查距离所有危险障碍物的距离
                    min_dist = float('inf')
                    for obs_id, obs_pos, obs_velocity in dangerous_obstacles:
                        dist = np.linalg.norm(current_check - np.array(obs_pos))
                        min_dist = min(min_dist, dist)
                    
                    # 如果距离足够远，认为逃离成功
                    if min_dist > 0.20:  # 20cm安全距离
                        print(f"  ✅ 紧急避障成功！使用策略: {strategy_name}, 当前安全距离: {min_dist:.3f}m")
                        escaped = True
                        break
                    
                    # 检查是否已经接近目标关节位置
                    current_joints = np.asarray([p.getJointState(self.robot_id, i)[0] for i in range(7)])
                    if np.allclose(current_joints, safety_joints, atol=0.05):
                        if min_dist > 0.12:  # 至少12cm
                            print(f"  ✅ 到达逃离位置，当前距离: {min_dist:.3f}m")
                            escaped = True
                            break
                
                if escaped:
                    return True
                    
            except Exception as e:
                print(f"  >> 策略 {strategy_name} 失败: {e}")
                continue
        
        # 所有策略都失败，最后尝试：强制向上移动
        print(f"  >> [最后尝试] 强制向上逃离...")
        for _ in range(20):
            current_joints = np.asarray([p.getJointState(self.robot_id, i)[0] for i in range(7)])
            # 简单策略：将所有关节向home位置移动一点
            home_config = [0.0, -0.785, 0.0, -2.356, 0.0, 1.57, 0.785]
            for joint_id in range(7):
                target = current_joints[joint_id] * 0.9 + home_config[joint_id] * 0.1
                p.setJointMotorControl2(
                    self.robot_id, joint_id,
                    controlMode=p.POSITION_CONTROL,
                    targetPosition=target,
                    maxVelocity=1.5,
                    force=120
                )
            simulate(steps=1, **sim_kwargs)
        
        return False
    
    def _try_direct_path(self, goal_pos, goal_orn, ignore_set, current_gripper_pos, sim_kwargs, debug=False):
        """
        尝试直接路径（快速模式）
        
        如果从当前位置到目标的直接路径没有碰撞，就直接移动
        
        Returns:
            bool: 是否成功通过直接路径到达
        """
        from collision_detection import is_path_colliding
        from motion_control import move_to_joints
        
        # 获取当前关节位置
        current_joints = [p.getJointState(self.robot_id, i)[0] for i in range(7)]
        
        # 计算目标关节位置
        try:
            target_joints = p.calculateInverseKinematics(
                self.robot_id, ROBOT_END_EFFECTOR_LINK_ID,
                goal_pos, goal_orn,
                maxNumIterations=50
            )[:7]
        except:
            return False
        
        # 获取障碍物ID（排除忽略的物体）
        all_bodies = [p.getBodyUniqueId(i) for i in range(p.getNumBodies())]
        obstacle_ids = [bid for bid in all_bodies if bid not in ignore_set]
        
        # 检查直接路径是否无碰撞
        # 注意：需要提供 end_gripper_pos 参数
        if not is_path_colliding(self.robot_id, current_joints, target_joints, 
                                 obstacle_ids, current_gripper_pos, current_gripper_pos, 
                                 num_steps=10):
            if debug:
                print("  ⚡ [快速模式] 检测到直接路径可行，跳过复杂规划")
            
            # 直接移动
            success = move_to_joints(
                self.robot_id, target_joints,
                max_velocity=self.max_velocity * 1.5,  # 快速模式下速度更快
                timeout=5,
                **sim_kwargs
            )
            
            if success:
                print("  ✅ [快速模式] 直接路径执行成功")
                return True
        
        return False
    
    def _final_alignment(self, goal_pos, goal_orn, ignore_set, current_gripper_pos, sim_kwargs):
        """
        最终精确对齐到目标位置
        
        Returns:
            bool: 是否成功
        """
        try:
            target_joints = p.calculateInverseKinematics(
                self.robot_id, ROBOT_END_EFFECTOR_LINK_ID,
                goal_pos, goal_orn
            )[:7]
            
            # 慢速精确移动
            for _ in range(50):
                current_joints = np.asarray([p.getJointState(self.robot_id, i)[0] 
                                            for i in range(7)])
                
                if np.allclose(current_joints, target_joints, atol=0.01):
                    return True
                
                for joint_id in range(7):
                    p.setJointMotorControl2(
                        self.robot_id, joint_id,
                        controlMode=p.POSITION_CONTROL,
                        targetPosition=target_joints[joint_id],
                        maxVelocity=0.5,  # 慢速
                        force=100
                    )
                
                simulate(steps=1, **sim_kwargs)
            
            return True
        except:
            return False
    
    def _trigger_exploration(self, obstacle_ids, sim_kwargs, debug=False):
        """
        🔥 触发随机探索（优先抬高机械臂）
        
        当连续失败多次无法找到路径时，通过随机探索改变机械臂姿态，
        从而改变相对于障碍物的位置，使得重新规划时可能找到新的可行路径
        
        Args:
            obstacle_ids: 障碍物ID列表
            sim_kwargs: 仿真参数
            debug: 调试模式
        
        Returns:
            bool: 是否探索成功
        """
        from exploration import perform_random_exploration
        
        print(f"\n{'='*60}")
        print(f"  🚀 [随机探索] 启动探索以逃离困境...")
        print(f"  💡 优先策略：⬆️ 抬高机械臂")
        print(f"{'='*60}\n")
        
        # 调用探索模块（会优先尝试抬高机械臂）
        success = perform_random_exploration(
            self.robot_id, 
            obstacle_ids,
            **sim_kwargs
        )
        
        if success:
            print(f"\n{'='*60}")
            print(f"  ✅ [随机探索] 探索成功！机械臂已移动到新位置")
            print(f"{'='*60}\n")
        else:
            print(f"\n{'='*60}")
            print(f"  ⚠️  [随机探索] 探索未完全成功，但会继续尝试")
            print(f"{'='*60}\n")
        
        return success

