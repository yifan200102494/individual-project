"""
动态避障执行器
集成实时感知、增量规划和运动执行的闭环控制系统
"""

import pybullet as p
import numpy as np
import time

from constants import (
    ROBOT_END_EFFECTOR_LINK_ID, 
    DELTA_T, 
    PROXIMITY_FAILSAFE_DISTANCE
)
from realtime_perception import AdaptivePerceptionSystem
from incremental_planner import IncrementalPlanner, ReactivePlanner
from motion_control import simulate
from collision_detection import is_state_colliding


class DynamicMotionExecutor:
    """
    动态运动执行器
    实现边感知边规划边执行的闭环控制
    """
    
    def __init__(self, robot_id, sensor_link_id):
        """
        初始化动态执行器
        
        Args:
            robot_id: 机器人ID
            sensor_link_id: 传感器连杆ID
        """
        self.robot_id = robot_id
        self.sensor_link_id = sensor_link_id
        
        # 初始化子系统
        self.perception = AdaptivePerceptionSystem(robot_id, sensor_link_id)
        self.local_planner = IncrementalPlanner(robot_id)
        self.reactive_planner = ReactivePlanner(robot_id)
        
        # 执行参数
        self.max_velocity = 1.0
        self.control_rate = 10  # 控制频率（Hz）
        self.perception_rate = 5  # 感知频率（Hz）
        
    def move_to_goal_dynamic(self, goal_pos, goal_orn, ignore_ids=None,
                            interferer_id=None, interferer_joints=None,
                            interferer_update_rate=120, max_time=30, debug=False):
        """
        动态移动到目标位置
        
        特点：
        - 持续感知障碍物
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
        
        # 当前状态
        current_joint_pos = np.asarray([p.getJointState(self.robot_id, i)[0] for i in range(7)])
        current_gripper_pos = [p.getJointState(self.robot_id, 9)[0], 
                               p.getJointState(self.robot_id, 10)[0]]
        
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
            if dist_to_goal < 0.03:
                print(f"  ✅ [动态执行器] 成功到达目标！")
                # 最后精确对齐
                return self._final_alignment(goal_pos, goal_orn, ignore_set, 
                                            current_gripper_pos, sim_kwargs)
            
            # ===============================================
            # 2. 实时感知（周期性）
            # ===============================================
            perception_counter += 1
            if perception_counter % (self.control_rate // self.perception_rate) == 0:
                perception_result = self.perception.perceive_with_prediction(
                    ignore_ids=ignore_set, debug=debug
                )
                current_obstacles = perception_result['current_obstacles']
                predicted_obstacles = perception_result['predicted_obstacles']
            else:
                # 使用上次的感知结果
                pass
            
            # 提取障碍物ID用于碰撞检测
            obstacle_ids = [obs[0] for obs in current_obstacles]
            
            # ===============================================
            # 3. 检查紧急情况
            # ===============================================
            dangerous_obstacles = self.reactive_planner.check_if_dangerous(
                current_pos, current_obstacles, danger_threshold=0.12
            )
            
            if dangerous_obstacles:
                print(f"  [!!] 检测到 {len(dangerous_obstacles)} 个危险障碍物，启动紧急避障")
                success = self._emergency_avoidance(
                    current_pos, goal_pos, goal_orn, dangerous_obstacles,
                    obstacle_ids, current_gripper_pos, sim_kwargs
                )
                if not success:
                    print(f"  [!!] 紧急避障失败")
                    return False
                continue
            
            # ===============================================
            # 4. 增量式路径规划（周期性）
            # ===============================================
            control_counter += 1
            if self.local_planner.should_replan() or control_counter == 1:
                if debug:
                    print(f"  >> [规划] 重新规划局部路径...")
                
                # 规划短期路径
                local_path = self.local_planner.plan_local_path(
                    current_pos, goal_pos, goal_orn, current_obstacles, num_waypoints=3
                )
                
                if not local_path:
                    print(f"  [!!] 局部路径规划失败")
                    simulate(steps=5, **sim_kwargs)
                    continue
                
                # 验证路径
                is_valid, joint_path = self.local_planner.validate_local_path(
                    local_path, goal_orn, obstacle_ids, current_gripper_pos
                )
                
                if not is_valid:
                    if debug:
                        print(f"  [!!] 局部路径验证失败，等待后重试")
                    simulate(steps=5, **sim_kwargs)
                    continue
                
                # 执行第一个路径点
                if joint_path:
                    target_joints = joint_path[0]
            else:
                # 继续朝当前目标移动
                try:
                    # 计算下一个路径点
                    next_waypoint = self.local_planner.compute_local_waypoint(
                        current_pos, goal_pos, current_obstacles, step_size=0.04
                    )
                    
                    # IK求解
                    target_joints = p.calculateInverseKinematics(
                        self.robot_id, ROBOT_END_EFFECTOR_LINK_ID,
                        next_waypoint, goal_orn
                    )[:7]
                except:
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
                simulate(steps=2, **sim_kwargs)
                continue
            
            # 更新当前状态
            current_joint_pos = np.asarray([p.getJointState(self.robot_id, i)[0] for i in range(7)])
        
        return False
    
    def _execute_single_step(self, target_joints, obstacle_ids, current_gripper_pos,
                            interferer_id, sim_kwargs):
        """
        执行单个运动步骤（带碰撞检测）
        
        Returns:
            bool: 是否成功
        """
        num_arm_joints = len(target_joints)
        
        # Failsafe 1: 碰撞检测
        if obstacle_ids:
            current_joint_pos_check = np.asarray([p.getJointState(self.robot_id, i)[0] 
                                                  for i in range(num_arm_joints)])
            
            if is_state_colliding(self.robot_id, current_joint_pos_check, 
                                 obstacle_ids, current_gripper_pos):
                return False
        
        # Failsafe 2: 近距离保护
        if interferer_id is not None:
            closest_points = p.getClosestPoints(self.robot_id, interferer_id, 
                                               PROXIMITY_FAILSAFE_DISTANCE)
            if closest_points:
                return False
        
        # 设置电机控制
        for joint_id in range(num_arm_joints):
            p.setJointMotorControl2(
                self.robot_id, joint_id, 
                controlMode=p.POSITION_CONTROL,
                targetPosition=target_joints[joint_id],
                maxVelocity=self.max_velocity,
                force=100
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
        
        # 生成多个逃离方向（主方向 + 上方 + 后方）
        escape_strategies = [
            ("主方向", primary_escape, 0.25),  # 增加逃离距离到0.25米
            ("向上", np.array([0, 0, 1.0]), 0.3),  # 直接向上
            ("主方向+上", primary_escape + np.array([0, 0, 0.5]), 0.25),
            ("后撤", -np.array([primary_escape[0], primary_escape[1], 0]) + np.array([0, 0, 0.3]), 0.2)
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
                
                # 强制逃离模式：直接设置关节位置，跳过碰撞检测
                escaped = False
                for step in range(30):
                    # 只检查近距离保护（不检查感知的碰撞）
                    if sim_kwargs.get("interferer_id") is not None:
                        closest_points = p.getClosestPoints(
                            self.robot_id, 
                            sim_kwargs.get("interferer_id"), 
                            PROXIMITY_FAILSAFE_DISTANCE * 0.5  # 放宽到一半
                        )
                        if closest_points:
                            # 即使很近，也继续移动（逃离优先）
                            pass
                    
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

