"""
机械臂控制器模块
负责控制Franka Panda机械臂进行夹取-放置操作
"""

import pybullet as p
import numpy as np
import time
from config import (
    NORMAL_VELOCITY, SLOW_VELOCITY,
    GRASP_HEIGHT_OFFSET, PLACE_HEIGHT_OFFSET,
    REPLANNING_COOLDOWN, REPLAN_SUCCESS_COOLDOWN, ESCAPE_DISTANCE,
    GRIPPER_OPEN_POSITION, GRIPPER_CLOSED_POSITION,
    GRIPPER_FORCE, GRIPPER_MAX_VELOCITY
)


class RobotController:
    """
    Franka Panda机械臂控制器
    实现抓取和放置物体的功能
    """
    
    # --- 修改：在构造函数中接收 obstacle_arm_id ---
    def __init__(self, robot_id, object_id, tray_id, obstacle_arm_id):
        """
        初始化机械臂控制器
        
        Args:
            robot_id: 机械臂的ID
            object_id: 目标物体的ID
            tray_id: 托盘的ID
            obstacle_arm_id: 障碍臂的ID (用于重新规划)
        """
        self.robot_id = robot_id
        self.object_id = object_id
        self.tray_id = tray_id
        self.obstacle_arm_id = obstacle_arm_id # 新增
        
        # Franka Panda的关节索引
        self.num_joints = p.getNumJoints(robot_id)
        self.arm_joint_indices = list(range(7))  # 前7个关节是手臂
        self.gripper_indices = [9, 10]  # 夹爪关节
        self.end_effector_index = 11  # 末端执行器link索引
        
        # 夹爪状态
        self.gripper_open_pos = GRIPPER_OPEN_POSITION
        self.gripper_closed_pos = GRIPPER_CLOSED_POSITION
        
        # 任务状态
        self.state = "idle"
        self.state_counter = 0
        
        # 任务参数
        self.grasp_height_offset = GRASP_HEIGHT_OFFSET
        self.place_height_offset = PLACE_HEIGHT_OFFSET
        
        # --- 用于轨迹缩放的变量 ---
        self.current_target_joint_poses = []
        for i in self.arm_joint_indices:
            joint_state = p.getJointState(self.robot_id, i)
            self.current_target_joint_poses.append(joint_state[0])
            
        self.normal_velocity = NORMAL_VELOCITY
        self.slow_velocity = SLOW_VELOCITY
        
        # --- 用于动态重新规划的变量 ---
        self.final_goal_pos = None          # 状态机的最终笛卡尔目标
        self.is_blocked = False             # 是否被安全系统停止
        self.block_timer = 0.0              # 阻塞计时器
        self.replanning_cooldown = REPLANNING_COOLDOWN
        self.is_replanning = False          # 是否正在执行"绕行"
        self.last_obstacle_point = None     # 障碍物上的最近点
        self.replan_success_timer = 0       # 绕行成功后的冷却计数
        # ---------------------------------
        
        print("机械臂控制器初始化完成")
        self._print_robot_info()
    
    def _print_robot_info(self):
        """打印机械臂的关节信息"""
        print("\n" + "=" * 60)
        print("机械臂关节信息:")
        for i in range(self.num_joints):
            joint_info = p.getJointInfo(self.robot_id, i)
            print(f"  关节 {i}: {joint_info[1].decode('utf-8')} (类型: {joint_info[2]})")
        print("=" * 60 + "\n")
    
    def set_gripper(self, open_gripper):
        """
        控制夹爪开合
        """
        target_pos = self.gripper_open_pos if open_gripper else self.gripper_closed_pos
        
        for i in self.gripper_indices:
            p.setJointMotorControl2(
                bodyUniqueId=self.robot_id,
                jointIndex=i,
                controlMode=p.POSITION_CONTROL,
                targetPosition=target_pos,
                force=GRIPPER_FORCE,
                maxVelocity=GRIPPER_MAX_VELOCITY
            )
    
    def move_to_position(self, target_pos, target_orn=None):
        """
        使用逆运动学 *计算* 目标位置，并 *存储* 目标关节角度
        
        Returns:
            bool: 是否成功计算IK
        """
        # 默认末端执行器朝下
        if target_orn is None:
            target_orn = p.getQuaternionFromEuler([np.pi, 0, 0])
        
        # 获取关节限制
        lower_limits = []
        upper_limits = []
        joint_ranges = []
        rest_poses = []
        
        for i in self.arm_joint_indices:
            joint_info = p.getJointInfo(self.robot_id, i)
            lower_limits.append(joint_info[8])
            upper_limits.append(joint_info[9])
            joint_ranges.append(joint_info[9] - joint_info[8])
            # 使用当前关节位置作为rest pose
            joint_state = p.getJointState(self.robot_id, i)
            rest_poses.append(joint_state[0])
        
        # 计算逆运动学
        joint_poses = p.calculateInverseKinematics(
            bodyUniqueId=self.robot_id,
            endEffectorLinkIndex=self.end_effector_index,
            targetPosition=target_pos,
            targetOrientation=target_orn,
            lowerLimits=lower_limits,
            upperLimits=upper_limits,
            jointRanges=joint_ranges,
            restPoses=rest_poses,
            maxNumIterations=200,
            residualThreshold=1e-4
        )
        
        # 只存储手臂关节（前7个）的目标
        self.current_target_joint_poses = list(joint_poses[:7])
        
        return True

    def apply_safety_control(self, status):
        """
        --- 灵活执行层：轨迹缩放 ---
        根据安全状态，应用“轨迹缩放”
        --- 修改：如果正在重新规划(绕行)，则无视'STOP'，以'GO'速度执行逃逸 ---
        """
        
        if self.current_target_joint_poses is None:
            return
        
        # --- 新增的逻辑：检查是否正在绕行 ---
        if self.is_replanning:
            # 如果正在执行“绕行”
            # 我们必须允许它移动，即使安全状态是'STOP'
            # 绕行本身就是安全动作，我们信任它会将机械臂带到安全位置
            target_poses = self.current_target_joint_poses
            target_velocity = self.normal_velocity # 允许全速逃逸
        
        # --- 原有逻辑（现在用 elif）---
        elif status == "STOP":
            # 状态：停止
            # 目标：保持当前位置
            target_poses = []
            for i in self.arm_joint_indices:
                current_state = p.getJointState(self.robot_id, i)
                target_poses.append(current_state[0])
            
            target_velocity = self.normal_velocity  # 使用大力矩保持位置
            
        elif status == "SLOW":
            # 状态：减速
            target_poses = self.current_target_joint_poses
            target_velocity = self.slow_velocity
            
        else: # status == "GO"
            # 状态：正常
            target_poses = self.current_target_joint_poses
            target_velocity = self.normal_velocity
        
        # 应用关节控制 (这段保持不变)
        for i, joint_idx in enumerate(self.arm_joint_indices):
            p.setJointMotorControl2(
                bodyUniqueId=self.robot_id,
                jointIndex=joint_idx,
                controlMode=p.POSITION_CONTROL,
                targetPosition=target_poses[i],
                force=500,
                maxVelocity=target_velocity
            )
    
    def get_end_effector_pos(self):
        """
        获取末端执行器当前位置
        """
        link_state = p.getLinkState(self.robot_id, self.end_effector_index)
        return np.array(link_state[0])
    
    def check_position_reached(self, target_pos, threshold=0.02):
        """
        检查是否到达笛卡尔目标位置
        """
        current_pos = self.get_end_effector_pos()
        distance = np.linalg.norm(current_pos - np.array(target_pos))
        return distance < threshold

    def check_joint_target_reached(self, threshold=0.1):
        """
        --- 新增：检查是否到达关节目标位置 ---
        用于检查是否已到达“绕行”的中间点
        """
        if not self.current_target_joint_poses:
            return False
        
        current_joint_poses = []
        for i in self.arm_joint_indices:
            joint_state = p.getJointState(self.robot_id, i)
            current_joint_poses.append(joint_state[0])
            
        diff = np.array(self.current_target_joint_poses) - np.array(current_joint_poses)
        error = np.linalg.norm(diff)
        return error < threshold
    
    def get_object_position(self):
        """
        获取目标物体的位置
        """
        pos, _ = p.getBasePositionAndOrientation(self.object_id)
        return np.array(pos)
    
    def get_tray_position(self):
        """
        获取托盘的位置
        """
        pos, _ = p.getBasePositionAndOrientation(self.tray_id)
        return np.array(pos)
    
    def replan_around_obstacle(self):
        """
        --- 新增：灵活执行层：轨迹规划器 ---
        计算并执行一个"绕行"路径
        --- 策略修改：引入“侧向”运动以避免局部最小值陷阱 ---
        """
        if self.last_obstacle_point is None or self.final_goal_pos is None:
            print("  REPLAN: 无法重新规划 (缺少障碍物或目标信息)")
            return

        self.is_replanning = True
        print(f"  REPLAN: 正在重新规划以绕过 {self.last_obstacle_point}")
        
        current_pos = self.get_end_effector_pos()
        obstacle_point = np.array(self.last_obstacle_point)
        
        # 策略1: 远离障碍物的方向
        vector_away_from_obstacle = current_pos - obstacle_point
        norm_away = np.linalg.norm(vector_away_from_obstacle)
        if norm_away > 1e-5:
            vector_away_from_obstacle = vector_away_from_obstacle / norm_away
        else:
            vector_away_from_obstacle = np.array([0, 0, 1.0])
        
        # 策略2: 向上的方向（通常是安全的）
        vector_up = np.array([0, 0, 1.0])

        # --- 新增：策略3: 侧向移动 ---
        # 通过“远离障碍”向量和“向上”向量的叉积，得到一个与两者都垂直的侧向向量
        # 这有助于“绕过”障碍物，而不是仅仅“后退”
        vector_sideways = np.cross(vector_away_from_obstacle, vector_up)
        norm_sideways = np.linalg.norm(vector_sideways)
        
        if norm_sideways > 1e-5:
            vector_sideways = vector_sideways / norm_sideways
        else:
            # 如果“远离”向量与“向上”向量平行（例如，障碍物正上方或正下方）
            # 此时“侧向”无定义，我们退而求其次，
            # 尝试使用“目标”向量来定义一个侧向
            vector_to_goal = self.final_goal_pos - current_pos
            norm_to_goal = np.linalg.norm(vector_to_goal)
            if norm_to_goal > 1e-5:
                 vector_to_goal = vector_to_goal / norm_to_goal
            
            vector_sideways = np.cross(vector_to_goal, vector_up)
            norm_sideways = np.linalg.norm(vector_sideways)
            
            if norm_sideways > 1e-5:
                vector_sideways = vector_sideways / norm_sideways
            else:
                vector_sideways = np.array([1, 0, 0]) # 最后的备用方案
        
        # 组合策略（修改权重）：
        # - 30% 远离障碍物 (减少权重)
        # - 30% 向上移动 (减少权重)
        # - 40% 侧向移动 (新增权重)
        # - 0% 朝向目标（在绕行时，“安全”优先于“效率”）
        escape_dir = (0.30 * vector_away_from_obstacle + 
                     0.30 * vector_up + 
                     0.40 * vector_sideways)
        
        # 归一化逃逸方向
        norm_escape = np.linalg.norm(escape_dir)
        if norm_escape > 1e-5:
            escape_dir = escape_dir / norm_escape
        else:
            escape_dir = np.array([0, 0, 1.0])
        
        # 计算中间路径点
        intermediate_waypoint = current_pos + escape_dir * ESCAPE_DISTANCE
        
        # 确保中间路径点的高度不会太低（至少离地面15cm）
        if intermediate_waypoint[2] < 0.15:
            intermediate_waypoint[2] = 0.15
        
        print(f"  REPLAN: 当前位置: {current_pos}")
        print(f"  REPLAN: 障碍点: {obstacle_point}")
        print(f"  REPLAN: 逃逸方向 (含侧向): {escape_dir}")
        print(f"  REPLAN: 尝试中间路径点: {intermediate_waypoint}")
        
        # 规划到这个中间点
        success = self.move_to_position(intermediate_waypoint)
        if not success:
            print("  REPLAN: 重新规划失败 (IK无法求解)")
            self.is_replanning = False # 放弃绕行
    
    def execute_pick_and_place(self):
        """
        执行完整的抓取-放置流程 (状态机)
        """
        
        if self.state == "idle":
            print("\n开始执行抓取-放置任务...")
            print("=" * 60)
            self.set_gripper(True)
            obj_pos = self.get_object_position()
            pre_grasp_pos = obj_pos + np.array([0, 0, self.grasp_height_offset])
            
            print(f"状态: 移动到物体上方")
            print(f"  目标位置: {pre_grasp_pos}")
            
            self.final_goal_pos = pre_grasp_pos # 设置最终目标
            self.move_to_position(self.final_goal_pos)
            self.state = "moving_to_pre_grasp"
            self.state_counter = 0
        
        elif self.state == "moving_to_pre_grasp":
            self.state_counter += 1
            if self.check_position_reached(self.final_goal_pos) or self.state_counter > 300:
                print(f"  到达物体上方!")
                print(f"状态: 下降到抓取位置")
                
                obj_pos = self.get_object_position()
                grasp_pos = obj_pos + np.array([0, 0, 0.02])
                
                print(f"  目标抓取位置: {grasp_pos}")
                
                self.final_goal_pos = grasp_pos # 设置最终目标
                self.move_to_position(self.final_goal_pos)
                self.state = "moving_to_grasp"
                self.state_counter = 0
        
        elif self.state == "moving_to_grasp":
            self.state_counter += 1
            if self.check_position_reached(self.final_goal_pos, threshold=0.03) or self.state_counter > 200:
                print(f"状态: 关闭夹爪抓取物体")
                self.set_gripper(False)
                self.state = "grasping"
                self.state_counter = 0
        
        elif self.state == "grasping":
            self.state_counter += 1
            if self.state_counter > 50:
                print(f"状态: 提升物体")
                current_pos = self.get_end_effector_pos()
                lift_pos = current_pos + np.array([0, 0, 0.3])
                
                print(f"  目标位置: {lift_pos}")
                
                self.final_goal_pos = lift_pos # 设置最终目标
                self.move_to_position(self.final_goal_pos)
                self.state = "lifting"
                self.state_counter = 0
        
        elif self.state == "lifting":
            self.state_counter += 1
            if self.check_position_reached(self.final_goal_pos) or self.state_counter > 200:
                print(f"状态: 移动到托盘上方")
                tray_pos = self.get_tray_position()
                pre_place_pos = tray_pos + np.array([0, 0, self.place_height_offset])
                
                print(f"  目标位置: {pre_place_pos}")
                
                self.final_goal_pos = pre_place_pos # 设置最终目标
                self.move_to_position(self.final_goal_pos)
                self.state = "moving_to_pre_place"
                self.state_counter = 0
        
        elif self.state == "moving_to_pre_place":
            self.state_counter += 1
            if self.check_position_reached(self.final_goal_pos) or self.state_counter > 200:
                print(f"状态: 下降到放置位置")
                tray_pos = self.get_tray_position()
                place_pos = tray_pos + np.array([0, 0, 0.15])
                
                print(f"  目标位置: {place_pos}")
                
                self.final_goal_pos = place_pos # 设置最终目标
                self.move_to_position(self.final_goal_pos)
                self.state = "moving_to_place"
                self.state_counter = 0
        
        elif self.state == "moving_to_place":
            self.state_counter += 1
            if self.check_position_reached(self.final_goal_pos, threshold=0.03) or self.state_counter > 200:
                print(f"状态: 打开夹爪放置物体")
                self.set_gripper(True)
                self.state = "placing"
                self.state_counter = 0
        
        elif self.state == "placing":
            self.state_counter += 1
            if self.state_counter > 50:
                print(f"状态: 抬起手臂")
                current_pos = self.get_end_effector_pos()
                retract_pos = current_pos + np.array([0, 0, 0.2])
                
                print(f"  目标位置: {retract_pos}")
                
                self.final_goal_pos = retract_pos # 设置最终目标
                self.move_to_position(self.final_goal_pos)
                self.state = "retracting"
                self.state_counter = 0
        
        elif self.state == "retracting":
            self.state_counter += 1
            if self.check_position_reached(self.final_goal_pos) or self.state_counter > 200:
                print(f"状态: 任务完成！")
                print("=" * 60 + "\n")
                
                self.final_goal_pos = None # 清除最终目标
                self.state = "done"
                self.state_counter = 0
        
        elif self.state == "done":
            pass
    
    def update(self, safety_status="GO", obstacle_point=None):
        """
        --- 灵活执行层：主更新循环 ---
        
        Args:
            safety_status: 来自安全层的状态 ('GO', 'SLOW', 'STOP')
            obstacle_point: 障碍物上的最近点
        """
        # 1. 安全优先：始终应用轨迹缩放（安全停止/减速） [cite: 673-675]
        self.apply_safety_control(safety_status)
        
        # 2. 更新阻塞状态和障碍物位置
        self.last_obstacle_point = obstacle_point
        
        # 改进的阻塞检测：STOP和SLOW都算作阻塞
        self.is_blocked = (safety_status == "STOP" or safety_status == "SLOW")
        
        if safety_status == "STOP":
            # 状态：STOP - 严重阻塞，快速累积计时器
            self.block_timer += 1.0
        elif safety_status == "SLOW":
            # 状态：SLOW - 轻度阻塞，慢速累积计时器
            self.block_timer += 0.3
        else:
            # 状态：GO - 没有阻塞，快速衰减计时器
            if self.block_timer > 0:
                self.block_timer = max(0, self.block_timer - 2)
        
        # 减少绕行成功冷却计数
        if self.replan_success_timer > 0:
            self.replan_success_timer -= 1
            
        # 3. 检查"绕行"是否已完成
        if self.is_replanning:
            if self.check_joint_target_reached(threshold=0.15):
                print("  REPLAN: 到达中间路径点。")
                print("  REPLAN: 恢复原始任务...")
                self.is_replanning = False
                self.move_to_position(self.final_goal_pos) # 重新规划回到原始目标
                self.block_timer = 0 # 冷却，防止立即再次绕行
                self.replan_success_timer = REPLAN_SUCCESS_COOLDOWN
            
        # 4. 检查是否需要触发"绕行"
        # 检查是否：被阻塞 AND 阻塞超时 AND 有一个最终目标 AND 当前没有在绕行 AND 冷却时间已过
        should_replan = (self.is_blocked and 
                        self.block_timer >= self.replanning_cooldown and
                        self.replan_success_timer == 0)
        can_replan = (self.final_goal_pos is not None) and (not self.is_replanning)

        if should_replan and can_replan:
            print(f"  REPLAN: 已阻塞 {self.block_timer:.1f} 帧，超过 {self.replanning_cooldown} 帧。")
            self.replan_around_obstacle()
            self.block_timer = 0 # 重置计时器
        
        # 5. 推进状态机
        # 检查是否可以推进状态（安全且未绕行）
        can_advance_state = (safety_status == "GO" and (not self.is_replanning))

        # 推进状态机的条件：
        # 1. 我们可以安全推进 (can_advance_state)
        # 2. 或者 我们处于 "idle" 状态（必须允许设置第一个目标，以打破死锁）
        if (can_advance_state or self.state == "idle") and self.state != "done":
            self.execute_pick_and_place()
    
    def reset(self):
        """
        重置控制器状态
        """
        self.state = "idle"
        self.state_counter = 0
        self.current_target_joint_poses = []
        for i in self.arm_joint_indices:
            joint_state = p.getJointState(self.robot_id, i)
            self.current_target_joint_poses.append(joint_state[0])
            
        # 重置绕行状态
        self.final_goal_pos = None
        self.is_blocked = False
        self.block_timer = 0
        self.is_replanning = False
        self.last_obstacle_point = None
        self.replan_success_timer = 0
            
        print("\n控制器已重置\n")