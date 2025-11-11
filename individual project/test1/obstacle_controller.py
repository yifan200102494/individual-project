"""
障碍臂控制器模块
负责控制障碍臂的随机运动，同时避免进入禁区
"""

import pybullet as p
import numpy as np
from config import OBSTACLE_UPDATE_INTERVAL, OBSTACLE_MOVE_STEP_RATIO, OBSTACLE_JOINT_FORCE


class ObstacleArmController:
    """
    控制障碍臂的随机运动，同时避免进入白色机械臂基座的前三节关节区域
    """
    
    def __init__(self, obstacle_arm_id, main_robot_id, forbidden_zone_center, forbidden_zone_radius):
        """
        初始化控制器
        
        Args:
            obstacle_arm_id: 障碍臂的ID
            main_robot_id: 主机械臂的ID
            forbidden_zone_center: 禁区中心位置 [x, y, z]
            forbidden_zone_radius: 禁区半径（米）
        """
        self.obstacle_arm_id = obstacle_arm_id
        self.main_robot_id = main_robot_id
        self.forbidden_zone_center = np.array(forbidden_zone_center)
        self.forbidden_zone_radius = forbidden_zone_radius
        
        # 获取所有可动关节
        self.joint_indices = []
        num_joints = p.getNumJoints(obstacle_arm_id)
        for i in range(num_joints):
            joint_info = p.getJointInfo(obstacle_arm_id, i)
            if joint_info[2] == p.JOINT_REVOLUTE:
                self.joint_indices.append(i)
        
        # 存储当前目标关节角度
        self.target_joint_angles = []
        for idx in self.joint_indices:
            joint_state = p.getJointState(obstacle_arm_id, idx)
            self.target_joint_angles.append(joint_state[0])
        
        # 运动参数
        self.update_counter = 0
        self.update_interval = OBSTACLE_UPDATE_INTERVAL
    
    def is_in_forbidden_zone(self):
        """
        检查障碍臂是否进入禁区
        
        Returns:
            bool: True表示在禁区内
        """
        # 检查障碍臂的每个链接
        num_links = p.getNumJoints(self.obstacle_arm_id)
        
        for link_idx in range(-1, num_links):  # -1是基座
            if link_idx == -1:
                # 检查基座
                base_pos, _ = p.getBasePositionAndOrientation(self.obstacle_arm_id)
                link_pos = np.array(base_pos)
            else:
                # 检查链接
                link_state = p.getLinkState(self.obstacle_arm_id, link_idx)
                link_pos = np.array(link_state[0])
            
            # 计算到禁区中心的距离
            distance = np.linalg.norm(link_pos - self.forbidden_zone_center)
            
            if distance < self.forbidden_zone_radius:
                return True
        
        return False
    
    def generate_random_target(self):
        """
        生成随机目标关节角度
        
        Returns:
            list: 随机关节角度列表
        """
        new_targets = []
        for idx in self.joint_indices:
            joint_info = p.getJointInfo(self.obstacle_arm_id, idx)
            lower_limit = joint_info[8]
            upper_limit = joint_info[9]
            
            # 生成随机角度
            random_angle = np.random.uniform(lower_limit, upper_limit)
            new_targets.append(random_angle)
        
        return new_targets
    
    def update(self):
        """
        更新障碍臂的运动
        每次调用会平滑地向目标角度移动
        """
        self.update_counter += 1
        
        # 定期生成新的随机目标
        if self.update_counter >= self.update_interval:
            self.update_counter = 0
            
            # --- 修改：移除is_in_forbidden_zone检查 ---
            # 为了真正测试主机械臂的避障，
            # 我们让障碍臂随机移动，*不管*它是否会进入禁区。
            
            # 直接生成并设置新目标
            print("  OBSTACLE: Generating new random target...")
            self.target_joint_angles = self.generate_random_target()
            
            # --- 旧逻辑（已注释掉）---
            # # 保存当前状态
            # saved_states = []
            # for idx in self.joint_indices:
            #     joint_state = p.getJointState(self.obstacle_arm_id, idx)
            #     saved_states.append(joint_state[0])
            
            # # 尝试生成新目标
            # max_attempts = 10
            # found_valid_target = False
            
            # for attempt in range(max_attempts):
            #     # 生成随机目标
            #     test_targets = self.generate_random_target()
                
            #     # 临时设置到目标位置进行测试
            #     for i, idx in enumerate(self.joint_indices):
            #         p.resetJointState(self.obstacle_arm_id, idx, test_targets[i])
                
            #     # 检查是否在禁区
            #     if not self.is_in_forbidden_zone():
            #         self.target_joint_angles = test_targets
            #         found_valid_target = True
            #         # 恢复当前状态，然后平滑移动过去
            #         for i, idx in enumerate(self.joint_indices):
            #             p.resetJointState(self.obstacle_arm_id, idx, saved_states[i])
            #         break
            #     else:
            #         # 恢复状态，继续尝试
            #         for i, idx in enumerate(self.joint_indices):
            #             p.resetJointState(self.obstacle_arm_id, idx, saved_states[i])
            
            # if not found_valid_target:
            #     # 如果找不到有效目标，保持当前目标
            #     self.target_joint_angles = saved_states
            # --- 结束旧逻辑 ---
        
        # 平滑移动到目标角度
        for i, idx in enumerate(self.joint_indices):
            current_state = p.getJointState(self.obstacle_arm_id, idx)
            current_angle = current_state[0]
            target_angle = self.target_joint_angles[i]
            
            # 计算差值并平滑插值
            angle_diff = target_angle - current_angle
            move_step = angle_diff * OBSTACLE_MOVE_STEP_RATIO
            
            new_angle = current_angle + move_step
            
            # 使用位置控制
            p.setJointMotorControl2(
                bodyUniqueId=self.obstacle_arm_id,
                jointIndex=idx,
                controlMode=p.POSITION_CONTROL,
                targetPosition=new_angle,
                force=OBSTACLE_JOINT_FORCE
            )