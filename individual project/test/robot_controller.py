"""
机械臂控制器模块
负责控制Franka Panda机械臂的运动、抓取、放置。
包含了使用末端摄像头进行障碍物识别的功能。
"""

import pybullet as p
import numpy as np
import time
from config import (
    # 导入摄像头相关参数
    END_EFFECTOR_LINK_INDEX, CAMERA_WIDTH, CAMERA_HEIGHT, CAMERA_FOV,
    CAMERA_NEAR_PLANE, CAMERA_FAR_PLANE,
    
    # 导入避障检测相关参数
    AVOIDANCE_DANGER_DISTANCE 
)


class RobotController:
    """
    控制Franka Panda机械臂执行任务，并包含视觉识别功能。
    """
    
    def __init__(self, robot_id, object_id, tray_id, obstacle_arm_id):
        """
        初始化机械臂控制器
        
        Args:
            robot_id (int): 机器人ID
            object_id (int): 目标物体ID
            tray_id (int): 托盘ID
            obstacle_arm_id (int): 障碍臂ID (这是关键)
        """
        self.robot_id = robot_id
        self.object_id = object_id
        self.tray_id = tray_id
        
        # 存储障碍臂的ID，用于视觉识别
        self.obstacle_arm_id = obstacle_arm_id 
        
        # 摄像头和末端执行器
        self.end_effector_link_index = END_EFFECTOR_LINK_INDEX
        self.cam_width = CAMERA_WIDTH
        self.cam_height = CAMERA_HEIGHT
        self.cam_fov = CAMERA_FOV
        self.cam_near = CAMERA_NEAR_PLANE
        self.cam_far = CAMERA_FAR_PLANE
        
        # 其他控制状态 (此处省略，专注于视觉)
        # ...


    def get_camera_view(self):
        """
        获取当前末端摄像头的图像数据，包括分割掩码。
        
        Returns:
            tuple: (rgb_img, depth_img, seg_mask)
                   rgb_img: RGB图像 (numpy array)
                   depth_img: 深度图像 (numpy array, 0-1 范围)
                   seg_mask: 分割掩码 (numpy array, 值为 bodyUniqueId)
        """
        
        # 1. 计算摄像头位姿
        # 获取末端执行器link的世界坐标和姿态
        link_state = p.getLinkState(self.robot_id, self.end_effector_link_index)
        link_pos = link_state[0]
        link_orn = link_state[1]
        
        # 计算旋转矩阵
        matrix = p.getMatrixFromQuaternion(link_orn)
        rot_matrix = np.array(matrix).reshape(3, 3)
        
        # 根据link的姿态计算摄像头的前方(Z轴)和上方(-Y轴)向量
        forward_vec = rot_matrix[:, 2] # Link的Z轴
        up_vec = -rot_matrix[:, 1]     # Link的-Y轴 (作为摄像头的"上")
        
        # 摄像头位置 (假设与link 11的 frame origin 重合)
        cam_pos = np.array(link_pos)
        
        # 摄像头目标点 (沿着Z轴正向看)
        cam_target = cam_pos + forward_vec
        
        # 2. 计算视图(View)和投影(Projection)矩阵
        view_matrix = p.computeViewMatrix(
            cameraEyePosition=cam_pos,
            cameraTargetPosition=cam_target,
            cameraUpVector=up_vec
        )
        
        aspect = self.cam_width / self.cam_height
        projection_matrix = p.computeProjectionMatrixFOV(
            fov=self.cam_fov,
            aspect=aspect,
            nearVal=self.cam_near,
            farVal=self.cam_far
        )
        
        # 3. 获取图像
        images = p.getCameraImage(
            width=self.cam_width,
            height=self.cam_height,
            viewMatrix=view_matrix,
            projectionMatrix=projection_matrix,
            # 必须使用 OpenGL 渲染器才能获取分割掩码
            renderer=p.ER_BULLET_HARDWARE_OPENGL 
        )
        
        # 解析返回的数据
        rgb_img = np.array(images[2]).reshape(self.cam_height, self.cam_width, 4)
        depth_img = np.array(images[3]).reshape(self.cam_height, self.cam_width)
        seg_mask = np.array(images[4]).reshape(self.cam_height, self.cam_width)
        
        return rgb_img, depth_img, seg_mask

    
    # --- 这是您请求的核心识别功能 ---
    
    def check_for_obstacle(self):
        """
        使用末端摄像头检查前方是否有*需要避让*的障碍物。
        
        此函数会识别视野中的所有物体，但 **只把 红色障碍臂 标记为障碍**。
        
        Returns:
            bool: True 如果检测到红色障碍臂 (self.obstacle_arm_id) 在危险距离内。
        """
        
        # 1. 获取摄像头的最新"视野"
        _, depth_img, seg_mask = self.get_camera_view()
        
        # 2. 【核心识别逻辑】
        # 检查分割掩码 (seg_mask) 中是否存在障碍臂的ID
        # seg_mask 上的每个像素值 对应于 它所看到的物体的 bodyUniqueId
        
        # 找到所有等于"障碍臂ID"的像素
        obstacle_pixels_mask = (seg_mask == self.obstacle_arm_id)
        
        # 3. 判断是否看到了障碍臂
        if np.any(obstacle_pixels_mask):
            # 如果看到了，我们还需要判断它是否 "太近了"
            
            # 3a. 获取所有障碍物像素的深度值 (0-1范围)
            obstacle_depth_values = depth_img[obstacle_pixels_mask]
            
            # 3b. 将深度值 (0-1) 转换回真实的米(m)
            # 这是标准OpenGL的深度转换公式
            near = self.cam_near
            far = self.cam_far
            real_depths_m = (far * near) / (far - (far - near) * obstacle_depth_values)
            
            # 3c. 检查是否有任何部分的障碍物在 "危险距离" 内
            min_obstacle_dist = np.min(real_depths_m)
            
            if min_obstacle_dist < AVOIDANCE_DANGER_DISTANCE:
                print(f"!!! 视觉检测：检测到红色障碍臂！距离: {min_obstacle_dist:.2f}m (危险！)")
                return True # 是障碍物，且在危险距离内
            else:
                 print(f"  视觉检测：看到红色障碍臂，但距离安全: {min_obstacle_dist:.2f}m")
        
        # 4. 【区分其他物体】
        # 检查是否看到了其他物体 (例如方块或托盘)
        # 根据您的要求，这些*不*应该触发避障
        unique_ids = np.unique(seg_mask)
        other_objects_seen = []
        for uid in unique_ids:
            # uid < 0 是背景
            if uid >= 0 and uid != self.obstacle_arm_id:
                if uid == self.object_id:
                    other_objects_seen.append("目标方块")
                elif uid == self.tray_id:
                    other_objects_seen.append("托盘")
                else:
                    # 比如地面或其他
                    other_objects_seen.append(f"其他物体 (ID: {uid})")
        
        if other_objects_seen:
            # 打印出来，但返回 False (因为它们不是需要避让的障碍)
            print(f"  视觉检测：看到 {', '.join(other_objects_seen)} (非障碍, 已忽略)。")
            
        # 默认：没有检测到*需要避让的*障碍物
        return False

    
    # -----------------------------------------------------
    # (此处省略了机械臂的其他控制方法, 如:
    #  - move_to_position()
    #  - control_gripper()
    #  - run_task_fsm() ... 等等
    # )
    # -----------------------------------------------------