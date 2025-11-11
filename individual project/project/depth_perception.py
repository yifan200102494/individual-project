"""
深度感知模块
基于外参标定和深度传感的障碍物占用检测

核心概念：
1. Extrinsic Calibration: 定义相机在机器人末端的位置和姿态
2. Depth Sensing: 使用深度相机获取深度图像
3. Occupancy Detection: 将深度信息转换为障碍物位置
"""

import pybullet as p
import numpy as np
from typing import List, Tuple, Optional, Dict


class DepthCamera:
    """
    深度相机类
    负责外参标定和深度图像获取
    """
    
    def __init__(self, 
                 robot_id: int,
                 sensor_link_id: int,
                 image_width: int = 128,
                 image_height: int = 128,
                 fov: float = 60.0,
                 near_plane: float = 0.01,
                 far_plane: float = 2.0):
        """
        初始化深度相机
        
        Args:
            robot_id: 机器人ID
            sensor_link_id: 传感器连接的link ID（通常是末端执行器）
            image_width: 图像宽度（像素）
            image_height: 图像高度（像素）
            fov: 视场角（度）
            near_plane: 近平面距离（米）
            far_plane: 远平面距离（米）
        """
        self.robot_id = robot_id
        self.sensor_link_id = sensor_link_id
        self.image_width = image_width
        self.image_height = image_height
        self.fov = fov
        self.near_plane = near_plane
        self.far_plane = far_plane
        
        # ==========================================
        # Extrinsic Calibration（外参标定）
        # ==========================================
        # 定义相机在传感器link坐标系中的相对位置和姿态
        # 这些参数需要根据实际的机器人配置进行标定
        
        # 相机相对于末端执行器的平移（米）
        # 假设相机安装在末端执行器前方稍微偏下的位置
        self.camera_offset_position = np.array([0.0, 0.0, 0.0])  # [x, y, z]
        
        # 相机相对于末端执行器的旋转（欧拉角，弧度）
        # 假设相机朝向与末端执行器一致，稍微向下倾斜
        self.camera_offset_orientation = np.array([0.0, 0.0, 0.0])  # [roll, pitch, yaw]
        
        # 计算投影矩阵（只需计算一次）
        self.projection_matrix = self._compute_projection_matrix()
        
        print(f"[深度相机] 初始化完成")
        print(f"  分辨率: {image_width}x{image_height}")
        print(f"  视场角: {fov}°")
        print(f"  深度范围: {near_plane}m - {far_plane}m")
    
    def _compute_projection_matrix(self) -> List[float]:
        """
        计算相机投影矩阵
        
        Returns:
            投影矩阵（4x4）
        """
        aspect = self.image_width / self.image_height
        projection_matrix = p.computeProjectionMatrixFOV(
            fov=self.fov,
            aspect=aspect,
            nearVal=self.near_plane,
            farVal=self.far_plane
        )
        return projection_matrix
    
    def get_camera_pose_in_world(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        获取相机在世界坐标系中的位置和姿态
        这是外参标定的核心：从机器人link状态推算相机位置
        
        Returns:
            Tuple[相机位置(3,), 相机姿态四元数(4,)]
        """
        # 1. 获取传感器link在世界坐标系中的状态
        link_state = p.getLinkState(
            self.robot_id, 
            self.sensor_link_id,
            computeForwardKinematics=True
        )
        link_pos_world = np.array(link_state[0])  # 位置
        link_orn_world = np.array(link_state[1])  # 四元数方向
        
        # 2. 应用相机的外参标定（相机相对于link的偏移）
        # 旋转偏移 - 使用四元数乘法组合旋转
        camera_offset_quat = p.getQuaternionFromEuler(self.camera_offset_orientation)
        
        # 四元数乘法: q_world = q_link * q_offset
        camera_orn_world = np.array(p.multiplyTransforms(
            [0, 0, 0], link_orn_world.tolist(),      # link的姿态
            [0, 0, 0], camera_offset_quat            # 相机偏移
        )[1])  # 只取四元数部分，转为numpy数组
        
        # 平移偏移（考虑link的旋转）
        link_rot_matrix = np.array(p.getMatrixFromQuaternion(link_orn_world)).reshape(3, 3)
        camera_pos_world = link_pos_world + link_rot_matrix @ self.camera_offset_position
        
        return camera_pos_world, camera_orn_world
    
    def capture_depth_image(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        捕获深度图像
        
        Returns:
            Tuple[depth_buffer(H, W), rgb_image(H, W, 3)]
        """
        # 获取相机在世界坐标系中的位置和姿态
        camera_pos, camera_orn = self.get_camera_pose_in_world()
        
        # 将四元数转换为旋转矩阵
        camera_rot_matrix = np.array(p.getMatrixFromQuaternion(camera_orn)).reshape(3, 3)
        
        # 计算相机的前方向（通常是-Z轴）和上方向（通常是Y轴）
        camera_forward = camera_rot_matrix @ np.array([0, 0, -1])
        camera_up = camera_rot_matrix @ np.array([0, 1, 0])
        
        # 计算目标点（相机朝向的点）
        target_pos = camera_pos + camera_forward * 1.0
        
        # 计算视图矩阵
        view_matrix = p.computeViewMatrix(
            cameraEyePosition=camera_pos.tolist(),
            cameraTargetPosition=target_pos.tolist(),
            cameraUpVector=camera_up.tolist()
        )
        
        # 获取相机图像（RGB + Depth）
        width, height, rgb_img, depth_buffer, seg_img = p.getCameraImage(
            width=self.image_width,
            height=self.image_height,
            viewMatrix=view_matrix,
            projectionMatrix=self.projection_matrix,
            renderer=p.ER_TINY_RENDERER  # 使用快速渲染器
        )
        
        # 将RGB图像转换为numpy数组
        rgb_array = np.array(rgb_img).reshape(height, width, 4)[:, :, :3]
        
        # 将深度缓冲区转换为numpy数组
        depth_array = np.array(depth_buffer).reshape(height, width)
        
        return depth_array, rgb_array
    
    def depth_buffer_to_distance(self, depth_buffer: np.ndarray) -> np.ndarray:
        """
        将深度缓冲区值转换为实际距离（米）
        
        PyBullet的深度缓冲区是归一化的非线性值，需要转换
        
        Args:
            depth_buffer: 深度缓冲区数组 (H, W)，值在[0, 1]
            
        Returns:
            距离数组 (H, W)，单位：米
        """
        # PyBullet深度缓冲区公式：
        # depth = far * near / (far - (far - near) * depth_buffer)
        distance = self.far_plane * self.near_plane / (
            self.far_plane - (self.far_plane - self.near_plane) * depth_buffer
        )
        return distance


class OccupancyDetector:
    """
    占用检测器
    将深度图像转换为3D空间中的障碍物位置
    """
    
    def __init__(self, depth_camera: DepthCamera):
        """
        初始化占用检测器
        
        Args:
            depth_camera: 深度相机实例
        """
        self.camera = depth_camera
        
        # 预计算像素坐标网格（提高效率）
        self.pixel_coords = self._create_pixel_grid()
        
        print(f"[占用检测器] 初始化完成")
    
    def _create_pixel_grid(self) -> np.ndarray:
        """
        创建像素坐标网格
        
        Returns:
            像素坐标数组 (H, W, 2)，存储每个像素的(u, v)坐标
        """
        u_coords = np.arange(self.camera.image_width)
        v_coords = np.arange(self.camera.image_height)
        u_grid, v_grid = np.meshgrid(u_coords, v_coords)
        pixel_coords = np.stack([u_grid, v_grid], axis=-1)
        return pixel_coords
    
    def depth_to_point_cloud(self, 
                            depth_distance: np.ndarray,
                            camera_pos: np.ndarray,
                            camera_orn: np.ndarray) -> np.ndarray:
        """
        将深度图像转换为3D点云（世界坐标系）
        
        Args:
            depth_distance: 深度距离数组 (H, W)
            camera_pos: 相机在世界坐标系中的位置 (3,)
            camera_orn: 相机在世界坐标系中的姿态（四元数） (4,)
            
        Returns:
            点云数组 (N, 3)，世界坐标系中的3D点
        """
        H, W = depth_distance.shape
        
        # 相机内参（从FOV计算）
        focal_length = (W / 2.0) / np.tan(np.deg2rad(self.camera.fov / 2.0))
        cx = W / 2.0
        cy = H / 2.0
        
        # 获取所有像素坐标
        u = self.pixel_coords[:, :, 0]
        v = self.pixel_coords[:, :, 1]
        
        # 反投影到相机坐标系
        # 相机坐标系：X右，Y下，Z前
        z_cam = depth_distance
        x_cam = (u - cx) * z_cam / focal_length
        y_cam = (v - cy) * z_cam / focal_length
        
        # 组合为相机坐标系中的点 (H, W, 3)
        points_cam = np.stack([x_cam, y_cam, z_cam], axis=-1)
        
        # 重塑为 (H*W, 3)
        points_cam_flat = points_cam.reshape(-1, 3)
        
        # 转换到世界坐标系
        camera_rot_matrix = np.array(p.getMatrixFromQuaternion(camera_orn)).reshape(3, 3)
        points_world = (camera_rot_matrix @ points_cam_flat.T).T + camera_pos
        
        return points_world
    
    def detect_obstacles_from_point_cloud(self,
                                         point_cloud: np.ndarray,
                                         ignore_ids: set,
                                         voxel_size: float = 0.05,
                                         min_points_threshold: int = 10,
                                         max_z_height: float = 1.5,
                                         min_z_height: float = 0.02,
                                         tray_position: Optional[np.ndarray] = None,
                                         tray_size: Optional[np.ndarray] = None,
                                         gripper_position: Optional[np.ndarray] = None) -> List[Tuple[int, np.ndarray, np.ndarray]]:
        """
        从点云中检测障碍物占用
        
        使用体素化(voxelization)方法简化点云，然后进行空间聚类
        
        Args:
            point_cloud: 点云 (N, 3) - 世界坐标系中的3D点
            ignore_ids: 要忽略的物体ID（用于确定是否需要过滤夹爪区域）
            voxel_size: 体素大小（米）
            min_points_threshold: 最小点数阈值，少于此数量的簇会被过滤
            max_z_height: 最大高度阈值（过滤天花板等）
            min_z_height: 最小高度阈值（过滤地面）
            tray_position: 托盘位置 [x, y, z]（可选）
            tray_size: 托盘尺寸 [length, width, height]（可选）
            gripper_position: 夹爪位置 [x, y, z]（可选，用于过滤被抓取物品）
            
        Returns:
            障碍物列表: [(obs_id, position, velocity), ...]
            注意：obs_id为虚拟ID（从1开始），velocity为零向量（点云无法直接获取速度）
        """
        # 1. 过滤无效点（超出范围的点）
        valid_mask = (
            (point_cloud[:, 2] > min_z_height) &  # 高于地面
            (point_cloud[:, 2] < max_z_height) &  # 低于天花板
            (np.abs(point_cloud[:, 0]) < 2.0) &   # X范围合理
            (np.abs(point_cloud[:, 1]) < 2.0) &   # Y范围合理
            (~np.isnan(point_cloud).any(axis=1)) & # 无NaN
            (~np.isinf(point_cloud).any(axis=1))   # 无Inf
        )
        
        # 🔥 2. 过滤夹爪附近的点云（被抓取的物品）
        # 如果提供了夹爪位置且ignore_ids不为空（说明有被抓取的物品），则过滤夹爪附近的点
        if gripper_position is not None and len(ignore_ids) > 0:
            # 计算每个点到夹爪的距离
            distances_to_gripper = np.linalg.norm(point_cloud - gripper_position, axis=1)
            # 过滤掉距离夹爪15cm以内的点（被抓取的物品通常在这个范围内）
            gripper_radius = 0.15
            is_not_near_gripper = distances_to_gripper > gripper_radius
            valid_mask = valid_mask & is_not_near_gripper
        
        # 3. 过滤托盘底部（只保留托盘的四壁）
        if tray_position is not None and tray_size is not None:
            # 托盘的边界
            tray_x_min = tray_position[0] - tray_size[0] / 2
            tray_x_max = tray_position[0] + tray_size[0] / 2
            tray_y_min = tray_position[1] - tray_size[1] / 2
            tray_y_max = tray_position[1] + tray_size[1] / 2
            tray_z_max = tray_position[2] + tray_size[2]
            
            # 定义托盘内部区域（缩小边界，留出边缘）
            # 边缘厚度约5cm，这样托盘的四壁会被保留
            edge_thickness = 0.05
            inner_x_min = tray_x_min + edge_thickness
            inner_x_max = tray_x_max - edge_thickness
            inner_y_min = tray_y_min + edge_thickness
            inner_y_max = tray_y_max - edge_thickness
            
            # 识别托盘底部的点（在托盘内部且高度较低）
            is_inside_tray = (
                (point_cloud[:, 0] > inner_x_min) &
                (point_cloud[:, 0] < inner_x_max) &
                (point_cloud[:, 1] > inner_y_min) &
                (point_cloud[:, 1] < inner_y_max) &
                (point_cloud[:, 2] < tray_z_max + 0.02)  # 托盘高度+2cm以内
            )
            
            # 过滤掉托盘底部的点（保留托盘边缘和其他障碍物）
            valid_mask = valid_mask & ~is_inside_tray
        
        filtered_points = point_cloud[valid_mask]
        
        if len(filtered_points) < min_points_threshold:
            return []
        
        # 2. 体素化 - 将点云降采样到规则网格
        # 计算每个点属于哪个体素
        voxel_indices = np.floor(filtered_points / voxel_size).astype(np.int32)
        
        # 使用字典存储每个体素中的点
        voxel_dict = {}
        for i, voxel_idx in enumerate(voxel_indices):
            key = tuple(voxel_idx)
            if key not in voxel_dict:
                voxel_dict[key] = []
            voxel_dict[key].append(filtered_points[i])
        
        # 3. 计算每个体素的中心点
        voxel_centers = []
        for voxel_points in voxel_dict.values():
            if len(voxel_points) >= 2:  # 至少2个点才认为是有效体素
                center = np.mean(voxel_points, axis=0)
                voxel_centers.append(center)
        
        if len(voxel_centers) == 0:
            return []
        
        voxel_centers = np.array(voxel_centers)
        
        # 4. 简单的空间聚类 - 基于距离的连通性
        # 使用DBSCAN思想，但简化实现
        clusters = self._simple_spatial_clustering(
            voxel_centers, 
            eps=voxel_size * 3,  # 聚类距离阈值
            min_samples=min_points_threshold // 5  # 最少体素数
        )
        
        # 5. 为每个簇生成障碍物信息
        obstacles = []
        obs_id_counter = 1
        
        for cluster_points in clusters:
            if len(cluster_points) >= 3:  # 至少3个体素
                # 计算簇的中心作为障碍物位置
                obstacle_center = np.mean(cluster_points, axis=0)
                
                # 速度设为零（点云无法直接测量速度）
                velocity = np.array([0.0, 0.0, 0.0])
                
                # 添加到障碍物列表
                # 格式: (obs_id, position, velocity)
                obstacles.append((obs_id_counter, obstacle_center, velocity))
                obs_id_counter += 1
        
        return obstacles
    
    def _simple_spatial_clustering(self, 
                                   points: np.ndarray, 
                                   eps: float, 
                                   min_samples: int) -> List[np.ndarray]:
        """
        简单的空间聚类算法（类DBSCAN）
        
        Args:
            points: 点云 (N, 3)
            eps: 邻域半径
            min_samples: 最小样本数
            
        Returns:
            簇列表，每个簇是一个点数组
        """
        n_points = len(points)
        if n_points == 0:
            return []
        
        # 标记每个点是否已被访问
        visited = np.zeros(n_points, dtype=bool)
        # 标记每个点属于哪个簇（-1表示噪声）
        labels = np.full(n_points, -1, dtype=np.int32)
        
        cluster_id = 0
        
        for i in range(n_points):
            if visited[i]:
                continue
            
            visited[i] = True
            
            # 找到当前点的邻居
            distances = np.linalg.norm(points - points[i], axis=1)
            neighbors = np.where(distances < eps)[0]
            
            if len(neighbors) < min_samples:
                # 噪声点
                labels[i] = -1
            else:
                # 开始新簇
                labels[i] = cluster_id
                
                # 扩展簇
                seed_set = list(neighbors)
                j = 0
                while j < len(seed_set):
                    neighbor_idx = seed_set[j]
                    
                    if not visited[neighbor_idx]:
                        visited[neighbor_idx] = True
                        
                        # 找邻居的邻居
                        neighbor_distances = np.linalg.norm(points - points[neighbor_idx], axis=1)
                        neighbor_neighbors = np.where(neighbor_distances < eps)[0]
                        
                        if len(neighbor_neighbors) >= min_samples:
                            seed_set.extend(neighbor_neighbors.tolist())
                    
                    if labels[neighbor_idx] == -1:
                        labels[neighbor_idx] = cluster_id
                    
                    j += 1
                
                cluster_id += 1
        
        # 组织成簇列表
        clusters = []
        for cid in range(cluster_id):
            cluster_mask = (labels == cid)
            if np.sum(cluster_mask) > 0:
                clusters.append(points[cluster_mask])
        
        return clusters


class DepthPerceptionSystem:
    """
    深度感知系统（主接口）
    整合深度相机和占用检测
    """
    
    def __init__(self,
                 robot_id: int,
                 sensor_link_id: int,
                 image_width: int = 128,
                 image_height: int = 128,
                 tray_position: Optional[np.ndarray] = None,
                 tray_size: Optional[np.ndarray] = None):
        """
        初始化深度感知系统
        
        Args:
            robot_id: 机器人ID
            sensor_link_id: 传感器link ID
            image_width: 图像宽度
            image_height: 图像高度
            tray_position: 托盘位置 [x, y, z]（可选）
            tray_size: 托盘尺寸 [length, width, height]（可选）
        """
        self.robot_id = robot_id
        self.sensor_link_id = sensor_link_id
        
        # 托盘信息（用于过滤托盘底部）
        self.tray_position = tray_position if tray_position is not None else np.array([0.5, 0.5, 0.0])
        self.tray_size = tray_size if tray_size is not None else np.array([0.4, 0.3, 0.05])  # 默认托盘尺寸
        
        # 初始化深度相机
        self.depth_camera = DepthCamera(
            robot_id=robot_id,
            sensor_link_id=sensor_link_id,
            image_width=image_width,
            image_height=image_height
        )
        
        # 初始化占用检测器
        self.occupancy_detector = OccupancyDetector(self.depth_camera)
        
        print(f"[深度感知系统] 初始化完成")
        print(f"  托盘位置: {self.tray_position}")
        print(f"  托盘尺寸: {self.tray_size}")
    
    def perceive_with_depth(self,
                           ignore_ids: Optional[set] = None,
                           debug: bool = False) -> Dict:
        """
        使用深度感知获取障碍物信息
        
        Args:
            ignore_ids: 要忽略的物体ID
            debug: 是否显示调试信息
            
        Returns:
            感知结果字典: {
                'current_obstacles': [(obs_id, position, velocity), ...],
                'predicted_obstacles': [(obs_id, predicted_position, confidence), ...],
                'depth_image': depth_array,
                'point_cloud': point_cloud_array
            }
        """
        if ignore_ids is None:
            ignore_ids = set()
        
        # 1. 捕获深度图像
        depth_buffer, rgb_image = self.depth_camera.capture_depth_image()
        
        # 2. 转换为距离
        depth_distance = self.depth_camera.depth_buffer_to_distance(depth_buffer)
        
        # 3. 获取相机位姿
        camera_pos, camera_orn = self.depth_camera.get_camera_pose_in_world()
        
        # 4. 转换为点云
        point_cloud = self.occupancy_detector.depth_to_point_cloud(
            depth_distance, camera_pos, camera_orn
        )
        
        # 🔥 5. 获取夹爪位置（用于过滤被抓取的物品）
        # 如果ignore_ids不为空，说明有物品被抓取，需要获取夹爪位置
        gripper_position = None
        if len(ignore_ids) > 0:
            try:
                # 获取末端执行器（夹爪）的位置
                ee_state = p.getLinkState(
                    self.robot_id, 
                    self.sensor_link_id,
                    computeForwardKinematics=True
                )
                gripper_position = np.array(ee_state[0])
            except Exception as e:
                if debug:
                    print(f"  [深度感知] 无法获取夹爪位置: {e}")
        
        # 6. 检测障碍物（传递托盘信息和夹爪位置以过滤）
        current_obstacles = self.occupancy_detector.detect_obstacles_from_point_cloud(
            point_cloud, ignore_ids,
            tray_position=self.tray_position,
            tray_size=self.tray_size,
            gripper_position=gripper_position
        )
        
        # 7. 预测未来位置（暂时返回空）
        predicted_obstacles = []
        
        if debug:
            print(f"  [深度感知] 捕获深度图像: {depth_buffer.shape}")
            print(f"  [深度感知] 生成点云: {point_cloud.shape[0]} 个点")
            if gripper_position is not None:
                print(f"  [深度感知] 🔥 已过滤夹爪附近15cm范围内的点云（被抓取物品）")
            print(f"  [深度感知] 检测到 {len(current_obstacles)} 个障碍物")
        
        return {
            'current_obstacles': current_obstacles,
            'predicted_obstacles': predicted_obstacles,
            'depth_image': depth_distance,
            'point_cloud': point_cloud
        }

