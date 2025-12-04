import numpy as np
import pybullet as p
import math
import time
import avoidance  # 确保导入了 avoidance
import predictor  # 障碍物运动预测模块

class CameraSystem:
    def __init__(self, robot_id, tray_id):
        self.robot_id = robot_id
        self.tray_id = tray_id  # 记录托盘ID，用于过滤
        self.plane_id = 0  # 地面ID通常是0（第一个加载的物体）
        self.ignored_ids = set()  # 动态忽略的物体ID集合（如抓取目标）
        
        # ==========================================
        # 主摄像头（俯视角度）
        # ==========================================
        self.camera_pos = np.array([0.8, 0, 0.6])
        self.target_pos = np.array([0.4, 0, 0.0])
        self.up_vector = np.array([0, 0, 1])
        
        self.view_matrix = p.computeViewMatrix(
            cameraEyePosition=self.camera_pos,
            cameraTargetPosition=self.target_pos,
            cameraUpVector=self.up_vector
        )
        
        self.fov = 60
        self.width = 128
        self.height = 128
        self.proj_matrix = p.computeProjectionMatrixFOV(
            fov=self.fov, aspect=1.0, nearVal=0.1, farVal=2.0
        )
        
        view_mat = np.array(self.view_matrix).reshape(4, 4).T
        proj_mat = np.array(self.proj_matrix).reshape(4, 4).T
        self.inv_pv_mat = np.linalg.inv(np.dot(proj_mat, view_mat))
        
        # ==========================================
        # 侧视摄像头（专门用于获取障碍物高度）
        # 从Y轴负方向观察，可以清晰看到障碍物的高度轮廓
        # ==========================================
        self.side_camera_pos = np.array([0.5, -0.8, 0.4])  # 侧面位置
        self.side_target_pos = np.array([0.5, 0.0, 0.3])   # 看向工作区中心
        self.side_up_vector = np.array([0, 0, 1])
        
        self.side_view_matrix = p.computeViewMatrix(
            cameraEyePosition=self.side_camera_pos,
            cameraTargetPosition=self.side_target_pos,
            cameraUpVector=self.side_up_vector
        )
        
        # 侧视摄像头使用更高分辨率以获取精确高度
        self.side_width = 160
        self.side_height = 120
        self.side_proj_matrix = p.computeProjectionMatrixFOV(
            fov=50, aspect=self.side_width/self.side_height, nearVal=0.1, farVal=2.0
        )
        
        # 预计算侧视摄像头的逆投影矩阵
        side_view_mat = np.array(self.side_view_matrix).reshape(4, 4).T
        side_proj_mat = np.array(self.side_proj_matrix).reshape(4, 4).T
        self.side_inv_pv_mat = np.linalg.inv(np.dot(side_proj_mat, side_view_mat))
        
        # 障碍物高度信息缓存
        self.obstacle_height_info = {
            "max_height": 0.0,          # 障碍物最高点Z坐标
            "min_height": 0.0,          # 障碍物最低点Z坐标
            "clearance_height": 0.0,    # 建议的安全越过高度
            "confidence": 0.0,          # 测量置信度
            "last_update": 0            # 上次更新时间
        }
        
        # 初始化障碍物运动预测器
        self.obstacle_predictor = predictor.ObstaclePredictor(
            history_size=20,        # 保存20帧历史
            prediction_horizon=0.8  # 预测0.8秒后的位置
        )

    def scan_obstacle_volume(self):
        """主摄像头扫描障碍物体积"""
        img_arr = p.getCameraImage(
            self.width, self.height, 
            self.view_matrix, self.proj_matrix, 
            renderer=p.ER_TINY_RENDERER
        )#调用 p.getCameraImage 拍了一张照片
        
        opengl_depth_buffer = np.reshape(img_arr[3], (self.height, self.width))#每个像素代表离相机有多远
        seg_buffer = np.reshape(img_arr[4], (self.height, self.width))#分割掩码
        
        # === 过滤掉地面、机器人、托盘和动态忽略物体 ===
        valid_mask = (seg_buffer != self.robot_id) & \
                     (seg_buffer != self.tray_id) & \
                     (seg_buffer != self.plane_id) & \
                     (opengl_depth_buffer < 0.95)
        
        # 过滤掉动态忽略的物体（如当前要抓取的目标）
        for ignored_id in self.ignored_ids:
            valid_mask = valid_mask & (seg_buffer != ignored_id)
        
        if not np.any(valid_mask):
            return None 
        
        step = 4 
        rows, cols = np.where(valid_mask)
        rows = rows[::step]
        cols = cols[::step]
        depths = opengl_depth_buffer[rows, cols]
        
        x_ndc = (2 * cols / self.width) - 1
        y_ndc = 1 - (2 * rows / self.height)
        z_ndc = 2 * depths - 1
        
        pixel_coords = np.vstack([x_ndc, y_ndc, z_ndc, np.ones_like(x_ndc)])
        world_coords_homo = np.dot(self.inv_pv_mat, pixel_coords)
        world_coords = world_coords_homo[:3] / world_coords_homo[3]
        points_3d = world_coords.T
        
        # 过滤掉接近地面的点（增加阈值以避免杂点）
        points_3d = points_3d[points_3d[:, 2] > 0.08]
        
        if len(points_3d) < 5:
            return None

        min_bound = np.min(points_3d, axis=0)
        max_bound = np.max(points_3d, axis=0)
        center = np.mean(points_3d, axis=0)
        
        self.draw_debug_box(min_bound, max_bound)
        
        # 更新预测器：将障碍物中心位置喂给预测器
        self.obstacle_predictor.update(center)
        
        return {
            "min": min_bound,
            "max": max_bound,
            "center": center
        }
    
    def scan_obstacle_height_from_side(self):
        """
        使用侧视摄像头扫描障碍物高度
        返回障碍物的精确高度信息，用于自动计算越过高度
        """
        img_arr = p.getCameraImage(
            self.side_width, self.side_height,
            self.side_view_matrix, self.side_proj_matrix,
            renderer=p.ER_TINY_RENDERER
        )
        
        opengl_depth_buffer = np.reshape(img_arr[3], (self.side_height, self.side_width))
        seg_buffer = np.reshape(img_arr[4], (self.side_height, self.side_width))
        
        # 过滤：只保留障碍物（排除机器人、托盘、地面、忽略物体）
        valid_mask = (seg_buffer != self.robot_id) & \
                     (seg_buffer != self.tray_id) & \
                     (seg_buffer != self.plane_id) & \
                     (opengl_depth_buffer < 0.95)
        
        for ignored_id in self.ignored_ids:
            valid_mask = valid_mask & (seg_buffer != ignored_id)
        
        if not np.any(valid_mask):
            # 没有检测到障碍物，返回默认安全高度
            self.obstacle_height_info = {
                "max_height": 0.0,
                "min_height": 0.0,
                "clearance_height": 0.15,  # 默认安全高度
                "confidence": 0.0,
                "last_update": time.time()
            }
            return self.obstacle_height_info
        
        # 获取所有有效像素的3D坐标
        step = 2  # 更密集采样以获取精确高度
        rows, cols = np.where(valid_mask)
        rows = rows[::step]
        cols = cols[::step]
        depths = opengl_depth_buffer[rows, cols]
        
        # 转换到NDC坐标
        x_ndc = (2 * cols / self.side_width) - 1
        y_ndc = 1 - (2 * rows / self.side_height)
        z_ndc = 2 * depths - 1
        
        # 反投影到世界坐标
        pixel_coords = np.vstack([x_ndc, y_ndc, z_ndc, np.ones_like(x_ndc)])
        world_coords_homo = np.dot(self.side_inv_pv_mat, pixel_coords)
        world_coords = world_coords_homo[:3] / world_coords_homo[3]
        points_3d = world_coords.T
        
        # 过滤地面附近的点
        points_3d = points_3d[points_3d[:, 2] > 0.05]
        
        if len(points_3d) < 3:
            self.obstacle_height_info["confidence"] = 0.0
            return self.obstacle_height_info
        
        # 计算高度统计信息
        z_coords = points_3d[:, 2]
        max_height = np.max(z_coords)
        min_height = np.min(z_coords)
        
        # 使用百分位数来排除噪点
        height_95 = np.percentile(z_coords, 95)  # 取95%分位数作为稳健最高点
        
        # 计算安全越过高度 = 最高点 + 安全裕度
        safety_margin = 0.08  # 8cm安全裕度
        clearance_height = height_95 + safety_margin
        
        # 计算置信度（基于检测到的点数）
        confidence = min(len(points_3d) / 100.0, 1.0)
        
        self.obstacle_height_info = {
            "max_height": float(max_height),
            "min_height": float(min_height),
            "height_95": float(height_95),
            "clearance_height": float(clearance_height),
            "confidence": float(confidence),
            "point_count": len(points_3d),
            "last_update": time.time()
        }
        
        return self.obstacle_height_info
    
    def get_obstacle_clearance_height(self):
        """
        获取机械臂需要升高到的安全高度
        这是给避障系统调用的主要接口
        """
        return self.obstacle_height_info.get("clearance_height", 0.15)
    
    def get_obstacle_height_info(self):
        """获取完整的障碍物高度信息"""
        return self.obstacle_height_info
    
    def get_predicted_obstacle_pos(self, robot_pos):
        """
        获取预测的障碍物位置（用于提前避障）
        
        Args:
            robot_pos: 机器人当前位置
            
        Returns:
            predicted_pos: 预测的障碍物位置
            prediction_info: 预测详情
        """
        return self.obstacle_predictor.get_avoidance_position(robot_pos)
    
    def get_motion_trend(self):
        """获取障碍物运动趋势"""
        return self.obstacle_predictor.get_motion_trend()
    
    def should_preemptive_avoid(self, robot_pos, robot_target):
        """判断是否需要提前避障"""
        return self.obstacle_predictor.should_preemptive_avoid(robot_pos, robot_target)

    def draw_debug_box(self, min_pos, max_pos):
        p.removeAllUserDebugItems()
        corners = [
            [min_pos[0], min_pos[1], min_pos[2]],
            [max_pos[0], min_pos[1], min_pos[2]],
            [min_pos[0], max_pos[1], min_pos[2]],
            [max_pos[0], max_pos[1], min_pos[2]],
            [min_pos[0], min_pos[1], max_pos[2]],
            [max_pos[0], min_pos[1], max_pos[2]],
            [min_pos[0], max_pos[1], max_pos[2]],
            [max_pos[0], max_pos[1], max_pos[2]]
        ]
        lines = [
            (0,1), (1,3), (3,2), (2,0),
            (4,5), (5,7), (7,6), (6,4),
            (0,4), (1,5), (2,6), (3,7)
        ]
        for start, end in lines:
            p.addUserDebugLine(corners[start], corners[end], [0, 1, 0], lineWidth=2, lifeTime=0.2)
        
        # 【新增】绘制安全越过高度线（蓝色水平线）
        clearance_z = self.obstacle_height_info.get("clearance_height", 0)
        if clearance_z > 0.1:  # 只有当检测到有效高度时才绘制
            # 绘制一个水平的蓝色平面表示安全高度
            clearance_corners = [
                [min_pos[0] - 0.1, min_pos[1] - 0.1, clearance_z],
                [max_pos[0] + 0.1, min_pos[1] - 0.1, clearance_z],
                [min_pos[0] - 0.1, max_pos[1] + 0.1, clearance_z],
                [max_pos[0] + 0.1, max_pos[1] + 0.1, clearance_z],
            ]
            # 绘制安全高度的边框
            p.addUserDebugLine(clearance_corners[0], clearance_corners[1], [0, 0.5, 1], lineWidth=3, lifeTime=0.2)
            p.addUserDebugLine(clearance_corners[1], clearance_corners[3], [0, 0.5, 1], lineWidth=3, lifeTime=0.2)
            p.addUserDebugLine(clearance_corners[3], clearance_corners[2], [0, 0.5, 1], lineWidth=3, lifeTime=0.2)
            p.addUserDebugLine(clearance_corners[2], clearance_corners[0], [0, 0.5, 1], lineWidth=3, lifeTime=0.2)
            
            # 在安全高度处显示文字标注
            text_pos = [(min_pos[0] + max_pos[0])/2, (min_pos[1] + max_pos[1])/2, clearance_z + 0.02]
            p.addUserDebugText(f"Safe: {clearance_z:.2f}m", text_pos, [0, 0.5, 1], textSize=1.2, lifeTime=0.2)


class RobotController:
    # === 修改点：初始化接收 tray_id ===
    def __init__(self, robot_id, tray_id):
        self.robot_id = robot_id
        self.eef_id = 11 
        self.critical_joints = [11, 6, 4] 
        self.finger_indices = [9, 10] 
        self.gripper_open_pos = 0.05 
        self.gripper_closed_pos = 0.03 
        
        # 将 tray_id 传给视觉系统
        self.vision_system = CameraSystem(robot_id, tray_id)
        self.avoider = avoidance.VisualAvoidanceSystem(safe_distance=0.40, stop_distance=0.15)
        
        self.sim_step_callback = None 
        
        p.setPhysicsEngineParameter(numSolverIterations=200, contactBreakingThreshold=0.001)
        self.ll = [-7]*7
        self.ul = [7]*7
        self.jr = [7]*7
        self.rp = [0, -math.pi/4, 0, -math.pi/2, 0, math.pi/3, 0]
        
        for finger in self.finger_indices:
            p.changeDynamics(self.robot_id, finger, lateralFriction=10.0, frictionAnchor=True)

    def set_ignored_object(self, object_id):
        """设置要忽略的物体ID（如当前要抓取的目标）"""
        self.vision_system.ignored_ids.add(object_id)
    
    def clear_ignored_object(self, object_id):
        """从忽略列表中移除物体ID"""
        self.vision_system.ignored_ids.discard(object_id)
    
    def get_critical_body_points(self):
        points = []
        for joint_index in self.critical_joints:
            state = p.getLinkState(self.robot_id, joint_index)
            pos = list(state[4]) 
            points.append(pos)
        return points

    def get_current_eef_pos(self):
        state = p.getLinkState(self.robot_id, self.eef_id)
        return list(state[4]) 

    def step_simulation_with_callback(self):
        p.stepSimulation()
        if self.sim_step_callback:
            self.sim_step_callback() 
        time.sleep(1./240.)

    def move_gripper(self, open_state=True):
        target_pos = self.gripper_open_pos if open_state else self.gripper_closed_pos
        for i in self.finger_indices:
            p.setJointMotorControl2(self.robot_id, i, p.POSITION_CONTROL, targetPosition=target_pos, force=500)
        for _ in range(20): 
            self.step_simulation_with_callback() 

    def move_arm_smart(self, target_pos, target_orn=None, timeout=10.0, debug=False):
        if target_orn is None:
            target_orn = p.getQuaternionFromEuler([math.pi, 0, math.pi/2])
        
        start_time = time.time()
        obs_aabb = None 
        step_counter = 0 
        vision_interval = 12 
        side_vision_interval = 24  # 侧视摄像头扫描间隔（比主摄像头略低频）
        last_status = ""
        last_prediction_info = None

        while True:
            current_eef_pos = self.get_current_eef_pos()
            
            # 1. 视觉感知（每次都更新，如果没检测到障碍物则清空）
            if step_counter % vision_interval == 0:
                scan_result = self.vision_system.scan_obstacle_volume()
                obs_aabb = scan_result  # 直接赋值，如果没检测到则为 None
            
            # 【新增】侧视摄像头扫描获取障碍物高度
            if step_counter % side_vision_interval == 0:
                height_info = self.vision_system.scan_obstacle_height_from_side()
                # 将高度信息传递给避障系统
                self.avoider.set_obstacle_height_info(height_info)
                
                if debug and height_info.get("confidence", 0) > 0.3:
                    print(f"  [侧视] 障碍物高度: {height_info.get('max_height', 0):.3f}m, "
                          f"安全越过高度: {height_info.get('clearance_height', 0):.3f}m, "
                          f"置信度: {height_info.get('confidence', 0):.2f}")
            
            step_counter += 1

            # 2. 【预测性避障核心】使用预测位置而非当前位置
            if obs_aabb is not None:
                obs_min = obs_aabb["min"]
                obs_max = obs_aabb["max"]
                
                # 获取预测的障碍物位置
                predicted_pos, prediction_info = self.vision_system.get_predicted_obstacle_pos(current_eef_pos)
                
                if predicted_pos is not None:
                    # 使用预测位置作为避障依据
                    effective_obs_pos = predicted_pos
                    
                    # 如果障碍物正在接近，增加安全裕度
                    if prediction_info.get("direction") == "approaching":
                        # 将预测位置向机器人方向偏移一点，更保守
                        approach_bias = 0.05  # 5cm 额外裕度
                        direction_to_robot = np.array(current_eef_pos) - np.array(predicted_pos)
                        dir_norm = np.linalg.norm(direction_to_robot)
                        if dir_norm > 0.01:
                            direction_to_robot = direction_to_robot / dir_norm
                            effective_obs_pos = [
                                predicted_pos[0] + direction_to_robot[0] * approach_bias,
                                predicted_pos[1] + direction_to_robot[1] * approach_bias,
                                predicted_pos[2] + direction_to_robot[2] * approach_bias
                            ]
                else:
                    # 预测器未初始化，使用传统方法
                    effective_obs_pos = []
                    for i in range(3):
                        val = max(obs_min[i], min(current_eef_pos[i], obs_max[i]))
                        effective_obs_pos.append(val)
                    
                    dist_to_clamped = math.sqrt(sum([(a-b)**2 for a,b in zip(current_eef_pos, effective_obs_pos)]))
                    if dist_to_clamped < 0.05:
                        effective_obs_pos = obs_aabb["center"].tolist()
                    prediction_info = {"status": "fallback"}
            else:
                effective_obs_pos = [10.0, 10.0, 10.0]
                prediction_info = {"status": "no_obstacle"}

            # 3. 检查是否需要提前避障
            should_avoid, threat_level, recommendation = self.vision_system.should_preemptive_avoid(
                current_eef_pos, target_pos
            )
            
            # 根据预判调整避障策略
            if should_avoid and recommendation == "emergency_avoid":
                # 紧急情况：强制增大安全距离
                self.avoider.d_th2 = 0.55  # 临时增大警戒距离
            elif should_avoid and recommendation == "preemptive_avoid":
                self.avoider.d_th2 = 0.48
            else:
                self.avoider.d_th2 = 0.40  # 恢复默认

            # 4. 将障碍物运动信息传递给避障系统
            motion_trend = self.vision_system.get_motion_trend()
            self.avoider.set_obstacle_motion(
                velocity=motion_trend.get("velocity"),
                is_moving=motion_trend.get("is_moving", False),
                direction=motion_trend.get("direction")
            )
            
            # 5. 避障向量计算
            body_points = self.get_critical_body_points()
            min_dist_to_obs = 999.0
            
            for point in body_points:
                dist = math.sqrt(sum([(a-b)**2 for a,b in zip(point, effective_obs_pos)]))
                if dist < min_dist_to_obs:
                    min_dist_to_obs = dist
            
            # 末端执行器到障碍物的距离也要考虑
            eef_dist = math.sqrt(sum([(a-b)**2 for a,b in zip(current_eef_pos, effective_obs_pos)]))
            min_dist_to_obs = min(min_dist_to_obs, eef_dist)
            
            # 计算下一步位置
            virtual_next_pos, status = self.avoider.compute_modified_step(
                current_eef_pos, target_pos, effective_obs_pos
            )
            
            # 调试输出（状态变化时或预测信息变化时打印）
            if debug:
                should_print = status != last_status
                # 检查预测状态变化
                if prediction_info != last_prediction_info:
                    if prediction_info.get("direction") != (last_prediction_info or {}).get("direction"):
                        should_print = True
                
                if should_print:
                    dist_to_target = math.sqrt(sum([(a-b)**2 for a,b in zip(current_eef_pos, target_pos)]))
                    pred_status = prediction_info.get("status", "unknown")
                    direction = prediction_info.get("direction", "-")
                    speed = prediction_info.get("speed", 0)
                    print(f"  [调试] 状态: {status}, 距目标: {dist_to_target:.3f}, 距障碍: {eef_dist:.3f}")
                    print(f"         预测: {pred_status}, 方向: {direction}, 速度: {speed:.4f}, 建议: {recommendation}")
                    last_status = status
                    last_prediction_info = prediction_info.copy() if isinstance(prediction_info, dict) else prediction_info
            
            # 4. 执行运动
            dist_to_target = math.sqrt(sum([(a-b)**2 for a,b in zip(current_eef_pos, target_pos)]))
            
            # 【改进】放宽到达判定阈值，避免因精度问题卡住
            if dist_to_target < 0.08:
                if debug:
                    print(f"  [调试] 到达目标！距离: {dist_to_target:.3f}")
                break
            
            joint_poses = p.calculateInverseKinematics(
                self.robot_id, self.eef_id, virtual_next_pos, target_orn,
                lowerLimits=self.ll, upperLimits=self.ul, jointRanges=self.jr, restPoses=self.rp
            )
            
            for i in range(7):
                p.setJointMotorControl2(self.robot_id, i, p.POSITION_CONTROL, 
                                        targetPosition=joint_poses[i], 
                                        maxVelocity=2.0, force=500)
            
            self.step_simulation_with_callback()
            
            if time.time() - start_time > timeout:
                print("移动超时，强制中断")
                break

    def move_arm_exact(self, target_pos, target_orn=None, steps=80):
        if target_orn is None:
            target_orn = p.getQuaternionFromEuler([math.pi, 0, math.pi/2])
        for _ in range(steps):
            joint_poses = p.calculateInverseKinematics(
                self.robot_id, self.eef_id, target_pos, target_orn,
                lowerLimits=self.ll, upperLimits=self.ul, jointRanges=self.jr, restPoses=self.rp
            )
            for i in range(7):
                p.setJointMotorControl2(self.robot_id, i, p.POSITION_CONTROL, targetPosition=joint_poses[i], maxVelocity=2.0, force=500)
            
            self.step_simulation_with_callback() 

    def execute_pick_and_place(self, cube_id, tray_id):
        p.changeDynamics(cube_id, -1, lateralFriction=10.0, mass=0.05)
        cube_pos, _ = p.getBasePositionAndOrientation(cube_id)
        tray_pos, _ = p.getBasePositionAndOrientation(tray_id)
        
        hover_z = 0.3 
        pre_grasp = [cube_pos[0], cube_pos[1], cube_pos[2] + hover_z]
        grasp_pos = [cube_pos[0], cube_pos[1], 0.045] 
        drop_pos = [tray_pos[0], tray_pos[1], tray_pos[2] + hover_z]
        drop_low = [tray_pos[0], tray_pos[1], tray_pos[2] + 0.15] 
        
        # 【关键修复】将目标方块加入忽略列表，避免被识别为障碍物
        self.set_ignored_object(cube_id)
        
        print(">>> 1. 智能移动：接近方块上方")
        self.move_gripper(open_state=True)
        self.move_arm_smart(pre_grasp, timeout=15.0) 

        print(">>> 2. 精准下降")
        self.move_arm_exact(grasp_pos, steps=200)
        
        for _ in range(40): self.step_simulation_with_callback()

        print(">>> 3. 闭合夹爪")
        self.move_gripper(open_state=False)
        for _ in range(100): self.step_simulation_with_callback()

        print(">>> 4. 抬起 (视觉避障开启)")
        self.move_arm_smart(pre_grasp, timeout=10.0, debug=True)

        print(">>> 5. 搬运至终点 (视觉避障开启)")
        self.move_arm_smart(drop_pos, timeout=30.0, debug=True)

        print(">>> 6. 放下")
        self.move_arm_exact(drop_low, steps=300)
        
        for _ in range(50): self.step_simulation_with_callback()
        self.move_gripper(open_state=True)
        print("任务完成")