import numpy as np
import pybullet as p
import math
import time
import avoidance  # 确保导入了 avoidance

class CameraSystem:
    def __init__(self, robot_id, tray_id):
        self.robot_id = robot_id
        self.tray_id = tray_id  # 记录托盘ID，用于过滤
        self.plane_id = 0  # 地面ID通常是0（第一个加载的物体）
        self.ignored_ids = set()  # 动态忽略的物体ID集合（如抓取目标）
        
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

    def scan_obstacle_volume(self):
        img_arr = p.getCameraImage(
            self.width, self.height, 
            self.view_matrix, self.proj_matrix, 
            renderer=p.ER_TINY_RENDERER
        )
        
        opengl_depth_buffer = np.reshape(img_arr[3], (self.height, self.width))
        seg_buffer = np.reshape(img_arr[4], (self.height, self.width))
        
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
        
        return {
            "min": min_bound,
            "max": max_bound,
            "center": center
        }

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
        vision_interval = 3 
        last_status = ""

        while True:
            current_eef_pos = self.get_current_eef_pos()
            
            # 1. 视觉感知（每次都更新，如果没检测到障碍物则清空）
            if step_counter % vision_interval == 0:
                scan_result = self.vision_system.scan_obstacle_volume()
                obs_aabb = scan_result  # 直接赋值，如果没检测到则为 None
            step_counter += 1

            # 2. 威胁点计算
            if obs_aabb is not None:
                obs_min = obs_aabb["min"]
                obs_max = obs_aabb["max"]
                
                effective_obs_pos = []
                for i in range(3):
                    val = max(obs_min[i], min(current_eef_pos[i], obs_max[i]))
                    effective_obs_pos.append(val)
                
                dist_to_clamped = math.sqrt(sum([(a-b)**2 for a,b in zip(current_eef_pos, effective_obs_pos)]))
                if dist_to_clamped < 0.05:
                     effective_obs_pos = obs_aabb["center"].tolist()
            else:
                effective_obs_pos = [10.0, 10.0, 10.0]

            # 3. 避障向量计算
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
            
            # 调试输出（状态变化时才打印）
            if debug and status != last_status:
                dist_to_target = math.sqrt(sum([(a-b)**2 for a,b in zip(current_eef_pos, target_pos)]))
                print(f"  [调试] 状态: {status}, 距目标: {dist_to_target:.3f}, 距障碍: {eef_dist:.3f}")
                last_status = status
            
            # 4. 执行运动
            dist_to_target = math.sqrt(sum([(a-b)**2 for a,b in zip(current_eef_pos, target_pos)]))
            
            # 【改进】更灵活的到达判定
            if dist_to_target < 0.03:
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
        self.move_arm_smart(drop_pos, timeout=20.0, debug=True)

        print(">>> 6. 放下")
        self.move_arm_exact(drop_low, steps=300)
        
        for _ in range(50): self.step_simulation_with_callback()
        self.move_gripper(open_state=True)
        print("任务完成")