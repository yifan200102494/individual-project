import pybullet as p
import time
import math
import numpy as np
import avoidance

# ==========================================
# 视觉系统 (包含矩阵逆变换修复)
# ==========================================
class CameraSystem:
    def __init__(self, robot_id):
        self.robot_id = robot_id
        
        # 相机位置
        self.camera_pos = np.array([0.8, 0, 0.6])
        self.target_pos = np.array([0.4, 0, 0.0])
        self.up_vector = np.array([0, 0, 1])
        
        # 计算视图矩阵和投影矩阵
        self.view_matrix = p.computeViewMatrix(
            cameraEyePosition=self.camera_pos,
            cameraTargetPosition=self.target_pos,
            cameraUpVector=self.up_vector
        )
        
        self.fov = 60
        self.proj_matrix = p.computeProjectionMatrixFOV(
            fov=self.fov, aspect=1.0, nearVal=0.1, farVal=2.0
        )
        
        self.width = 64
        self.height = 64

    def get_closest_obstacle(self):
        img_arr = p.getCameraImage(
            self.width, self.height, 
            self.view_matrix, self.proj_matrix, 
            renderer=p.ER_TINY_RENDERER
        )
        
        opengl_depth_buffer = np.reshape(img_arr[3], (self.height, self.width))
        seg_buffer = np.reshape(img_arr[4], (self.height, self.width))
        
        # 过滤掉无穷远(1.0)和自身(robot_id)
        valid_mask = (seg_buffer != self.robot_id) & (opengl_depth_buffer < 0.99)
        
        if not np.any(valid_mask):
            return None 
        
        masked_depth = np.where(valid_mask, opengl_depth_buffer, 1.0)
        min_idx = np.argmin(masked_depth)
        
        min_row = min_idx // self.width
        min_col = min_idx % self.width
        min_depth_val = masked_depth[min_row, min_col]
        
        if min_depth_val >= 0.99:
            return None

        # --- 矩阵逆变换计算 3D 坐标 ---
        x_ndc = (2 * min_col / self.width) - 1
        y_ndc = 1 - (2 * min_row / self.height) 
        z_ndc = 2 * min_depth_val - 1
        
        clip_pos = np.array([x_ndc, y_ndc, z_ndc, 1.0])
        
        view_mat = np.array(self.view_matrix).reshape(4, 4).T 
        proj_mat = np.array(self.proj_matrix).reshape(4, 4).T
        pv_mat = np.dot(proj_mat, view_mat)
        inv_pv_mat = np.linalg.inv(pv_mat)
        
        world_pos_homo = np.dot(inv_pv_mat, clip_pos)
        world_pos = world_pos_homo[:3] / world_pos_homo[3] 
        
        final_pos = world_pos.tolist()
        final_pos[2] = max(final_pos[2], 0.3) 
        
        # 画出调试线
        p.addUserDebugLine(self.camera_pos, final_pos, [0, 1, 0], lifeTime=0.5, lineWidth=2)
        
        return final_pos

# ==========================================
# 机器人控制器 (包含全身检测和 move_gripper)
# ==========================================
class RobotController:
    def __init__(self, robot_id):
        self.robot_id = robot_id
        self.eef_id = 11 
        # 全身关键点：末端(11), 肘部(6), 大臂(4)
        self.critical_joints = [11, 6, 4] 
        
        self.finger_indices = [9, 10] 
        self.gripper_open_pos = 0.05 
        self.gripper_closed_pos = 0.03 
        
        self.vision_system = CameraSystem(robot_id)
        # 避障参数
        self.avoider = avoidance.VisualAvoidanceSystem(safe_distance=0.40, stop_distance=0.15)
        
        self.sim_step_callback = None 
        
        p.setPhysicsEngineParameter(numSolverIterations=200, contactBreakingThreshold=0.001)
        self.ll = [-7]*7
        self.ul = [7]*7
        self.jr = [7]*7
        self.rp = [0, -math.pi/4, 0, -math.pi/2, 0, math.pi/3, 0]
        
        for finger in self.finger_indices:
            p.changeDynamics(self.robot_id, finger, lateralFriction=10.0, frictionAnchor=True)

    def get_critical_body_points(self):
        """获取全身关键部位坐标"""
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
        """控制夹爪开合"""
        target_pos = self.gripper_open_pos if open_state else self.gripper_closed_pos
        for i in self.finger_indices:
            p.setJointMotorControl2(self.robot_id, i, p.POSITION_CONTROL, targetPosition=target_pos, force=500)
        for _ in range(20): 
            self.step_simulation_with_callback() 

    def move_arm_smart(self, target_pos, target_orn=None, timeout=10.0):
        if target_orn is None:
            target_orn = p.getQuaternionFromEuler([math.pi, 0, math.pi/2])
        
        start_time = time.time()
        
        current_obs_pos = None      
        last_obs_pos = None         
        obs_velocity = np.array([0.0, 0.0, 0.0])
        
        step_counter = 0 
        vision_interval = 2 

        while True:
            current_eef_pos = self.get_current_eef_pos()
            body_points = self.get_critical_body_points() # 获取全身点
            
            # === 视觉感知 ===
            if step_counter % vision_interval == 0:
                detected_pos = self.vision_system.get_closest_obstacle()
                
                if detected_pos is not None:
                    current_obs_pos = np.array(detected_pos)
                    if last_obs_pos is not None:
                        move_vec = current_obs_pos - last_obs_pos
                        obs_velocity = 0.7 * obs_velocity + 0.3 * move_vec
                    last_obs_pos = current_obs_pos
                else:
                    obs_velocity = obs_velocity * 0.9 
            
            step_counter += 1

            # === 预测与全身避障 ===
            if current_obs_pos is not None:
                prediction_factor = 10.0 
                predicted_obs_pos = current_obs_pos + obs_velocity * prediction_factor
                effective_obs_pos = predicted_obs_pos.tolist()
            else:
                effective_obs_pos = [10.0, 10.0, 10.0]

            # 找到离障碍物最近的身体部位
            closest_body_point = current_eef_pos 
            min_dist_to_obs = 999.0
            
            for point in body_points:
                dist = math.sqrt(sum([(a-b)**2 for a,b in zip(point, effective_obs_pos)]))
                if dist < min_dist_to_obs:
                    min_dist_to_obs = dist
                    closest_body_point = point
            
            # 计算避障
            # 我们用"最近的那个部位"去计算它想往哪里躲
            virtual_next_pos, status = self.avoider.compute_modified_step(closest_body_point, target_pos, effective_obs_pos)
            
            # 计算偏移量 (避障要往哪里偏)
            avoidance_delta = np.array(virtual_next_pos) - np.array(closest_body_point)
            
            current_eef_arr = np.array(current_eef_pos)
            
            # 如果处于避障状态，把偏移量加到末端上
            if status in ["AVOIDING_ACTIVE", "EMERGENCY_RETREAT"] and np.linalg.norm(avoidance_delta) > 0.001:
                # 叠加避障偏移
                next_step_pos = (current_eef_arr + avoidance_delta).tolist()
                # 保持一点点向目标的牵引力(20%)，防止完全跑丢
                target_vec = np.array(target_pos) - current_eef_arr
                if np.linalg.norm(target_vec) > 0:
                    next_step_pos = np.array(next_step_pos) + 0.02 * (target_vec / np.linalg.norm(target_vec))
            else:
                # 正常导航
                next_step_pos, _ = self.avoider.compute_modified_step(current_eef_pos, target_pos, effective_obs_pos)

            if status == "ARRIVED": break
            
            dist_to_target = math.sqrt(sum([(a-b)**2 for a,b in zip(current_eef_pos, target_pos)]))
            if dist_to_target < 0.05: break
            
            if time.time() - start_time > timeout:
                print("移动超时，强制结束")
                break
                
            joint_poses = p.calculateInverseKinematics(
                self.robot_id, self.eef_id, next_step_pos, target_orn,
                lowerLimits=self.ll, upperLimits=self.ul, jointRanges=self.jr, restPoses=self.rp
            )
            
            for i in range(7):
                p.setJointMotorControl2(self.robot_id, i, p.POSITION_CONTROL, targetPosition=joint_poses[i], maxVelocity=3.0)
            
            self.step_simulation_with_callback() 

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
        self.move_arm_smart(pre_grasp, timeout=10.0)

        print(">>> 5. 搬运至终点 (视觉避障开启)")
        self.move_arm_smart(drop_pos, timeout=20.0)

        print(">>> 6. 放下")
        self.move_arm_exact(drop_low, steps=300)
        
        for _ in range(50): self.step_simulation_with_callback()
        self.move_gripper(open_state=True)
        print("任务完成")