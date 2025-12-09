import numpy as np
import pybullet as p
import math
import time
import avoidance
import predictor

class CameraSystem:
    def __init__(self, robot_id, tray_id):
        self.robot_id, self.tray_id, self.plane_id = robot_id, tray_id, 0
        self.ignored_ids = set()
        
        # 主摄像头（俯视）
        self.width, self.height = 128, 128
        self.view_matrix, self.proj_matrix, self.inv_pv_mat = self._init_camera(
            [0.8, 0, 0.6], [0.4, 0, 0], 60, 1.0)
        
        # 侧视摄像头（测量高度）
        self.side_width, self.side_height = 160, 120
        self.side_view_matrix, self.side_proj_matrix, self.side_inv_pv_mat = self._init_camera(
            [0.5, -0.8, 0.4], [0.5, 0, 0.3], 50, 160/120)
        
        self.obstacle_height_info = {"max_height": 0.0, "min_height": 0.0, 
            "clearance_height": 0.0, "confidence": 0.0, "last_update": 0}
        self.obstacle_predictor = predictor.ObstaclePredictor(history_size=20, prediction_horizon=0.8)
    
    def _init_camera(self, eye_pos, target_pos, fov, aspect):
        
        view_mat = p.computeViewMatrix(eye_pos, target_pos, [0, 0, 1])
        proj_mat = p.computeProjectionMatrixFOV(fov, aspect, 0.1, 2.0)
        view_np = np.array(view_mat).reshape(4, 4).T
        proj_np = np.array(proj_mat).reshape(4, 4).T
        return view_mat, proj_mat, np.linalg.pinv(proj_np @ view_np)

    def _scan_camera(self, width, height, view_mat, proj_mat, inv_pv, step=4, z_thresh=0.08):
        
        img = p.getCameraImage(width, height, view_mat, proj_mat, renderer=p.ER_TINY_RENDERER)
        depth = np.reshape(img[3], (height, width))
        seg = np.reshape(img[4], (height, width))
        
        # 过滤无效物体
        mask = (seg != self.robot_id) & (seg != self.tray_id) & (seg != self.plane_id) & (depth < 0.95)
        for ign_id in self.ignored_ids:
            mask &= (seg != ign_id)
        if not np.any(mask):
            return None
        
        # 采样并转换到世界坐标
        rows, cols = np.where(mask)
        rows, cols = rows[::step], cols[::step]
        depths = depth[rows, cols]
        
        x_ndc = (2 * cols / width) - 1
        y_ndc = 1 - (2 * rows / height)
        z_ndc = 2 * depths - 1
        
        world = np.dot(inv_pv, np.vstack([x_ndc, y_ndc, z_ndc, np.ones_like(x_ndc)]))
        points = (world[:3] / world[3]).T
        return points[points[:, 2] > z_thresh]

    def scan_obstacle_volume(self):
        
        points = self._scan_camera(self.width, self.height, self.view_matrix, 
                                   self.proj_matrix, self.inv_pv_mat, step=4, z_thresh=0.08)
        if points is None or len(points) < 5:
            return None
        
        min_bound, max_bound = np.min(points, axis=0), np.max(points, axis=0)
        center = np.mean(points, axis=0)
        self.draw_debug_box(min_bound, max_bound)
        self.obstacle_predictor.update(center)
        return {"min": min_bound, "max": max_bound, "center": center}
    
    def scan_obstacle_height_from_side(self):
        
        points = self._scan_camera(self.side_width, self.side_height, self.side_view_matrix,
                                   self.side_proj_matrix, self.side_inv_pv_mat, step=2, z_thresh=0.05)
        if points is None or len(points) < 3:
            self.obstacle_height_info = {"max_height": 0.0, "min_height": 0.0,
                "clearance_height": 0.15, "confidence": 0.0, "last_update": time.time()}
            return self.obstacle_height_info
        
        z = points[:, 2]
        self.obstacle_height_info = {
            "max_height": float(np.max(z)), "min_height": float(np.min(z)),
            "height_95": float(np.percentile(z, 95)),
            "clearance_height": float(np.percentile(z, 95) + 0.08),
            "confidence": min(len(points) / 100.0, 1.0),
            "point_count": len(points), "last_update": time.time()
        }
        return self.obstacle_height_info
    
    def get_obstacle_clearance_height(self):
        return self.obstacle_height_info.get("clearance_height", 0.15)
    
    def get_obstacle_height_info(self):
        return self.obstacle_height_info
    
    def get_predicted_obstacle_pos(self, robot_pos):
        return self.obstacle_predictor.get_avoidance_position(robot_pos)
    
    def get_motion_trend(self):
        return self.obstacle_predictor.get_motion_trend()
    
    def should_preemptive_avoid(self, robot_pos, robot_target):
        return self.obstacle_predictor.should_preemptive_avoid(robot_pos, robot_target)

    def draw_debug_box(self, min_pos, max_pos):
        p.removeAllUserDebugItems()
        mn, mx = min_pos, max_pos
        corners = [[mn[0],mn[1],mn[2]], [mx[0],mn[1],mn[2]], [mn[0],mx[1],mn[2]], [mx[0],mx[1],mn[2]],
                   [mn[0],mn[1],mx[2]], [mx[0],mn[1],mx[2]], [mn[0],mx[1],mx[2]], [mx[0],mx[1],mx[2]]]
        for s, e in [(0,1),(1,3),(3,2),(2,0),(4,5),(5,7),(7,6),(6,4),(0,4),(1,5),(2,6),(3,7)]:
            p.addUserDebugLine(corners[s], corners[e], [0,1,0], lineWidth=2, lifeTime=0.2)
        
        cz = self.obstacle_height_info.get("clearance_height", 0)
        if cz > 0.1:
            cc = [[mn[0]-0.1,mn[1]-0.1,cz], [mx[0]+0.1,mn[1]-0.1,cz], 
                  [mn[0]-0.1,mx[1]+0.1,cz], [mx[0]+0.1,mx[1]+0.1,cz]]
            for s, e in [(0,1),(1,3),(3,2),(2,0)]:
                p.addUserDebugLine(cc[s], cc[e], [0,0.5,1], lineWidth=3, lifeTime=0.2)
            p.addUserDebugText(f"Safe: {cz:.2f}m", [(mn[0]+mx[0])/2,(mn[1]+mx[1])/2,cz+0.02], 
                              [0,0.5,1], textSize=1.2, lifeTime=0.2)


class RobotController:
    def __init__(self, robot_id, tray_id):
        self.robot_id, self.eef_id = robot_id, 11
        self.critical_joints, self.finger_indices = [11, 6, 4], [9, 10]
        self.gripper_open_pos, self.gripper_closed_pos = 0.05, 0.03
        
        self.vision_system = CameraSystem(robot_id, tray_id)
        self.avoider = avoidance.VisualAvoidanceSystem(safe_distance=0.40, stop_distance=0.15)
        self.sim_step_callback = None
        
        # 被抓物品信息 - 用于避障时考虑物品体积
        self.grabbed_object_id = None
        self.grabbed_object_size = None  # [半宽, 半深, 半高]
        self.grabbed_object_offset = 0.0  # 物品底部相对于末端执行器的偏移
        
        p.setPhysicsEngineParameter(numSolverIterations=200, contactBreakingThreshold=0.001)
        self.ll, self.ul, self.jr = [-7]*7, [7]*7, [7]*7
        self.rp = [0, -math.pi/4, 0, -math.pi/2, 0, math.pi/3, 0]
        
        for f in self.finger_indices:
            p.changeDynamics(self.robot_id, f, lateralFriction=10.0, frictionAnchor=True)

    def set_ignored_object(self, obj_id):
        self.vision_system.ignored_ids.add(obj_id)
    
    def clear_ignored_object(self, obj_id):
        self.vision_system.ignored_ids.discard(obj_id)
    
    def set_grabbed_object(self, obj_id):
        """设置被抓物品，自动获取其尺寸用于避障计算"""
        self.grabbed_object_id = obj_id
        if obj_id is not None:
            # 获取物品的AABB边界框
            aabb_min, aabb_max = p.getAABB(obj_id)
            # 计算半尺寸
            half_size = [(aabb_max[i] - aabb_min[i]) / 2 for i in range(3)]
            self.grabbed_object_size = half_size
            # 物品底部到中心的距离（物品被抓住后，底部会在夹爪下方）
            self.grabbed_object_offset = half_size[2] + 0.02  # 加上夹爪到物品中心的间隙
            print(f"  [物品体积] 尺寸: {[f'{s*2:.3f}' for s in half_size]}m, 底部偏移: {self.grabbed_object_offset:.3f}m")
        else:
            self.grabbed_object_size = None
            self.grabbed_object_offset = 0.0
    
    def clear_grabbed_object(self):
        """清除被抓物品信息"""
        self.grabbed_object_id = None
        self.grabbed_object_size = None
        self.grabbed_object_offset = 0.0
    
    def get_effective_collision_bounds(self):
        """获取考虑被抓物品后的有效碰撞边界扩展量"""
        if self.grabbed_object_size is None:
            return {"radius_extend": 0.0, "bottom_extend": 0.0}
        
        # 水平方向扩展（物品的最大水平半径）
        radius_extend = max(self.grabbed_object_size[0], self.grabbed_object_size[1])
        # 向下扩展（物品底部相对于末端执行器的距离）
        bottom_extend = self.grabbed_object_offset
        
        return {"radius_extend": radius_extend, "bottom_extend": bottom_extend}
    
    def get_critical_body_points(self):
        return [list(p.getLinkState(self.robot_id, j)[4]) for j in self.critical_joints]

    def get_current_eef_pos(self):
        return list(p.getLinkState(self.robot_id, self.eef_id)[4])

    def step_simulation_with_callback(self):
        p.stepSimulation()
        if self.sim_step_callback:
            self.sim_step_callback()
        time.sleep(1./240.)

    def move_gripper(self, open_state=True):
        pos = self.gripper_open_pos if open_state else self.gripper_closed_pos
        for i in self.finger_indices:
            p.setJointMotorControl2(self.robot_id, i, p.POSITION_CONTROL, targetPosition=pos, force=500)
        for _ in range(20):
            self.step_simulation_with_callback() 

    def _get_effective_obstacle_pos(self, obs_aabb, current_eef_pos):
        
        if obs_aabb is None:
            return [10.0, 10.0, 10.0], {"status": "no_obstacle"}
        
        predicted_pos, pred_info = self.vision_system.get_predicted_obstacle_pos(current_eef_pos)
        if predicted_pos is not None:
            effective_pos = predicted_pos
            if pred_info.get("direction") == "approaching":
                bias = 0.05
                d = np.array(current_eef_pos) - np.array(predicted_pos)
                if np.linalg.norm(d) > 0.01:
                    d = d / np.linalg.norm(d)
                    effective_pos = [predicted_pos[i] + d[i] * bias for i in range(3)]
            return effective_pos, pred_info
        
        # 回退方法
        effective_pos = [max(obs_aabb["min"][i], min(current_eef_pos[i], obs_aabb["max"][i])) for i in range(3)]
        if math.sqrt(sum((a-b)**2 for a,b in zip(current_eef_pos, effective_pos))) < 0.05:
            effective_pos = obs_aabb["center"].tolist()
        return effective_pos, {"status": "fallback"}

    def move_arm_smart(self, target_pos, target_orn=None, timeout=10.0, debug=False):
        if target_orn is None:
            target_orn = p.getQuaternionFromEuler([math.pi, 0, math.pi/2])
        
        start_time = time.time()
        obs_aabb, step_counter = None, 0
        last_status, last_pred_info = "", None

        while True:
            current_eef_pos = self.get_current_eef_pos()
            
            # 视觉感知
            if step_counter % 24 == 0:
                obs_aabb = self.vision_system.scan_obstacle_volume()
            if step_counter % 48 == 0:
                h_info = self.vision_system.scan_obstacle_height_from_side()
                self.avoider.set_obstacle_height_info(h_info)
                if debug and h_info.get("confidence", 0) > 0.3:
                    print(f"  [侧视] 高度:{h_info.get('max_height',0):.3f}m, 安全:{h_info.get('clearance_height',0):.3f}m")
            step_counter += 1

            # 获取有效障碍物位置
            eff_obs_pos, pred_info = self._get_effective_obstacle_pos(obs_aabb, current_eef_pos)

            # 预判调整安全距离
            _, _, rec = self.vision_system.should_preemptive_avoid(current_eef_pos, target_pos)
            self.avoider.d_th2 = {"emergency_avoid": 0.55, "preemptive_avoid": 0.48}.get(rec, 0.40)

            # 传递运动信息
            trend = self.vision_system.get_motion_trend()
            self.avoider.set_obstacle_motion(trend.get("velocity"), trend.get("is_moving", False), trend.get("direction"))
            
            # 传递被抓物品的碰撞边界扩展量
            bounds = self.get_effective_collision_bounds()
            self.avoider.set_grabbed_object_bounds(bounds["radius_extend"], bounds["bottom_extend"])
            
            # 计算下一步
            virtual_next, status = self.avoider.compute_modified_step(current_eef_pos, target_pos, eff_obs_pos)
            
            # 调试输出
            if debug and (status != last_status or pred_info.get("direction") != (last_pred_info or {}).get("direction")):
                eef_dist = math.sqrt(sum((a-b)**2 for a,b in zip(current_eef_pos, eff_obs_pos)))
                print(f"  [调试] 状态:{status}, 距障碍:{eef_dist:.3f}, 方向:{pred_info.get('direction','-')}")
                last_status, last_pred_info = status, pred_info.copy() if isinstance(pred_info, dict) else pred_info
            
            # 到达检测
            dist = math.sqrt(sum((a-b)**2 for a,b in zip(current_eef_pos, target_pos)))
            if dist < 0.08:
                if debug: print(f"  [调试] 到达目标！")
                break
            
            # 执行运动
            joints = p.calculateInverseKinematics(self.robot_id, self.eef_id, virtual_next, target_orn,
                lowerLimits=self.ll, upperLimits=self.ul, jointRanges=self.jr, restPoses=self.rp)
            for i in range(7):
                p.setJointMotorControl2(self.robot_id, i, p.POSITION_CONTROL, targetPosition=joints[i], maxVelocity=2.0, force=500)
            
            self.step_simulation_with_callback()
            if time.time() - start_time > timeout:
                print("移动超时")
                break

    def move_arm_exact(self, target_pos, target_orn=None, steps=80):
        if target_orn is None:
            target_orn = p.getQuaternionFromEuler([math.pi, 0, math.pi/2])
        for _ in range(steps):
            joints = p.calculateInverseKinematics(self.robot_id, self.eef_id, target_pos, target_orn,
                lowerLimits=self.ll, upperLimits=self.ul, jointRanges=self.jr, restPoses=self.rp)
            for i in range(7):
                p.setJointMotorControl2(self.robot_id, i, p.POSITION_CONTROL, targetPosition=joints[i], maxVelocity=2.0, force=500)
            self.step_simulation_with_callback()

    def execute_pick_and_place(self, cube_id, tray_id):
        p.changeDynamics(cube_id, -1, lateralFriction=10.0, mass=0.05)
        cube_pos, _ = p.getBasePositionAndOrientation(cube_id)
        tray_pos, _ = p.getBasePositionAndOrientation(tray_id)
        
        pre_grasp = [cube_pos[0], cube_pos[1], cube_pos[2] + 0.3]
        grasp_pos = [cube_pos[0], cube_pos[1], 0.045]
        drop_pos = [tray_pos[0], tray_pos[1], tray_pos[2] + 0.3]
        drop_low = [tray_pos[0], tray_pos[1], tray_pos[2] + 0.15]
        
        self.set_ignored_object(cube_id)
        
        # 定义抓取后的回调 - 记录被抓物品体积
        def grasp_and_record():
            self.move_gripper(False)
            for _ in range(100):
                self.step_simulation_with_callback()
            # 抓取成功后，记录物品体积用于避障
            self.set_grabbed_object(cube_id)
        
        # 定义放下后的回调 - 清除物品体积信息
        def release_and_clear():
            self.move_arm_exact(drop_low, steps=300)
            self.move_gripper(True)
            # 放下后清除物品体积信息
            self.clear_grabbed_object()
        
        steps = [("1. 接近方块上方", lambda: (self.move_gripper(True), self.move_arm_smart(pre_grasp, timeout=15.0))),
                 ("2. 精准下降", lambda: self.move_arm_exact(grasp_pos, steps=200)),
                 ("3. 闭合夹爪并记录物品体积", grasp_and_record),
                 ("4. 抬起(避障考虑物品体积)", lambda: self.move_arm_smart(pre_grasp, timeout=10.0, debug=True)),
                 ("5. 搬运至终点(避障考虑物品体积)", lambda: self.move_arm_smart(drop_pos, timeout=30.0, debug=True)),
                 ("6. 放下并清除体积信息", release_and_clear)]
        
        for name, action in steps:
            print(f">>> {name}")
            action()
        print("任务完成")