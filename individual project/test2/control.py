import pybullet as p
import time
import math
import environmen

class RobotController:
    def __init__(self, robot_id):
        self.robot_id = robot_id
        self.eef_id = 11 
        self.finger_indices = [9, 10] 
        self.gripper_open_pos = 0.05 
        
        # ========================================================
        # [核心修复] 修改闭合目标值，解决穿模
        # ========================================================
        # 0.0  = 完全闭合 (两指接触) -> 会导致穿模
        # 0.02 = 保留 4cm 缝隙 (方块大概 6cm 宽) -> 既能夹紧，又不会插太深
        self.gripper_closed_pos = 0.03 
        
        # 提高物理引擎求解精度 (保持之前的优化)
        p.setPhysicsEngineParameter(numSolverIterations=200, contactBreakingThreshold=0.001)

        # 关节限制
        self.ll = [-7]*7
        self.ul = [7]*7
        self.jr = [7]*7
        self.rp = [0, -math.pi/4, 0, -math.pi/2, 0, math.pi/3, 0]

        # 增加手指摩擦力 (保持之前的优化)
        for finger in self.finger_indices:
            p.changeDynamics(self.robot_id, finger, 
                             lateralFriction=10.0, 
                             spinningFriction=1.0,
                             rollingFriction=1.0, 
                             frictionAnchor=True)

    def move_gripper(self, open_state=True, step_callback=None):
        target_pos = self.gripper_open_pos if open_state else self.gripper_closed_pos
        
        # 发送指令
        for i in self.finger_indices:
            p.setJointMotorControl2(self.robot_id, i, p.POSITION_CONTROL, 
                                    targetPosition=target_pos, 
                                    force=500) # 保持大力度
        
        # 等待物理执行
        for _ in range(40): 
            p.stepSimulation()
            if step_callback: step_callback()
            time.sleep(1./240.)

    def move_arm(self, target_pos, target_orn=None, steps=50, step_callback=None, keep_gripper_closed=False):
        if target_orn is None:
            target_orn = p.getQuaternionFromEuler([math.pi, 0, math.pi/2])
        
        for _ in range(steps):
            # 实时 IK
            joint_poses = p.calculateInverseKinematics(
                self.robot_id, self.eef_id, target_pos, target_orn,
                lowerLimits=self.ll, upperLimits=self.ul, jointRanges=self.jr, restPoses=self.rp
            )
            for i in range(7):
                p.setJointMotorControl2(self.robot_id, i, p.POSITION_CONTROL, 
                                        targetPosition=joint_poses[i], maxVelocity=2.0)
            
            # 持续保持夹紧状态
            if keep_gripper_closed:
                for finger in self.finger_indices:
                    p.setJointMotorControl2(self.robot_id, finger, p.POSITION_CONTROL, 
                                            targetPosition=self.gripper_closed_pos, # 这里现在是 0.02
                                            force=500)

            p.stepSimulation()
            if step_callback: step_callback()
            time.sleep(1./240.)

    def execute_pick_and_place(self, cube_id, tray_id, loop_callback=None):
        # 保持方块的高摩擦力设置
        p.changeDynamics(cube_id, -1, 
                         lateralFriction=10.0, 
                         spinningFriction=1.0,
                         rollingFriction=1.0,
                         frictionAnchor=True,
                         mass=0.05)

        cube_pos, _ = p.getBasePositionAndOrientation(cube_id)
        tray_pos, _ = p.getBasePositionAndOrientation(tray_id)
        
        pre_grasp_pos = [cube_pos[0], cube_pos[1], cube_pos[2] + 0.30]
        approach_pos = [cube_pos[0], cube_pos[1], cube_pos[2] + 0.10]
        
        # 抓取高度保持 0.05
        grasp_pos = [cube_pos[0], cube_pos[1], 0.05]
        drop_pos = [tray_pos[0], tray_pos[1], tray_pos[2] + 0.3] 

        down_orn = p.getQuaternionFromEuler([math.pi, 0, math.pi/2])

        print(">>> 1. 移动到方块上方")
        self.move_gripper(open_state=True, step_callback=loop_callback)
        self.move_arm(pre_grasp_pos, target_orn=down_orn, steps=100, step_callback=loop_callback)

        print(">>> 2. 下降")
        self.move_arm(approach_pos, target_orn=down_orn, steps=80, step_callback=loop_callback)
        self.move_arm(grasp_pos, target_orn=down_orn, steps=100, step_callback=loop_callback)

        print(">>> 稳定对齐")
        for _ in range(60):
            self.move_arm(grasp_pos, target_orn=down_orn, steps=1, step_callback=loop_callback)

        print(">>> 3. 抓取 (无穿模版)")
        self.move_gripper(open_state=False) 
        # 闭合等待
        for _ in range(50):
            self.move_arm(grasp_pos, target_orn=down_orn, steps=1, step_callback=loop_callback, keep_gripper_closed=True)
        
        print(">>> 4. 抬起")
        self.move_arm(pre_grasp_pos, target_orn=down_orn, steps=100, step_callback=loop_callback, keep_gripper_closed=True)

        print(">>> 5. 搬运")
        self.move_arm(drop_pos, target_orn=down_orn, steps=200, step_callback=loop_callback, keep_gripper_closed=True)

        print(">>> 6. 放下")
        self.move_gripper(open_state=True, step_callback=loop_callback)
        
        exit_pos = [drop_pos[0], drop_pos[1], drop_pos[2] + 0.1]
        self.move_arm(exit_pos, target_orn=down_orn, steps=50, step_callback=loop_callback)

        print("任务完成！")

if __name__ == "__main__":
    robot_id, tray_id, cube_id = environmen.setup_environment()
    controller = RobotController(robot_id)
    time.sleep(1)
    controller.execute_pick_and_place(cube_id, tray_id)
    while True:
        p.stepSimulation()
        time.sleep(1./240.)