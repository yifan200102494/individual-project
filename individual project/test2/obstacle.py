import pybullet as p
import random

class DynamicObstacle:
    def __init__(self):
        """
        初始化线性伸缩障碍物 (纯净版 - 无底座)
        """
        # ==========================================
        # 1. 设置参数
        # ==========================================
        # 障碍物所在的 Y 轴和 Z 轴位置
        self.base_y = 0.0
        self.base_z = 0.3
        
        # 伸缩范围 (X轴坐标)
        self.retract_x = 1.2   # 完全收回的位置 (藏在远处)
        self.extend_x = 0.45   # 伸出时的位置 (阻挡机器人)
        
        # 尺寸
        self.arm_length = 0.8 # 臂长
        self.arm_width = 0.06 # 臂宽
        
        # ==========================================
        # 2. 创建伸缩臂 (只保留这个红色障碍物)
        # ==========================================
        
        # 创建视觉形状 (红色)
        arm_visual = p.createVisualShape(p.GEOM_BOX, 
                                         halfExtents=[self.arm_length/2, self.arm_width, self.arm_width], 
                                         rgbaColor=[0.8, 0.1, 0.1, 1.0])
        
        # 创建碰撞形状 (用于物理检测)
        arm_col = p.createCollisionShape(p.GEOM_BOX, 
                                         halfExtents=[self.arm_length/2, self.arm_width, self.arm_width])
        
        # 初始位置：完全收回状态
        # 注意：物体坐标中心是它的几何中心，所以要加上半长
        self.current_pos = [self.retract_x + self.arm_length/2, self.base_y, self.base_z]
        
        self.body_id = p.createMultiBody(
            baseMass=0, # 0质量 = 运动学物体(Kinematic)，悬空不掉落
            baseCollisionShapeIndex=arm_col,
            baseVisualShapeIndex=arm_visual,
            basePosition=self.current_pos
        )

        # ==========================================
        # 3. 状态机控制参数
        # ==========================================
        self.state = "IDLE"     # 初始状态
        self.wait_timer = 0     # 计时器
        self.speed = 0.01       # 伸缩速度
        self.target_x = self.current_pos[0] # 目标X坐标

        print(f"红色障碍臂已加载 (无底座版, ID: {self.body_id})")

    def get_position(self):
        """返回障碍物最前端(危险点)的坐标"""
        tip_x = self.current_pos[0] - self.arm_length/2
        return [tip_x, self.current_pos[1], self.current_pos[2]]

    def get_id(self):
        return self.body_id

    def update(self):
        """
        更新伸缩逻辑
        """
        # 1. 状态机
        if self.state == "IDLE":
            self.wait_timer += 1
            # 随机等待一段时间后伸出
            if self.wait_timer > random.randint(50, 200): 
                self.state = "EXTENDING"
                self.target_x = self.extend_x + self.arm_length/2
                self.wait_timer = 0
                print(">>> 警告：障碍物进入工作区！")

        elif self.state == "EXTENDING":
            if abs(self.current_pos[0] - self.target_x) < 0.01:
                self.state = "HOLDING"
                self.wait_timer = 0

        elif self.state == "HOLDING":
            self.wait_timer += 1
            # 阻挡一段时间后收回
            if self.wait_timer > random.randint(100, 300):
                self.state = "RETRACTING"
                self.target_x = self.retract_x + self.arm_length/2
                print("<<< 解除：障碍物离开。")

        elif self.state == "RETRACTING":
            if abs(self.current_pos[0] - self.target_x) < 0.01:
                self.state = "IDLE"

        # 2. 移动逻辑
        if self.current_pos[0] < self.target_x:
            self.current_pos[0] += self.speed
        elif self.current_pos[0] > self.target_x:
            self.current_pos[0] -= self.speed
            
        p.resetBasePositionAndOrientation(self.body_id, self.current_pos, [0,0,0,1])