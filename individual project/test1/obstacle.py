import pybullet as p
import random
import math

class DynamicObstacle:
    def __init__(self):
        # 障碍物基础位置范围
        self.base_y = 0.0
        self.base_z = 0.25
        
        # ========== 活动范围 - 限制在机械臂检测区域内 ==========
        # X轴活动范围 (机械臂工作区: 0.3~0.6)
        self.x_min = 0.35   # 靠近机械臂
        self.x_max = 0.65   # 工作区边缘
        
        # Y轴活动范围 (方块Y=-0.3, 托盘Y=0.4, 运动路径中间)
        self.y_min = -0.2
        self.y_max = 0.25
        
        # Z轴活动范围 (抓取高度附近, 避障检测高度)
        self.z_min = 0.15
        self.z_max = 0.35
        
        # 尺寸
        self.arm_length = 0.8
        self.arm_width = 0.06
               
        # 创建视觉形状 (红色)
        arm_visual = p.createVisualShape(p.GEOM_BOX, 
                                         halfExtents=[self.arm_length/2, self.arm_width, self.arm_width], 
                                         rgbaColor=[0.8, 0.1, 0.1, 1.0])
        
        # 创建碰撞形状
        arm_col = p.createCollisionShape(p.GEOM_BOX, 
                                         halfExtents=[self.arm_length/2, self.arm_width, self.arm_width])
        
        # 初始位置：在检测区域边缘
        self.current_pos = [self.x_max + self.arm_length/2, self.base_y, self.base_z]
        
        self.body_id = p.createMultiBody(
            baseMass=0,
            baseCollisionShapeIndex=arm_col,
            baseVisualShapeIndex=arm_visual,
            basePosition=self.current_pos
        )

        # ========== 动态随机移动参数 ==========
        self.state = "IDLE"
        self.wait_timer = 0
        
        # 速度参数
        self.base_speed = 0.002
        self.current_speed = self.base_speed
        self.speed_variation = 0.003  # 速度变化幅度
        
        # 目标位置
        self.target_pos = list(self.current_pos)
        
        # 运动模式
        self.movement_mode = "LINEAR"  # LINEAR, ZIGZAG, CIRCULAR, RANDOM_WALK
        
        # 圆形运动参数
        self.circle_center = [0.6, 0.0, 0.3]
        self.circle_radius = 0.2
        self.circle_angle = 0
        self.circle_speed = 0.02
        
        # 蛇形运动参数
        self.zigzag_amplitude = 0.2
        self.zigzag_frequency = 0.05
        self.zigzag_progress = 0
        
        # 随机方向改变计时器
        self.direction_change_timer = 0
        self.direction_change_interval = random.randint(30, 100)
        
        # 速度变化计时器
        self.speed_change_timer = 0
        
        print(f"动态随机障碍物已加载 (ID: {self.body_id})")

    def get_position(self):
        """获取障碍物尖端位置"""
        tip_x = self.current_pos[0] - self.arm_length/2
        return [tip_x, self.current_pos[1], self.current_pos[2]]

    def get_id(self):
        return self.body_id

    def is_in_work_area(self, work_area_x_threshold=0.8):
        """检查是否在工作区内"""
        tip_x = self.current_pos[0] - self.arm_length/2
        return tip_x < work_area_x_threshold

    def get_state_info(self):
        tip_pos = self.get_position()
        return {
            "state": self.state,
            "mode": self.movement_mode,
            "tip_x": tip_pos[0],
            "tip_y": tip_pos[1],
            "tip_z": tip_pos[2],
            "in_work_area": self.is_in_work_area()
        }

    def _generate_random_target(self):
        """生成随机目标位置 - 限制在机械臂检测区域内"""
        # 在检测区域内随机生成目标
        target_x = random.uniform(self.x_min, self.x_max) + self.arm_length/2
        target_y = random.uniform(self.y_min, self.y_max)
        target_z = random.uniform(self.z_min, self.z_max)
        
        return [target_x, target_y, target_z]

    def _choose_movement_mode(self):
        """随机选择运动模式"""
        modes = ["LINEAR", "ZIGZAG", "CIRCULAR", "RANDOM_WALK"]
        weights = [0.3, 0.25, 0.2, 0.25]  # 各模式的概率权重
        self.movement_mode = random.choices(modes, weights=weights)[0]
        
        if self.movement_mode == "CIRCULAR":
            # 初始化圆形运动参数 - 限制在检测区域内
            self.circle_center = [
                random.uniform(0.45, 0.55),  # X轴中心区域
                random.uniform(-0.05, 0.1),  # Y轴中心区域
                random.uniform(0.22, 0.28)   # Z轴中心区域
            ]
            self.circle_radius = random.uniform(0.08, 0.12)  # 减小半径防止越界
            self.circle_angle = random.uniform(0, 2 * math.pi)
            self.circle_speed = random.uniform(0.01, 0.04)
            
        elif self.movement_mode == "ZIGZAG":
            # 初始化蛇形运动参数 - 调整幅度适应小范围
            self.zigzag_amplitude = random.uniform(0.05, 0.12)  # 减小摆动幅度
            self.zigzag_frequency = random.uniform(0.02, 0.05)
            self.zigzag_progress = 0
            self.target_pos = self._generate_random_target()

    def _update_speed(self):
        """随机更新速度"""
        self.speed_change_timer += 1
        if self.speed_change_timer > random.randint(20, 60):
            # 随机调整速度
            speed_factor = random.uniform(0.2, 1.5)
            self.current_speed = self.base_speed * speed_factor
            self.speed_change_timer = 0

    def _move_linear(self):
        """直线移动到目标"""
        dx = self.target_pos[0] - self.current_pos[0]
        dy = self.target_pos[1] - self.current_pos[1]
        dz = self.target_pos[2] - self.current_pos[2]
        
        distance = math.sqrt(dx*dx + dy*dy + dz*dz)
        
        if distance < 0.01:
            return True  # 到达目标
        
        # 归一化方向并移动
        if distance > 0:
            self.current_pos[0] += (dx / distance) * self.current_speed
            self.current_pos[1] += (dy / distance) * self.current_speed
            self.current_pos[2] += (dz / distance) * self.current_speed
        
        return False

    def _move_zigzag(self):
        """蛇形移动"""
        # 主方向移动
        dx = self.target_pos[0] - self.current_pos[0]
        dy_base = self.target_pos[1] - self.current_pos[1]
        dz = self.target_pos[2] - self.current_pos[2]
        
        distance = math.sqrt(dx*dx + dy_base*dy_base + dz*dz)
        
        if distance < 0.02:
            return True
        
        # 添加横向摆动
        self.zigzag_progress += self.zigzag_frequency
        lateral_offset = math.sin(self.zigzag_progress) * self.zigzag_amplitude * self.current_speed * 2
        
        if distance > 0:
            # 主方向移动
            move_x = (dx / distance) * self.current_speed
            move_y = (dy_base / distance) * self.current_speed + lateral_offset
            move_z = (dz / distance) * self.current_speed
            
            self.current_pos[0] += move_x
            self.current_pos[1] += move_y
            self.current_pos[2] += move_z
            
            # 边界检查
            self.current_pos[1] = max(self.y_min, min(self.y_max, self.current_pos[1]))
        
        return False

    def _move_circular(self):
        """圆形运动"""
        self.circle_angle += self.circle_speed
        
        # 计算圆形轨迹上的新位置
        new_x = self.circle_center[0] + self.circle_radius * math.cos(self.circle_angle) + self.arm_length/2
        new_y = self.circle_center[1] + self.circle_radius * math.sin(self.circle_angle)
        new_z = self.circle_center[2] + self.circle_radius * 0.3 * math.sin(self.circle_angle * 2)  # 添加Z轴起伏
        
        # 平滑过渡到目标位置
        self.current_pos[0] += (new_x - self.current_pos[0]) * 0.1
        self.current_pos[1] += (new_y - self.current_pos[1]) * 0.1
        self.current_pos[2] += (new_z - self.current_pos[2]) * 0.1
        
        # 边界检查
        self.current_pos[2] = max(self.z_min, min(self.z_max, self.current_pos[2]))
        
        return False  # 圆形运动不会"到达"

    def _move_random_walk(self):
        """随机游走"""
        # 检查是否需要改变方向
        self.direction_change_timer += 1
        if self.direction_change_timer > self.direction_change_interval:
            # 随机选择新方向
            self.target_pos = self._generate_random_target()
            self.direction_change_interval = random.randint(30, 100)
            self.direction_change_timer = 0
        
        # 向目标移动
        return self._move_linear()

    def update(self):
        """更新障碍物状态和位置"""
        
        # 更新速度
        self._update_speed()
        
        if self.state == "IDLE":
            self.wait_timer += 1
            if self.wait_timer > random.randint(30, 150):
                self.state = "MOVING"
                self._choose_movement_mode()
                self.target_pos = self._generate_random_target()
                self.wait_timer = 0
                
                mode_names = {
                    "LINEAR": "直线",
                    "ZIGZAG": "蛇形",
                    "CIRCULAR": "圆形",
                    "RANDOM_WALK": "随机游走"
                }
                print(f">>> 障碍物开始 {mode_names[self.movement_mode]} 运动！")

        elif self.state == "MOVING":
            reached_target = False
            
            if self.movement_mode == "LINEAR":
                reached_target = self._move_linear()
            elif self.movement_mode == "ZIGZAG":
                reached_target = self._move_zigzag()
            elif self.movement_mode == "CIRCULAR":
                # 圆形运动持续一段时间后切换
                self._move_circular()
                self.wait_timer += 1
                if self.wait_timer > random.randint(200, 400):
                    reached_target = True
                    self.wait_timer = 0
            elif self.movement_mode == "RANDOM_WALK":
                reached_target = self._move_random_walk()
                # 随机游走持续一段时间后切换
                self.wait_timer += 1
                if self.wait_timer > random.randint(150, 300):
                    reached_target = True
                    self.wait_timer = 0
            
            if reached_target:
                # 决定下一步：继续移动还是暂停
                if random.random() < 0.3:
                    self.state = "HOLDING"
                    self.wait_timer = 0
                    print("--- 障碍物暂停移动...")
                else:
                    # 直接切换到新的运动模式
                    self._choose_movement_mode()
                    self.target_pos = self._generate_random_target()

        elif self.state == "HOLDING":
            self.wait_timer += 1
            if self.wait_timer > random.randint(50, 200):
                # 随机决定是退出还是继续移动 (降低撤退概率，让障碍物更多停留在检测区域内)
                if random.random() < 0.2:
                    self.state = "RETREATING"
                    # 撤退到检测区域边缘
                    self.target_pos = [self.x_max + self.arm_length/2, 
                                       random.uniform(self.y_min, self.y_max), 
                                       self.base_z]
                    print("<<< 障碍物移动到边缘...")
                else:
                    self.state = "MOVING"
                    self._choose_movement_mode()
                    self.target_pos = self._generate_random_target()
                    print(">>> 障碍物继续移动！")
                self.wait_timer = 0

        elif self.state == "RETREATING":
            if self._move_linear():
                self.state = "IDLE"
                print("--- 障碍物已移至边缘，进入待机状态")

        # 应用位置更新
        p.resetBasePositionAndOrientation(self.body_id, self.current_pos, [0,0,0,1])
        
        # 边界安全检查
        self._clamp_position()

    def _clamp_position(self):
        """确保位置在安全边界内"""
        self.current_pos[0] = max(self.x_min + self.arm_length/2, 
                                   min(self.x_max + self.arm_length/2, self.current_pos[0]))
        self.current_pos[1] = max(self.y_min, min(self.y_max, self.current_pos[1]))
        self.current_pos[2] = max(self.z_min, min(self.z_max, self.current_pos[2]))
