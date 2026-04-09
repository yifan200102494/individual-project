import pybullet as p
import random
import math

class DynamicObstacle:
    # 运动模式名称映射
    MODE_NAMES = {"LINEAR": "直线", "ZIGZAG": "蛇形", "CIRCULAR": "圆形", "RANDOM_WALK": "随机游走"}
    
    def __init__(self):
        # 活动范围
        self.bounds = {
            'x': (0.35, 0.65), 'y': (-0.2, 0.25), 'z': (0.15, 0.35)
        }
        self.base_y, self.base_z = 0.0, 0.25
        self.arm_length, self.arm_width = 0.8, 0.06
        
        # 创建障碍物
        half_ext = [self.arm_length/2, self.arm_width, self.arm_width]
        visual = p.createVisualShape(p.GEOM_BOX, halfExtents=half_ext, rgbaColor=[0.8, 0.1, 0.1, 1.0])
        collision = p.createCollisionShape(p.GEOM_BOX, halfExtents=half_ext)
        
        self.current_pos = [self.bounds['x'][1] + self.arm_length/2, self.base_y, self.base_z]
        self.body_id = p.createMultiBody(baseMass=0, baseCollisionShapeIndex=collision,
                                          baseVisualShapeIndex=visual, basePosition=self.current_pos)
        
        # 状态与运动参数
        self.state, self.movement_mode = "IDLE", "LINEAR"
        self.wait_timer, self.target_pos = 0, list(self.current_pos)
        self.base_speed, self.current_speed = 0.001, 0.001
        self.speed_change_timer, self.direction_change_timer = 0, 0
        self.direction_change_interval = random.randint(30, 100)
        
        # 圆形/蛇形运动参数
        self.circle = {'center': [0.6, 0.0, 0.3], 'radius': 0.2, 'angle': 0, 'speed': 0.01}
        self.zigzag = {'amplitude': 0.2, 'frequency': 0.025, 'progress': 0}
        
        print(f"动态随机障碍物已加载 (ID: {self.body_id})")

    # ============ 基础接口 ============
    def get_id(self): return self.body_id
    def get_position(self): return [self.current_pos[0] - self.arm_length/2, self.current_pos[1], self.current_pos[2]]
    def is_in_work_area(self, threshold=0.8): return self.current_pos[0] - self.arm_length/2 < threshold
    
    def get_state_info(self):
        pos = self.get_position()
        return {"state": self.state, "mode": self.movement_mode, 
                "tip_x": pos[0], "tip_y": pos[1], "tip_z": pos[2], "in_work_area": self.is_in_work_area()}

    # ============ 内部工具 ============
    def _rand(self, key): return random.uniform(self.bounds[key][0], self.bounds[key][1])
    def _clamp(self, val, key): return max(self.bounds[key][0], min(self.bounds[key][1], val))
    def _distance(self, p1, p2): return math.sqrt(sum((a-b)**2 for a, b in zip(p1, p2)))
    
    def _generate_random_target(self):
        return [self._rand('x') + self.arm_length/2, self._rand('y'), self._rand('z')]

    def _move_toward(self, threshold=0.01):
        """向目标移动，返回是否到达"""
        dist = self._distance(self.current_pos, self.target_pos)
        if dist < threshold: return True
        for i in range(3):
            self.current_pos[i] += (self.target_pos[i] - self.current_pos[i]) / dist * self.current_speed
        return False

    # ============ 运动模式 ============
    def _choose_movement_mode(self):
        self.movement_mode = random.choices(
            ["LINEAR", "ZIGZAG", "CIRCULAR", "RANDOM_WALK"], 
            weights=[0.3, 0.25, 0.2, 0.25]
        )[0]
        
        if self.movement_mode == "CIRCULAR":
            self.circle = {
                'center': [random.uniform(0.45, 0.55), random.uniform(-0.05, 0.1), random.uniform(0.22, 0.28)],
                'radius': random.uniform(0.08, 0.12),
                'angle': random.uniform(0, 2 * math.pi),
                'speed': random.uniform(0.005, 0.02)
            }
        elif self.movement_mode == "ZIGZAG":
            self.zigzag = {'amplitude': random.uniform(0.05, 0.12), 'frequency': random.uniform(0.01, 0.025), 'progress': 0}
            self.target_pos = self._generate_random_target()

    def _move_linear(self): return self._move_toward()

    def _move_zigzag(self):
        dist = self._distance(self.current_pos, self.target_pos)
        if dist < 0.02: return True
        
        self.zigzag['progress'] += self.zigzag['frequency']
        lateral = math.sin(self.zigzag['progress']) * self.zigzag['amplitude'] * self.current_speed * 2
        
        for i in range(3):
            delta = (self.target_pos[i] - self.current_pos[i]) / dist * self.current_speed
            self.current_pos[i] += delta + (lateral if i == 1 else 0)
        self.current_pos[1] = self._clamp(self.current_pos[1], 'y')
        return False

    def _move_circular(self):
        c = self.circle
        c['angle'] += c['speed']
        
        target = [
            c['center'][0] + c['radius'] * math.cos(c['angle']) + self.arm_length/2,
            c['center'][1] + c['radius'] * math.sin(c['angle']),
            c['center'][2] + c['radius'] * 0.3 * math.sin(c['angle'] * 2)
        ]
        for i in range(3):
            self.current_pos[i] += (target[i] - self.current_pos[i]) * 0.1
        self.current_pos[2] = self._clamp(self.current_pos[2], 'z')
        return False

    def _move_random_walk(self):
        self.direction_change_timer += 1
        if self.direction_change_timer > self.direction_change_interval:
            self.target_pos = self._generate_random_target()
            self.direction_change_interval = random.randint(30, 100)
            self.direction_change_timer = 0
        return self._move_linear()

    # ============ 主更新循环 ============
    def _update_speed(self):
        self.speed_change_timer += 1
        if self.speed_change_timer > random.randint(20, 60):
            self.current_speed = self.base_speed * random.uniform(0.2, 1.5)
            self.speed_change_timer = 0

    def update(self):
        self._update_speed()
        
        if self.state == "IDLE":
            self.wait_timer += 1
            if self.wait_timer > random.randint(30, 150):
                self.state, self.wait_timer = "MOVING", 0
                self._choose_movement_mode()
                self.target_pos = self._generate_random_target()
                print(f">>> 障碍物开始 {self.MODE_NAMES[self.movement_mode]} 运动！")

        elif self.state == "MOVING":
            move_funcs = {"LINEAR": self._move_linear, "ZIGZAG": self._move_zigzag, 
                          "CIRCULAR": self._move_circular, "RANDOM_WALK": self._move_random_walk}
            reached = move_funcs[self.movement_mode]()
            
            # 圆形和随机游走有时间限制
            if self.movement_mode in ("CIRCULAR", "RANDOM_WALK"):
                self.wait_timer += 1
                limit = random.randint(200, 400) if self.movement_mode == "CIRCULAR" else random.randint(150, 300)
                if self.wait_timer > limit:
                    reached, self.wait_timer = True, 0
            
            if reached:
                if random.random() < 0.3:
                    self.state, self.wait_timer = "HOLDING", 0
                    print("--- 障碍物暂停移动...")
                else:
                    self._choose_movement_mode()
                    self.target_pos = self._generate_random_target()

        elif self.state == "HOLDING":
            self.wait_timer += 1
            if self.wait_timer > random.randint(50, 200):
                self.wait_timer = 0
                if random.random() < 0.2:
                    self.state = "RETREATING"
                    self.target_pos = [self.bounds['x'][1] + self.arm_length/2, self._rand('y'), self.base_z]
                    print("<<< 障碍物移动到边缘...")
                else:
                    self.state = "MOVING"
                    self._choose_movement_mode()
                    self.target_pos = self._generate_random_target()
                    print(">>> 障碍物继续移动！")

        elif self.state == "RETREATING":
            if self._move_linear():
                self.state = "IDLE"
                print("--- 障碍物已移至边缘，进入待机状态")

        # 应用位置并边界检查
        p.resetBasePositionAndOrientation(self.body_id, self.current_pos, [0,0,0,1])
        self.current_pos[0] = max(self.bounds['x'][0] + self.arm_length/2, 
                                   min(self.bounds['x'][1] + self.arm_length/2, self.current_pos[0]))
        self.current_pos[1] = self._clamp(self.current_pos[1], 'y')
        self.current_pos[2] = self._clamp(self.current_pos[2], 'z')
