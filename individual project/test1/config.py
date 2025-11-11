"""
配置文件
集中管理所有仿真参数和常量
"""

# ========== 安全距离配置 ==========
# 模拟速度与分离监控 (SSM)
STOP_DISTANCE = 0.05  # 危险距离 (米): 必须停止
SLOW_DISTANCE = 0.1   # 警告距离 (米): 必须减速

# ========== 机械臂控制参数 ==========
# 速度控制
NORMAL_VELOCITY = 1.0   # 正常速度
SLOW_VELOCITY = 0.2     # 减速时的速度

# 任务参数
GRASP_HEIGHT_OFFSET = 0.15   # 抓取时在物体上方的高度
PLACE_HEIGHT_OFFSET = 0.20   # 放置时在托盘上方的高度

# 避障重规划参数
REPLANNING_COOLDOWN = 30      # 触发重新规划前需阻塞的帧数
REPLAN_SUCCESS_COOLDOWN = 80  # 绕行成功后的冷却时间
ESCAPE_DISTANCE = 0.25        # 绕行时的逃逸距离

# ========== 物体位置配置 ==========
# 机械臂位置
ROBOT_BASE_POSITION = [0, 0, 0]
ROBOT_BASE_ORIENTATION = [0, 0, 0, 1]

# 小方块位置
OBJECT_POSITION = [0.5, -0.3, 0.025]
OBJECT_ORIENTATION = [0, 0, 0, 1]

# 托盘位置
TRAY_POSITION = [0.5, 0.5, 0.0]
TRAY_ORIENTATION = [0, 0, 0, 1]

# 障碍臂位置
OBSTACLE_ARM_POSITION = [0.7, 0.1, 0.0]
OBSTACLE_ARM_YAW = 1.5708  # π/2 弧度
OBSTACLE_ARM_SCALE = 0.7    # 缩放到70%

# ========== 禁区配置 ==========
# 用于障碍臂避让主机械臂基座区域
FORBIDDEN_ZONE_CENTER = [0.0, 0.0, 0.2]  # 基座上方20cm处
FORBIDDEN_ZONE_RADIUS = 0.35              # 半径35cm

# ========== 可视化配置 ==========
CAMERA_DISTANCE = 1.7
CAMERA_YAW = 60
CAMERA_PITCH = -30
CAMERA_TARGET = [0.2, 0.2, 0.25]

# 机械臂颜色
ROBOT_COLOR = [0.8, 0.8, 0.8, 1]      # 白色（默认）
OBSTACLE_ARM_COLOR = [1, 0.2, 0.2, 1]  # 红色

# ========== 仿真参数 ==========
SIMULATION_FREQUENCY = 240  # 240Hz
GRAVITY = [0, 0, -9.81]

# 调试输出频率
DEBUG_PRINT_INTERVAL = 50  # 每50步打印一次调试信息

# ========== 夹爪参数 ==========
GRIPPER_OPEN_POSITION = 0.04    # 夹爪打开位置
GRIPPER_CLOSED_POSITION = 0.0   # 夹爪关闭位置
GRIPPER_FORCE = 20              # 夹爪力量
GRIPPER_MAX_VELOCITY = 0.1      # 夹爪最大速度

# ========== 障碍臂运动参数 ==========
OBSTACLE_UPDATE_INTERVAL = 120  # 障碍臂每120步更新一次目标
OBSTACLE_MOVE_STEP_RATIO = 0.02 # 每步移动2%的差距
OBSTACLE_JOINT_FORCE = 50       # 障碍臂关节控制力

