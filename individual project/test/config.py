"""
配置文件
集中管理所有仿真参数和常量
"""



# 任务参数
GRASP_HEIGHT_OFFSET = 0.15   # 抓取时在物体上方的高度
PLACE_HEIGHT_OFFSET = 0.20   # 放置时在托盘上方的高度



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
OBSTACLE_UPDATE_INTERVAL = 120  # 障碍臂每120步更新一次*目标*
OBSTACLE_MOVE_INTERVAL = 20      # <-- 【新增】障碍臂每N个仿真步才*移动*一次 (1=每步都移)
OBSTACLE_MOVE_STEP_RATIO = 0.02 # 每次移动时，移动2%的差距
OBSTACLE_JOINT_FORCE = 50       # 障碍臂关节控制力

# ========== 末端摄像头参数 ==========
END_EFFECTOR_LINK_INDEX = 11    # Panda机械臂末端执行器链接索引
CAMERA_WIDTH = 640              # 摄像头图像宽度
CAMERA_HEIGHT = 480             # 摄像头图像高度
CAMERA_FOV = 60                 # 视场角（度）
CAMERA_NEAR_PLANE = 0.01        # 近裁剪面（米）
CAMERA_FAR_PLANE = 2.0          # 远裁剪面（米）
CAMERA_DISTANCE_FROM_EE = 0.05  # 摄像头距离末端的距离（米）
CAMERA_DETECTION_THRESHOLD = 0.5  # 障碍物检测深度阈值（米）

# ========== 反应式避障参数 ==========
AVOIDANCE_ENABLED = True              # 是否启用反应式避障
AVOIDANCE_DANGER_DISTANCE = 0.12      # 危险距离阈值（米），小于此距离触发避障（已调大，提前触发）
AVOIDANCE_SAFE_DISTANCE = 0.2         # 安全距离（米），大于此距离才继续前进
AVOIDANCE_CHECK_INTERVAL = 5          # 避障检测间隔（仿真步）
AVOIDANCE_RETREAT_STEPS = 30          # 后退步数
AVOIDANCE_RETREAT_RATIO = 0.02        # 每步后退的关节角度比例
AVOIDANCE_RANDOM_ROTATION_RANGE = 0.3 # 随机转向的角度范围（弧度）±
AVOIDANCE_MAX_RETRIES = 5             # 最大重试次数
AVOIDANCE_MOVE_TO_BASE_DISTANCE = 0.2 # 往基座方向移动的距离（米）
AVOIDANCE_MOVE_TO_BASE_STEPS = 50     # 往基座方向移动的步数