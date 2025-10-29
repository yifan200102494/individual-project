"""
机器人控制工具库 - 统一入口
将所有功能模块整合在一起，提供向后兼容的接口

模块结构：
- constants.py: 常量定义
- collision_detection.py: 碰撞检测
- perception.py: 传感器感知
- path_planning.py: 路径规划（PFM、弧形路径等）
- exploration.py: 随机探索
- motion_control.py: 运动控制（仿真、关节运动、夹爪）
- planner.py: 高层规划器
"""

# ============================================================
# 导入所有常量
# ============================================================
from constants import (
    JOINT_TYPES,
    ROBOT_HOME_CONFIG,
    ROBOT_END_EFFECTOR_LINK_ID,
    NUM_ARM_JOINTS,
    DELTA_T,
    DEFAULT_NULL_SPACE_PARAMS,
    WORKSPACE_LIMITS,
    PROXIMITY_FAILSAFE_DISTANCE
)

# ============================================================
# 导入碰撞检测功能
# ============================================================
from collision_detection import (
    is_state_colliding,
    is_path_colliding
)

# ============================================================
# 导入感知功能
# ============================================================
from perception import (
    perceive_obstacles_with_rays
)

# ============================================================
# 导入路径规划功能
# ============================================================
from path_planning import (
    calc_attractive_force,
    calc_anisotropic_repulsive_force,
    prepare_obstacles_info,
    plan_path_with_pfm,
    generate_arc_path,
    validate_workspace_path,
    generate_detour_strategies,
    add_path_to_history,
    is_path_similar_to_history,
    PATH_HISTORY,
    MAX_PATH_HISTORY
)

# ============================================================
# 导入探索功能
# ============================================================
from exploration import (
    perform_random_exploration,
    generate_workspace_exploration_targets,
    generate_obstacle_avoidance_targets,
    generate_height_level_targets,
    generate_safe_retreat_targets,
    generate_spiral_targets
)

# ============================================================
# 导入运动控制功能
# ============================================================
from motion_control import (
    simulate,
    move_to_joints,
    move_to_pose,
    gripper_open,
    gripper_close
)

# ============================================================
# 导入高层规划器
# ============================================================
from planner import (
    plan_and_execute_motion
)

# ============================================================
# 导入实时动态系统（新）
# ============================================================
try:
    from dynamic_executor import DynamicMotionExecutor
    from realtime_perception import AdaptivePerceptionSystem
    from incremental_planner import IncrementalPlanner, ReactivePlanner
    _DYNAMIC_SYSTEM_AVAILABLE = True
except ImportError:
    _DYNAMIC_SYSTEM_AVAILABLE = False

# ============================================================
# 向后兼容：保留原有的私有变量名
# ============================================================
_NUM_ARM_JOINTS = NUM_ARM_JOINTS
_DEFAULT_NULL_SPACE_PARAMS = DEFAULT_NULL_SPACE_PARAMS
_GLOBAL_SIM_STEP_COUNTER = 0  # 注意：这个变量在 motion_control 模块中

# ============================================================
# 模块说明
# ============================================================
__all__ = [
    # 常量
    'JOINT_TYPES',
    'ROBOT_HOME_CONFIG',
    'ROBOT_END_EFFECTOR_LINK_ID',
    'NUM_ARM_JOINTS',
    'DELTA_T',
    'DEFAULT_NULL_SPACE_PARAMS',
    'WORKSPACE_LIMITS',
    'PROXIMITY_FAILSAFE_DISTANCE',
    
    # 碰撞检测
    'is_state_colliding',
    'is_path_colliding',
    
    # 感知
    'perceive_obstacles_with_rays',
    
    # 路径规划
    'calc_attractive_force',
    'calc_anisotropic_repulsive_force',
    'prepare_obstacles_info',
    'plan_path_with_pfm',
    'generate_arc_path',
    'validate_workspace_path',
    'generate_detour_strategies',
    'add_path_to_history',
    'is_path_similar_to_history',
    'PATH_HISTORY',
    'MAX_PATH_HISTORY',
    
    # 探索
    'perform_random_exploration',
    'generate_workspace_exploration_targets',
    'generate_obstacle_avoidance_targets',
    'generate_height_level_targets',
    'generate_safe_retreat_targets',
    'generate_spiral_targets',
    
    # 运动控制
    'simulate',
    'move_to_joints',
    'move_to_pose',
    'gripper_open',
    'gripper_close',
    
    # 高层规划
    'plan_and_execute_motion',
    
    # 实时动态系统（新）
    'DynamicMotionExecutor',
    'AdaptivePerceptionSystem',
    'IncrementalPlanner',
    'ReactivePlanner',
]

# ============================================================
# 版本信息
# ============================================================
__version__ = '2.0.0'
__author__ = 'Refactored Modular Version'
