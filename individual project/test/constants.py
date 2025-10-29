"""
常量定义模块
包含机器人配置和仿真参数
"""

import numpy as np

# --- 关节类型 ---
JOINT_TYPES = ["REVOLUTE", "PRISMATIC", "SPHERICAL", "PLANAR", "FIXED"]

# --- 机器人配置 ---
ROBOT_HOME_CONFIG = [0.0, -0.7854, 0.0, -2.3562, 0.0, 1.5708, 0.7854]
ROBOT_END_EFFECTOR_LINK_ID = 8
NUM_ARM_JOINTS = 7

# --- 仿真参数 ---
DELTA_T = 1.0 / 240.0

# --- 零空间参数 ---
DEFAULT_NULL_SPACE_PARAMS = {
    "lowerLimits": [-np.pi * 2] * NUM_ARM_JOINTS,
    "upperLimits": [np.pi * 2] * NUM_ARM_JOINTS,
    "jointRanges": [np.pi * 4] * NUM_ARM_JOINTS,
    "restPoses": list(ROBOT_HOME_CONFIG)
}

# --- 工作空间限制 ---
WORKSPACE_LIMITS = {
    "X_MIN": 0.1,
    "X_MAX": 0.8,
    "Y_MIN": -0.6,
    "Y_MAX": 0.6,
    "Z_MIN": 0.15,
    "Z_MAX": 0.8
}

# --- 安全参数 ---
PROXIMITY_FAILSAFE_DISTANCE = 0.03  # 3cm

