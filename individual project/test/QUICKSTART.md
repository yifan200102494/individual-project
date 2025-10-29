# 🚀 快速开始指南

## 📋 目录
1. [安装依赖](#安装依赖)
2. [基础使用](#基础使用)
3. [完整示例](#完整示例)
4. [常见问题](#常见问题)

## 安装依赖

```bash
pip install pybullet numpy
```

## 基础使用

### 1️⃣ 导入模块

```python
# 方式1：导入所有功能（简单）
from util import *

# 方式2：按需导入（推荐）
from util import (
    ROBOT_END_EFFECTOR_LINK_ID,
    perceive_obstacles_with_rays,
    plan_and_execute_motion,
    gripper_open,
    gripper_close
)
```

### 2️⃣ 感知障碍物

```python
# 使用多方向射线检测障碍物
perceived_ids = perceive_obstacles_with_rays(
    robot_id=robot_id,
    sensor_link_id=ROBOT_END_EFFECTOR_LINK_ID,
    ray_range=1.5,      # 射线范围（米）
    grid_size=7,        # 网格密度
    fov_width=0.8,      # 视场宽度
    debug=True          # 显示调试射线
)

# 过滤掉地面（ID为-1）
obstacle_list = list(perceived_ids - {-1})
```

### 3️⃣ 规划并执行运动

```python
import numpy as np

# 定义目标位置和姿态
goal_position = [0.5, 0.2, 0.3]
goal_orientation = p.getQuaternionFromEuler([0, np.pi, 0])

# 自动规划并执行
success = plan_and_execute_motion(
    robot_id=robot_id,
    goal_pos=goal_position,
    goal_orn=goal_orientation,
    obstacle_ids=obstacle_list,
    interferer_id=interferer_id  # 动态障碍物（可选）
)

if success:
    print("✅ 成功到达目标！")
else:
    print("❌ 规划失败")
```

### 4️⃣ 控制夹爪

```python
# 打开夹爪
gripper_open(robot_id)

# 闭合夹爪（抓取物体）
gripper_close(robot_id)
```

## 完整示例

### 示例1：简单的抓取任务

```python
import pybullet as p
import numpy as np
from util import *

# 初始化 PyBullet
p.connect(p.GUI)
p.setGravity(0, 0, -9.8)

# 加载机器人和环境
robot_id = p.loadURDF("franka_panda/panda.urdf", useFixedBase=True)
table_id = p.loadURDF("table/table.urdf", [0.5, 0, 0])
cube_id = p.loadURDF("cube.urdf", [0.5, 0.2, 0.5])

# 1. 移动到抓取准备位置
print("📍 移动到抓取准备位置...")
pre_grasp_pos = [0.5, 0.2, 0.6]
pre_grasp_orn = p.getQuaternionFromEuler([0, np.pi, 0])

perceived_ids = perceive_obstacles_with_rays(robot_id, ROBOT_END_EFFECTOR_LINK_ID)
obstacles = list(perceived_ids - {-1, cube_id})  # 排除地面和目标物体

success = plan_and_execute_motion(
    robot_id, pre_grasp_pos, pre_grasp_orn, obstacles
)

# 2. 打开夹爪
print("✋ 打开夹爪...")
gripper_open(robot_id)

# 3. 移动到抓取位置
print("🎯 移动到抓取位置...")
grasp_pos = [0.5, 0.2, 0.5]
success = plan_and_execute_motion(
    robot_id, grasp_pos, pre_grasp_orn, obstacles
)

# 4. 闭合夹爪
print("🤏 闭合夹爪...")
gripper_close(robot_id)

# 5. 提起物体
print("⬆️ 提起物体...")
lift_pos = [0.5, 0.2, 0.7]
success = plan_and_execute_motion(
    robot_id, lift_pos, pre_grasp_orn, obstacles
)

print("✅ 抓取任务完成！")
```

### 示例2：动态避障

```python
import pybullet as p
import numpy as np
from util import *

# 初始化环境
p.connect(p.GUI)
p.setGravity(0, 0, -9.8)

robot_id = p.loadURDF("franka_panda/panda.urdf", useFixedBase=True)
moving_obstacle_id = p.loadURDF("cube.urdf", [0.4, 0, 0.3])

# 目标位置
goal_pos = [0.6, 0.3, 0.4]
goal_orn = p.getQuaternionFromEuler([0, np.pi, 0])

# 主循环：持续感知和规划
for i in range(100):
    # 实时感知障碍物
    perceived_ids = perceive_obstacles_with_rays(
        robot_id, 
        ROBOT_END_EFFECTOR_LINK_ID,
        debug=True
    )
    
    obstacles = list(perceived_ids - {-1})
    
    # 动态规划和执行
    success = plan_and_execute_motion(
        robot_id,
        goal_pos,
        goal_orn,
        obstacle_ids=obstacles,
        interferer_id=moving_obstacle_id  # 指定动态障碍物
    )
    
    if success:
        print(f"✅ 第 {i+1} 次尝试成功！")
        break
    else:
        print(f"⚠️ 第 {i+1} 次尝试失败，重新规划...")
```

### 示例3：自定义路径规划参数

```python
from util import *
import pybullet as p

# 使用自定义 PFM 参数
from path_planning import plan_path_with_pfm

# 规划路径
workspace_path = plan_path_with_pfm(
    start_pos=[0.3, 0, 0.5],
    goal_pos=[0.6, 0.3, 0.4],
    obstacle_ids=[obstacle1_id, obstacle2_id],
    step_size=0.015,        # 更小的步长（更精细）
    max_steps=500,          # 更多的最大步数
    k_att=1.5,              # 更强的吸引力
    k_rep=0.8,              # 更弱的排斥力
    randomize=True          # 启用随机化
)

if workspace_path:
    print(f"✅ 生成了 {len(workspace_path)} 个路径点")
else:
    print("❌ 路径规划失败")
```

## 常见问题

### ❓ Q1: 为什么路径规划总是失败？

**A:** 可能的原因：
1. 障碍物检测范围太小 → 增加 `ray_range` 参数
2. PFM 陷入局部最小值 → 启用 `randomize=True`
3. 目标位置不可达 → 检查是否在工作空间内

```python
# 增加感知范围
perceived_ids = perceive_obstacles_with_rays(
    robot_id, 
    ROBOT_END_EFFECTOR_LINK_ID,
    ray_range=2.0,  # 从 1.5 增加到 2.0
    grid_size=9     # 增加网格密度
)
```

### ❓ Q2: 运动执行超时怎么办？

**A:** 调整超时参数和速度：

```python
from motion_control import move_to_joints

success = move_to_joints(
    robot_id,
    target_joints,
    max_velocity=2.0,  # 增加速度
    timeout=10,        # 增加超时时间
)
```

### ❓ Q3: 如何可视化传感器射线？

**A:** 设置 `debug=True`：

```python
perceived_ids = perceive_obstacles_with_rays(
    robot_id,
    ROBOT_END_EFFECTOR_LINK_ID,
    debug=True  # 显示绿色/红色射线
)
```

### ❓ Q4: 如何只使用特定的探索策略？

**A:** 直接调用探索策略函数：

```python
from exploration import generate_safe_retreat_targets

# 只使用安全撤退策略
safe_positions = generate_safe_retreat_targets()
for pos in safe_positions:
    # 尝试移动到安全位置
    pass
```

### ❓ Q5: 如何添加自定义障碍物？

**A:** 

```python
# 加载自定义障碍物
custom_obstacle = p.loadURDF("my_obstacle.urdf", [0.5, 0, 0.3])

# 手动添加到障碍物列表
obstacle_list = [custom_obstacle, table_id]

# 或者通过感知自动检测
perceived_ids = perceive_obstacles_with_rays(...)
obstacle_list = list(perceived_ids - {-1})  # 自动包含所有检测到的障碍物
```

## 🎓 进阶技巧

### 技巧1：路径平滑

```python
from path_planning import plan_path_with_pfm, validate_workspace_path

# 先规划粗略路径
rough_path = plan_path_with_pfm(
    start_pos, goal_pos, obstacles,
    step_size=0.05  # 大步长
)

# 验证并细化
smooth_path, joint_path = validate_workspace_path(
    rough_path, robot_id, goal_orn, obstacles,
    current_gripper_pos, sampling_step=1  # 更密集的采样
)
```

### 技巧2：多目标点序列

```python
waypoints = [
    [0.3, 0, 0.5],
    [0.4, 0.2, 0.5],
    [0.5, 0.2, 0.4],
    [0.6, 0, 0.3]
]

for i, wp in enumerate(waypoints):
    print(f"移动到路径点 {i+1}/{len(waypoints)}...")
    success = plan_and_execute_motion(
        robot_id, wp, goal_orn, obstacles
    )
    if not success:
        print(f"无法到达路径点 {i+1}，停止执行")
        break
```

### 技巧3：优先尝试简单策略

```python
from collision_detection import is_path_colliding

# 1. 先检查直接路径
current_joints = [p.getJointState(robot_id, i)[0] for i in range(7)]
target_joints = p.calculateInverseKinematics(robot_id, 8, goal_pos, goal_orn)[:7]

if not is_path_colliding(robot_id, current_joints, target_joints, obstacles, ...):
    # 直接路径可行，直接执行
    move_to_joints(robot_id, target_joints)
else:
    # 需要复杂规划
    plan_and_execute_motion(robot_id, goal_pos, goal_orn, obstacles)
```

## 📚 更多资源

- [README.md](README.md) - 项目概述和模块说明
- [ARCHITECTURE.md](ARCHITECTURE.md) - 详细架构设计
- [pybullet_starter.py](pybullet_starter.py) - 完整示例代码

## 💡 提示

- 🔍 使用 `debug=True` 来可视化传感器和路径
- ⚡ 调整 PFM 参数以适应不同场景
- 🔄 利用路径历史避免重复规划
- 📊 监控执行时间和成功率

祝你使用愉快！🎉

