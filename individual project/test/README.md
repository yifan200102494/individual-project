# 机器人路径规划与控制库

## 📁 项目结构

```
test/
├── constants.py              # 常量定义
├── collision_detection.py    # 碰撞检测模块
├── perception.py             # 传感器感知模块
├── path_planning.py          # 路径规划模块（PFM、弧形路径等）
├── exploration.py            # 随机探索模块
├── motion_control.py         # 运动控制模块（仿真、关节运动、夹爪）
├── planner.py                # 高层规划器
├── util.py                   # 统一入口（向后兼容）
├── pybullet_starter.py       # 主程序
├── environment.py            # 环境配置
└── README.md                 # 本文档
```

## 🔧 模块说明

### 1. **constants.py** - 常量定义
- 机器人配置参数（HOME位置、末端执行器ID等）
- 仿真参数（时间步长等）
- 工作空间限制
- 安全参数

### 2. **collision_detection.py** - 碰撞检测
- `is_state_colliding()`: 检查给定状态是否碰撞
- `is_path_colliding()`: 检查关节空间路径是否碰撞

### 3. **perception.py** - 传感器感知
- `perceive_obstacles_with_rays()`: 使用多方向射线检测障碍物
- 支持6个轴向 + 4个对角线方向的全方位感知

### 4. **path_planning.py** - 路径规划
- **势场法 (PFM)**:
  - `plan_path_with_pfm()`: 主规划函数
  - `calc_attractive_force()`: 计算吸引力
  - `calc_anisotropic_repulsive_force()`: 计算各向异性排斥力
  
- **弧形路径**:
  - `generate_arc_path()`: 生成弧形路径
  
- **路径验证**:
  - `validate_workspace_path()`: 验证工作空间路径在关节空间中是否可行
  
- **绕行策略**:
  - `generate_detour_strategies()`: 生成多种绕行策略
  
- **路径历史**:
  - `add_path_to_history()`: 添加路径到历史记录
  - `is_path_similar_to_history()`: 检查路径是否与历史相似

### 5. **exploration.py** - 随机探索
- `perform_random_exploration()`: 主探索函数
- **多种探索策略**:
  - 大范围工作空间采样
  - 远离障碍物策略
  - 多层级高度探索
  - 安全撤退位置
  - 螺旋式探索
  - 关节空间随机移动

### 6. **motion_control.py** - 运动控制
- **仿真**:
  - `simulate()`: 执行仿真步进
  
- **关节运动**:
  - `move_to_joints()`: 移动到目标关节位置（含碰撞检测）
  - `move_to_pose()`: 移动到目标末端执行器位姿
  
- **夹爪控制**:
  - `gripper_open()`: 打开夹爪
  - `gripper_close()`: 闭合夹爪

### 7. **planner.py** - 高层规划器
- `plan_and_execute_motion()`: 整合所有模块的主规划函数
- **规划流程**:
  1. 尝试直接路径
  2. 使用 PFM 路径规划
  3. 尝试绕行策略 (Plan B)
  4. 随机探索 + 重新规划

### 8. **util.py** - 统一入口
- 导入并重新导出所有模块的功能
- 保持向后兼容性
- 可以继续使用 `from util import *`

## 📦 使用方法

### 方式1：通过 util.py 统一入口（推荐）
```python
from util import (
    perceive_obstacles_with_rays,
    plan_and_execute_motion,
    move_to_joints,
    gripper_open,
    gripper_close
)
```

### 方式2：直接导入特定模块
```python
from perception import perceive_obstacles_with_rays
from planner import plan_and_execute_motion
from motion_control import move_to_joints, gripper_open, gripper_close
```

## 🎯 示例代码

```python
import pybullet as p
from util import (
    ROBOT_END_EFFECTOR_LINK_ID,
    perceive_obstacles_with_rays,
    plan_and_execute_motion,
    gripper_open,
    gripper_close
)

# 1. 感知障碍物
perceived_ids = perceive_obstacles_with_rays(
    robot_id, 
    ROBOT_END_EFFECTOR_LINK_ID,
    debug=True
)

# 2. 规划并执行运动
success = plan_and_execute_motion(
    robot_id,
    goal_pos=[0.5, 0.2, 0.3],
    goal_orn=p.getQuaternionFromEuler([0, np.pi, 0]),
    obstacle_ids=list(perceived_ids - {-1}),
    interferer_id=interferer_id
)

# 3. 控制夹爪
gripper_open(robot_id)
gripper_close(robot_id)
```

## ✨ 优势

### 模块化设计
- ✅ 每个模块职责单一、清晰
- ✅ 易于理解和维护
- ✅ 方便单元测试
- ✅ 便于扩展新功能

### 代码复用
- ✅ 消除了重复代码
- ✅ 统一的接口设计
- ✅ 更好的代码组织

### 维护性
- ✅ 修改某个功能只需要修改对应模块
- ✅ 减少了模块间的耦合
- ✅ 更容易定位和修复bug

## 📊 代码统计

| 模块 | 行数 | 主要功能 |
|------|------|----------|
| constants.py | ~40 | 常量定义 |
| collision_detection.py | ~90 | 碰撞检测 |
| perception.py | ~165 | 传感器感知 |
| path_planning.py | ~450 | 路径规划 |
| exploration.py | ~270 | 随机探索 |
| motion_control.py | ~200 | 运动控制 |
| planner.py | ~280 | 高层规划 |
| util.py | ~140 | 统一入口 |
| **总计** | **~1635** | **原1268行，重构后更模块化** |

## 🔄 版本历史

- **v2.0.0** (当前版本): 完全模块化重构
- **v1.0.0**: 单文件版本 (util.py, 1268行)

## 📝 注意事项

1. 所有模块都需要安装 `pybullet` 和 `numpy`
2. `util.py` 保持向后兼容，原有代码无需修改
3. 推荐使用模块化导入方式以获得更好的IDE支持
4. 各模块之间的依赖关系已经优化，避免循环导入

## 🚀 未来改进方向

- [ ] 添加单元测试
- [ ] 添加性能分析工具
- [ ] 支持更多路径规划算法（RRT、RRT*等）
- [ ] 添加可视化工具
- [ ] 支持配置文件

