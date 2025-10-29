# 项目架构说明

## 📐 模块依赖关系图

```
┌─────────────────────────────────────────────────────────┐
│                      util.py                            │
│                   (统一入口层)                           │
│  - 导入并重新导出所有模块                                 │
│  - 提供向后兼容性                                        │
└──────────────────┬──────────────────────────────────────┘
                   │
                   │ (导入所有模块)
                   ↓
    ┌──────────────────────────────────────────────┐
    │                                              │
    ↓                                              ↓
┌─────────────────┐                      ┌─────────────────┐
│  planner.py     │                      │ constants.py    │
│  (高层规划器)   │←─────────────────────│ (常量定义)      │
│                 │                      └─────────────────┘
│ - 整合所有模块  │                               ↑
│ - 主规划逻辑    │                               │
└────────┬────────┘                               │
         │                                        │
         │ (使用)                                 │
         ↓                                        │
    ┌────────────────────────────────────────┐   │
    │                                        │   │
    ↓                ↓              ↓        ↓   │
┌──────────┐  ┌──────────────┐  ┌──────────────┐│
│path_     │  │exploration.py│  │motion_       ││
│planning  │  │(随机探索)    │  │control.py    ││
│.py       │  │              │  │(运动控制)    ││
│          │  │- 工作空间探索│  │              ││
│- PFM     │  │- 避障探索    │  │- simulate()  ││
│- 弧形路径│  │- 多层级探索  │  │- 关节运动    ││
│- 路径验证│  │- 螺旋探索    │  │- 夹爪控制    ││
└────┬─────┘  └──────┬───────┘  └──────┬───────┘│
     │               │                  │        │
     │               │                  │        │
     └───────┬───────┴──────────────────┘        │
             │                                   │
             │ (依赖)                            │
             ↓                                   │
    ┌─────────────────────────┐                 │
    │                         │                 │
    ↓                         ↓                 │
┌──────────────────┐  ┌──────────────┐         │
│collision_        │  │perception.py │         │
│detection.py      │  │(传感器感知)  │─────────┘
│(碰撞检测)        │  │              │
│                  │  │- 多向射线    │
│- 状态碰撞        │  │- 障碍物检测  │
│- 路径碰撞        │  └──────────────┘
└──────────────────┘
         ↑
         │
         │ (依赖)
         │
    ┌────┴─────┐
    │constants │
    │.py       │
    └──────────┘
```

## 🔄 数据流图

```
用户代码
   ↓
util.py (统一入口)
   ↓
planner.plan_and_execute_motion()
   ↓
┌──────────────────────────────────────────┐
│ 1. 获取当前状态                          │
│ 2. 检查直接路径 (collision_detection)    │
│    ├─ 成功 → 执行 (motion_control)       │
│    └─ 失败 ↓                             │
│                                          │
│ 3. PFM 路径规划 (path_planning)          │
│    ├─ 成功 → 验证 → 执行                 │
│    └─ 失败 ↓                             │
│                                          │
│ 4. Plan B 绕行策略 (path_planning)       │
│    ├─ 成功 → 执行                        │
│    └─ 失败 ↓                             │
│                                          │
│ 5. 随机探索 (exploration)                │
│    ├─ 成功 → 从新位置重新规划            │
│    └─ 失败 → 返回失败                    │
└──────────────────────────────────────────┘
```

## 🧩 模块职责矩阵

| 模块 | 主要职责 | 对外接口 | 依赖模块 |
|------|---------|---------|---------|
| **constants.py** | 定义全局常量 | 常量值 | 无 |
| **collision_detection.py** | 碰撞检测 | `is_state_colliding()`<br>`is_path_colliding()` | constants |
| **perception.py** | 传感器感知 | `perceive_obstacles_with_rays()` | constants |
| **path_planning.py** | 路径规划算法 | `plan_path_with_pfm()`<br>`generate_arc_path()`<br>`validate_workspace_path()` | constants<br>collision_detection |
| **exploration.py** | 随机探索策略 | `perform_random_exploration()` | constants<br>collision_detection<br>motion_control |
| **motion_control.py** | 运动执行 | `simulate()`<br>`move_to_joints()`<br>`gripper_*()` | constants<br>collision_detection |
| **planner.py** | 高层规划逻辑 | `plan_and_execute_motion()` | 所有模块 |
| **util.py** | 统一入口 | 重新导出所有接口 | 所有模块 |

## 🎯 设计原则

### 1. 单一职责原则 (SRP)
每个模块只负责一个功能领域：
- `collision_detection.py` 只处理碰撞检测
- `perception.py` 只处理传感器感知
- `motion_control.py` 只处理运动执行

### 2. 依赖倒置原则 (DIP)
- 高层模块（`planner.py`）不依赖低层模块的具体实现
- 通过清晰的函数接口进行交互

### 3. 开闭原则 (OCP)
- 对扩展开放：可以轻松添加新的探索策略、路径规划算法
- 对修改封闭：修改某个策略不影响其他模块

### 4. 接口隔离原则 (ISP)
- 每个模块只暴露必要的接口
- 用户可以只导入需要的功能

## 🔧 扩展指南

### 添加新的路径规划算法

1. 在 `path_planning.py` 中添加新函数：
```python
def plan_path_with_rrt(start_pos, goal_pos, obstacle_ids, **kwargs):
    # 实现 RRT 算法
    pass
```

2. 在 `planner.py` 中集成：
```python
# 在 plan_and_execute_motion() 中添加
workspace_path = plan_path_with_rrt(...)
```

3. 在 `util.py` 中导出（如果需要外部访问）：
```python
from path_planning import plan_path_with_rrt
```

### 添加新的探索策略

1. 在 `exploration.py` 中添加生成函数：
```python
def generate_custom_targets(current_pos):
    # 生成新的探索目标
    pass
```

2. 在 `perform_random_exploration()` 中调用：
```python
exploration_candidates.extend(generate_custom_targets(current_pos))
```

### 添加新的传感器类型

1. 在 `perception.py` 中添加新函数：
```python
def perceive_with_lidar(robot_id, sensor_link_id, **kwargs):
    # 实现激光雷达感知
    pass
```

2. 在需要的地方调用新传感器

## 📦 导入建议

### 场景1：完整功能开发
```python
# 通过 util.py 导入所有功能
from util import *
```

### 场景2：只需要特定功能
```python
# 只导入需要的模块
from perception import perceive_obstacles_with_rays
from planner import plan_and_execute_motion
```

### 场景3：开发新算法
```python
# 直接导入底层模块
from path_planning import prepare_obstacles_info, calc_attractive_force
from collision_detection import is_path_colliding
```

## 🐛 调试建议

### 问题：路径规划失败
1. 检查 `perception.py` - 障碍物是否正确检测
2. 检查 `path_planning.py` - PFM 参数是否合适
3. 检查 `collision_detection.py` - 碰撞检测是否过于严格

### 问题：运动执行超时
1. 检查 `motion_control.py` - 速度限制是否合理
2. 检查 `collision_detection.py` - 实时碰撞检测逻辑
3. 检查 `constants.py` - 超时参数设置

### 问题：循环导入错误
- 当前架构已避免循环导入
- 如果出现，检查是否在函数内部导入而不是文件顶部

## 🚀 性能优化建议

1. **感知模块**：减少射线数量或采样密度
2. **路径规划**：调整 PFM 步长和最大步数
3. **碰撞检测**：减少插值步数
4. **探索模块**：减少候选目标数量

## 📊 代码质量指标

- **模块内聚性**：高 ✅
- **模块耦合度**：低 ✅
- **代码复用率**：高 ✅
- **可测试性**：高 ✅
- **可维护性**：高 ✅
- **可扩展性**：高 ✅

