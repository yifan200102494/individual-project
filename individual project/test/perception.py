"""
感知模块
使用多方向射线传感器检测障碍物
"""

import pybullet as p
import numpy as np


def perceive_obstacles_with_rays(robot_id, sensor_link_id, 
                                 ray_range=1.5, grid_size=7, fov_width=0.8, 
                                 debug=False):
    """
    使用多方向射线检测障碍物
    
    Args:
        robot_id: 机器人ID
        sensor_link_id: 传感器所在连杆ID
        ray_range: 射线范围（米）
        grid_size: 网格大小
        fov_width: 视场宽度
        debug: 是否显示调试射线
    
    Returns:
        set: 检测到的物体ID集合
    """
    
    # 获取传感器位姿
    try:
        link_state = p.getLinkState(robot_id, sensor_link_id, computeForwardKinematics=True)
    except Exception as e:
        print(f"  [感知错误] 无法获取 link state: {e}")
        return set()
    
    sensor_pos_world = np.array(link_state[0])
    sensor_orn_world = np.array(link_state[1])
    sensor_rot_matrix = np.array(p.getMatrixFromQuaternion(sensor_orn_world)).reshape(3, 3)

    # 定义传感器方向：6个轴向 + 4个对角线方向
    sensor_directions = [
        (2, 1.0),   # 向下 (+Z)
        (0, 1.0),   # 向前 (+X)
        (0, -1.0),  # 向后 (-X)
        (1, 1.0),   # 向左 (+Y)
        (1, -1.0),  # 向右 (-Y)
        (2, -1.0),  # 向上 (-Z)
        ('diagonal', [1.0, 1.0, 0.0]),    # 前左对角
        ('diagonal', [1.0, -1.0, 0.0]),   # 前右对角
        ('diagonal', [-1.0, 1.0, 0.0]),   # 后左对角
        ('diagonal', [-1.0, -1.0, 0.0])   # 后右对角
    ]
    
    # 生成射线
    ray_froms_world, ray_tos_world = _generate_rays(
        sensor_pos_world, sensor_rot_matrix, sensor_directions,
        ray_range, grid_size, fov_width
    )
    
    # 执行批量射线检测
    results = p.rayTestBatch(ray_froms_world, ray_tos_world)
    
    # 处理结果
    perceived_object_ids = set()
    
    if debug:
        p.removeAllUserDebugItems()
    
    for i, res in enumerate(results):
        hit_id = res[0]
        perceived_object_ids.add(hit_id)
        
        if debug:
            hit_pos = res[3]
            from_pos = ray_froms_world[i]
            to_pos = ray_tos_world[i]
            
            if hit_id == -1:
                p.addUserDebugLine(from_pos, to_pos, [0.0, 1.0, 0.0], lifeTime=0)
            else:
                p.addUserDebugLine(from_pos, hit_pos, [1.0, 0.0, 0.0], lifeTime=0)
    
    return perceived_object_ids


def _generate_rays(sensor_pos, rot_matrix, directions, ray_range, grid_size, fov_width):
    """
    生成射线起点和终点
    
    Returns:
        (ray_froms, ray_tos): 射线起点和终点列表
    """
    ray_froms_world = []
    ray_tos_world = []
    
    grid_coords_1 = np.linspace(-fov_width, fov_width, grid_size)
    grid_coords_2 = np.linspace(-fov_width, fov_width, grid_size)
    start_offset = 0.01
    
    for sensor_dir in directions:
        if sensor_dir[0] == 'diagonal':
            # 对角线方向
            direction_vec = np.array(sensor_dir[1])
            direction_vec = direction_vec / np.linalg.norm(direction_vec)
            
            for u_grid in grid_coords_1:
                for v_grid in grid_coords_2:
                    ray_from_local = direction_vec * start_offset
                    ray_to_local = direction_vec * ray_range
                    
                    # 添加垂直扩散
                    perpendicular_1 = np.array([-direction_vec[1], direction_vec[0], 0])
                    perpendicular_2 = np.array([0, 0, 1])
                    ray_to_local += perpendicular_1 * u_grid * 0.5
                    ray_to_local += perpendicular_2 * v_grid * 0.5
                    
                    ray_from_world = sensor_pos + rot_matrix.dot(ray_from_local)
                    ray_to_world = sensor_pos + rot_matrix.dot(ray_to_local)
                    
                    ray_froms_world.append(ray_from_world)
                    ray_tos_world.append(ray_to_world)
        else:
            # 轴向方向
            axis_idx, direction = sensor_dir
            grid_axis_1 = (axis_idx + 1) % 3
            grid_axis_2 = (axis_idx + 2) % 3
            
            for u_grid in grid_coords_1:
                for v_grid in grid_coords_2:
                    ray_from_local = np.array([0.0, 0.0, 0.0])
                    ray_from_local[axis_idx] = direction * start_offset
                    
                    ray_to_local = np.array([0.0, 0.0, 0.0])
                    ray_to_local[axis_idx] = direction * ray_range
                    ray_to_local[grid_axis_1] = u_grid
                    ray_to_local[grid_axis_2] = v_grid
                    
                    ray_from_world = sensor_pos + rot_matrix.dot(ray_from_local)
                    ray_to_world = sensor_pos + rot_matrix.dot(ray_to_local)
                    
                    ray_froms_world.append(ray_from_world)
                    ray_tos_world.append(ray_to_world)
    
    return ray_froms_world, ray_tos_world

