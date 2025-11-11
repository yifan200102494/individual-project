"""
测试深度相机和外参标定
简单的测试脚本，验证深度图像获取是否正常工作
"""

import pybullet as p
import pybullet_data
import numpy as np
from depth_perception import DepthPerceptionSystem
from constants import ROBOT_END_EFFECTOR_LINK_ID

def test_depth_camera():
    """测试深度相机基本功能"""
    print("=" * 60)
    print("测试深度相机和外参标定")
    print("=" * 60)
    
    # 1. 初始化PyBullet
    p.connect(p.GUI)
    p.setGravity(0, 0, -9.81)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)
    p.resetDebugVisualizerCamera(1.5, 45, -30, [0.3, 0.0, 0.3])
    
    # 2. 加载场景
    print("\n[1] 加载场景...")
    plane_id = p.loadURDF('plane.urdf')
    robot_id = p.loadURDF('franka_panda/panda.urdf', useFixedBase=True)
    
    # 加载一些障碍物用于测试
    cube1_id = p.loadURDF('cube_small.urdf', 
                          basePosition=[0.5, 0.0, 0.025],
                          baseOrientation=[0, 0, 0, 1])
    cube2_id = p.loadURDF('cube_small.urdf',
                          basePosition=[0.5, 0.3, 0.025],
                          baseOrientation=[0, 0, 0, 1])
    
    print(f"  机器人ID: {robot_id}")
    print(f"  障碍物ID: {cube1_id}, {cube2_id}")
    
    # 3. 设置机器人到合适的位置（能看到障碍物）
    print("\n[2] 设置机器人姿态...")
    home_config = [0.0, -0.785, 0.0, -2.356, 0.0, 1.57, 0.785]
    for i, angle in enumerate(home_config):
        p.resetJointState(robot_id, i, angle)
    
    # 4. 初始化深度感知系统
    print("\n[3] 初始化深度感知系统...")
    perception_system = DepthPerceptionSystem(
        robot_id=robot_id,
        sensor_link_id=ROBOT_END_EFFECTOR_LINK_ID,
        image_width=128,
        image_height=128
    )
    
    # 5. 测试相机位姿计算（外参标定）
    print("\n[4] 测试外参标定 - 计算相机在世界坐标系中的位姿...")
    camera_pos, camera_orn = perception_system.depth_camera.get_camera_pose_in_world()
    print(f"  相机位置: {camera_pos}")
    print(f"  相机姿态(四元数): {camera_orn}")
    
    # 在场景中可视化相机位置
    p.addUserDebugLine(camera_pos, camera_pos + [0, 0, 0.1], 
                      [1, 0, 0], lineWidth=3, lifeTime=0)
    p.addUserDebugText("CAMERA", camera_pos + [0, 0, 0.05],
                      textColorRGB=[1, 0, 0], textSize=1.5, lifeTime=0)
    
    # 6. 捕获深度图像
    print("\n[5] 捕获深度图像...")
    depth_buffer, rgb_image = perception_system.depth_camera.capture_depth_image()
    print(f"  深度缓冲区形状: {depth_buffer.shape}")
    print(f"  深度缓冲区范围: [{depth_buffer.min():.3f}, {depth_buffer.max():.3f}]")
    print(f"  RGB图像形状: {rgb_image.shape}")
    
    # 7. 转换为实际距离
    depth_distance = perception_system.depth_camera.depth_buffer_to_distance(depth_buffer)
    print(f"  深度距离形状: {depth_distance.shape}")
    print(f"  深度距离范围: [{depth_distance.min():.3f}m, {depth_distance.max():.3f}m]")
    
    # 8. 生成点云
    print("\n[6] 生成3D点云...")
    point_cloud = perception_system.occupancy_detector.depth_to_point_cloud(
        depth_distance, camera_pos, camera_orn
    )
    print(f"  点云形状: {point_cloud.shape}")
    print(f"  点云X范围: [{point_cloud[:, 0].min():.3f}, {point_cloud[:, 0].max():.3f}]")
    print(f"  点云Y范围: [{point_cloud[:, 1].min():.3f}, {point_cloud[:, 1].max():.3f}]")
    print(f"  点云Z范围: [{point_cloud[:, 2].min():.3f}, {point_cloud[:, 2].max():.3f}]")
    
    # 9. 可视化一些点云（采样显示）
    print("\n[7] 可视化点云样本...")
    sample_indices = np.random.choice(point_cloud.shape[0], 
                                     min(100, point_cloud.shape[0]), 
                                     replace=False)
    for idx in sample_indices:
        point = point_cloud[idx]
        # 过滤掉太远的点
        if point[2] > 0.01 and point[2] < 1.5:
            p.addUserDebugLine(point, point + [0, 0, 0.01],
                             [0, 1, 0], lineWidth=1, lifeTime=0)
    
    print("\n[8] 测试完成！")
    print("  检查PyBullet窗口中的可视化效果")
    print("  红色标记: 相机位置")
    print("  绿色点: 点云样本")
    print("  按Enter键关闭...")
    input()
    
    p.disconnect()
    print("\n✅ 测试成功完成！")

if __name__ == "__main__":
    test_depth_camera()

