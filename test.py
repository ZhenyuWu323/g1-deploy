import torch
import numpy as np
from scipy.spatial.transform import Rotation

def apriltag_to_isaaclab_format(pose_matrix, flip_coordinate_system=True):
    """
    将AprilTag检测的pose转换为Isaac Lab仿真格式
    
    AprilTag检测返回的是tag在camera frame中的pose，
    这正好对应仿真中的 object_position_in_camera_frame 和 object_orientation_in_camera_frame
    
    Args:
        pose_matrix: 4x4 AprilTag pose matrix from detection_pose()
        flip_coordinate_system: 是否需要坐标系转换
    
    Returns:
        position: 3D position tensor 对应 object_position_in_camera_frame()
        quaternion: quaternion tensor 对应 object_orientation_in_camera_frame()
    """
    
    # 提取位置和旋转
    position = pose_matrix[:3, 3]  # [x, y, z]
    rotation_matrix = pose_matrix[:3, :3]  # 3x3 rotation matrix
    
    if flip_coordinate_system:
        # Isaac Lab通常使用右手坐标系，但具体的轴向定义可能不同
        # 常见的OpenCV到机器人学坐标系转换
        
        # 方案1: 标准转换 (最常见)
        # OpenCV: X-right, Y-down, Z-forward
        # Robotics: X-forward, Y-left, Z-up
        transform = np.array([
            [0,  0,  1],   # camera Z -> robot X (前)
            [-1, 0,  0],   # camera -X -> robot Y (左)  
            [0, -1,  0]    # camera -Y -> robot Z (上)
        ])
        
        # 方案2: 如果只需要Y轴翻转 (uncomment if needed)
        # transform = np.array([
        #     [1,  0,  0],
        #     [0, -1,  0],
        #     [0,  0,  1]
        # ])
        
        # 方案3: 如果只需要Z轴翻转 (uncomment if needed)
        # transform = np.array([
        #     [1,  0,  0],
        #     [0,  1,  0],
        #     [0,  0, -1]
        # ])
        
        # 应用变换
        position = transform @ position
        rotation_matrix = transform @ rotation_matrix @ transform.T
    
    # 转换为四元数
    r = Rotation.from_matrix(rotation_matrix)
    quaternion = r.as_quat()  # [x, y, z, w] format
    
    # Isaac Lab通常使用 [w, x, y, z] 格式
    quaternion_wxyz = np.array([quaternion[3], quaternion[0], quaternion[1], quaternion[2]])
    
    return position, quaternion_wxyz

def create_observation_from_apriltags(detections, detector, camera_params, tag_size, 
                                    target_tag_id=None, device='cpu'):
    """
    从AprilTag检测结果创建Isaac Lab格式的观察数据
    
    Args:
        detections: AprilTag检测结果
        detector: AprilTag检测器
        camera_params: 相机参数
        tag_size: 标签尺寸
        target_tag_id: 目标标签ID（如果只关心特定标签）
        device: PyTorch设备
    
    Returns:
        position_tensor: torch.Tensor shape (3,) 或 (N, 3)
        orientation_tensor: torch.Tensor shape (4,) 或 (N, 4) 
        valid_detection: bool 是否有有效检测
    """
    
    positions = []
    orientations = []
    
    for detection in detections:
        # 如果指定了目标ID，只处理该ID
        if target_tag_id is not None and detection.tag_id != target_tag_id:
            continue
            
        try:
            # 获取pose
            pose_matrix, init_error, final_error = detector.detection_pose(
                detection=detection,
                camera_params=camera_params,
                tag_size=tag_size
            )
            
            # 转换为Isaac Lab格式
            position, quaternion = apriltag_to_isaaclab_format(pose_matrix)
            
            positions.append(position)
            orientations.append(quaternion)
            
            print(f"Tag {detection.tag_id}:")
            print(f"  Position in camera frame: [{position[0]:.3f}, {position[1]:.3f}, {position[2]:.3f}]")
            print(f"  Quaternion (w,x,y,z): [{quaternion[0]:.3f}, {quaternion[1]:.3f}, {quaternion[2]:.3f}, {quaternion[3]:.3f}]")
            print(f"  Pose errors: init={init_error:.6f}, final={final_error:.6f}")
            
        except Exception as e:
            print(f"Error processing tag {detection.tag_id}: {e}")
            continue
    
    if positions:
        position_tensor = torch.tensor(np.array(positions), device=device, dtype=torch.float32)
        orientation_tensor = torch.tensor(np.array(orientations), device=device, dtype=torch.float32)
        
        # 如果只有一个检测结果，返回单个tensor而不是batch
        if len(positions) == 1:
            position_tensor = position_tensor.squeeze(0)  # (3,)
            orientation_tensor = orientation_tensor.squeeze(0)  # (4,)
        
        return position_tensor, orientation_tensor, True
    else:
        # 没有检测到，返回零张量
        if target_tag_id is not None:
            return (torch.zeros(3, device=device, dtype=torch.float32),
                   torch.tensor([1., 0., 0., 0.], device=device, dtype=torch.float32),  # identity quaternion
                   False)
        else:
            return (torch.empty(0, 3, device=device, dtype=torch.float32),
                   torch.empty(0, 4, device=device, dtype=torch.float32),
                   False)

def debug_compare_with_sim(apriltag_position, apriltag_quaternion, sim_position=None, sim_quaternion=None):
    """
    调试函数：比较AprilTag检测结果与仿真数据
    """
    print("=== AprilTag vs Simulation Comparison ===")
    print(f"AprilTag Position: [{apriltag_position[0]:.3f}, {apriltag_position[1]:.3f}, {apriltag_position[2]:.3f}]")
    print(f"AprilTag Quaternion: [{apriltag_quaternion[0]:.3f}, {apriltag_quaternion[1]:.3f}, {apriltag_quaternion[2]:.3f}, {apriltag_quaternion[3]:.3f}]")
    
    if sim_position is not None:
        print(f"Sim Position: [{sim_position[0]:.3f}, {sim_position[1]:.3f}, {sim_position[2]:.3f}]")
        pos_diff = np.linalg.norm(apriltag_position - sim_position)
        print(f"Position difference: {pos_diff:.3f}")
    
    if sim_quaternion is not None:
        print(f"Sim Quaternion: [{sim_quaternion[0]:.3f}, {sim_quaternion[1]:.3f}, {sim_quaternion[2]:.3f}, {sim_quaternion[3]:.3f}]")
        
        # 计算角度差异
        r1 = Rotation.from_quat([apriltag_quaternion[1], apriltag_quaternion[2], apriltag_quaternion[3], apriltag_quaternion[0]])  # xyzw
        r2 = Rotation.from_quat([sim_quaternion[1], sim_quaternion[2], sim_quaternion[3], sim_quaternion[0]])  # xyzw
        angle_diff = (r1.inv() * r2).magnitude() * 180 / np.pi
        print(f"Orientation difference: {angle_diff:.1f} degrees")
    
    print("=" * 45)

# 在主循环中的使用示例:
def process_apriltags_for_policy(detections, detector, camera_params, tag_size, target_tag_id=0):
    """
    处理AprilTag检测结果，输出policy可用的观察数据
    
    这个函数模拟你的仿真中的：
    - object_position_in_camera_frame()
    - object_orientation_in_camera_frame()
    """
    
    position, orientation, valid = create_observation_from_apriltags(
        detections, detector, camera_params, tag_size, 
        target_tag_id=target_tag_id
    )
    
    if valid:
        # 这里的position和orientation格式与仿真中的obs完全一致
        # 可以直接传入policy
        
        # 如果需要添加batch维度（模拟多环境）
        if len(position.shape) == 1:  # single detection
            position = position.unsqueeze(0)  # (1, 3)
            orientation = orientation.unsqueeze(0)  # (1, 4)
        
        return {
            'object_position_in_camera_frame': position,
            'object_orientation_in_camera_frame': orientation,
            'detection_valid': True
        }
    else:
        return {
            'object_position_in_camera_frame': torch.zeros(1, 3),
            'object_orientation_in_camera_frame': torch.tensor([[1., 0., 0., 0.]]),  # identity quat
            'detection_valid': False
        }

# 坐标系验证函数
def verify_coordinate_system():
    """
    帮助验证坐标系转换是否正确的函数
    在已知位置放置AprilTag然后验证检测结果
    """
    print("=== Coordinate System Verification ===")
    print("Place AprilTag at known positions and verify:")
    print("1. Tag at camera front, 1m away: expect position ≈ [0, 0, 1] or [1, 0, 0]")
    print("2. Tag to camera right, 0.5m: expect position ≈ [0.5, 0, 0] or [0, -0.5, 0]")
    print("3. Tag above camera, 0.3m: expect position ≈ [0, -0.3, 0] or [0, 0, 0.3]")
    print("Compare with simulation results to determine correct transform")
    print("=" * 45)