import numpy as np
import torch
from torchtyping import TensorType
from nerfstudio.cameras.cameras import Cameras


def normalize(x):
    """Normalization helper function."""
    return x / torch.linalg.norm(x)


def viewmatrix(lookdir, up, position):
    """Construct lookat view matrix."""
    vec2 = normalize(lookdir)
    vec0 = normalize(torch.cross(up, vec2))
    vec1 = normalize(torch.cross(vec2, vec0))
    m = torch.stack([vec0, vec1, vec2, position], dim=1)
    return m


def project_points_to_image_space(points: TensorType["num_pts":..., 3], camera: Cameras):
    camera_to_world = camera.camera_to_worlds.numpy()
    intrinsic = camera.get_intrinsics_matrices().numpy()
    # 转换为齐次坐标
    points = points.numpy()
    n = points.shape[0]
    points_hom = np.hstack((points, np.ones((n, 1))))
    camera_to_world = np.vstack((camera_to_world, np.array([0.0, 0.0, 0.0, 1.0])))

    # 从相机坐标系转换到世界坐标系
    world_to_camera = np.linalg.inv(camera_to_world)
    points_world_hom = points_hom.dot(world_to_camera.T)
    points_world = points_world_hom[:, :3]
    points_world[:, 1] *= -1
    points_world[:, 2] *= -1
    visibility_front = points_world[:, 2] > 0
    # 使用相机内参将点投影到相机平面
    points_image_hom = points_world.dot(intrinsic.T)

    points_image_hom /= points_image_hom[:, 2:3]

    # 去除齐次坐标
    points_image = points_image_hom[:, :2]

    return points_image, visibility_front


def adjust_cameras_lookat(cameras: Cameras, center: TensorType[3, 1]):
    # 给定相机，重新计算其外参，使得相机始终看向scene center
    # cameras: Cameras
    # scene_center: 场景中心
    # return: new_cameras
    scene_center_ = center.squeeze(-1)
    new_poses = []
    Ks = cameras.get_intrinsics_matrices()
    new_poses = torch.zeros_like(cameras.camera_to_worlds, device=cameras.camera_to_worlds.device)
    for i in range(0, len(cameras)):
        cam_pos = cameras[i].camera_to_worlds[:, -1]
        up = torch.tensor([0.0, 0.0, 1.0], device=scene_center_.device)
        new_pose = viewmatrix(cam_pos - scene_center_, up, cam_pos)
        new_poses[i, :, :] = new_pose
    new_cameras = Cameras(fx=Ks[:, 0, 0], fy=Ks[:, 1, 1], cx=Ks[:, 0, 2], cy=Ks[:, 1, 2], camera_to_worlds=new_poses)

    return new_cameras
