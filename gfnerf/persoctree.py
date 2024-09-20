"""Perspective Octree"""

import torch
from torch import nn
import torch.nn.functional as F
import numpy as np

from nerfstudio.cameras.cameras import Cameras
from nerfstudio.cameras.rays import RaySamples
from nerfstudio.data.scene_box import SceneBox

from torchtyping import TensorType
from nerfstudio.utils.tensor_dataclass import TensorDataclass
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Union
import gfnerf.utils as utils

N_PROS = 12  


@dataclass(init=False)
class TransInfo(TensorDataclass):
    w2xz: TensorType[N_PROS, 2, 4]  # world to image space
    weight: TensorType[3, N_PROS]
    center: TensorType[3, 1]
    dis_summary: float
    def __init__(self,w2xz=None,weight=None,center=None,dis_summary=None) -> None:
        super().__init__()
        self.w2xz = w2xz
        self.weight = weight
        self.center = center
        self.dis_summary = dis_summary


@dataclass(init=False)
class TreeNode(TensorDataclass):
    center: torch.Tensor
    side_len: float
    parent: int = -1
    childs: List[int] = field(default_factory=list)
    is_leaf_node: bool
    pers_trans: TransInfo
    visi_cams: int
    def __init__(self,center=None,side_len=None,parent=None,childs=None,is_leaf_node=None,pers_trans=None,visi_cams=None) -> None:
        super().__init__()
        self.center = center
        self.side_len = side_len
        self.parent = parent
        self.childs = childs
        self.is_leaf_node = is_leaf_node
        self.pers_trans = pers_trans
        self.visi_cams = visi_cams


class PersOctree:
    """Perspective Octree"""


    def __init__(
        self,
        cameras: Cameras,
        bbox_side_len: float,
        bounds: TensorType["num_cameras":..., 2],
        max_depth=8,
        split_dist_thres=1.5,
    ) -> None:
        print("PersOctree::PersOctree")

        # training camera tranforms
        self.cameras = cameras
        self.device = self.cameras.camera_to_worlds.device

        # params for octree
        self.tree_nodes_ = []
        self.max_depth = max_depth
        self.split_dist_thres = split_dist_thres
        self.bounds_ = bounds
        # construct octree
        root = TreeNode()
        root.parent = -1
        self.tree_nodes_.append(TreeNode)
        self.construct_tree_node(0, 0, torch.zeros((3, 1), device=self.device), side_len=bbox_side_len)
        self.vis_octree("octree.obj")
        self.vis_cam_pos("cam_pos.obj")

    def construct_trans(
        self, visible_cameras: Cameras, rand_pts: TensorType["num_pts":..., 3], center: TensorType[3, 1]
    ) -> TransInfo:
        n_virt_cams = N_PROS // 2
        n_cur_cams = len(visible_cameras)
        n_pts = rand_pts.shape[0]
        intri = visible_cameras.get_intrinsics_matrices()[0]

        # align distance for camera, see supplemental material of f2-nerf
        cam_pos = visible_cameras.camera_to_worlds[:, :, -1]
        dis = torch.linalg.norm(cam_pos - center.permute(1, 0), ord=2, dim=-1, keepdim=True)
        dis_summary = self.distance_summary(dis)
        cam_scale = (dis / dis_summary).clip(1.0, 1e9)
        orig_cam_axis = torch.linalg.inv(visible_cameras.camera_to_worlds[:, 0:3, 0:3]).contiguous()
        normed_cam_pos = (cam_pos - center.permute(1, 0)) / dis
        rel_cam_pos = (cam_pos - center.permute(1, 0)) / dis * dis.clip(dis_summary, 1e9)

        # select cameras using farest points sampling
        dis_pairs = torch.linalg.norm(normed_cam_pos.unsqueeze(0) - normed_cam_pos.unsqueeze(1), dim=-1)
        good_cams = []
        cam_marks = torch.zeros(n_cur_cams, dtype=torch.int32)
        good_cams.append(torch.randint(n_cur_cams, (1,), dtype=torch.int32).item())
        cam_marks[good_cams[0]] = 1
        for cnt in range(1, n_virt_cams if n_virt_cams < n_cur_cams else n_cur_cams):
            candi = -1
            max_dis = -1.0
            for i in range(n_cur_cams):
                if cam_marks[i]:
                    continue
                cur_dis = 1e8
                for j in range(n_cur_cams):
                    if cam_marks[j]:
                        cur_dis = min(cur_dis, dis_pairs[i, j])
                if cur_dis > max_dis:
                    max_dis = cur_dis
                    candi = i
            assert candi >= 0
            cam_marks[candi] = 1
            good_cams.append(candi)
        good_cam_indices = torch.tensor(good_cams, dtype=torch.int64) 

        # align cameras to look at the center

        visible_cameras_lookat = utils.adjust_cameras_lookat(visible_cameras, center)
        # project points to image space
        good_cameras = visible_cameras_lookat[good_cam_indices]
        good_cam_scale = torch.ones((n_virt_cams,), device=self.device)
        good_cam_pos = rel_cam_pos[good_cam_indices] + center.permute(1, 0)
        good_rel_cam_pos = (
            (good_cam_pos - center.permute(1, 0)) / dis[good_cam_indices] * dis[good_cam_indices].clip(dis_summary, 1e9)
        )
        good_cam_scale = cam_scale[good_cam_indices]
        good_cam_axis = torch.linalg.inv(good_cameras.camera_to_worlds[:, 0:3, 0:3]).contiguous()
        expect_z_axis = good_rel_cam_pos / torch.linalg.norm(good_rel_cam_pos, 2, -1, True)

        x_axis = good_cam_axis[:, 0, :]
        y_axis = good_cam_axis[:, 1, :]
        z_axis = good_cam_axis[:, 2, :]
        diff = z_axis - expect_z_axis
        assert diff.abs().max().item() <= 1e-3
        focal = intri[0][0] / intri[0][2] 
        x_axis *= focal
        y_axis *= focal
        x_axis *= good_cam_scale
        y_axis *= good_cam_scale

        x_axis = torch.cat((x_axis, y_axis), dim=0)  # (12,3)
        z_axis = torch.cat((z_axis, z_axis), dim=0)  # (12,3)

        wp_cam_pos = torch.cat((good_cam_pos, good_cam_pos), dim=0)  # (12,3)
        frame_trans = torch.zeros((N_PROS, 2, 4), device=self.device)
        frame_trans[:, 0, 0:3] = x_axis
        frame_trans[:, 1, 0:3] = z_axis
        frame_trans[:, 0, 3] = -(x_axis * wp_cam_pos).sum(-1)
        frame_trans[:, 1, 3] = -(z_axis * wp_cam_pos).sum(-1)
        # (1,12,2,3) *(n,1,3,1)
        transed_pts = torch.matmul(
            frame_trans[:, :, 0:3].unsqueeze(0), rand_pts.unsqueeze(-1).unsqueeze(-1).permute(0, 2, 1, 3)
        )  # (n,12,2,1)
        transed_pts = transed_pts[..., 0] + frame_trans.unsqueeze(0)[:, :, :, 3]  # (n,12,2)
        dv_da = 1.0 / transed_pts[:, :, 1]  # (n,12)
        dv_db = transed_pts[:, :, 0] / -transed_pts[:, :, 1].square()  # (n,12)
        dv_dab = torch.stack((dv_da, dv_db), -1)  # (n,12,2)
        dab_dxyz = frame_trans[:, :, 0:3].unsqueeze(0).clone()  # (1,12,2,3)
        dv_dxyz = torch.matmul(dv_dab.unsqueeze(2), dab_dxyz)[:, :, 0, :]  # (n,12,3)

        assert torch.max(transed_pts[:, :, 1] < 0)
        transed_pts = (
            transed_pts[:, :, 0] / transed_pts[:, :, 1]
        )  
        assert torch.sum(torch.isnan(transed_pts)) == 0
        L, V = self.PCA(transed_pts)
        V = V.permute(1, 0)[0:3].contiguous()  
        jac = torch.matmul(V.unsqueeze(0), dv_dxyz)  # （1,3，12） * （n,12,3） = (n,3,3)
        jac_warp2world = torch.linalg.inv(jac)
        jac_warp2image = torch.matmul(dv_dxyz, jac_warp2world)

        jac_abs = jac_warp2image.abs()
        jac_max, max_tmp = torch.max(jac_abs, 1)
        exp_step = 1.0 / jac_max
        mean_step = exp_step.mean(0)
        V /= mean_step.unsqueeze(-1)

        assert torch.sum(torch.isnan(V)) == 0
        assert torch.sum(torch.isnan(frame_trans)) == 0
        ret = TransInfo()
        ret.center = center
        ret.w2xz = frame_trans.clone()
        ret.weight = V.clone()
        ret.dis_summary = dis_summary
        return ret

    def PCA(self, pts):
        mean = torch.mean(pts, dim=0, keepdim=True)
        moved = pts - mean
        cov = torch.matmul(moved.unsqueeze(-1), moved.unsqueeze(1))
        cov = torch.mean(cov, dim=0)
        L, V = torch.linalg.eigh(cov)  
        L = L.to(torch.float32)
        V = V.to(torch.float32)
        L_sorted, indices = torch.sort(L, dim=0, descending=True)
        V = V.permute(1, 0).contiguous().index_select(0, indices).permute(1, 0).contiguous()
        L = torch.index_select(L, 0, indices).contiguous()
        return L, V

    def construct_tree_node(self, u: int, depth: int, center: TensorType[3, 1], side_len: float) -> None:
        assert u < len(self.tree_nodes_)

        # initialize
        print("depth:", depth)
        self.tree_nodes_[u].center = center
        self.tree_nodes_[u].side_len = side_len
        self.tree_nodes_[u].is_leaf_node = False
        self.tree_nodes_[u].pers_trans = None
        self.tree_nodes_[u].childs = [-1, -1, -1, -1, -1, -1, -1, -1]
        n_rand_pts = 32 * 32 * 32
        rand_pts = (torch.rand((n_rand_pts, 3)).to(self.device) - 0.5) * side_len + center.permute(1, 0)
        # achieve max depth and stop subdivision
        if depth > self.max_depth:
            self.tree_nodes_[u].is_leaf_node = True
            self.tree_nodes_[u].pers_trans = None
            return
        # calculate visibility
        visi_cams = self.get_visi_cams(side_len, center, bounds=self.bounds_)
        self.tree_nodes_[u].visi_cams = torch.sum(visi_cams).item()
        # calculate camera distance
        cam_pos = self.cameras.camera_to_worlds[:, :, -1]
        cam_dis = torch.linalg.norm(cam_pos - center.permute(1, 0), ord=2, dim=-1, keepdim=True)
        visi_dis = cam_dis[visi_cams]
        dis_summary = self.distance_summary(visi_dis)
        visible_cameras = self.cameras[visi_cams]
        exist_unaddressed_cams = (visi_dis.shape[0] >= (N_PROS // 2)) and (
            dis_summary < (side_len * self.split_dist_thres)
        )   

        # subdivide the tree node
        if exist_unaddressed_cams:
            for i in range(0, 8):
                v = len(self.tree_nodes_)
                self.tree_nodes_.append(TreeNode())
                offset = torch.tensor(
                    [((i >> 2) & 1) - 0.5, ((i >> 1) & 1) - 0.5, (i & 1) - 0.5], device=self.device
                ).view(3, 1)
                sub_center = center + side_len * 0.5 * offset
                self.tree_nodes_[u].childs[i] = v
                self.tree_nodes_[v].parent = u
                self.construct_tree_node(v, depth + 1, sub_center, side_len * 0.5)
        elif visi_dis.shape[0] < (N_PROS // 2):
            self.tree_nodes_[u].is_leaf_node = True
            self.tree_nodes_[u].pers_trans = None  # is leaf node but not valid - not enough visible cameras
        else:
            self.tree_nodes_[u].is_leaf_node = True
            pers_trans = self.construct_trans(rand_pts=rand_pts, center=center, visible_cameras=visible_cameras)
            self.tree_nodes_[u].pers_trans = pers_trans

    def distance_summary(self, dis):
        if dis.view(-1).size(0) <= 0:
            return 1e8

        log_dis = torch.log(dis)
        thres = torch.quantile(log_dis, 0.25).item()
        mask = (log_dis < thres).to(torch.float32)

        if mask.sum().item() < 1e-3:
            return torch.exp(log_dis.mean()).item()

        return torch.exp((log_dis * mask).sum() / mask.sum()).item()

    def get_visi_cams(
        self, bbox_side_len: float, center: TensorType[3, 1], bounds: torch.Tensor
    ) -> TensorType["num_cameras":..., 1]:
        # calculate box
        assert center.shape == torch.Size([3, 1])
        a = center - bbox_side_len * 0.5
        b = center + bbox_side_len * 0.5
        aabb = SceneBox(torch.cat((a, b), dim=0))

        # get image coords with interval
        interval = 100
        image_height = self.cameras.image_height[-1].item()
        image_width = self.cameras.image_width[-1].item()
        image_coords = torch.meshgrid(
            torch.arange(image_height // interval) * interval,
            torch.arange(image_width // interval) * interval,
            indexing="ij",
        )
        image_coords = torch.stack(image_coords, dim=-1)

        # calculate visibility for each camera
        visi_cams = torch.zeros(
            (
                len(
                    self.cameras,
                )
            ),
            device=self.device,
            dtype=torch.bool,
        )

        for i in range(0, len(self.cameras)):
            ray_bundle = self.cameras.generate_rays(camera_indices=i, coords=image_coords, aabb_box=aabb)
            fars = torch.minimum(ray_bundle.fars, bounds[i][1])
            nears = torch.maximum(ray_bundle.nears, bounds[i][0])
            mask = fars > nears
            good = torch.sum(mask)
            visi_cams[i] = good > 0  
        return visi_cams

    def vis_octree(self, output_path):
        with open(output_path, "w") as f:
            n_nodes = len(self.tree_nodes_)

            for node in self.tree_nodes_:
                for st in range(8):
                    offset = torch.tensor([(st >> 2 & 1) - 0.5, (st >> 1 & 1) - 0.5, (st >> 0 & 1) - 0.5]).view(3, 1)
                    
                    xyz = node.center.cpu() + offset * node.side_len
                    f.write(f"v {xyz[0].item()} {xyz[1].item()} {xyz[2].item()}\n")
            count = 0
            for i in range(n_nodes):
                if self.tree_nodes_[i].pers_trans is None:
                    count += 1
                    continue
                for a in range(8):
                    for b in range(a + 1, 8):
                        st = a ^ b
                        if st == 1 or st == 2 or st == 4:
                            f.write(f"l {i * 8 + a + 1} {i * 8 + b + 1}\n")
            print("total nodes:", n_nodes)
            print("invalid count:", count)

    def vis_cam_pos(self, output_path):
        with open(output_path, "w") as f:
            n_nodes = len(self.tree_nodes_)
            for i in range(len(self.cameras)):
                pos = self.cameras[i].camera_to_worlds[:, 3]
                f.write(f"v {pos[0].item()} {pos[1].item()} {pos[2].item()}\n")

    def vis_transed_pts(self, tree_node: TreeNode, output_path):

        n_rand_pts = 32 * 32 * 32

        def gen_grid_points(resolution, scene_box):
            # return: (resolution ^ 3, 3)

            x = np.linspace(scene_box[0][0], scene_box[0][1], resolution, dtype=np.float32)
            y = np.linspace(scene_box[1][0], scene_box[1][1], resolution, dtype=np.float32)
            z = np.linspace(scene_box[2][0], scene_box[2][1], resolution, dtype=np.float32)
            X, Y, Z = np.meshgrid(x, y, z)
            points = np.column_stack(
                (X.ravel(), Y.ravel(), Z.ravel()),
            )
            return points

        # rand_pts = (torch.rand((n_rand_pts, 3)) - 0.5) * tree_node.side_len + tree_node.center.permute(1, 0)
        rand_pts = (
            torch.from_numpy(gen_grid_points(resolution=32, scene_box=[[-0.5, 0.5], [-0.5, 0.5], [-0.5, 0.5]]))
        ) * tree_node.side_len + tree_node.center.permute(1, 0)

        frame_trans = tree_node.pers_trans.w2xz
        transed_pts = torch.matmul(
            frame_trans[:, :, 0:3].unsqueeze(0), rand_pts.unsqueeze(-1).unsqueeze(-1).permute(0, 2, 1, 3)
        )  # (n,12,2,1)
        transed_pts = transed_pts[..., 0] + frame_trans.unsqueeze(0)[:, :, :, 3]  # (n,12,2)
        transed_pts = transed_pts[:, :, 0] / transed_pts[:, :, 1]  # (n,12)
        transed_pts = transed_pts.permute(1, 0)  # (12,n)
        warped_pts = torch.matmul(tree_node.pers_trans.weight, transed_pts)
        warped_pts = warped_pts.permute(1, 0)  # (n,3)
        with open(output_path, "w") as f:
            for i in range(warped_pts.shape[0]):
                pos = warped_pts[i, :]
                f.write(f"v {pos[0].item()} {pos[1].item()} {pos[2].item()}\n")

    def oct_search_nearset(self, rand_pts: TensorType["num_pts":..., 3]) -> TensorType["num_pts":..., 1]:
        pass


if __name__ == "__main__":
    from nerfstudio.cameras.camera_paths import get_path_from_json
    import json

    # camera_path = "/home/smq/data/dataset/nerf_publish/B01FE199E2/base_cam_render.json"
    # camera_path = "/home/smq/data/dataset/nerf_publish/taishan/colmap/base_cam_render.json"
    camera_path = "/home/smq/data/projects/f2-nerf/data/data_smq/beijing/base_cam_render.json"

    with open(camera_path, "r", encoding="utf-8") as f:
        camera_path = json.load(f)
        cameras = get_path_from_json(camera_path)
        bounds = torch.zeros((len(cameras), 2))
        bounds[:, 0] = 0.01
        bounds[:, 1] = 1000
    print(cameras.device)
    octree = PersOctree(cameras=cameras, bounds=bounds, bbox_side_len=512, max_depth=16)
    octree.vis_octree("octree.obj")
    octree.vis_cam_pos("cam_pos.obj")
    # for i in range(0, len(octree.tree_nodes_)):
    #     if octree.tree_nodes_[i].pers_trans is not None:
    #         octree.vis_transed_pts(octree.tree_nodes_[i], f"{i}_transed_pts.obj")
    # print(octree.device)
