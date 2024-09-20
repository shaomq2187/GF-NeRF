import gc
import importlib
import os
import warnings
from typing import Any, Dict, Iterator, List, Mapping, Optional, Tuple

import numpy as np
import tinycudann as tcnn
import torch
from torch import nn
from torch.nn.parameter import Parameter
from torchtyping import TensorType
from dataclasses import dataclass

from nerfstudio.utils.tensor_dataclass import TensorDataclass
from nerfstudio.cameras.cameras import Cameras
from nerfstudio.cameras.rays import (
    F2Frustums,
    Frustums,
    RayBundle,
    RaySamples,
    WarpedSamples,
)
from nerfstudio.model_components.ray_samplers import Sampler
from einops import rearrange, repeat, reduce
from sklearn.cluster import KMeans,SpectralClustering
from gfnerf.cluster.spectral_equal_size_clustering import SpectralEqualSizeClustering
from gfnerf.persoctree import TreeNode,TransInfo

module_path = "../GF-NeRF/gfnerf/bindings/f2nerf-bindings.so"
torch.classes.load_library(module_path)

@dataclass(init=False)
class TreeNodes(TensorDataclass):
    center: TensorType["num_nodes":...,3]
    side_len: TensorType["num_nodes":...,1]
    block_idx: TensorType["num_nodes":...,1]
    trans_idx: TensorType["num_nodes":...,1]
    is_leaf_node: TensorType["num_nodes":...,1]
    def __init__(self,center,side_len,block_idx,trans_idx,is_leaf_node):
        self.center = center
        self.side_len = side_len
        self.block_idx = block_idx
        self.trans_idx = trans_idx
        self.is_leaf_node = is_leaf_node

class PersSampler(Sampler):
    def __init__(
        self,
        cameras: Cameras,
        n_split_dataset,
        steps_per_split_dataset,
        steps_perssampler_init ,
        bounds: TensorType["n":..., 2],
        split_dist_thres: float = 1.5,
        sub_div_milestones: List[int] = [2000*1, 4000*1, 6000*1, 8000*1,10000],
        compact_freq: int = 1000*1,
        max_oct_intersect_per_ray: int = 1024,
        global_near: float = 0.01,
        scale_by_dis: bool = True,
        bbox_levels: int = 8,  # bbox_side_len = 2^(n-1)=512 
        sample_l: float = 1.0 / 256,
        max_level: int = 16,  # octree depth
        mode: int = 0,  # 0:train, 1:eval
        sampled_oct_per_ray: int = 512,
        ray_march_fineness: float = 1.0,
        ray_march_init_fineness=16.0,
        ray_march_fineness_decay_end_iter=10000,
        device=None,

    ) -> None:
        super().__init__()
        if not device:
            self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        else:
            self.device = device

        self.cameras = cameras
        n_cameras = len(cameras)
        c2w = cameras.camera_to_worlds.to(self.device)  # (n,3,4)
        w2c = torch.eye(4, device=self.device).unsqueeze(0).repeat((n_cameras), 1, 1).contiguous()  # (n,4,4)
        w2c[:, 0:3, :] = c2w
        w2c = torch.linalg.inv(w2c)
        w2c = w2c[:, 0:3, :].contiguous()  # (n,3,4)
        intri = cameras.get_intrinsics_matrices().to(self.device)  # (n,3,3)
        self.bounds = bounds.to(self.device)  


        self.max_pts_per_ray = 1024


        # clustering params
        self.register_buffer("cameras_rays_nodes_centers",None)
        self.register_buffer("cameras_labels",None)
        self.register_buffer("cameras_valid_mask",None)


        # update parameters according to the steps of init stage
        ray_march_fineness_decay_end_iter = int(ray_march_fineness_decay_end_iter * max(steps_perssampler_init // 30000,1))
        sub_div_milestones = [int(x  * max(steps_perssampler_init // 30000,1)) for x in sub_div_milestones]

        # init persampler
        self.sampler = torch.classes.my_classes.PersSampler()
        self.sampler.InitSampler(
            split_dist_thres,
            sub_div_milestones,
            compact_freq,
            max_oct_intersect_per_ray,
            global_near,
            scale_by_dis,
            bbox_levels,
            sample_l,
            max_level,
            c2w,
            w2c,
            intri,
            self.bounds,
            mode,
            sampled_oct_per_ray,
            ray_march_fineness,
            ray_march_init_fineness,
            ray_march_fineness_decay_end_iter,
        )
        self._register_state_dict_hook(self.state_dict_hook)
        self.register_buffer("tree_nodes_gpu", None)
        self.register_buffer("pers_trans_gpu", None)
        self.register_buffer("tree_visit_cnt", None)
        self.register_buffer("milestones_ts", None)

        self.register_buffer("c2w",c2w)
        


        self.n_split_dataset = n_split_dataset
        self.steps_per_split_dataset = steps_per_split_dataset
        self.steps_perssampler_init = steps_perssampler_init

    def get_nearest_split_dataset(self,origin,direction):
        # origin:(1,3)
        n_cameras = self.c2w.shape[0] # 7834
        print('n_cameras:',n_cameras)
        cameras_pos = self.c2w[:,0:3,-1] # (n,3)

        cameras_dirs = self.c2w[:,:,2]
        

        dists = torch.norm(cameras_pos - origin,dim=1,p=2)
        if False:
            # only used for matrix city dataset to get the nearest embedding
            sorted_dists, sorted_indices = torch.sort(dists)
            print('sorted values:',sorted_dists[:10])
            print('sorted sorted_indices:',sorted_indices[:10])
            print('sorted dirs:',cameras_dirs[sorted_indices[:10],:])
            assert torch.all((sorted_dists[:5] - sorted_dists[0]) < 1e-4) 

            dir_dists = torch.norm(direction - cameras_dirs[sorted_indices[:5]],dim=1,p=2)
            nearest_image_idx = sorted_indices[torch.argmin(dir_dists)].item()
            print('nearest dirs:',cameras_dirs[nearest_image_idx])
            print('nearest_image_idx:',nearest_image_idx)

        else:
            nearest_image_idx = torch.argmin(dists).item()
        cur_split_idx = self.cameras_labels[nearest_image_idx].item()

        return cur_split_idx,nearest_image_idx
    


    
    def get_distance_matrix_oct(self):
        # get the distance matrix of all the train cameras instersect octree nodes
        # generate ray bundles of all train cameras with their optical center direciton
        n_cams = self.c2w.shape[0]

        rays_o = self.c2w[:,0:3,-1] # (n,3)
        rays_d = self.c2w[:,:, 2]
        bounds = self.bounds
        (
            sampled_world_pts,
            sampled_pts,
            sampled_dirs,
            sampled_dists,
            sampled_t,
            sampled_anchors,
            pts_idx_start_end,
            first_oct_dis,
        ) = self.sampler.GetSamples(rays_o, rays_d, bounds)
        anchors_index = sampled_anchors[:,:,0].view(-1,1).contiguous() # (n_cams * 1024, 3)
            
        cameras_rays_nodes_centers = self.query_tree_nodes_centers(anchors_index)
        cameras_rays_nodes_centers = rearrange(cameras_rays_nodes_centers,'(n m) c -> n m c',n=n_cams, m=self.max_pts_per_ray,c=3)
        
        self.cameras_rays_nodes_centers = cameras_rays_nodes_centers

        valid_mask = sampled_anchors[:,:,0] > 0 #(n_cams,1024,1)
        self.cameras_valid_mask = valid_mask

        distance_matrix = [[0.0 for j in range(n_cams)] for i in range(n_cams)]

        distance_mask_1d = [True for _ in range(n_cams)]
        for i in range(0,n_cams): 
 
            
            for j in range(i,n_cams):

                cur_distance = torch.norm(rays_o[i,:] - rays_o[j,:],p=2,dim=-1) 
  
                distance_matrix[i][j] = cur_distance.item()
                distance_matrix[j][i] = cur_distance.item()

        distance_matrix = torch.tensor(distance_matrix)
        distance_mask_1d = torch.tensor(distance_mask_1d).view(-1)
        

        return distance_matrix,distance_mask_1d
    def train_cameras_clustering_oct(self,k):
        # clustering
        print('[debug] perssampler run train_cameras clustering...')
        assert self.cameras_labels is None
        # 1. get distance matrix
        distance_matrix,distance_mask_1d = self.get_distance_matrix_oct()
    
        # 2. mask invalid distances
        distance_matrix = distance_matrix[distance_mask_1d,:][:,distance_mask_1d].cpu().numpy()


        # 3. run spectral clustering
        clustering = SpectralEqualSizeClustering(nclusters=k,
                                                nneighbors=int(distance_matrix.shape[0] * 0.1),
                                                equity_fraction=1,
                                                seed=1234
                                                )
        valid_labels = clustering.fit(distance_matrix)
        

        self.cameras_labels = torch.zeros([len(self.c2w),1],dtype=torch.int64)
        self.cameras_labels[distance_mask_1d,:] = torch.tensor(valid_labels).view(-1,1)
        
        self.num_cams_per_cluster = []
        for i in range(k):
            self.num_cams_per_cluster.append(torch.sum(self.cameras_labels==i).item())
        assert 0 not in self.num_cams_per_cluster
        
    def get_nearest_split_dataset_orig(self,origin):
        # origin:(1,3)
        n_cameras = self.c2w.shape[0] # 7834
        print('n_cameras:',n_cameras)
        n_images_per_split_dataset = n_cameras // self.n_split_dataset # 783
        cameras_pos = self.c2w[:,0:3,-1] # (n,3)



        
        dists = (cameras_pos - origin) # (n,3)
        dists = torch.norm(dists,dim=1)
        nearest_image_idx = torch.argmin(dists)
        cur_split_idx = nearest_image_idx // n_images_per_split_dataset
        cur_split_idx = min(cur_split_idx,self.n_split_dataset - 1)

        return cur_split_idx,nearest_image_idx

        
    def visualize_split_cameras(self,output_dir):
        assert self.cameras_labels is not None
        from plyfile import PlyData,PlyElement

        k = torch.max(self.cameras_labels) + 1
        for i in range(k):
            cur_mask = (self.cameras_labels == i).view(-1)
            cur_cams_origins = self.c2w[cur_mask,0:3,-1].cpu().numpy() # (n,3)
            cur_cams_dirs = self.c2w[cur_mask,:,2].cpu().numpy() # (n,3)
            cam_origins = [(cur_cams_origins[i,0], cur_cams_origins[i,1], cur_cams_origins[i,2],cur_cams_dirs[i,0],cur_cams_dirs[i,1],cur_cams_dirs[i,2]) for i in range(cur_cams_origins.shape[0])]

            vertex = np.array(cam_origins,
                      dtype=[('x', 'f4'), ('y', 'f4'),
                             ('z', 'f4'),('nx', 'f4'), ('ny', 'f4'),
                             ('nz', 'f4')])
    
            vertices = PlyElement.describe(vertex, 'vertex', comments=['vertices'])

            save_path = os.path.join(output_dir,f'camera_cluster_{i}.ply')
            PlyData([vertices]).write(save_path)






    def get_oct_split_dataset(self,cam_origin,cam_direction):   
        # cam_origin: (1,3)
        # cam_direction: (1,3)
        import time
        n_cams = self.c2w.shape[0]

        cam_origin = cam_origin.view(1,3)
        cam_direction = cam_direction.view(1,3)
        bound = self.bounds[0,:].view(1,2)

        (
            sampled_world_pts,
            sampled_pts,
            sampled_dirs,
            sampled_dists,
            sampled_t,
            sampled_anchors,
            pts_idx_start_end,
            first_oct_dis,
        ) = self.sampler.GetSamples(cam_origin, cam_direction, bound)

        

        anchors_index = sampled_anchors[:,:,0].view(-1,1).contiguous() # (n_cams * 1024, 1)

        cur_rays_nodes_centers = self.query_tree_nodes_centers(anchors_index)

        cur_mask = sampled_anchors[:,:,0] > 0 
        # print('cur camera mask sum:',torch.sum(cur_mask))
        cur_mask = repeat(cur_mask,'n m -> (n repeat) m',n=1,m=self.max_pts_per_ray,repeat=n_cams)
        assert cur_mask.shape == self.cameras_valid_mask.shape
        cur_mask = cur_mask & self.cameras_valid_mask # (n_cams,1024)
        # print('cur camera mask sum:',torch.sum(cur_mask))

        # (1024,3) - (n_cams,1024,3)
        cur_rays_nodes_centers = repeat(cur_rays_nodes_centers.unsqueeze(0),'n m c -> (n repeat) m c',n=1,m=self.max_pts_per_ray,c=3,repeat=n_cams)
        cur_distances = torch.norm(cur_rays_nodes_centers - self.cameras_rays_nodes_centers,p=2,dim=-1) # (n_cams,1024)



        cur_distances_list = []
        for i in range(n_cams):
            cur_mask_row = cur_mask[i,:]
            if torch.sum(cur_mask_row) > 0:
                temp_distance = cur_distances[i,:][cur_mask_row]
                temp_distance = torch.mean(temp_distance).item()
            else:
                temp_distance = 1e9
            cur_distances_list.append(temp_distance)


        cur_distances = torch.tensor(cur_distances_list)
        camera_idx = torch.argmin(cur_distances).item()
        splix_idx = self.cameras_labels[camera_idx].item()



        

        return splix_idx,camera_idx
        
    def generate_ray_samples(
        self,
        ray_bundle: RayBundle,
    ) -> RaySamples:
        import time
        start = time.time()
        rays_o = ray_bundle.origins
        rays_d = ray_bundle.directions
        rays_lookat_d = ray_bundle.lookat_directions
        max_pts_per_ray = self.max_pts_per_ray  # same as f2-nerf definition: MAX_SAMPLE_PER_RAY
        cur_split_idx = -1 
        cur_step = -1
        if ray_bundle.steps is not None:
            cur_step = ray_bundle.steps[0][0].item()
            if cur_step >= self.steps_perssampler_init:
                cur_split_idx = (cur_step - self.steps_perssampler_init) // self.steps_per_split_dataset
                cur_split_idx = cur_split_idx % self.n_split_dataset
        

        if cur_split_idx == -1:
            # eval mode or init stage
            if self.cameras_labels is not None:
                # eval mode with splited labels
                cur_split_idx,nearest_image_idx = self.get_nearest_split_dataset(rays_o[0],rays_lookat_d[0])
            else:
                # eval mode without splited labels
                cur_split_idx,nearest_image_idx = self.get_nearest_split_dataset_orig(rays_o[0])

                
            # if self.cameras_labels is not None:
            #     # eval mode
            #     cur_split_idx, nearest_image_idx = self.get_oct_split_dataset(rays_o[0],rays_lookat_d[0])
            #     print(f'[debug] using oct_split_dataset cur_split_idx:{cur_split_idx} ')
  
                

        # if ray_bundle.nears is not None and ray_bundle.fars is not None:
        #     bounds = torch.stack((ray_bundle.nears, ray_bundle.fars), dim=1)
        # else:
        #     # TODO: here we assume each camera has same bounds
        bounds = self.bounds[0, :].repeat((rays_o.shape[0], 1))  # (n_rays,2)

        (
            sampled_world_pts,
            sampled_pts,
            sampled_dirs,
            sampled_dists,
            sampled_t,
            sampled_anchors,
            pts_idx_start_end,
            first_oct_dis,
        ) = self.sampler.GetSamples(rays_o, rays_d, bounds)

        pts_idx_start_end = pts_idx_start_end.unsqueeze(1).repeat((1, max_pts_per_ray, 1))
        first_oct_dis = first_oct_dis.unsqueeze(1).repeat((1, max_pts_per_ray, 1))
        sampled_dists = sampled_dists.unsqueeze(-1)
        sampled_t = sampled_t.unsqueeze(-1)
        f2samples = WarpedSamples(
            sampled_world_pts=sampled_world_pts,
            sampled_pts=sampled_pts,
            sampled_dirs=sampled_dirs,
            sampled_dists=sampled_dists,
            sampled_t=sampled_t,
            sampled_anchors=sampled_anchors,
            pts_idx_start_end=pts_idx_start_end,
            first_oct_dis=first_oct_dis,
        )

        rays_o = rays_o.unsqueeze(1).repeat((1, max_pts_per_ray, 1))
        rays_d = rays_d.unsqueeze(1).repeat((1, max_pts_per_ray, 1))
        frustums = Frustums(
            origins=rays_o,
            directions=rays_d,
            starts=sampled_t,
            ends=sampled_t,
            pixel_area=ray_bundle.pixel_area.unsqueeze(1).repeat((1, max_pts_per_ray, 1)),
        )
        camera_indices = ray_bundle.camera_indices.unsqueeze(1).repeat((1, max_pts_per_ray, 1))
        if ray_bundle.rel_camera_indices is None:
            print('ray_bundle rel_camera_indices is None: test mode')
            rel_camera_indices = torch.ones_like(camera_indices) * nearest_image_idx
            # rel_camera_indices = camera_indices
        else:
            rel_camera_indices = ray_bundle.rel_camera_indices.unsqueeze(1).repeat((1, max_pts_per_ray, 1))

        ray_samples = RaySamples(
            f2samples=f2samples,
            frustums=frustums,
            camera_indices=camera_indices,
            rel_camera_indices=rel_camera_indices,
            deltas=sampled_dists,
            cur_step=(torch.ones_like(camera_indices)*cur_step).to(camera_indices),
            cur_split_dataset_idx=(torch.ones_like(camera_indices)*cur_split_idx).to(camera_indices),
        )
        end = time.time()
        # print(f"[debug] perssamper gey_samples time: {end-start}")

        return ray_samples

    def vis_octree(self, base_exp_dir):
        assert os.path.isdir(base_exp_dir)
        with open(os.path.join(base_exp_dir, "cam_pos.obj"), "w") as f:
            for i in range(len(self.cameras)):
                pos = self.cameras[i].camera_to_worlds[:, 3]
                f.write(f"v {pos[0].item()} {pos[1].item()} {pos[2].item()}\n")
        self.sampler.VisOctree(base_exp_dir)

    def update_oct_nodes(
        self,
        sampled_anchors: TensorType,
        pts_idx_bounds: TensorType,
        sampled_weights: TensorType,
        sampled_alpha: TensorType,
        iter_step: int,
    ):
        self.sampler.UpdateOctNodes(sampled_anchors, pts_idx_bounds, sampled_weights, sampled_alpha, iter_step)

    def update_ray_march(self, cur_step: int):
        self.sampler.UpdateRayMarch(cur_step)

    def update_mode(self, mode: int):
        self.sampler.UpdateMode(mode)

    def update_block_idx(self,block_centers):
        self.sampler.UpdateBlockIdxs(block_centers)

    def get_points_anchors(self,rays_o,rays_d,t_starts,t_ends):
        #  rays_origins, const Tensor rays_dirs, const Tensor t_starts, const Tensor t_ends
        rays_o = rays_o.contiguous()
        rays_d = rays_d.contiguous()
        t_starts = t_starts.contiguous()
        t_ends = t_ends.contiguous()

        assert rays_o.shape == rays_d.shape
        assert rays_o.shape[0] == rays_d.shape[0] == t_starts.shape[0] == t_ends.shape[0]
        
        return self.sampler.get_points_anchors(rays_o,rays_d,t_starts,t_ends)

    def trans_query_frame(self,world_positions_flat,anchors_flat):
        world_positions_flat = world_positions_flat.contiguous()
        anchors_flat = anchors_flat.contiguous()
        assert world_positions_flat.shape[0] == anchors_flat.shape[0]
        return self.sampler.trans_query_frame(world_positions_flat,anchors_flat)


    def get_edge_samples(self, n_pts: int) -> Tuple[TensorType]:
        sample_pts, sample_idxs = self.sampler.GetEdgeSamples(n_pts)
        return (sample_pts, sample_idxs)
    
    def query_tree_nodes_centers(self,anchors):
        assert anchors.shape[1] == 1
        assert anchors.dtype == torch.int64
        return self.sampler.qurey_tree_nodes_centers(anchors)
    
    def state_dict_hook(self, *args):
        print("[PersSampler] state_dict_hook is called")

        destination = args[1]
        prefix = args[2]
        tree_nodes_gpu_, pers_trans_gpu_, tree_visit_cnt_, milestones_ts_ = self.states()
        destination[prefix + "tree_nodes_gpu"] = tree_nodes_gpu_
        destination[prefix + "pers_trans_gpu"] = pers_trans_gpu_
        destination[prefix + "tree_visit_cnt"] = tree_visit_cnt_
        destination[prefix + "milestones_ts"] = milestones_ts_

        return destination

    def load_state_dict(self, state_dict: Mapping[str, Any], strict: bool = True):
        print("[PersSampler] load_state_dict")
        states = []
        states.append(state_dict["persampler.tree_nodes_gpu"])
        states.append(state_dict["persampler.pers_trans_gpu"])
        states.append(state_dict["persampler.tree_visit_cnt"])
        states.append(state_dict["persampler.milestones_ts"])
        del state_dict["persampler.tree_nodes_gpu"]
        del state_dict["persampler.pers_trans_gpu"]
        del state_dict["persampler.tree_visit_cnt"]
        del state_dict["persampler.milestones_ts"]
        self.load_states(states, idx=0)


        self.c2w = state_dict['field.persampler.c2w'].cuda()
        del state_dict['field.persampler.c2w']

        # # for oct split
        # print(state_dict.keys())
        # self.cameras_rays_nodes_centers = state_dict['field.persampler.cameras_rays_nodes_centers'].cuda()
        # self.cameras_labels = state_dict['field.persampler.cameras_labels'].cuda()
        # self.cameras_valid_mask = state_dict['field.persampler.cameras_valid_mask'].cuda()

        # del state_dict['field.persampler.cameras_rays_nodes_centers']
        # del state_dict['field.persampler.cameras_labels']
        # del state_dict['field.persampler.cameras_valid_mask']



        # super().load_state_dict()
        return None



    def to_proposal_sampler_oct_tree_nodes(self) -> List[TreeNode]:
        # convert perssampler treenodes to the tree_nodes in proposal_sampler_oct.py
        w2xz,weight,center,side_len,dis_summary= self.sampler.get_pers_trans_info()
        
        tree_nodes = []
        centers = self.sampler.get_tree_nodes_center_()
        block_idxs = self.sampler.get_tree_nodes_block_idx_()
        side_lens = self.sampler.get_tree_nodes_side_len_()
        trans_idx = self.sampler.get_tree_nodes_trans_idx_()
        is_leaf_node = self.sampler.get_tree_nodes_is_leaf_node_()
        
        n_tree_nodes = len(centers)
        for i in range(n_tree_nodes):
            if trans_idx[i] != -1:
                cur_w2xz_trans = torch.tensor(w2xz[trans_idx[i]]) # (12,2,4)
                cur_center_trans = torch.tensor(center[trans_idx[i]]).unsqueeze(-1) # (3,1)
                cur_weight_trans = torch.tensor(weight[trans_idx[i]]) # (3,12)
                cur_dis_summary_trans = dis_summary[trans_idx[i]]

                cur_trans_info = TransInfo(w2xz=cur_w2xz_trans,center=cur_center_trans,weight=cur_weight_trans,dis_summary=cur_dis_summary_trans)
            else:
                cur_trans_info = None
                
            cur_center = torch.tensor(centers[i]).unsqueeze(-1)
            cur_side_len = side_lens[i]

            cur_is_leaf_node = is_leaf_node[i]
            cur_node = TreeNode(center=cur_center,side_len=cur_side_len,childs=[],parent=-1,is_leaf_node=cur_is_leaf_node,visi_cams=0,pers_trans=cur_trans_info)
            tree_nodes.append(cur_node)

        return tree_nodes     
        
    def load_states(self, states: List[TensorType], idx: int) -> int:
        return self.sampler.LoadStates(states, idx)

    def states(self) -> List[TensorType]:
        return self.sampler.States()

    @property
    def sub_div_milestones_(self) -> List[int]:
        return self.sampler.get_sub_div_milestones_()

    @property
    def compact_freq_(self) -> int:
        return self.sampler.get_compact_freq_()

    @property
    def max_oct_intersect_per_ray_(self) -> int:
        return self.sampler.get_max_oct_intersect_per_ray_()

    @property
    def global_near_(self) -> float:
        return self.sampler.get_global_near_()

    @property
    def sample_l_(self) -> float:
        return self.sampler.get_sample_l_()

    @property
    def scale_by_dis_(self) -> bool:
        return self.sampler.get_scale_by_dis_()

    @property
    def mode_(self) -> int:
        return self.sampler.get_mode_()

    @property
    def n_volumes_(self) -> int:
        return self.sampler.get_n_volumes_()

    @property
    def sampled_oct_per_ray_(self) -> float:
        return self.sampler.sampled_oct_per_ray_()

    @property
    def ray_march_fineness_(self) -> float:
        return self.sampler.get_ray_march_fineness_()

    @property
    def tree_nodes_(self):
        centers = torch.tensor(self.sampler.get_tree_nodes_center_())
        block_idxs = torch.tensor(self.sampler.get_tree_nodes_block_idx_()).unsqueeze(-1)
        side_lens = torch.tensor(self.sampler.get_tree_nodes_side_len_()).unsqueeze(-1)
        trans_idx = torch.tensor(self.sampler.get_tree_nodes_trans_idx_()).unsqueeze(-1)
        is_leaf_node = torch.tensor(self.sampler.get_tree_nodes_is_leaf_node_()).unsqueeze(-1)

        return TreeNodes(center=centers,side_len = side_lens,block_idx = block_idxs,trans_idx = trans_idx,is_leaf_node=is_leaf_node)





if __name__ == "__main__":
    import json

    from nerfstudio.cameras.camera_paths import get_path_from_json
    from nerfstudio.cameras.cameras import Cameras

    camera_path = "/home/smq/data/projects/f2-nerf/data/data_smq/beijing/base_cam_render.json"
    with open(camera_path, "r", encoding="utf-8") as f:
        camera_file = json.load(f)
        cameras = get_path_from_json(camera_file)
    bounds = torch.zeros((len(cameras), 2))
    bounds[:, 0] = 0.01
    bounds[:, 1] = 512
    sampelr = PersSampler(cameras=cameras, bounds=bounds)
    sampelr.vis_octree("/home/smq/data/")
