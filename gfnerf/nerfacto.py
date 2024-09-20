# Copyright 2022 The Nerfstudio Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
NeRF implementation that combines many recent advancements.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Optional, Tuple, Type
import os
import numpy as np
import torch
from einops import rearrange
from torch.nn import Parameter
from torchmetrics import PeakSignalNoiseRatio
from torchmetrics.functional import structural_similarity_index_measure
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
from typing_extensions import Literal
import tinycudann as tcnn
from gfnerf.nerfacto_field import GFNeRFField
from gfnerf.perssampler import PersSampler

from nerfstudio.cameras.rays import RayBundle
from nerfstudio.engine.callbacks import (
    TrainingCallback,
    TrainingCallbackAttributes,
    TrainingCallbackLocation,
)
from nerfstudio.field_components.field_heads import FieldHeadNames
from nerfstudio.field_components.spatial_distortions import SceneContraction
from nerfstudio.fields.density_fields import HashMLPDensityField
from nerfstudio.model_components.losses import (
    CharbonnierLoss,
    L1Loss,
    MSELoss,
    distortion_loss,
    interlevel_loss,
    orientation_loss,
    pred_normal_loss,
    ScaleAndShiftInvariantLoss,
    S3IM,
)
from nerfstudio.model_components.ray_samplers import UniformSampler
from nerfstudio.model_components.renderers import (
    AccumulationRenderer,
    DepthRenderer,
    NormalsRenderer,
    RGBRenderer,
    SemanticRenderer,
)
from nerfstudio.model_components.scene_colliders import AABBBoxCollider, NearFarCollider
from nerfstudio.model_components.shaders import NormalsShader
from nerfstudio.models.base_model import Model, ModelConfig
from nerfstudio.utils import colormaps
from rich.progress import Console
from pathlib import Path
from nerfstudio.cameras.cameras import Cameras
from torchvision.utils import save_image

CONSOLE = Console(width=120)

@dataclass
class GFNeRFModelConfig(ModelConfig):
    """Nerfacto Model Config"""

    _target: Type = field(default_factory=lambda: GFNeRFModel)

    near_plane: float = 0.05
    """How far along the ray to start sampling."""
    far_plane: float = 300.0
    """How far along the ray to stop sampling."""
    background_color: Literal["random", "last_sample", "black", "white"] = "last_sample"
    """Whether to randomize the background color."""
    patch_size: int = 32
    """size of patch sampling"""
    hidden_dim: int = 64
    """Dimension of hidden layers"""
    hidden_dim_color: int = 64
    """Dimension of hidden layers for color network"""
    hidden_dim_transient: int = 64
    """Dimension of hidden layers for transient network"""
    num_levels: int = 16
    """Number of levels of the hashmap for the base mlp."""
    max_res: int = 2048
    """Maximum resolution of the hashmap for the base mlp."""
    log2_hashmap_size: int = 19
    """Size of the hashmap for the base mlp"""
    num_proposal_samples_per_ray: Tuple[int, ...] = (256, 96)
    """Number of samples per ray for each proposal network."""
    num_nerf_samples_per_ray: int = 48
    """Number of samples per ray for the nerf network."""
    proposal_update_every: int = 5
    """Sample every n steps after the warmup"""
    proposal_warmup: int = 5000
    """Scales n from 1 to proposal_update_every over this many steps"""
    num_proposal_iterations: int = 2
    """Number of proposal network iterations."""
    use_same_proposal_network: bool = False
    """Use the same proposal network. Otherwise use different ones."""
    proposal_net_args_list: List[Dict] = field(
        default_factory=lambda: [
            {"hidden_dim": 16 * 4, "log2_hashmap_size": 19, "num_levels": 5*2, "max_res": 128 * 4, "use_linear": False},
            {"hidden_dim": 16 * 4, "log2_hashmap_size": 19, "num_levels": 5*2, "max_res": 256 * 4, "use_linear": False},
        ]
    )
    """Arguments for the proposal density fields."""
    proposal_initial_sampler: Literal["piecewise", "uniform"] = "piecewise"
    """Initial sampler for the proposal network. Piecewise is preferred for unbounded scenes."""
    interlevel_loss_mult: float = 1.0
    """Proposal loss multiplier."""
    distortion_loss_mult: float = 0.002
    """Distortion loss multiplier."""
    orientation_loss_mult: float = 0.0001
    """Orientation loss multiplier on computed normals."""
    pred_normal_loss_mult: float = 0.001
    """Predicted normal loss multiplier."""
    perceptual_loss_mult: float = 1.0
    """Perceptual loss multiplier."""
    mono_depth_loss_mult: float = 0.0
    use_proposal_weight_anneal: bool = True
    """Whether to use proposal weight annealing."""
    use_average_appearance_embedding: bool = True
    """Whether to use average appearance embedding or zeros for inference."""
    proposal_weights_anneal_slope: float = 10.0
    """Slope of the annealing function for the proposal weights."""
    proposal_weights_anneal_max_num_iters: int = 1000
    """Max num iterations for the annealing function."""
    use_single_jitter: bool = True
    """Whether use single jitter or not for the proposal networks."""
    predict_normals: bool = False
    """Whether to predict normals or not."""
    disable_scene_contraction: bool = False
    """Whether to disable scene contraction or not."""
    use_ch_loss: bool = False
    """Whether to use CharbonnierLoss or not."""
    use_perceptual_loss: bool = False
    """Whether to use perceptual or not."""
    use_aabb_collider: bool = False
    """Whether to use aabb collider or not."""
    use_mask: bool = False
    """Whether to use aabb collider or not."""
    use_normal_loss: bool = False
    """Whether to use normal loss or not."""
    use_depth_loss: bool = False

    use_semantics: bool = False

    num_semantic_classes: int = 2

    semantic_loss_weight: float = 0.1


    """ for block-f2nerf"""
    n_blocks: int = 1

    n_split_dataset: int=1


    steps_per_split_dataset: int = 30000


    n_activate_block: int = 3

    steps_perssampler_init: int = 10000

    use_appearance_embedding: bool = False


    """ for perssampler split"""
    scale_factor: float = 1.0


    """ for S3IM Loss """

    s3im_loss_mult: float = 0.0
    """S3IM loss multiplier."""
    s3im_kernel_size: int = 4
    """S3IM kernel size."""
    s3im_stride: int = 4
    """S3IM stride."""
    s3im_repeat_time: int = 10
    """S3IM repeat time."""
    s3im_patch_height: int = 32
    """S3IM virtual patch height."""


class GFNeRFModel(Model):
    """Nerfacto model

    Args:
        config: Nerfacto configuration to instantiate model
    """

    config: GFNeRFModelConfig

    def populate_modules(self):
        """Set the fields and modules."""
        super().populate_modules()

        if self.config.disable_scene_contraction:
            scene_contraction = None
        else:
            scene_contraction = SceneContraction(order=float("inf"))
            # scene_contraction = SceneContraction()
        self.cameras = self.kwargs["cameras"]
        bounds = self.kwargs["bounds"]
        base_dir = self.kwargs["base_dir"]
        self.base_dir = base_dir
        self.density_fns = []
        self.persampler = PersSampler(cameras=self.cameras,bounds=bounds,bbox_levels=10,max_level=16,
                                      steps_per_split_dataset=self.config.steps_per_split_dataset,
                                      steps_perssampler_init=self.config.steps_perssampler_init,
                                      n_split_dataset=self.config.n_split_dataset ,
                                      )
        
        self.scale_factor = self.config.scale_factor
        # importance sampler
        self.sample_tmp_dir = None
        n_blocks = self.config.n_blocks
        n_cameras = len(self.cameras)
        step_n = n_cameras // n_blocks
        block_centers = []
        for i in range(n_blocks):
            print(f'block {i} center at camera: ',i*step_n)
            block_centers.append(self.cameras.camera_to_worlds[i*step_n,:,3])


        self.block_centers = torch.stack(block_centers,dim=0).cuda()

        self.n_blocks = n_blocks


        # Fields
        self.field = GFNeRFField(
            self.scene_box.aabb,
            hidden_dim=self.config.hidden_dim,
            num_levels=self.config.num_levels,
            max_res=self.config.max_res,
            log2_hashmap_size=self.config.log2_hashmap_size,
            hidden_dim_color=self.config.hidden_dim_color,
            hidden_dim_transient=self.config.hidden_dim_transient,
            spatial_distortion=scene_contraction,
            num_images=self.num_train_data,
            use_pred_normals=self.config.predict_normals,
            use_average_appearance_embedding=self.config.use_average_appearance_embedding,
            n_blocks = self.n_blocks,
            block_centers=self.block_centers,
            n_active_block = self.config.n_activate_block,
            steps_perssampler_init = self.config.steps_perssampler_init,
            use_semantics = self.config.use_semantics,
            num_semantic_classes = self.config.num_semantic_classes,
            base_dir=base_dir,
            use_appearance_embedding = self.config.use_appearance_embedding,
            n_volumes = len(self.persampler.to_proposal_sampler_oct_tree_nodes()),
        )
        # self.persampler.vis_octree('/home/smq/data')

        # build proposal sampler with octree

        # Collider
        if self.config.use_aabb_collider:
            self.collider = AABBBoxCollider(scene_box=self.scene_box)
        else:
            self.collider = NearFarCollider(near_plane=self.config.near_plane, far_plane=self.config.far_plane)

        # renderers
        self.renderer_rgb = RGBRenderer(background_color=self.config.background_color)
        self.renderer_accumulation = AccumulationRenderer()
        self.renderer_depth = DepthRenderer(method="expected")
        self.renderer_normals = NormalsRenderer()
        self.renderer_semantics = SemanticRenderer()

        # shaders
        self.normals_shader = NormalsShader()

        # losses
        if self.config.use_ch_loss:
            self.rgb_loss = CharbonnierLoss()
        else:
            self.rgb_loss = MSELoss()
        self.lpips = LearnedPerceptualImagePatchSimilarity()
        self.depth_loss = ScaleAndShiftInvariantLoss(alpha=0.0, scales=1)
        self.cross_entropy_loss = torch.nn.CrossEntropyLoss(reduction="mean")
        self.s3im_loss = S3IM(s3im_kernel_size=self.config.s3im_kernel_size, s3im_stride=self.config.s3im_stride, s3im_repeat_time=self.config.s3im_repeat_time, s3im_patch_height=self.config.s3im_patch_height)

        # metrics
        self.psnr = PeakSignalNoiseRatio(data_range=1.0)
        self.ssim = structural_similarity_index_measure

    def get_param_groups(self) -> Dict[str, List[Parameter]]:
        param_groups = {}
        n_named_parms = len(list(self.field.named_parameters()))
        a = list(self.field.named_parameters())
        params = list(self.field.parameters())
        param_groups["fields"] = params[0:n_named_parms] # only applicable to single mlp
        param_groups["base_encoding_init"] = [params[n_named_parms]]

        if hasattr(self,'proposal_sampler'):
            param_groups["proposal_networks"] = list(self.proposal_networks.parameters())
            param_groups["proposal_encodings"] = []
            for encoding in self.proposal_encodings:
                for p in encoding.parameters():
                    param_groups["proposal_encodings"].append(p)




        return param_groups

    def get_training_callbacks(
        self, training_callback_attributes: TrainingCallbackAttributes
    ) -> List[TrainingCallback]:
        callbacks = []

        if hasattr(self,'proposal_sampler'):
            if self.config.use_proposal_weight_anneal:
                # anneal the weights of the proposal network before doing PDF sampling
                N = self.config.proposal_weights_anneal_max_num_iters

                def set_anneal(step):
                    # https://arxiv.org/pdf/2111.12077.pdf eq. 18
                    train_frac = np.clip(step / N, 0, 1)
                    bias = lambda x, b: (b * x) / ((b - 1) * x + 1)
                    anneal = bias(train_frac, self.config.proposal_weights_anneal_slope)
                    self.proposal_sampler.set_anneal(anneal)

                callbacks.append(
                    TrainingCallback(
                        where_to_run=[TrainingCallbackLocation.BEFORE_TRAIN_ITERATION],
                        update_every_num_iters=1,
                        func=set_anneal,
                    )
                )
                callbacks.append(
                    TrainingCallback(
                        where_to_run=[TrainingCallbackLocation.AFTER_TRAIN_ITERATION],
                        update_every_num_iters=1,
                        func=self.proposal_sampler.step_cb,
                    )
                )
        def train_cameras_clustering(step,*args,**kwargs):
            if self.field.cur_stage == 'block_stage' and self.persampler.cameras_labels is None:
                # block阶段且未进行clustering
                k = kwargs['n_blocks']
                self.persampler.train_cameras_clustering_oct(k)
                # self.persampler.visualize_split_cameras('/home/smq/data/cluster_test')

        def render_init_error_maps(step,*args,**kwargs):
            # 1. render train images
            # 2. calculate and save error map(npy, png)
        
            if self.field.cur_stage == 'block_stage' and self.persampler.cameras_labels is None:
                sample_tmp_dir = Path(self.base_dir) / "sample_tmp"
                self.sample_tmp_dir = str(sample_tmp_dir)
        
                gt_dir = sample_tmp_dir / "gt"
                pred_dir = sample_tmp_dir / "pred"
                npy_dir = sample_tmp_dir / "npy"
                png_dir = sample_tmp_dir / "png"


                os.makedirs(npy_dir,exist_ok=False)
                os.makedirs(png_dir,exist_ok=False)
                os.makedirs(gt_dir,exist_ok=False)
                os.makedirs(pred_dir,exist_ok=False)
                datamanager = kwargs["pipeline"].datamanager
                filenames = datamanager.train_dataparser_outputs.image_filenames
                cameras:Cameras = datamanager.train_dataparser_outputs.cameras.to(self.device)
                n_cameras = len(cameras)
                down_scale_factor = 8
                image_coords = cameras.get_image_coords()
                for image_idx in range(n_cameras):
                    base_name = os.path.basename(filenames[image_idx])
                    npy_path = str(npy_dir / base_name) + '.npy'
                    png_path = str(png_dir / base_name) + '.png'
                    gt_path = str(gt_dir / base_name) + '.png'
                    pred_path = str(pred_dir / base_name) + '.png'

                    gt_image = datamanager.train_dataset[image_idx]["image"].to(self.device) # (H,W,C)
                    H,W,_ = gt_image.shape

                    gt_image = rearrange(gt_image,'h w c -> 1 c h w')

                    # render image from init stage
                    down_scale_image_coords = image_coords[::down_scale_factor,::down_scale_factor]
                    camera_ray_bundle = cameras.generate_rays(camera_indices=image_idx, coords=down_scale_image_coords,keep_shape=True)
                    camera_ray_bundle.rel_camera_indices = camera_ray_bundle.camera_indices
                    camera_ray_bundle.steps = torch.full_like(
                        camera_ray_bundle.origins, step, dtype=torch.int32, device=camera_ray_bundle.origins.device
                    )
                    outputs = self.get_outputs_for_camera_ray_bundle(camera_ray_bundle)
                    predict_image = rearrange(outputs["rgb"],'h w c -> 1 c h w')
                    predict_image = torch.nn.functional.interpolate(predict_image,(H,W),mode='nearest')

                    # get error map
                    error = torch.abs(gt_image - predict_image)
                    error = torch.sum(error,dim=1)

                    # save error map 
                    save_image(error,png_path)
                    save_image(gt_image,gt_path)
                    save_image(predict_image,pred_path)
                    np.save(npy_path,error.cpu().numpy())
                
                    

                    
                    


                    
                

                    
            

            
            
        def update_field_stage(step,*args,**kwargs):
            if (self.config.steps_perssampler_init >0 and step>=0 and step < self.config.steps_perssampler_init):
                self.field.cur_stage = 'init_stage'  
            else:
                self.field.cur_stage = 'block_stage'  

        def update_datamanager(step,*args,**kwargs):
            split_idx = self.field.cur_split_dataset_idx
            pipeline = kwargs['pipeline']
            
            
            pipeline.datamanager.setup_train_split_oct(self.persampler.cameras_labels,split_idx,self.sample_tmp_dir)


                
                
        def update_optimizer(step,*args,**kwargs):
            # delete and add optimizer according to active blocks to save memory
            # print('update_optimizer!!!')
            optimizers = kwargs['optimizer']
            list_in_train = self.field.active_block_idxs
            list_in_test = self.field.active_block_idxs_test
            list_not_in_train_or_test = list(set(range(self.n_blocks)).difference(set(list_in_train + list_in_test)))
            stage = self.field.cur_stage
            if stage == 'init_stage':
                # keep only init encoding
                optimizers.add_optimizer(f"base_encoding_init")
                optimizers.add_optimizer(f"fields")

                if hasattr(self,'proposal_sampler'):
                    optimizers.add_optimizer(f'proposal_networks')
                    optimizers.add_optimizer(f'proposal_encodings')

                for idx in range(self.n_blocks):
                    optimizers.delete_optimizer(f"base_encoding_{idx}")
            else:

                # save last encoding ckpt
                if step == (self.config.steps_perssampler_init + self.config.n_split_dataset * self.config.steps_per_split_dataset) - 1:
                    for idx in list_in_train:
                        self.field.save_table(idx)
                        self.field.del_table(idx)
                else:
                    optimizers.delete_optimizer("base_encoding_init")
                    optimizers.delete_optimizer("fields")
                    if hasattr(self,'proposal_sampler'):
                        optimizers.delete_optimizer(f'proposal_networks')
                        optimizers.delete_optimizer(f'proposal_encodings')

                    for idx in list_in_train:
                        params = getattr(self.field,f"base_encoding_{idx}").parameters()
                        optimizers.add_optimizer(f"base_encoding_{idx}",lr_init = 5e-3,params = params )

                    for idx in list_not_in_train_or_test:
                        optimizers.delete_optimizer(f"base_encoding_{idx}")
                print('[debug] update optimizer list in train:',list_in_train)
                print('[debug] update optimizer list_not_in_train_or_test:',list_not_in_train_or_test)

            
        update_optimizer_callback = TrainingCallback(where_to_run=[TrainingCallbackLocation.AFTER_TRAIN_ITERATION],
                         func=update_optimizer,
                         update_every_num_iters=True,
                         args=None,
                         kwargs={'optimizer':training_callback_attributes.optimizers},
                         )
        train_cameras_clustering_callback = TrainingCallback(where_to_run=[TrainingCallbackLocation.AFTER_TRAIN_ITERATION],
                         func=train_cameras_clustering,
                         update_every_num_iters=True,
                         args=None,
                         kwargs={'n_blocks':self.n_blocks},
                         )
        update_datamanager_callback = TrainingCallback(where_to_run=[TrainingCallbackLocation.AFTER_TRAIN_ITERATION],
                         func=update_datamanager,
                         update_every_num_iters=True,
                         args=None,
                         kwargs={'pipeline':training_callback_attributes.pipeline},
                         )
        render_init_error_maps_callback = TrainingCallback(where_to_run=[TrainingCallbackLocation.AFTER_TRAIN_ITERATION],
                         func=render_init_error_maps,
                         update_every_num_iters=True,
                         args=None,
                         kwargs={'pipeline':training_callback_attributes.pipeline},
                         )

        callbacks.append(update_optimizer_callback)
        callbacks.append(render_init_error_maps_callback)
        callbacks.append(train_cameras_clustering_callback)
        callbacks.append(update_datamanager_callback)
        return callbacks

    def get_outputs(self, ray_bundle: RayBundle):
        if hasattr(self,'proposal_sampler'):
            ray_samples, weights_list,ray_samples_list = self.proposal_sampler(ray_bundle,proposal_encodings = self.proposal_encodings,proposal_networks = self.proposal_networks,perssampler = self.persampler)
        else:
            ray_samples = self.persampler(ray_bundle)

  

        # get filed outputs
        # TODO:debug here
        if self.field.persampler is None:
            self.field.persampler = self.persampler
        
        field_outputs = self.field(ray_samples, compute_normals=(self.config.use_normal_loss or self.config.predict_normals))
        weights,alphas,trans = ray_samples.get_weights_f2nerf(field_outputs[FieldHeadNames.DENSITY])


        if hasattr(self,'proposal_sampler'):
            weights_list.append(weights)
            ray_samples_list.append(ray_samples)



        rgb = self.renderer_rgb(rgb=field_outputs[FieldHeadNames.RGB], weights=weights)
        depth = self.renderer_depth(weights=weights, ray_samples=ray_samples) / self.scale_factor
        accumulation = self.renderer_accumulation(weights=weights)
        if hasattr(self,'proposal_sampler'):        
            outputs = {
                "rgb": rgb,
                "accumulation": accumulation,
                "depth": depth,
            }
        else:
            oct_depth = ray_samples.f2samples.first_oct_dis[:,0,:] / self.scale_factor
            assert torch.min(oct_depth) > 0


            outputs = {
                "rgb": rgb,
                "accumulation": accumulation,
                "depth": depth,
                "oct_depth": oct_depth
            }


        if self.config.predict_normals:
            normals = self.renderer_normals(normals=field_outputs[FieldHeadNames.NORMALS], weights=weights)
            pred_normals = self.renderer_normals(field_outputs[FieldHeadNames.PRED_NORMALS], weights=weights)
            outputs["normals"] = self.normals_shader(normals)
            outputs["pred_normals"] = self.normals_shader(pred_normals)
        if self.config.use_normal_loss:
            normals = self.renderer_normals(normals=field_outputs[FieldHeadNames.NORMALS], weights=weights)
            outputs["normals"] = self.normals_shader(normals)

        # These use a lot of GPU memory, so we avoid storing them for eval.
        if hasattr(self,'proposal_sampler') and self.training:
            outputs["weights_list"] = weights_list
            outputs["ray_samples_list"] = ray_samples_list

        if self.training and self.config.predict_normals:
            outputs["rendered_orientation_loss"] = orientation_loss(
                weights.detach(), field_outputs[FieldHeadNames.NORMALS], ray_bundle.directions
            )

            outputs["rendered_pred_normal_loss"] = pred_normal_loss(
                weights.detach(),
                field_outputs[FieldHeadNames.NORMALS].detach(),
                field_outputs[FieldHeadNames.PRED_NORMALS],
            )
        # semantics
        if self.config.use_semantics:
            outputs["semantics"] = self.renderer_semantics(
                field_outputs[FieldHeadNames.SEMANTICS], weights=weights
            )

        # # update f2nerf sampler
        if not hasattr(self,'proposal_sampler'):
            if self.training:
                cur_step = ray_bundle.steps[0, 0].item()

                # print("[debug] update ray march rel_step:",rel_step)
                self.persampler.update_ray_march(cur_step)  
                self.persampler.update_mode(0)
                if self.field.cur_stage == "init_stage":
                    # only update oct nodes on init stage
                    self.persampler.update_oct_nodes(
                        sampled_anchors=ray_samples.f2samples.sampled_anchors,
                        pts_idx_bounds=ray_samples.f2samples.pts_idx_start_end,
                        sampled_weights=weights,
                        sampled_alpha=alphas,
                        iter_step=cur_step,
                    )
                else:
                    # pass
                    self.persampler.update_mode(1)


        return outputs

    def get_metrics_dict(self, outputs, batch):
        metrics_dict = {}
        image = batch["image"].to(self.device)
        metrics_dict["psnr"] = self.psnr(outputs["rgb"], image)

        return metrics_dict

    def get_loss_dict(self, outputs, batch, metrics_dict=None):
        loss_dict = {}
        image = batch["image"].to(self.device)
        if self.config.use_mask:
            mask = batch["mask"].to(self.device)
            nonzero_indices = torch.nonzero(mask[..., 0], as_tuple=False)
            loss_dict["rgb_loss"] = self.rgb_loss(outputs["rgb"][nonzero_indices], image[nonzero_indices])
        else:
            loss_dict["rgb_loss"] = self.rgb_loss(image, outputs["rgb"])

        if self.config.use_perceptual_loss:
            out_patches = (
                outputs["rgb"].view(-1, self.config.patch_size, self.config.patch_size, 3).permute(0, 3, 1, 2) * 2 - 1
            ).clamp(-1, 1)
            gt_patches = (
                image.view(-1, self.config.patch_size, self.config.patch_size, 3).permute(0, 3, 1, 2) * 2 - 1
            ).clamp(-1, 1)
            loss_dict["lpips_loss"] = self.config.perceptual_loss_mult * self.lpips(out_patches, gt_patches)
        
        if self.config.use_normal_loss:
            pass
        
        # monocular depth loss
        if "depth" in batch and self.config.mono_depth_loss_mult > 0.0:
            # TODO check it's true that's we sample from only a single image
            # TODO only supervised pixel that hit the surface and remove hard-coded scaling for depth
            depth_gt = batch["depth"].to(self.device)[..., None] / 100 # (2048,1,1)
            a = torch.max(depth_gt)
            b = torch.min(depth_gt)
            depth_pred = outputs["depth"][...,None] / 0.02662921932698809 # (2048,1)
            c = torch.max(depth_pred)
            d = torch.min(depth_pred)
            road_mask = batch["road_mask"].to(self.device)[...,None].bool()
            road_mask = road_mask & (depth_gt > 0)
            road_mask = road_mask.reshape(1, 32, -1).bool()
            
            # road_mask = torch.ones_like(depth_gt).reshape(1,32,-1).bool()
            loss_dict["depth_loss"] = (
                self.depth_loss(depth_pred.reshape(1, 32, -1), (depth_gt).reshape(1, 32, -1), road_mask)
                * self.config.mono_depth_loss_mult 
            )
            CONSOLE.print(f"[debug] depth loss:",loss_dict["depth_loss"])
        
        # semantic loss
        if self.config.use_semantics:
            gt_semantics = batch["road_mask"][...,0].long() # (2048,1)
            if self.config.use_semantics:
                loss_dict["semantics_loss"] = self.config.semantic_loss_weight * self.cross_entropy_loss(
                    outputs["semantics"], gt_semantics
                ) 
        

        # proposal loss
        if hasattr(self,'proposal_sampler') and self.training:
            loss_dict['interlevel_loss'] = self.config.interlevel_loss_mult * interlevel_loss(outputs["weights_list"],outputs["ray_samples_list"])


        
        # s3im loss
        if self.config.s3im_loss_mult > 0:
            loss_dict["s3im_loss"] = self.s3im_loss(image, outputs["rgb"]) * self.config.s3im_loss_mult
        return loss_dict

    def load_state_dict(self, state_dict: Mapping[str, Any], strict: bool = True):
        # 手动加载state_dict
        base_encoding_init = getattr(self.field,f"base_encoding_init")
        base_encoding_init.load_state_dict(state_dict,prefix = 'field.base_encoding_init')

 

        del state_dict[f"field.persampler.tree_nodes_gpu"]
        del state_dict[f"field.persampler.pers_trans_gpu"]
        del state_dict[f"field.persampler.tree_visit_cnt"]
        del state_dict[f"field.persampler.milestones_ts"]

        self.persampler.load_state_dict(state_dict)

        # load porposampler state dict
        if hasattr(self,'proposal_sampler'):
            for i in range(len(self.proposal_encodings)):
                proposal_encoding_i = self.proposal_encodings[i]
                proposal_encoding_i.load_state_dict(state_dict,prefix = f'proposal_encodings.{i}')
            

  
        
        return super().load_state_dict(state_dict, strict)

    def get_image_metrics_and_images(
        self, outputs: Dict[str, torch.Tensor], batch: Dict[str, torch.Tensor]
    ) -> Tuple[Dict[str, float], Dict[str, torch.Tensor]]:
        image = batch["image"].to(self.device)
        rgb = outputs["rgb"]
        acc = colormaps.apply_colormap(outputs["accumulation"])
        depth = colormaps.apply_depth_colormap(
            outputs["depth"],
            accumulation=outputs["accumulation"],
        )

        combined_rgb = torch.cat([image, rgb], dim=1)
        combined_acc = torch.cat([acc], dim=1)
        combined_depth = torch.cat([depth], dim=1)

        # Switch images from [H, W, C] to [1, C, H, W] for metrics computations
        image = torch.moveaxis(image, -1, 0)[None, ...]
        rgb = torch.moveaxis(rgb, -1, 0)[None, ...]

        psnr = self.psnr(image, rgb)
        ssim = self.ssim(image, rgb)
        lpips = self.lpips(image, rgb)

        # all of these metrics will be logged as scalars
        metrics_dict = {"psnr": float(psnr.item()), "ssim": float(ssim)}  # type: ignore
        metrics_dict["lpips"] = float(lpips)

        images_dict = {"img": combined_rgb, "accumulation": combined_acc, "depth": combined_depth}



        return metrics_dict, images_dict
