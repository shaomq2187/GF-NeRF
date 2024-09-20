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
Put all the method implementations in one location.
"""

from __future__ import annotations

from typing import Dict

import tyro

# custom files
from gfnerf.gf_pipeline import GFNerfPipelineConfig
from gfnerf.nerfacto import GFNeRFModelConfig
from gfnerf.ori_dataparser import NerfstudioDataParserConfig

# custom class in nerfstudio
from nerfstudio.data.datamanagers.base_datamanager import GFNerfDataManagerConfig
from nerfstudio.engine.schedulers import GFNerfExponentialDecaySchedulerConfig,ExponentialDecaySchedulerConfig

# nerfstudio class
from nerfstudio.cameras.camera_optimizers import CameraOptimizerConfig
from nerfstudio.configs.base_config import ViewerConfig
from nerfstudio.engine.optimizers import AdamOptimizerConfig
from nerfstudio.plugins.registry import discover_methods
from nerfstudio.plugins.types import MethodSpecification
from nerfstudio.engine.trainer import TrainerConfig

# from nerfstudio.fields.sdf_field import SDFFieldConfig
N_BLOCKS = 10
N_SPLIT_DATASET = 10
N_ACTIVE_BLOCK = 1
N_DATASET_CIRCLES = 1 # each subdataset is trained with N_DATA_CIRCLES times
STEPS_PERSSAMPLER_INIT = 30000  # steps focr update octnodes of perssampler ，-1 disable init stage
STEPS_PER_SPLIT_DATASET = 10000 
SCALE_FACTOR = 10.0
use_appearance_embedding = True
# 0-63 64-127 128-192   193-256
gfnerf_config = MethodSpecification(
    TrainerConfig(  
        method_name="gf-nerf", 
        steps_per_eval_batch=1000,
        steps_per_save=2000,
    
        max_num_iterations=STEPS_PERSSAMPLER_INIT + N_DATASET_CIRCLES * STEPS_PER_SPLIT_DATASET * N_SPLIT_DATASET,

        mixed_precision=False,
        pipeline=GFNerfPipelineConfig(
            datamanager=GFNerfDataManagerConfig(
                n_split_dataset=N_SPLIT_DATASET,
                steps_per_split_dataset=STEPS_PER_SPLIT_DATASET,
                steps_perssampler_init = STEPS_PERSSAMPLER_INIT,
                dataparser=NerfstudioDataParserConfig(
                    scale_factor=SCALE_FACTOR,
                    scene_scale=1.0,
                    downscale_factor=1,
                    orientation_method="vertical",
                    train_split_fraction=1.0,  # all cameras
                ),  # type: ignore
                patch_size=1,
                eval_image_indices=(0,),
                train_num_rays_per_batch=2048*4,
                eval_num_rays_per_batch=2048,
                train_num_images_to_sample_from=500,
                train_num_times_to_repeat_images=1000,
                semantic_sample_weights=None,
      

                camera_optimizer=CameraOptimizerConfig(
                    mode="off",
                    optimizer=AdamOptimizerConfig(lr=6e-4, eps=1e-8, weight_decay=1e-2),
                    scheduler=ExponentialDecaySchedulerConfig(lr_final=6e-6, max_steps=30000),
                ),
            ),
            model=GFNeRFModelConfig(
                n_blocks=N_BLOCKS,
                n_split_dataset=N_SPLIT_DATASET,
                steps_per_split_dataset=STEPS_PER_SPLIT_DATASET,
                n_activate_block=N_ACTIVE_BLOCK,
                steps_perssampler_init = STEPS_PERSSAMPLER_INIT,
                scale_factor=SCALE_FACTOR,


                # for s3im loss
                s3im_loss_mult = 1.0,
                s3im_kernel_size = 4,
                s3im_stride = 4,
                s3im_repeat_time = 10,
                s3im_patch_height = 32,


                use_appearance_embedding=use_appearance_embedding,
                mono_depth_loss_mult = 0.00,
                semantic_loss_weight = 0.0,
                patch_size=1,
                eval_num_rays_per_chunk=2048 ,
                predict_normals=False,
                use_normal_loss = False,
                use_depth_loss = False,
                use_ch_loss=True,
                use_perceptual_loss=False,
                disable_scene_contraction=False,
                use_semantics = False,
                num_semantic_classes = 2,
                use_mask=False,
                log2_hashmap_size=21,
                num_levels=16,
                far_plane=50,
                num_nerf_samples_per_ray=256,
                num_proposal_samples_per_ray=(512, 256),
                hidden_dim=128,
                hidden_dim_color=128,
                hidden_dim_transient=128,
                max_res=4096,
                proposal_weights_anneal_max_num_iters=5000,
                background_color="black",
            ),
        ),
        optimizers={
            "fields": {
                "optimizer": AdamOptimizerConfig(lr=1e-2, eps=1e-15),
                "scheduler": GFNerfExponentialDecaySchedulerConfig(lr_final=1e-4, max_steps=STEPS_PERSSAMPLER_INIT,
                                                                        n_split_dataset = N_SPLIT_DATASET,
                                                                        n_dataset_circles = N_DATASET_CIRCLES,
                                                                        steps_per_split_dataset = STEPS_PER_SPLIT_DATASET,
                                                                        steps_perssampler_init = STEPS_PERSSAMPLER_INIT,
                                                                        ),
            },

        },
        viewer=ViewerConfig(num_rays_per_chunk=1 << 8),
        vis="viewer",  
    ),
    description="Nerfacto with depth supervision for multi cameras",
)
