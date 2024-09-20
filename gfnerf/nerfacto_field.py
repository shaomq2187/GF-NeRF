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
Field for compound nerf model, adds scene contraction and image embeddings to instant ngp
"""
import time
import os

from typing import Any, Dict, Iterator, List, Mapping, Optional, Tuple
from rich.progress import Console
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn.parameter import Parameter
from torchtyping import TensorType

from nerfstudio.cameras.rays import RaySamples
from nerfstudio.data.scene_box import SceneBox
from nerfstudio.field_components.activations import trunc_exp
from nerfstudio.field_components.embedding import Embedding
from nerfstudio.field_components.encodings import Encoding, HashEncoding, SHEncoding
from nerfstudio.field_components.field_heads import (
    DensityFieldHead,
    FieldHead,
    FieldHeadNames,
    PredNormalsFieldHead,
    RGBFieldHead,
    SemanticFieldHead,
    TransientDensityFieldHead,
    TransientRGBFieldHead,
    UncertaintyFieldHead,
)
from nerfstudio.field_components.mlp import MLP
from nerfstudio.field_components.spatial_distortions import (
    SceneContraction,
    SpatialDistortion,
)
from nerfstudio.fields.base_field import Field
from nerfstudio.utils import profiler, writer

try:
    import tinycudann as tcnn
except ImportError:
    # tinycudann module doesn't exist
    pass
from gfnerf.hash_3d_anchored import Hash3DAnchored
from gfnerf.mlp import MLPNetwork

CONSOLE = Console(width=120)
def get_normalized_directions(directions: TensorType["bs":..., 3]) -> TensorType["bs":..., 3]:
    """SH encoding must be in the range [0, 1]

    Args:
        directions: batch of directions
    """
    return (directions + 1.0) / 2.0


class GFNeRFField(Field):
    """Compound Field that uses TCNN

    Args:
        aabb: parameters of scene aabb bounds
        num_images: number of images in the dataset
        num_layers: number of hidden layers
        hidden_dim: dimension of hidden layers
        geo_feat_dim: output geo feat dimensions
        num_levels: number of levels of the hashmap for the base mlp
        max_res: maximum resolution of the hashmap for the base mlp
        log2_hashmap_size: size of the hashmap for the base mlp
        num_layers_color: number of hidden layers for color network
        num_layers_transient: number of hidden layers for transient network
        hidden_dim_color: dimension of hidden layers for color network
        hidden_dim_transient: dimension of hidden layers for transient network
        appearance_embedding_dim: dimension of appearance embedding
        transient_embedding_dim: dimension of transient embedding
        use_transient_embedding: whether to use transient embedding
        use_semantics: whether to use semantic segmentation
        num_semantic_classes: number of semantic classes
        use_pred_normals: whether to use predicted normals
        use_average_appearance_embedding: whether to use average appearance embedding or zeros for inference
        spatial_distortion: spatial distortion to apply to the scene
    """

    def __init__(
        self,
        aabb: TensorType,
        num_images: int,
        num_layers: int = 2,
        hidden_dim: int = 64,
        geo_feat_dim: int = 15,
        num_levels: int = 16,
        max_res: int = 2048,
        log2_hashmap_size: int = 19,
        num_layers_color: int = 3,
        num_layers_transient: int = 2,
        hidden_dim_color: int = 64,
        hidden_dim_transient: int = 64,
        appearance_embedding_dim: int = 32,
        transient_embedding_dim: int = 16,
        use_transient_embedding: bool = False,
        use_semantics: bool = False,
        num_semantic_classes: int = 100,
        pass_semantic_gradients: bool = False,
        use_pred_normals: bool = False,
        use_average_appearance_embedding: bool = False,
        use_appearance_embedding: bool = False,
        spatial_distortion: SpatialDistortion = None,
        n_blocks: int = 1,
        n_active_block: int=3,
        steps_perssampler_init: int=10000,
        block_centers: TensorType["n_blocks":...,3] = None,
        base_dir: str = "",
        n_volumes: int = 0,
    ) -> None:
        super().__init__()

        self.register_buffer("aabb", aabb)
        self.geo_feat_dim = geo_feat_dim

        self.register_buffer("max_res", torch.tensor(max_res))
        self.register_buffer("num_levels", torch.tensor(num_levels))
        self.register_buffer("log2_hashmap_size", torch.tensor(log2_hashmap_size))
        self.spatial_distortion = spatial_distortion
        self.num_images = num_images
        self.appearance_embedding_dim = appearance_embedding_dim
        self.embedding_appearance = Embedding(self.num_images, self.appearance_embedding_dim)
        self.use_average_appearance_embedding = use_average_appearance_embedding
        self.use_transient_embedding = use_transient_embedding
        self.use_semantics = use_semantics
        self.use_pred_normals = use_pred_normals
        self.pass_semantic_gradients = pass_semantic_gradients
        self.use_appearance_embedding = use_appearance_embedding

        base_res: int = 16
        features_per_level: int = 2
        growth_factor = np.exp((np.log(max_res) - np.log(base_res)) / (num_levels - 1))
        self.direction_encoding = tcnn.Encoding(
            n_input_dims=3,
            encoding_config={
                "otype": "SphericalHarmonics",
                "degree": 4,
            },
        )

        self.persampler = None  # for trans_query_frame speed up
        self.cur_split_dataset_idx = None
        ###################### encoding and mlp base init ######################
        self.position_encoding = tcnn.Encoding(
            n_input_dims=3,
            encoding_config={"otype": "Frequency", "n_frequencies": 2},
        )
        network_config ={
                "otype": "FullyFusedMLP",
                "activation": "ReLU",
                "output_activation": "None",
                "n_neurons": hidden_dim,
                "n_hidden_layers": num_layers - 1,
            }
        self.base_network = MLPNetwork(
            n_input_dims=16 * 2,  # n_level 和 n_channel写死在了c++头文件里
            # n_input_dims=self.base_encoding.n_output_dims,
            n_output_dims=1 + self.geo_feat_dim,
            network_config=network_config,
        )

        
        # for encodings ckpt
        self.base_dir = base_dir
        self.encodings_ckpt_dir = Path(self.base_dir) / "encodings_ckpt"
        os.makedirs(self.encodings_ckpt_dir,exist_ok=True)
        self.n_volumes = n_volumes
        self.log2_table_size = log2_hashmap_size

        # for block init
        self.n_blocks = n_blocks
        self.block_centers = block_centers
        self.n_active_block = n_active_block
        self.steps_perssampler_init = steps_perssampler_init
        self.single_mlp = True

        self.__setattr__(name=f'base_encoding_init',value = Hash3DAnchored(
                log2_table_size=log2_hashmap_size,
                n_volumes=n_volumes,
                ) )  
        self.base_encoding_init.reset()
        init_feat_pool_, init_prime_pool_, init_bias_pool_, init_n_volume_ = self.base_encoding_init.states()
            






        self.active_block_idxs = []
        self.active_block_idxs_test = []

        self.update_active_blocks(-1) #  active block initialization
        


        #################### mlp head init ##################
        self.mlp_head = MLPNetwork(
            n_input_dims=self.direction_encoding.n_output_dims + self.geo_feat_dim + self.appearance_embedding_dim,
            n_output_dims=3,
            network_config={
                "otype": "FullyFusedMLP",
                "activation": "ReLU",
                "output_activation": "Sigmoid",
                "n_neurons": hidden_dim_color,
                "n_hidden_layers": num_layers_color - 1,
            },
        )
        # semantics head
        if self.use_semantics:
            self.mlp_semantics = MLPNetwork(
                n_input_dims=self.geo_feat_dim,
                n_output_dims=hidden_dim_transient,
                network_config={
                    "otype": "FullyFusedMLP",
                    "activation": "ReLU",
                    "output_activation": "None",
                    "n_neurons": 64,
                    "n_hidden_layers": 1,
                },
            )
            self.field_head_semantics = SemanticFieldHead(
                in_dim=self.mlp_semantics.n_output_dims, num_classes=num_semantic_classes
            )

        


    def update_active_blocks(self,cur_split_data_idx,ray_samples = None):
        # everey subdataset has a block and activate corresponding block

        if cur_split_data_idx == -1:
            # -1 indicates initialization process
            active_idxs = []
        else:
            active_idxs = [cur_split_data_idx]
        if self.training:
            list_in_train = active_idxs
            list_in_test = self.active_block_idxs_test
        else:
            list_in_train = self.active_block_idxs
            list_in_test = active_idxs
        list_not_in_train_or_test = list(set(range(self.n_blocks)).difference(set(list_in_train + list_in_test)))

        # block in train : enable all: to cuda, enable grad
        # block in test: to cuda, disable grad
        # block not in train and test: disable all: to cpu and disable grad


 
        # disable require_grad
        for i in list_in_test:

            # not loaded
            if not hasattr(self,f"base_encoding_{i}"):
                self.add_table(i)
                self.load_table(i,strict=True) # must exist
            
            encoding_field = self.__getattr__(f"base_encoding_{i}")
            encoding_field.to("cuda:0")
            encoding_field.set_require_grad(False)  # set require_grad must operate on cuda
            if not self.single_mlp:
                base_network = self.__getattr__(f"base_network_{i}")
                base_network = base_network.cuda()
                base_network.requires_grad_(False)
                mlp_head = self.__getattr__(f"mlp_head_{i}")
                mlp_head = mlp_head.cuda()
                mlp_head.requires_grad_(False)

        # enable all
        for i in list_in_train:
            if not hasattr(self,f"base_encoding_{i}"):
                self.add_table(i)
                self.load_table(i,strict=False) 
            
            encoding_field = self.__getattr__(f"base_encoding_{i}")
            encoding_field.to("cuda:0")
            encoding_field.set_require_grad(True)   
            encoding_field.train()

            if not self.single_mlp:
                base_network = self.__getattr__(f"base_network_{i}")
                base_network = base_network.cuda()
                base_network.requires_grad_(True)

                mlp_head = self.__getattr__(f"mlp_head_{i}")
                mlp_head = mlp_head.cuda()
                mlp_head.requires_grad_(True)
        # disable_all
                
        for i in list_not_in_train_or_test:
            if hasattr(self,f"base_encoding_{i}"):
                self.save_table(i)
                self.del_table(i)


            

            if not self.single_mlp:
                base_network = self.__getattr__(f"base_network_{i}")
                base_network.requires_grad_(False)

                mlp_head = self.__getattr__(f"mlp_head_{i}")
                mlp_head.requires_grad_(False)
        

        # update activate idxs
        if self.training:
            self.active_block_idxs = active_idxs
        else:
            self.active_block_idxs_test = active_idxs

        



    def add_table(self,table_idx):
        # add table to the field
        print(f'[debug] add table {table_idx}')
        assert not hasattr(self,f'base_encoding_{table_idx}')

        encoding_field = Hash3DAnchored(
                        log2_table_size=self.log2_table_size,
                        n_volumes=self.n_volumes,
                        )
        encoding_field.zero()
        encoding_field.set_require_grad(True)
        self.__setattr__(name=f'base_encoding_{table_idx}',value = encoding_field)

    def del_table(self,table_idx):
        # 删除一个已经存在的哈希表，调用前保证optimizer已经删除
        # delete table from field
        print(f'[debug] delete table {table_idx}')
        assert hasattr(self,f'base_encoding_{table_idx}')

        # unregister hooks to remove references
        base_encoding = getattr(self,f'base_encoding_{table_idx}')
        base_encoding.unregister_hooks()
        base_encoding.release_resources()


        self.__delattr__(f'base_encoding_{table_idx}')
 

        torch.cuda.empty_cache() # 释放显存


    
    def save_table(self,table_idx):
        # save table to disk
        print(f'[debug] save_table table {table_idx}')

        assert hasattr(self,f'base_encoding_{table_idx}')

        encoding_field = self.__getattr__(f'base_encoding_{table_idx}')
        states = encoding_field.state_dict()

        ckpt_path: Path = self.encodings_ckpt_dir / f"base_encoding_{table_idx}.ckpt"
        if not os.path.exists(ckpt_path):
            print('[debug] save ckpt:')
            torch.save(states,ckpt_path)
        else:
            print('[debug] warning! ckpt has exist, no overwriting!')
            torch.save(states,ckpt_path)
    
    
    def load_table(self,table_idx,strict = False):
        # load table from disk
        assert hasattr(self,f'base_encoding_{table_idx}')
        encoding_field = self.__getattr__(f'base_encoding_{table_idx}')

        ckpt_path: Path = self.encodings_ckpt_dir / f"base_encoding_{table_idx}.ckpt"
        print('ckpt_pah:',ckpt_path)
        if strict :
            assert os.path.exists(ckpt_path)
        if os.path.exists(ckpt_path):
            states = torch.load(ckpt_path,map_location=lambda storage, loc: storage)
            encoding_field.load_state_dict(states)
            del states
            torch.cuda.empty_cache()

        else:
            # TODO: train from checkpoint will have bug here
            print('[debug] !!!warning! ckpt not exists!!!')


    def memory_stats(self):
        allocated_memory = torch.cuda.memory_allocated()
        cached_memory = torch.cuda.memory_cached()
        print(f"Allocated Memory: {allocated_memory / 1024**3:.2f} GB")
        print(f"Cached Memory: {cached_memory / 1024**3:.2f} GB")
        
    def get_density(self, ray_samples: RaySamples) -> Tuple[TensorType, TensorType]:
        """Computes and returns the densities."""
        import time 
        start = time.time()

        # # proposal sampler
        # positions = ray_samples.frustums.get_positions()
        # tree_node_idxs = ray_samples.frustums.tree_node_idx
        # positions_flat = positions.view(-1,3)
        # tree_node_idxs_flat = tree_node_idxs.view(-1).contiguous().to(torch.int64)
        # sampled_pts_flat = self.persampler.trans_query_frame(positions_flat,tree_node_idxs_flat).contiguous()
        # sampled_pts_flat = (sampled_pts_flat + 1.5) / 3.0



        # for tree_idx in self.tree_idx_to_pers_idx:
        #     anchors[torch.where(tree_node_idxs_flat == tree_idx)] = self.tree_idx_to_pers_idx[tree_idx]
        
        sampled_pts = ray_samples.f2samples.sampled_pts.contiguous()
        sampled_pts = (sampled_pts + 1.5) / 3.0
        self._sample_locations = sampled_pts
        if not self._sample_locations.requires_grad:
            self._sample_locations.requires_grad = True
 
        sampled_pts_flat = sampled_pts.contiguous().view(-1, 3)
        anchors = ray_samples.f2samples.sampled_anchors[:, :, 0].contiguous().view(-1)
        anchors_block_idx = ray_samples.f2samples.sampled_anchors[:, :, 2].contiguous().view(-1)



        h = torch.zeros((anchors.shape[0], 1 + self.geo_feat_dim), dtype=torch.float32, device=anchors.device)
        valid_mask = anchors > -1
        block_valid_mask = valid_mask


        self.cur_step = ray_samples.cur_step[0][0].item()
        self.cur_stage = 'init_stage' if (self.steps_perssampler_init >0 and self.cur_step>=0 and self.cur_step < self.steps_perssampler_init) else 'block_stage'
        self.cur_stage = 'init_stage'
        if self.cur_stage == 'init_stage':
            # init stage
            self.base_encoding_init.set_require_grad(True)
            self.base_network.requires_grad_(True)
            hash_feats = self.base_encoding_init([sampled_pts_flat[valid_mask],anchors[valid_mask]])
            h[valid_mask] = self.base_network(hash_feats)
            self.cur_split_dataset_idx = -1
        
        elif self.cur_stage == 'block_stage':
            # 1.disable init encoding 
            self.base_encoding_init.set_require_grad(False)
            self.base_network.requires_grad_(False)

            # 2.qurey from blocks
            cur_split_dataset_idx = ray_samples.cur_split_dataset_idx[0][0].item()
            self.cur_split_dataset_idx = cur_split_dataset_idx

            self.update_active_blocks(cur_split_dataset_idx)
            
            valid_count = 0
            active_block_idxs = self.active_block_idxs if self.training else self.active_block_idxs_test
 
      
            

            # 利用残差结构的多个哈希表生成
            assert len(active_block_idxs) == 1
            hash_feats = self.base_encoding_init([sampled_pts_flat[valid_mask],anchors[valid_mask]])
            h[valid_mask] = self.base_network(hash_feats)

            cur_active_block = active_block_idxs[0]
            for i in active_block_idxs:
                valid_count += torch.sum(valid_mask).item()
                if torch.sum(valid_mask) > 0: 
                    encoding_field = self.__getattr__(f"base_encoding_{i}")
                    residual_hash_feats = encoding_field([sampled_pts_flat[valid_mask], anchors[valid_mask]])
                    hash_feats += residual_hash_feats # redisual in hast feats level

            assert self.single_mlp 
            h[valid_mask] = self.base_network(hash_feats)
        h = h.view(*ray_samples.frustums.shape, -1)
        density_before_activation, base_mlp_out = torch.split(h, [1, self.geo_feat_dim], dim=-1)
        
        self._density_before_activation = density_before_activation
        

        # Rectifying the density with an exponential is much more stable than a ReLU or
        # softplus, because it enables high post-activation (float32) density outputs
        # from smaller internal (float16) parameters.
        density = trunc_exp(density_before_activation + 1.0)
        valid_mask = anchors > -1
        valid_mask = valid_mask & block_valid_mask # block_valid_mask
        valid_mask = valid_mask.view(*ray_samples.shape, -1)
        density = density * valid_mask


        
        return density, base_mlp_out

    def get_outputs(
        self, ray_samples: RaySamples, density_embedding: Optional[TensorType] = None
    ) -> Dict[FieldHeadNames, TensorType]:
        assert density_embedding is not None
        outputs = {}
        if ray_samples.camera_indices is None:
            raise AttributeError("Camera indices are not provided.")
        camera_indices = ray_samples.camera_indices.squeeze()
        rel_camera_indices = ray_samples.rel_camera_indices.squeeze()
        # print(f'[debug] rel_camera_indices: min{torch.min(rel_camera_indices)} max{torch.max(rel_camera_indices)}')
        directions = get_normalized_directions(ray_samples.frustums.directions)
        directions_flat = directions.view(-1, 3)
        d = self.direction_encoding(directions_flat)

        outputs_shape = ray_samples.frustums.directions.shape[:-1]

        # appearance
        if self.use_appearance_embedding:
            if self.cur_stage == 'block_stage':
                self.embedding_appearance.embedding.requires_grad_(False)
            if self.training:
                embedded_appearance = self.embedding_appearance(rel_camera_indices)
            else:
                print('[debug] using embedding:',rel_camera_indices)
                embedded_appearance = self.embedding_appearance(rel_camera_indices) # test mode use nearest rel camera dincies
        else:
            # disable embedding
            embedded_appearance = torch.zeros(
                (*directions.shape[:-1], self.appearance_embedding_dim), device=directions.device
            )

        h = torch.cat(
            [
                d,
                density_embedding.view(-1, self.geo_feat_dim),
                embedded_appearance.view(-1, self.appearance_embedding_dim),
            ],
            dim=-1,
        )
        if self.cur_stage == "block_stage":
            self.mlp_head.requires_grad_(False) # disable color mlp
        else:
            self.mlp_head.requires_grad_(True)


        if self.cur_stage == "init_stage":
            rgb = self.mlp_head(h).view(*outputs_shape, -1).to(directions)
        else:
            rgb = torch.zeros_like(self.mlp_head(h))
            if hasattr(self,'proposal_sampler'):
                anchors = ray_samples.frustums.tree_node_idx.view(-1).contiguous().to(torch.int64)
            else:
                anchors = ray_samples.f2samples.sampled_anchors[:, :, 0].contiguous().view(-1)

            valid_mask = anchors > -10 
            block_valid_mask = valid_mask

            if self.training:
                active_block_idxs = self.active_block_idxs
            else:
                active_block_idxs = self.active_block_idxs_test

            for i in active_block_idxs:
                cur_mask = valid_mask
                if torch.sum(cur_mask) > 0:
                    if self.single_mlp:
                        rgb[cur_mask] = self.mlp_head(h[cur_mask])
                    else:
                        cur_mlp_head = self.__getattr__(f"mlp_head_{i}")
                        rgb[cur_mask] = cur_mlp_head(h[cur_mask])
            rgb = rgb.view(*outputs_shape, -1).to(directions)

        # semantics
        if self.use_semantics:
            semantics_input = density_embedding.view(-1, self.geo_feat_dim)
            if not self.pass_semantic_gradients:
                semantics_input = semantics_input.detach()

            x = self.mlp_semantics(semantics_input).view(*outputs_shape, -1).to(directions)
            outputs[FieldHeadNames.SEMANTICS] = self.field_head_semantics(x)
        outputs.update({FieldHeadNames.RGB: rgb})

        return outputs

    def parameters(self, recurse: bool = True) -> Iterator[Parameter]:
        print("[debug] nerfacto paramter")
        for param in super().parameters(recurse):
            yield param
        
        for params in self.__getattr__(f"base_encoding_init").parameters(recurse):
            yield params




