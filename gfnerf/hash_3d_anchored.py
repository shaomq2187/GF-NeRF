import gc
import importlib
import warnings
from typing import Any, Dict, Iterator, List, Mapping, Optional, Tuple

import numpy as np
import tinycudann as tcnn
import torch
from torch import nn
from torch.nn.parameter import Parameter
from torchtyping import TensorType

module_path = "../GF-NeRF/gfnerf/bindings/f2nerf-bindings.so"
torch.classes.load_library(module_path)


class Hash3DAnchored(nn.Module):
    def __init__(
        self,
        log2_table_size: int,
        n_volumes: int,
    ) -> None:
        super().__init__()
        print(f"[Hash3DAnchored] Init log2_table size:{log2_table_size} n_volumes:{n_volumes}")
        self.hash_3d = torch.classes.my_classes.Hash3DAnchored(log2_table_size, n_volumes, 1e-1)
        self.register_parameter("feat_pool", None)
        self.register_buffer("prime_pool", None)
        self.register_buffer("bias_pool", None)
        self.register_buffer("n_volumes", None)

        self.hook_handles = []
        state_dict_hook_handle = self._register_state_dict_hook(self.state_dict_hook)
        self.hook_handles.append(state_dict_hook_handle)

    def __del__(self):
        print("[Hash3DAnchored] __del__ is called")
        
        del self.hash_3d


    def unregister_hooks(self):
        print("[Hash3DAnchored] unregister_hooks is called")
        for hook_handle in self.hook_handles:
            hook_handle.remove()
        
    def state_dict_hook(self, *args):
        print("[Hash3DAnchored] state_dict is called")

        destination = args[1]
        prefix = args[2]
        feat_pool_, prime_pool_, bias_pool_, n_volume_ = self.states()
        destination[prefix + "feat_pool"] = feat_pool_
        destination[prefix + "prime_pool"] = prime_pool_
        destination[prefix + "bias_pool"] = bias_pool_
        destination[prefix + "n_volumes"] = n_volume_

        return destination

    def forward(self, input):
        points, anchors = input
        assert len(anchors.shape) == 1
        assert points.shape[1] == 3
        assert len(points.shape) == 2
        assert anchors.dtype == torch.int64
        results = self.anchored_query(points, anchors)
        # print("[debug] self.bias_pool min max:", torch.min(self.bias_pool), torch.max(self.bias_pool))
        # _, prime_pool, bias_pool, n_volume_ = self.states()
        # print("[debug] self.states bias_pool min max:", torch.min(bias_pool), torch.max(bias_pool))
        # print("[debug] hash3danchored results mean:", torch.mean(results))

        return results

    def parameters(self, recurse: bool = True) -> Iterator[Parameter]:
        print("[debug] Hash3DAnchored.parameters()")
        # 必须要显式指定parameters
        params = self.hash_3d.GetParams()
        for p in params:
            yield p

    def anchored_query(
        self, points: TensorType["num_pts":..., 3], anchors: TensorType["num_pts":..., 1]
    ) -> TensorType["num_pts":..., "mlp_out_dim":...]:
        out = self.hash_3d.AnchoredQuery(points, anchors)
        return out
    
    def load_state_dict(self, state_dict: Mapping[str, Any], strict: bool = True,prefix = ''):
        print("[Hash3DAnchored] load_state_dict")
        states = []
        if prefix == '':
            states.append(state_dict[f"feat_pool"])
            states.append(state_dict[f"prime_pool"])
            states.append(state_dict[f"bias_pool"])
            states.append(state_dict[f"n_volumes"])
            del state_dict[f"feat_pool"]
            del state_dict[f"prime_pool"]
            del state_dict[f"bias_pool"]
            del state_dict[f"n_volumes"]        
        else:
            states.append(state_dict[f"{prefix}.feat_pool"])
            states.append(state_dict[f"{prefix}.prime_pool"])
            states.append(state_dict[f"{prefix}.bias_pool"])
            states.append(state_dict[f"{prefix}.n_volumes"])
            del state_dict[f"{prefix}.feat_pool"]
            del state_dict[f"{prefix}.prime_pool"]
            del state_dict[f"{prefix}.bias_pool"]
            del state_dict[f"{prefix}.n_volumes"]

        self.load_states(states, idx=0)
        return None
    def set_require_grad(self,require_grad):
        assert type(require_grad) == bool
        self.hash_3d.SetFeatPoolRequireGrad(require_grad)

    def to(self,device):
        assert type(device) == str
        self.hash_3d.to(device)
        
        
    def load_states(self, states: List[TensorType], idx: int) -> int:
        return self.hash_3d.LoadStates(states, idx)

    # def to(self, device):
    #     return self

    def states(self) -> List[TensorType]:
        return self.hash_3d.States()

    def get_params(self) -> List[TensorType]:
        return self.hash_3d.GetParams()

    def reset(self) -> None:
        self.hash_3d.Reset()
    def zero(self) -> None:
        print('[Hash3DAnchored] zero feat_pool')
        self.hash_3d.Zero()
    def release_resources(self) -> None:
        print('[Hash3DAnchored] release_resources')

        self.hash_3d.ReleaseResources()


if __name__ == "__main__":
    hash3d = Hash3DAnchored(19, 500)
    base_network = tcnn.Network(
        n_input_dims=16 * 2,  # n_level 和 n_channel写死在了c++头文件里
        n_output_dims=1 + 16,
        network_config={
            "otype": "FullyFusedMLP",
            "activation": "ReLU",
            "output_activation": "None",
            "n_neurons": 128,
            "n_hidden_layers": 2 - 1,
        },
    )
    a = hash3d.state_dict()

    # encoing_config = {
    #     "otype": "HashGrid",
    #     "n_levels": 18,
    #     "n_features_per_level": 2,
    #     "log2_hashmap_size": 19,
    #     "base_resolution": 4096,
    #     "per_level_scale": 1.5,
    # }
    # hash3d = tcnn.Encoding(n_input_dims=3, encoding_config=encoing_config)  # 所有不在八叉树内的点所对应的hash grid
    device = torch.device("cuda:0")
    opt_params = []
    for p in hash3d.parameters():
        opt_params.append(p)
    # for p in base_network.parameters():
    #     opt_params.append(p)

    optimizer = torch.optim.Adam(opt_params, lr=1e-2)
    optimizer_2 = torch.optim.Adam(base_network.parameters(), lr=1e-2)

    points = 2 * torch.rand((100, 3), device=device, dtype=torch.float32)
    print("max:", torch.max(points))
    print("min:", torch.min(points))

    anchors = torch.randint_like(points, 1000, device=device, dtype=torch.int)
    anchors = anchors[:, 0].reshape(-1)

    for i in range(0, 1000):
        feats = hash3d.anchored_query(points, anchors)  # (100,32)
        results = base_network(feats)
        # print("results sum:", results[0][0])
        # results = hash3d(points)
        loss = (torch.abs(results - 1)).sum()
        optimizer.zero_grad()
        optimizer_2.zero_grad()
        loss.backward()
        optimizer.step()
        optimizer_2.step()

        # print("paramas sum:", torch.sum(hash3d.get_params()[0]))
        # print("paramas【0】 sum:", torch.sum(params[0]))

        print("loss:", loss)
        # # print("grad 0:", torch.max(torch.abs(params[0].grad)))
        # # print("grad 1:", torch.max(torch.abs(params[1].grad)))
        # # print("params[0] sum:", torch.sum(params[0]))
        print("results sum:", torch.sum(results))
        print(" ")
