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
Dataset.
"""
from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Dict, List

import numpy as np
import numpy.typing as npt
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchtyping import TensorType

from nerfstudio.data.dataparsers.base_dataparser import DataparserOutputs
from nerfstudio.data.utils.data_utils import (
get_image_mask_tensor_from_path,
get_normal_image_from_path,
get_depth_image_from_path,
get_semantics_from_path,
get_error_map_from_path,
)


class InputDataset(Dataset):
    """Dataset that returns images.

    Args:
        dataparser_outputs: description of where and how to read input images.
        scale_factor: The scaling factor for the dataparser outputs
    """

    def __init__(self, dataparser_outputs: DataparserOutputs, scale_factor: float = 1.0):
        super().__init__()
        self._dataparser_outputs = dataparser_outputs
        self.has_masks = dataparser_outputs.mask_filenames is not None
        self.has_depths = dataparser_outputs.metadata["depth_filenames"] is not None
        self.has_normals = dataparser_outputs.metadata["normal_filenames"] is not None
        self.has_road_masks = dataparser_outputs.metadata["road_mask_filenames"] is not None
        self.has_all_masks = dataparser_outputs.metadata["all_mask_filenames"] is not None
        self.has_global_image_indices = dataparser_outputs.metadata["global_image_indices"] is not None
        self.has_error_maps = "error_map_filenames" in dataparser_outputs.metadata and dataparser_outputs.metadata["error_map_filenames"] is not None

        if self.has_global_image_indices:
            self.global_image_indices = dataparser_outputs.metadata["global_image_indices"]
        self.scale_factor = scale_factor
        self.scene_box = deepcopy(dataparser_outputs.scene_box)
        self.metadata = deepcopy(dataparser_outputs.metadata)
        self.cameras = deepcopy(dataparser_outputs.cameras)
        self.cameras.rescale_output_resolution(scaling_factor=scale_factor)

    def __len__(self):
        return len(self._dataparser_outputs.image_filenames)

    def get_numpy_image(self, image_idx: int) -> npt.NDArray[np.uint8]:
        """Returns the image of shape (H, W, 3 or 4).

        Args:
            image_idx: The image index in the dataset.
        """
        image_filename = self._dataparser_outputs.image_filenames[image_idx]
        pil_image = Image.open(image_filename)
        if self.scale_factor != 1.0:
            width, height = pil_image.size
            newsize = (int(width * self.scale_factor), int(height * self.scale_factor))
            pil_image = pil_image.resize(newsize, resample=Image.BILINEAR)
        image = np.array(pil_image, dtype="uint8")  # shape is (h, w) or (h, w, 3 or 4)
        if len(image.shape) == 2:
            image = image[:, :, None].repeat(3, axis=2)
        assert len(image.shape) == 3
        assert image.dtype == np.uint8
        assert image.shape[2] in [3, 4], f"Image shape of {image.shape} is in correct."
        return image

    def get_image(self, image_idx: int) -> TensorType["image_height", "image_width", "num_channels"]:
        """Returns a 3 channel image.

        Args:
            image_idx: The image index in the dataset.
        """
        image = torch.from_numpy(self.get_numpy_image(image_idx).astype("float32") / 255.0)
        if self._dataparser_outputs.alpha_color is not None and image.shape[-1] == 4:
            assert image.shape[-1] == 4
            image = image[:, :, :3] * image[:, :, -1:] + self._dataparser_outputs.alpha_color * (1.0 - image[:, :, -1:])
        else:
            image = image[:, :, :3]
        return image

    def get_data(self, image_idx: int) -> Dict:
        """Returns the ImageDataset data as a dictionary.

        Args: 
            image_idx: The image index in the dataset.
        """
        image = self.get_image(image_idx) # (H,W,3)
        data = {"image_idx": image_idx}

        if self.has_global_image_indices:
            data["rel_camera_idx"] = self.global_image_indices[image_idx]

        data["image"] = image
        if self.has_masks:
            mask_filepath = self._dataparser_outputs.mask_filenames[image_idx]
            data["mask"] = get_image_mask_tensor_from_path(filepath=mask_filepath, scale_factor=self.scale_factor)
            assert (
                data["mask"].shape[:2] == data["image"].shape[:2]
            ), f"Mask and image have different shapes. Got {data['mask'].shape[:2]} and {data['image'].shape[:2]}"
        # print(f'image idx:{image_idx} len depth:{len(self._dataparser_outputs.metadata["depth_filenames"])}')
        if self.has_depths:
            depth_filepath = self._dataparser_outputs.metadata["depth_filenames"][image_idx]
            height,width,_ = image.shape
            data["depth"] = get_depth_image_from_path(filepath=depth_filepath, height=height, width=width, scale_factor=1.0)
            assert (
                data["depth"].shape[:2] == data["image"].shape[:2]
            ), f"depth and image have different shapes. Got {data['depth'].shape[:2]} and {data['image'].shape[:2]}"
        
        if self.has_normals:
            normal_filepath = self._dataparser_outputs.metadata["normal_filenames"][image_idx]
            data["normal"] = get_normal_image_from_path(filepath=normal_filepath, scale_factor=self.scale_factor)
            assert (
                data["normal"].shape[:2] == data["image"].shape[:2]
            ), f"normal and image have different shapes. Got {data['normal'].shape[:2]} and {data['image'].shape[:2]}"
        
        if self.has_road_masks:
            road_mask_filepath = self._dataparser_outputs.metadata['road_mask_filenames'][image_idx]
            data["road_mask"] = get_image_mask_tensor_from_path(filepath=road_mask_filepath, scale_factor=self.scale_factor)
            assert (
                data["road_mask"].shape[:2] == data["image"].shape[:2]
            ), f"road_mask and image have different shapes. Got {data['road_mask'].shape[:2]} and {data['image'].shape[:2]}"  
        if self.has_all_masks:
            all_mask_filepath = self._dataparser_outputs.metadata['all_mask_filenames'][image_idx]
            data["all_mask"] = get_semantics_from_path(filepath=all_mask_filepath, mask_indices=[],scale_factor=self.scale_factor)
            assert (
                data["all_mask"].shape[:2] == data["image"].shape[:2]
            ), f"all_mask and image have different shapes. Got {data['all_mask'].shape[:2]} and {data['image'].shape[:2]}"  
        if self.has_error_maps:
            error_map_path = self._dataparser_outputs.metadata["error_map_filenames"][image_idx]
            data["error_map"] = get_error_map_from_path(error_map_path)
            # data["error_map"] = torch.ones_like(data["error_map"]) * 0.1
        metadata = self.get_metadata(data)
        data.update(metadata)
        return data

    # pylint: disable=no-self-use
    def get_metadata(self, data: Dict) -> Dict:
        """Method that can be used to process any additional metadata that may be part of the model inputs.

        Args:
            image_idx: The image index in the dataset.
        """
        del data
        return {}

    def __getitem__(self, image_idx: int) -> Dict:
        data = self.get_data(image_idx)
        return data

    @property
    def image_filenames(self) -> List[Path]:
        """
        Returns image filenames for this dataset.
        The order of filenames is the same as in the Cameras object for easy mapping.
        """

        return self._dataparser_outputs.image_filenames
