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
Code for sampling pixels.
"""

import random
from typing import Dict, Optional, Union

import torch
from torchtyping import TensorType
import math
import numpy as np
class PixelSampler:  # pylint: disable=too-few-public-methods
    """Samples 'pixel_batch's from 'image_batch's.

    Args:
        num_rays_per_batch: number of rays to sample per batch
        keep_full_image: whether or not to include a reference to the full image in returned batch
    """

    def __init__(self, num_rays_per_batch: int, keep_full_image: bool = False, **kwargs) -> None:
        self.kwargs = kwargs
        self.num_rays_per_batch = num_rays_per_batch
        self.keep_full_image = keep_full_image

    def set_num_rays_per_batch(self, num_rays_per_batch: int):
        """Set the number of rays to sample per batch.

        Args:
            num_rays_per_batch: number of rays to sample per batch
        """
        self.num_rays_per_batch = num_rays_per_batch

    def sample_method(  # pylint: disable=no-self-use
        self,
        batch_size: int,
        num_images: int,
        image_height: int,
        image_width: int,
        mask: Optional[TensorType] = None,
        device: Union[torch.device, str] = "cpu",
    ) -> TensorType["batch_size", 3]:
        """
        Naive pixel sampler, uniformly samples across all possible pixels of all possible images.

        Args:
            batch_size: number of samples in a batch
            num_images: number of images to sample over
            mask: mask of possible pixels in an image to sample from.
        """
        if isinstance(mask, torch.Tensor):
            nonzero_indices = torch.nonzero(mask[..., 0].cpu(), as_tuple=False)
            chosen_indices = random.sample(range(len(nonzero_indices)), k=batch_size)
            indices = nonzero_indices[chosen_indices].to(mask.device)

            # nonzero_indices = torch.nonzero(mask[..., 0], as_tuple=False)
            # chosen_indices = random.sample(range(len(nonzero_indices)), k=batch_size)
            # indices = nonzero_indices[chosen_indices]
        else:
            indices = torch.floor(
                torch.rand((batch_size, 3), device=device)
                * torch.tensor([num_images, image_height, image_width], device=device)
            ).long()

        return indices

    def collate_image_dataset_batch(self, batch: Dict, num_rays_per_batch: int, keep_full_image: bool = False):
        """
        Operates on a batch of images and samples pixels to use for generating rays.
        Returns a collated batch which is input to the Graph.
        It will sample only within the valid 'mask' if it's specified.

        Args:
            batch: batch of images to sample from
            num_rays_per_batch: number of rays to sample per batch
            keep_full_image: whether or not to include a reference to the full image in returned batch
        """

        device = batch["image"].device
        num_images, image_height, image_width, _ = batch["image"].shape

        if "mask_bak" in batch:
            indices = self.sample_method(
                num_rays_per_batch, num_images, image_height, image_width, mask=batch["mask"], device=device
            )
        else:
            indices = self.sample_method(num_rays_per_batch, num_images, image_height, image_width, device=device)

        c, y, x = (i.flatten() for i in torch.split(indices, 1, dim=-1))
        collated_batch = {
            key: value[c, y, x] for key, value in batch.items() if key not in ["image_idx",'rel_camera_idx'] and value is not None
        }

        assert collated_batch["image"].shape == (num_rays_per_batch, 3), collated_batch["image"].shape

        # Needed to correct the random indices to their actual camera idx locations.
        c = torch.clone(c)
        indices[:, 0] = batch["image_idx"][c]

        collated_batch["indices"] = indices  # with the abs camera indices
        collated_batch['rel_camera_indices'] = batch["rel_camera_idx"][c] # with the relative camera indices
        if keep_full_image:
            collated_batch["full_image"] = batch["image"]
        return collated_batch

    def collate_image_dataset_batch_list(self, batch: Dict, num_rays_per_batch: int, keep_full_image: bool = False):
        """
        Does the same as collate_image_dataset_batch, except it will operate over a list of images / masks inside
        a list.

        We will use this with the intent of DEPRECIATING it as soon as we find a viable alternative.
        The intention will be to replace this with a more efficient implementation that doesn't require a for loop, but
        since pytorch's ragged tensors are still in beta (this would allow for some vectorization), this will do.

        Args:
            batch: batch of images to sample from
            num_rays_per_batch: number of rays to sample per batch
            keep_full_image: whether or not to include a reference to the full image in returned batch
        """

        device = batch["image"][0].device
        num_images = len(batch["image"])

        # only sample within the mask, if the mask is in the batch
        all_indices = []
        all_images = []

        if "mask_bak" in batch:
            num_rays_in_batch = num_rays_per_batch // num_images
            for i in range(num_images):
                image_height, image_width, _ = batch["image"][i].shape

                if i == num_images - 1:
                    num_rays_in_batch = num_rays_per_batch - (num_images - 1) * num_rays_in_batch

                indices = self.sample_method(
                    num_rays_in_batch, 1, image_height, image_width, mask=batch["mask"][i], device=device
                )
                indices[:, 0] = i
                all_indices.append(indices)
                all_images.append(batch["image"][i][indices[:, 1], indices[:, 2]])

        else:
            num_rays_in_batch = num_rays_per_batch // num_images
            for i in range(num_images):
                image_height, image_width, _ = batch["image"][i].shape
                if i == num_images - 1:
                    num_rays_in_batch = num_rays_per_batch - (num_images - 1) * num_rays_in_batch
                indices = self.sample_method(num_rays_in_batch, 1, image_height, image_width, device=device)
                indices[:, 0] = i
                all_indices.append(indices)
                all_images.append(batch["image"][i][indices[:, 1], indices[:, 2]])

        indices = torch.cat(all_indices, dim=0)

        c, y, x = (i.flatten() for i in torch.split(indices, 1, dim=-1))

        collated_batch = {
            key: value[c, y, x]
            for key, value in batch.items()
            if key not in  ["image_idx",'rel_camera_idx'] and key != "image" and key != "mask" and value is not None
        }

        collated_batch["image"] = torch.cat(all_images, dim=0)

        assert collated_batch["image"].shape == (num_rays_per_batch, 3), collated_batch["image"].shape

        # Needed to correct the random indices to their actual camera idx locations.
        c = torch.clone(c)
        indices[:, 0] = batch["image_idx"][c] # !!这里会修改c的值,所以之前要clone
        collated_batch["indices"] = indices  # with the abs camera indices
        collated_batch['rel_camera_indices'] = batch["rel_camera_idx"][c] # with the relative camera indices

        if keep_full_image:
            collated_batch["full_image"] = batch["image"]

        return collated_batch

    def sample(self, image_batch: Dict):
        """Sample an image batch and return a pixel batch.

        Args:
            image_batch: batch of images to sample from
        """
        if isinstance(image_batch["image"], list):
            image_batch = dict(image_batch.items())  # copy the dictionary so we don't modify the original
            pixel_batch = self.collate_image_dataset_batch_list(
                image_batch, self.num_rays_per_batch, keep_full_image=self.keep_full_image
            )
        elif isinstance(image_batch["image"], torch.Tensor):
            pixel_batch = self.collate_image_dataset_batch(
                image_batch, self.num_rays_per_batch, keep_full_image=self.keep_full_image
            )
        else:
            raise ValueError("image_batch['image'] must be a list or torch.Tensor")
        return pixel_batch


class EquirectangularPixelSampler(PixelSampler):  # pylint: disable=too-few-public-methods
    """Samples 'pixel_batch's from 'image_batch's. Assumes images are
    equirectangular and the sampling is done uniformly on the sphere.

    Args:
        num_rays_per_batch: number of rays to sample per batch
        keep_full_image: whether or not to include a reference to the full image in returned batch
    """

    # overrides base method
    def sample_method(  # pylint: disable=no-self-use
        self,
        batch_size: int,
        num_images: int,
        image_height: int,
        image_width: int,
        mask: Optional[TensorType] = None,
        device: Union[torch.device, str] = "cpu",
    ) -> TensorType["batch_size", 3]:
        if isinstance(mask, torch.Tensor):
            # Note: if there is a mask, sampling reduces back to uniform sampling, which gives more
            # sampling weight to the poles of the image than the equators.
            # TODO(kevinddchen): implement the correct mask-sampling method.

            indices = super().sample_method(batch_size, num_images, image_height, image_width, mask=mask, device=device)
        else:
            # We sample theta uniformly in [0, 2*pi]
            # We sample phi in [0, pi] according to the PDF f(phi) = sin(phi) / 2.
            # This is done by inverse transform sampling.
            # http://corysimon.github.io/articles/uniformdistn-on-sphere/
            num_images_rand = torch.rand(batch_size, device=device)
            phi_rand = torch.acos(1 - 2 * torch.rand(batch_size, device=device)) / torch.pi
            theta_rand = torch.rand(batch_size, device=device)
            indices = torch.floor(
                torch.stack((num_images_rand, phi_rand, theta_rand), dim=-1)
                * torch.tensor([num_images, image_height, image_width], device=device)
            ).long()

        return indices


class PatchPixelSampler(PixelSampler):  # pylint: disable=too-few-public-methods
    """Samples 'pixel_batch's from 'image_batch's. Samples square patches
    from the images randomly. Useful for patch-based losses.

    Args:
        num_rays_per_batch: number of rays to sample per batch
        keep_full_image: whether or not to include a reference to the full image in returned batch
        patch_size: side length of patch. This must be consistent in the method
        config in order for samples to be reshaped into patches correctly.
    """

    def __init__(self, num_rays_per_batch: int, keep_full_image: bool = False, **kwargs) -> None:
        self.patch_size = kwargs["patch_size"]
        num_rays = (num_rays_per_batch // (self.patch_size**2)) * (self.patch_size**2)
        super().__init__(num_rays, keep_full_image, **kwargs)

    def set_num_rays_per_batch(self, num_rays_per_batch: int):
        """Set the number of rays to sample per batch. Overrided to deal with patch-based sampling.

        Args:
            num_rays_per_batch: number of rays to sample per batch
        """
        self.num_rays_per_batch = (num_rays_per_batch // (self.patch_size**2)) * (self.patch_size**2)

    # overrides base method
    def sample_method(  # pylint: disable=no-self-use
        self,
        batch_size: int,
        num_images: int,
        image_height: int,
        image_width: int,
        mask: Optional[TensorType] = None,
        device: Union[torch.device, str] = "cpu",
    ) -> TensorType["batch_size", 3]:
        dilation = 20
        if mask:
            # Note: if there is a mask, sampling reduces back to uniform sampling
            indices = super().sample_method(batch_size, num_images, image_height, image_width, mask=mask, device=device)
        else:
            sub_bs = batch_size // (self.patch_size**2)
            indices = torch.rand((sub_bs, 3), device=device) * torch.tensor(
                [num_images, image_height - self.patch_size * dilation, image_width - self.patch_size * dilation],
                device=device,
            )

            indices = indices.view(sub_bs, 1, 1, 3).broadcast_to(sub_bs, self.patch_size, self.patch_size, 3).clone()

            yys, xxs = torch.meshgrid(
                torch.arange(self.patch_size, device=device), torch.arange(self.patch_size, device=device)
            )
            indices[:, ..., 1] += yys * dilation
            indices[:, ..., 2] += xxs * dilation

            indices = torch.floor(indices).long()
            indices = indices.flatten(0, 2)

        return indices


class EpipolarPixelSampler(PixelSampler):  # pylint: disable=too-few-public-methods
    """
    利用极线信息进行采样，给定n个anchors，
    Args:
        num_rays_per_batch: number of rays to sample per batch
        keep_full_image: whether or not to include a reference to the full image in returned batch
        patch_size: side length of patch. This must be consistent in the method
        config in order for samples to be reshaped into patches correctly.
    """

    def __init__(self, num_rays_per_batch: int, keep_full_image: bool = False, **kwargs) -> None:
        self.cameras = kwargs["cameras"]
        self.num_rays_per_epipolar = kwargs["num_rays_per_epipolar"]  # 每条极线上采样像素点数

        num_rays = (num_rays_per_batch // (self.cameras * self.num_rays_per_epipolar)) * (self.patch_size**2)
        super().__init__(num_rays, keep_full_image, **kwargs)

    def set_num_rays_per_batch(self, num_rays_per_batch: int):
        """Set the number of rays to sample per batch. Overrided to deal with patch-based sampling.

        Args:
            num_rays_per_batch: number of rays to sample per batch
        """
        self.num_rays_per_batch = (num_rays_per_batch // (self.patch_size**2)) * (self.patch_size**2)

    # overrides base method
    def sample_method(  # pylint: disable=no-self-use
        self,
        batch_size: int,
        num_images: int,
        image_height: int,
        image_width: int,
        mask: Optional[TensorType] = None,
        device: Union[torch.device, str] = "cpu",
    ) -> TensorType["batch_size", 3]:
        dilation = 20
        if mask:
            # Note: if there is a mask, sampling reduces back to uniform sampling
            indices = super().sample_method(batch_size, num_images, image_height, image_width, mask=mask, device=device)
        else:
            sub_bs = batch_size // (self.patch_size**2)
            indices = torch.rand((sub_bs, 3), device=device) * torch.tensor(
                [num_images, image_height - self.patch_size * dilation, image_width - self.patch_size * dilation],
                device=device,
            )

            indices = indices.view(sub_bs, 1, 1, 3).broadcast_to(sub_bs, self.patch_size, self.patch_size, 3).clone()

            yys, xxs = torch.meshgrid(
                torch.arange(self.patch_size, device=device), torch.arange(self.patch_size, device=device)
            )
            indices[:, ..., 1] += yys * dilation
            indices[:, ..., 2] += xxs * dilation

            indices = torch.floor(indices).long()
            indices = indices.flatten(0, 2)

        return indices
    

class SemanticPixelSampler:  # pylint: disable=too-few-public-methods
    """Samples 'pixel_batch's from 'image_batch's.

    Args:
        num_rays_per_batch: number of rays to sample per batch
        keep_full_image: whether or not to include a reference to the full image in returned batch
    """

    def __init__(self, num_rays_per_batch: int, keep_full_image: bool = False, **kwargs) -> None:
        self.kwargs = kwargs
        self.num_rays_per_batch = num_rays_per_batch
        self.keep_full_image = keep_full_image

        self.semantic_sample_weights = torch.tensor(list(self.kwargs['semantic_sample_weights'].values())).cuda()
    def set_num_rays_per_batch(self, num_rays_per_batch: int):
        """Set the number of rays to sample per batch.

        Args:
            num_rays_per_batch: number of rays to sample per batch
        """
        self.num_rays_per_batch = num_rays_per_batch

    def sample_method(  # pylint: disable=no-self-use
        self,
        batch_size: int,
        num_images: int,
        image_height: int,
        image_width: int,
        sementic_mask: torch.Tensor,
        device: Union[torch.device, str] = "cpu",
    ) -> TensorType["batch_size", 3]:
        """
        Naive pixel sampler, uniformly samples across all possible pixels of all possible images.

        Args:
            batch_size: number of samples in a batch
            num_images: number of images to sample over
            mask: mask of possible pixels in an image to sample from.
        """
        sementic_mask = sementic_mask.squeeze(-1) # （n_image,h,w）
        device = sementic_mask.device


        # 1. get the number of pixels of each class
        # num_pixels_per_class = {}
        # for i,class_name in enumerate(self.semantic_sample_weights.keys()):
        #     num_pixels_per_class[class_name] = torch.floor(torch.sum(sementic_mask == i) * self.semantic_sample_weights[class_name]).item()
        num_pixels_per_class = torch.bincount(sementic_mask.view(-1)) # (19,1)
        num_pixels_per_class = torch.floor(num_pixels_per_class * self.semantic_sample_weights)
        
        scale_factor = batch_size * 1.0 / torch.sum(num_pixels_per_class)
        num_pixels_per_class = (num_pixels_per_class * scale_factor).floor().to(torch.int).tolist()



        # for i,class_name in enumerate(num_pixels_per_class.keys()):
        #     num_pixels_per_class[class_name] = math.floor(num_pixels_per_class[class_name] * scale_factor)
   

        # 2. sample pixels in each class

        indices = []




        for i,num_pixels in enumerate(num_pixels_per_class):
            if num_pixels > 0:
                nonzero_indices = torch.nonzero((sementic_mask == i), as_tuple=False)
                chosen_indices = random.sample(range(len(nonzero_indices)), k=num_pixels)
                cur_indices = nonzero_indices[chosen_indices].long()
                indices.append(cur_indices)




        # 3. uniform sample for the left rays
        num_left_pixels = batch_size - sum(num_pixels_per_class)
        assert num_left_pixels >= 0
        left_indices = torch.floor(
            torch.rand((num_left_pixels, 3), device=device)
            * torch.tensor([num_images, image_height, image_width], device=device)
        ).long()
        indices.append(left_indices)
        
        # 4.gather indices
        indices = torch.cat(indices,dim=0).cpu()
        
        assert indices.shape[0] == batch_size

        return indices

    def collate_image_dataset_batch(self, batch: Dict, num_rays_per_batch: int, keep_full_image: bool = False):
        """
        Operates on a batch of images and samples pixels to use for generating rays.
        Returns a collated batch which is input to the Graph.
        It will sample only within the valid 'mask' if it's specified.

        Args:
            batch: batch of images to sample from
            num_rays_per_batch: number of rays to sample per batch
            keep_full_image: whether or not to include a reference to the full image in returned batch
        """

        device = batch["image"].device
        num_images, image_height, image_width, _ = batch["image"].shape

        indices = self.sample_method(
            num_rays_per_batch, num_images, image_height, image_width, sementic_mask=batch["all_mask"], device=device
        )


        c, y, x = (i.flatten() for i in torch.split(indices, 1, dim=-1))
        collated_batch = {
            key: value[c, y, x] for key, value in batch.items() if key not in ["image_idx",'rel_camera_idx'] and value is not None
        }

        assert collated_batch["image"].shape == (num_rays_per_batch, 3), collated_batch["image"].shape

        # Needed to correct the random indices to their actual camera idx locations.
        c = torch.clone(c)
        indices[:, 0] = batch["image_idx"][c]

        collated_batch["indices"] = indices  # with the abs camera indices
        collated_batch['rel_camera_indices'] = batch["rel_camera_idx"][c] # with the relative camera indices
        if keep_full_image:
            collated_batch["full_image"] = batch["image"]

        return collated_batch

    def collate_image_dataset_batch_list(self, batch: Dict, num_rays_per_batch: int, keep_full_image: bool = False):
        """
        Does the same as collate_image_dataset_batch, except it will operate over a list of images / masks inside
        a list.

        We will use this with the intent of DEPRECIATING it as soon as we find a viable alternative.
        The intention will be to replace this with a more efficient implementation that doesn't require a for loop, but
        since pytorch's ragged tensors are still in beta (this would allow for some vectorization), this will do.

        Args:
            batch: batch of images to sample from
            num_rays_per_batch: number of rays to sample per batch
            keep_full_image: whether or not to include a reference to the full image in returned batch
        """

        device = batch["image"][0].device
        num_images = len(batch["image"])

        # only sample within the mask, if the mask is in the batch
        all_indices = []
        all_images = []

        if "mask_bak" in batch:
            num_rays_in_batch = num_rays_per_batch // num_images
            for i in range(num_images):
                image_height, image_width, _ = batch["image"][i].shape

                if i == num_images - 1:
                    num_rays_in_batch = num_rays_per_batch - (num_images - 1) * num_rays_in_batch

                indices = self.sample_method(
                    num_rays_in_batch, 1, image_height, image_width, mask=batch["mask"][i], device=device
                )
                indices[:, 0] = i
                all_indices.append(indices)
                all_images.append(batch["image"][i][indices[:, 1], indices[:, 2]])

        else:
            num_rays_in_batch = num_rays_per_batch // num_images
            for i in range(num_images):
                image_height, image_width, _ = batch["image"][i].shape
                if i == num_images - 1:
                    num_rays_in_batch = num_rays_per_batch - (num_images - 1) * num_rays_in_batch
                indices = self.sample_method(num_rays_in_batch, 1, image_height, image_width, device=device)
                indices[:, 0] = i
                all_indices.append(indices)
                all_images.append(batch["image"][i][indices[:, 1], indices[:, 2]])

        indices = torch.cat(all_indices, dim=0)

        c, y, x = (i.flatten() for i in torch.split(indices, 1, dim=-1))
        collated_batch = {
            key: value[c, y, x]
            for key, value in batch.items()
            if key not in  ["image_idx",'rel_camera_idx'] and key != "image" and key != "mask" and value is not None
        }

        collated_batch["image"] = torch.cat(all_images, dim=0)

        assert collated_batch["image"].shape == (num_rays_per_batch, 3), collated_batch["image"].shape

        # Needed to correct the random indices to their actual camera idx locations.
        c = torch.clone(c)
        indices[:, 0] = batch["image_idx"][c]
        collated_batch["indices"] = indices  # with the abs camera indices
        collated_batch['rel_camera_indices'] = batch["rel_camera_idx"][c] # with the relative camera indices

        if keep_full_image:
            collated_batch["full_image"] = batch["image"]

        return collated_batch

    def sample(self, image_batch: Dict):
        """Sample an image batch and return a pixel batch.

        Args:
            image_batch: batch of images to sample from
        """
        if isinstance(image_batch["image"], list):
            image_batch = dict(image_batch.items())  # copy the dictionary so we don't modify the original
            pixel_batch = self.collate_image_dataset_batch_list(
                image_batch, self.num_rays_per_batch, keep_full_image=self.keep_full_image
            )
        elif isinstance(image_batch["image"], torch.Tensor):
            pixel_batch = self.collate_image_dataset_batch(
                image_batch, self.num_rays_per_batch, keep_full_image=self.keep_full_image
            )
        else:
            raise ValueError("image_batch['image'] must be a list or torch.Tensor")
        return pixel_batch


class ErrorPixelSampler:  # pylint: disable=too-few-public-methods
    """Samples 'pixel_batch's from 'image_batch's.

    Args:
        num_rays_per_batch: number of rays to sample per batch
        keep_full_image: whether or not to include a reference to the full image in returned batch
    """

    def __init__(self, num_rays_per_batch: int, keep_full_image: bool = False, **kwargs) -> None:
        self.kwargs = kwargs
        self.num_rays_per_batch = num_rays_per_batch
        self.keep_full_image = keep_full_image
        self.weighted_choice_ratio = 0.2 # 只有50%的概率去权重采样，其他部分依然要随机采样，保证初始值不被破坏

    def set_num_rays_per_batch(self, num_rays_per_batch: int):
        """Set the number of rays to sample per batch.

        Args:
            num_rays_per_batch: number of rays to sample per batch
        """
        self.num_rays_per_batch = num_rays_per_batch

    def weighted_choice(self, grid, dist, size,device):
        """Do inverse transform sampling to sample from a grid with given probabilities
        https://en.wikipedia.org/wiki/Inverse_transform_sampling

        Args:
            grid (tensor): Random are samples pulled from this 1D tensor
            dist (tensor): Probability of each item in the grid.  Should sum to 1
            size (tuple): shape of sampled output

        Returns:
            sampled (tensor): sampled values from `grid` with shape `size`
        """
        grid = torch.tensor(grid, device=device)
        dist = torch.tensor(dist, device=device)
        dist_cum = torch.cumsum(dist, 0)
        rand_ind = torch.searchsorted(dist_cum, torch.rand(size=size).to(device))
        
        return grid[rand_ind]
    
    def weighted_choice_multinomial(self,dist, size,device):
        n = dist.shape[0]
        chunk_size = 2 ** 24
        num_chunks = n // chunk_size


        chosen_indices_list = []
        for i in range((num_chunks)):
            chosen_indices = torch.multinomial(dist[i*chunk_size:(i+1)*chunk_size],size // num_chunks,False)
            chosen_indices += i*chunk_size
            chosen_indices = chosen_indices.to(device)
            chosen_indices_list.append(chosen_indices)
        
        if num_chunks == 0:
            # 不足chunk size时，直接在dist上采样n个点即可
            chosen_indices = torch.multinomial(dist, size,False)
            chosen_indices = chosen_indices.to(device)
            chosen_indices_list.append(chosen_indices)
        else:
            # left
            if size % num_chunks != 0:
                if n % num_chunks != 0:
                    chosen_indices = torch.multinomial(dist[num_chunks * chunk_size:],size % num_chunks,False)
                    chosen_indices += num_chunks * chunk_size
                    chosen_indices = chosen_indices.to(device)
                    chosen_indices_list.append(chosen_indices)
                else:
                    chosen_indices = random.sample(range(n), k=size % num_chunks)
                    chosen_indices = torch.tensor(chosen_indices)
                    chosen_indices = chosen_indices.to(device)
                    chosen_indices_list.append(chosen_indices)

        chosen_indices = torch.cat(chosen_indices_list,dim=0).long().to(device)
        assert chosen_indices.shape[0] == size
        return chosen_indices
        
            
    def sample_method(  # pylint: disable=no-self-use
        self,
        batch_size: int,
        num_images: int,
        image_height: int,
        image_width: int,
        error_map: torch.Tensor,
        device: Union[torch.device, str] = "cpu",
    ) -> TensorType["batch_size", 3]:
        """
        Naive pixel sampler, uniformly samples across all possible pixels of all possible images.

        Args:
            batch_size: number of samples in a batch
            num_images: number of images to sample over
            mask: mask of possible pixels in an image to sample from.
        """
        error_map = error_map.squeeze(-1) # （n_image,h,w）
        weights = error_map.view(-1)

        n_pixels = weights.shape[0]

        nonzero_indices = torch.nonzero(error_map>=0.0, as_tuple=False) # (n,h,w)
        error_sample_size = int(batch_size * self.weighted_choice_ratio)
        random_sample_size = batch_size - error_sample_size
        # 1. generate indices from error map

        error_chosen_indices = self.weighted_choice_multinomial(weights,error_sample_size,device)


        # 2. generate indices randomly
        random_chosen_indices = random.sample(range(n_pixels), k=random_sample_size)
        random_chosen_indices = torch.tensor(random_chosen_indices).to(device)
        
        # 3. cat chosen_indices
        chosen_indices = torch.cat((error_chosen_indices,random_chosen_indices),dim=0).long().to(device)

        # 4. get real sampled indices
        indices = nonzero_indices[chosen_indices].long().to(device)

        
        assert indices.shape[0] == batch_size

        return indices

    def collate_image_dataset_batch(self, batch: Dict, num_rays_per_batch: int, keep_full_image: bool = False):
        """
        Operates on a batch of images and samples pixels to use for generating rays.
        Returns a collated batch which is input to the Graph.
        It will sample only within the valid 'mask' if it's specified.

        Args:
            batch: batch of images to sample from
            num_rays_per_batch: number of rays to sample per batch
            keep_full_image: whether or not to include a reference to the full image in returned batch
        """

        device = batch["image"].device
        num_images, image_height, image_width, _ = batch["image"].shape

        indices = self.sample_method(
            num_rays_per_batch, num_images, image_height, image_width, error_map=batch["error_map"], device=device
        )


        c, y, x = (i.flatten() for i in torch.split(indices, 1, dim=-1))
        collated_batch = {
            key: value[c, y, x] for key, value in batch.items() if key not in ["image_idx",'rel_camera_idx'] and value is not None
        }

        assert collated_batch["image"].shape == (num_rays_per_batch, 3), collated_batch["image"].shape

        # Needed to correct the random indices to their actual camera idx locations.
        c = torch.clone(c)
        indices[:, 0] = batch["image_idx"][c]

        collated_batch["indices"] = indices  # with the abs camera indices
        collated_batch['rel_camera_indices'] = batch["rel_camera_idx"][c] # with the relative camera indices
        if keep_full_image:
            collated_batch["full_image"] = batch["image"]

        return collated_batch

    def collate_image_dataset_batch_list(self, batch: Dict, num_rays_per_batch: int, keep_full_image: bool = False):
        """
        Does the same as collate_image_dataset_batch, except it will operate over a list of images / masks inside
        a list.

        We will use this with the intent of DEPRECIATING it as soon as we find a viable alternative.
        The intention will be to replace this with a more efficient implementation that doesn't require a for loop, but
        since pytorch's ragged tensors are still in beta (this would allow for some vectorization), this will do.

        Args:
            batch: batch of images to sample from
            num_rays_per_batch: number of rays to sample per batch
            keep_full_image: whether or not to include a reference to the full image in returned batch
        """

        device = batch["image"][0].device
        num_images = len(batch["image"])

        # only sample within the mask, if the mask is in the batch
        all_indices = []
        all_images = []

        if "mask_bak" in batch:
            num_rays_in_batch = num_rays_per_batch // num_images
            for i in range(num_images):
                image_height, image_width, _ = batch["image"][i].shape

                if i == num_images - 1:
                    num_rays_in_batch = num_rays_per_batch - (num_images - 1) * num_rays_in_batch

                indices = self.sample_method(
                    num_rays_in_batch, 1, image_height, image_width, mask=batch["mask"][i],error_map=batch["error_map"][i], device=device
                )
                indices[:, 0] = i
                all_indices.append(indices)
                all_images.append(batch["image"][i][indices[:, 1], indices[:, 2]])

        else:
            num_rays_in_batch = num_rays_per_batch // num_images
            for i in range(num_images):
                image_height, image_width, _ = batch["image"][i].shape
                if i == num_images - 1:
                    num_rays_in_batch = num_rays_per_batch - (num_images - 1) * num_rays_in_batch
                indices = self.sample_method(num_rays_in_batch, 1, image_height, image_width, device=device)
                indices[:, 0] = i
                all_indices.append(indices)
                all_images.append(batch["image"][i][indices[:, 1], indices[:, 2]])

        indices = torch.cat(all_indices, dim=0)

        c, y, x = (i.flatten() for i in torch.split(indices, 1, dim=-1))
        collated_batch = {
            key: value[c, y, x]
            for key, value in batch.items()
            if key not in  ["image_idx",'rel_camera_idx'] and key != "image" and key != "mask" and value is not None
        }

        collated_batch["image"] = torch.cat(all_images, dim=0)

        assert collated_batch["image"].shape == (num_rays_per_batch, 3), collated_batch["image"].shape

        # Needed to correct the random indices to their actual camera idx locations.
        c = torch.clone(c)
        indices[:, 0] = batch["image_idx"][c]
        collated_batch["indices"] = indices  # with the abs camera indices
        collated_batch['rel_camera_indices'] = batch["rel_camera_idx"][c] # with the relative camera indices

        if keep_full_image:
            collated_batch["full_image"] = batch["image"]

        return collated_batch

    def sample(self, image_batch: Dict):
        """Sample an image batch and return a pixel batch.

        Args:
            image_batch: batch of images to sample from
        """
        if isinstance(image_batch["image"], list):
            image_batch = dict(image_batch.items())  # copy the dictionary so we don't modify the original
            pixel_batch = self.collate_image_dataset_batch_list(
                image_batch, self.num_rays_per_batch, keep_full_image=self.keep_full_image
            )
        elif isinstance(image_batch["image"], torch.Tensor):
            pixel_batch = self.collate_image_dataset_batch(
                image_batch, self.num_rays_per_batch, keep_full_image=self.keep_full_image
            )
        else:
            raise ValueError("image_batch['image'] must be a list or torch.Tensor")
        return pixel_batch
