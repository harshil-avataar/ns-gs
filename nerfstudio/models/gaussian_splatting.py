# Copyright 2022 the Regents of the University of California, Nerfstudio Team and contributors. All rights reserved.
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
from typing import Dict, List, Optional, Tuple, Type, Union
import torch
from nerfstudio.data.scene_box import OrientedBox
from copy import deepcopy
from nerfstudio.cameras.rays import RayBundle
from torch.nn import Parameter
from torchmetrics.image import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
from nerfstudio.cameras.cameras import Cameras
from gsplat._torch_impl import quat_to_rotmat
from nerfstudio.engine.callbacks import TrainingCallback, TrainingCallbackAttributes, TrainingCallbackLocation
from nerfstudio.engine.optimizers import Optimizers
from nerfstudio.models.base_model import Model, ModelConfig
import math
import numpy as np
from sklearn.neighbors import NearestNeighbors
import viser.transforms as vtf
from nerfstudio.model_components.losses import scale_gauss_gradients_by_distance_squared
from nerfstudio.cameras.camera_optimizers import CameraOptimizer, CameraOptimizerConfig

from torchmetrics.image import StructuralSimilarityIndexMeasure
from gsplat.rasterize import RasterizeGaussians
from gsplat.project_gaussians import ProjectGaussians
from nerfstudio.model_components.losses import depth_ranking_loss
from gsplat.sh import SphericalHarmonics, num_sh_bases
from pytorch_msssim import SSIM


def random_quat_tensor(N, **kwargs):
    u = torch.rand(N, **kwargs)
    v = torch.rand(N, **kwargs)
    w = torch.rand(N, **kwargs)
    return torch.stack(
        [
            torch.sqrt(1 - u) * torch.sin(2 * math.pi * v),
            torch.sqrt(1 - u) * torch.cos(2 * math.pi * v),
            torch.sqrt(u) * torch.sin(2 * math.pi * w),
            torch.sqrt(u) * torch.sin(2 * math.pi * w),
        ],
        dim=-1,
    )


def RGB2SH(rgb):
    C0 = 0.28209479177387814
    return (rgb - 0.5) / C0


def SH2RGB(sh):
    C0 = 0.28209479177387814
    return sh * C0 + 0.5


def projection_matrix(znear, zfar, fovx, fovy, device="cpu"):
    t = znear * math.tan(0.5 * fovy)
    b = -t
    r = znear * math.tan(0.5 * fovx)
    l = -r
    n = znear
    f = zfar
    return torch.tensor(
        [
            [2 * n / (r - l), 0.0, (r + l) / (r - l), 0.0],
            [0.0, 2 * n / (t - b), (t + b) / (t - b), 0.0],
            [0.0, 0.0, (f + n) / (f - n), -1.0 * f * n / (f - n)],
            [0.0, 0.0, 1.0, 0.0],
        ],
        device=device,
    )


@dataclass
class GaussianSplattingModelConfig(ModelConfig):
    """Gaussian Splatting Model Config"""

    # og:  = 0.005

    _target: Type = field(default_factory=lambda: GaussianSplattingModel)
    warmup_length: int = 500  # densify_from in og: same
    """period of steps where refinement is turned off"""
    refine_every: int = 100  # densification iterval: same
    """period of steps where gaussians are culled and densified"""
    resolution_schedule: int = 250  # not present
    """training starts at 1/d resolution, every n steps this is doubled"""
    num_downscales: int = 2  # not present
    """at the beginning, resolution is 1/2^d, where d is this number"""
    cull_alpha_thresh: float = 0.1
    """threshold of opacity for culling gaussians"""
    cull_scale_thresh: float = 0.5
    """threshold of scale for culling gaussians"""
    reset_alpha_every: int = 30  # 3000 in original code.
    """Every this many refinement steps, reset the alpha"""
    densify_grad_thresh: float = 0.0002  # same
    """threshold of positional gradient norm for  densifying gaussians"""
    densify_size_thresh: float = 0.01  # percentage split
    """below this size, gaussians are *duplicated*, otherwise split"""
    n_split_samples: int = 2
    """number of samples to split gaussians into"""
    sh_degree_interval: int = 1000  # same
    """every n intervals turn on another sh degree"""
    cull_screen_size: float = 0.15
    """if a gaussian is more than this percent of screen space, cull it"""
    split_screen_size: float = 0.05
    """if a gaussian is more than this percent of screen space, split it"""
    stop_screen_size_at: int = 4000
    """stop culling/splitting at this step WRT screen size of gaussians"""
    random_init: bool = False
    """whether to initialize the positions uniformly randomly (not SFM points)"""
    extra_points: int = 0
    """number of extra points to add to the model in addition to sfm points, randomly distributed"""
    ssim_lambda: float = 0.2
    """weight of ssim loss"""
    stop_split_at: int = 15000  # densify_until_iter
    """stop splitting at this step"""
    sh_degree: int = 3
    """maximum degree of spherical harmonics to use"""
    camera_optimizer: CameraOptimizerConfig = CameraOptimizerConfig(mode="off")
    """camera optimizer config"""
    eval_og: bool = False 


class GaussianSplattingModel(Model):
    """Gaussian Splatting model

    Args:
        config: Gaussian Splatting configuration to instantiate model
    """

    config: GaussianSplattingModelConfig

    def __init__(self, *args, **kwargs):

        self.transform_matrix = kwargs['metadata']['transform_matrix']
        if "seed_points" in kwargs:
            self.seed_pts = kwargs["seed_points"]
        else:
            self.seed_pts = None
        super().__init__(*args, **kwargs)

    def populate_modules(self):
        if self.seed_pts is not None and not self.config.random_init:
            extra_means = (torch.rand((self.config.extra_points, 3)) - 0.5) * 10
            self.means = torch.nn.Parameter(torch.cat([self.seed_pts[0], extra_means]))  # (Location, Color)
        else:
            self.means = torch.nn.Parameter((torch.rand((500000, 3)) - 0.5) * 10)
        self.xys_grad_norm = None
        self.max_2Dsize = None
        distances, _ = self.k_nearest_sklearn(self.means.data, 3)
        distances = torch.from_numpy(distances)

        # find the average of the three nearest neighbors for each point and use that as the scale
        avg_dist = distances.mean(dim=-1, keepdim=True)
        self.scales = torch.nn.Parameter(torch.log(avg_dist.repeat(1, 3)))

        self.quats = torch.nn.Parameter(random_quat_tensor(self.num_points))
        dim_sh = num_sh_bases(self.config.sh_degree)

        if self.seed_pts is not None and not self.config.random_init:
            fused_color = RGB2SH(self.seed_pts[1] / 255)
            shs = torch.zeros((fused_color.shape[0], dim_sh, 3)).float().cuda()
            shs[:, 0, :3] = fused_color
            shs[:, 1:, 3:] = 0.0
            # concat on extra_points amount of random ones at the end
            extra_shs = torch.rand((self.config.extra_points, dim_sh, 3)).float().cuda()
            extra_shs[:, 1:, :] = 0.0  # zero out the higher freq
            shs = torch.cat([shs, extra_shs])
            self.colors_all = torch.nn.Parameter(shs)
        else:
            colors = torch.nn.Parameter(torch.rand(self.num_points, 1, 3))
            shs_rest = torch.nn.Parameter(torch.zeros((self.num_points, dim_sh - 1, 3)))
            self.colors_all = torch.nn.Parameter(torch.cat([colors, shs_rest], dim=1))

        self.opacities = torch.nn.Parameter(torch.logit(0.1 * torch.ones(self.num_points, 1)))

        # metrics
        self.psnr = PeakSignalNoiseRatio(data_range=1.0)
        self.ssim = SSIM(data_range=1.0, size_average=True, channel=3)
        self.lpips = LearnedPerceptualImagePatchSimilarity(normalize=True)
        self.step = 0

        self.crop_box: Optional[OrientedBox] = None
        self.back_color = torch.zeros(3)

        self.camera_optimizer: CameraOptimizer = self.config.camera_optimizer.setup(
            num_cameras=self.num_train_data, device="cpu"
        )

    @property
    def colors(self):
        return SH2RGB(self.colors_all[:, 0, :])

    @property
    def shs_rest(self):
        return self.colors_all[:, 1:, :]

    def load_state_dict(self, dict, **kwargs):
        # resize the parameters to match the new number of points
        self.step = 30000
        newp = dict["means"].shape[0]
        self.means = torch.nn.Parameter(torch.zeros(newp, 3, device=self.device))
        self.scales = torch.nn.Parameter(torch.zeros(newp, 3, device=self.device))
        self.quats = torch.nn.Parameter(torch.zeros(newp, 4, device=self.device))
        self.opacities = torch.nn.Parameter(torch.zeros(newp, 1, device=self.device))
        self.colors_all = torch.nn.Parameter(
            torch.zeros(newp, num_sh_bases(self.config.sh_degree), 3, device=self.device)
        )
        super().load_state_dict(dict, **kwargs)
        # self.save_ply()

    def k_nearest_sklearn(self, x: torch.Tensor, k: int):
        """
        Find k-nearest neighbors using sklearn's NearestNeighbors.
        x: The data tensor of shape [num_samples, num_features]
        k: The number of neighbors to retrieve
        """
        # Convert tensor to numpy array
        x_np = x.cpu().numpy()

        # Build the nearest neighbors model
        nn_model = NearestNeighbors(n_neighbors=k + 1, algorithm="auto", metric="euclidean").fit(x_np)

        # Find the k-nearest neighbors
        distances, indices = nn_model.kneighbors(x_np)

        # Exclude the point itself from the result and return
        return distances[:, 1:].astype(np.float32), indices[:, 1:].astype(np.float32)

    def remove_from_optim(self, optimizer, deleted_mask, new_params):
        """removes the deleted_mask from the optimizer provided"""
        assert len(new_params) == 1
        # assert isinstance(optimizer, torch.optim.Adam), "Only works with Adam"

        param = optimizer.param_groups[0]["params"][0]
        param_state = optimizer.state[param]
        del optimizer.state[param]

        # Modify the state directly without deleting and reassigning.
        param_state["exp_avg"] = param_state["exp_avg"][~deleted_mask]
        param_state["exp_avg_sq"] = param_state["exp_avg_sq"][~deleted_mask]

        # Update the parameter in the optimizer's param group.
        del optimizer.param_groups[0]["params"][0]
        del optimizer.param_groups[0]["params"]
        optimizer.param_groups[0]["params"] = new_params
        optimizer.state[new_params[0]] = param_state

    def dup_in_optim(self, optimizer, dup_mask, new_params, n=2):
        """adds the parameters to the optimizer : SAME as 3DGS expect this 'n' factor here"""
        param = optimizer.param_groups[0]["params"][0]
        param_state = optimizer.state[param]
        repeat_dims = (n,) + tuple(1 for _ in range(param_state["exp_avg"].dim() - 1))
        param_state["exp_avg"] = torch.cat(
            [param_state["exp_avg"], torch.zeros_like(param_state["exp_avg"][dup_mask.squeeze()]).repeat(*repeat_dims)],
            dim=0,
        )
        param_state["exp_avg_sq"] = torch.cat(
            [
                param_state["exp_avg_sq"],
                torch.zeros_like(param_state["exp_avg_sq"][dup_mask.squeeze()]).repeat(*repeat_dims),
            ],
            dim=0,
        )
        del optimizer.state[param]
        optimizer.state[new_params[0]] = param_state
        optimizer.param_groups[0]["params"] = new_params
        del param

    def after_train(self, step):
        with torch.no_grad():
            # keep track of a moving average of grad norms
            visible_mask = (self.radii > 0).flatten()
            grads = self.xys.grad.detach().norm(dim=-1)  # TODO fill in
            # print(f"grad norm min {grads.min().item()} max {grads.max().item()} mean {grads.mean().item()} size {grads.shape}")
            if self.xys_grad_norm is None:
                self.xys_grad_norm = grads
                self.vis_counts = torch.ones_like(self.xys_grad_norm)
            else:
                self.vis_counts[visible_mask] = self.vis_counts[visible_mask] + 1
                self.xys_grad_norm[visible_mask] = grads[visible_mask] + self.xys_grad_norm[visible_mask]

            # update the max screen size, as a ratio of number of pixels
            if self.max_2Dsize is None:
                self.max_2Dsize = torch.zeros_like(self.radii, dtype=torch.float32)
            newradii = self.radii.detach()[visible_mask]
            self.max_2Dsize[visible_mask] = torch.maximum(
                self.max_2Dsize[visible_mask], newradii / float(max(self.last_size[0], self.last_size[1]))
            )

    def set_crop(self, crop_box: OrientedBox):
        self.crop_box = crop_box

    def set_background(self, back_color: torch.Tensor):
        assert back_color.shape == (3,)
        self.back_color = back_color

    def refinement_after(self, optimizers: Optimizers, step):
        if self.step >= self.config.warmup_length:
            with torch.no_grad():
                # only split/cull if we've seen every image since opacity reset
                reset_interval = self.config.reset_alpha_every * self.config.refine_every
                if (
                    self.step < self.config.stop_split_at
                    and self.step % reset_interval > self.num_train_data + self.config.refine_every
                ):
                    # then we densify
                    avg_grad_norm = (
                        (self.xys_grad_norm / self.vis_counts) * 0.5 * max(self.last_size[0], self.last_size[1])
                    )  # vis_counts is self.denom, why 0.5 mult? last term is max(H,W)
                    # print(avg_grad_norm)
                    # hb: this norm seems different

                    high_grads = (avg_grad_norm > self.config.densify_grad_thresh).squeeze()
                    splits = (self.scales.exp().max(dim=-1).values > self.config.densify_size_thresh).squeeze()
                    # exp here is the scaling activation.

                    if self.step < self.config.stop_screen_size_at:
                        splits |= (self.max_2Dsize > self.config.split_screen_size).squeeze()
                    splits &= high_grads  # same as 3DGS

                    nsamps = self.config.n_split_samples
                    (
                        split_means,
                        split_colors,
                        split_opacities,
                        split_scales,
                        split_quats,
                    ) = self.split_gaussians(splits, nsamps)

                    dups = (self.scales.exp().max(dim=-1).values <= self.config.densify_size_thresh).squeeze()
                    dups &= high_grads
                    dup_means, dup_colors, dup_opacities, dup_scales, dup_quats = self.dup_gaussians(dups)
                    self.means = Parameter(torch.cat([self.means.detach(), split_means, dup_means], dim=0))

                    # where do they add the gradient to the means of the gaussians ?
                    self.colors_all = Parameter(torch.cat([self.colors_all.detach(), split_colors, dup_colors], dim=0))

                    self.opacities = Parameter(
                        torch.cat([self.opacities.detach(), split_opacities, dup_opacities], dim=0)
                    )
                    self.scales = Parameter(torch.cat([self.scales.detach(), split_scales, dup_scales], dim=0))
                    self.quats = Parameter(torch.cat([self.quats.detach(), split_quats, dup_quats], dim=0))
                    # append zeros to the max_2Dsize tensor
                    self.max_2Dsize = torch.cat(
                        [self.max_2Dsize, torch.zeros_like(split_scales[:, 0]), torch.zeros_like(dup_scales[:, 0])],
                        dim=0,
                    )
                    split_idcs = torch.where(splits)[0]
                    param_groups = self.get_gaussian_param_groups()
                    for group, param in param_groups.items():
                        self.dup_in_optim(optimizers.optimizers[group], split_idcs, param, n=nsamps)
                    dup_idcs = torch.where(dups)[0]

                    param_groups = self.get_gaussian_param_groups()
                    for group, param in param_groups.items():
                        self.dup_in_optim(optimizers.optimizers[group], dup_idcs, param, 1)

                # Offset all the opacity reset logic by refine_every so that we don't
                # save checkpoints right when the opacity is reset (saves every 2k)
                if self.step % reset_interval > self.num_train_data + self.config.refine_every:
                    # then cull
                    deleted_mask = self.cull_gaussians()
                    param_groups = self.get_gaussian_param_groups()
                    for group, param in param_groups.items():
                        self.remove_from_optim(optimizers.optimizers[group], deleted_mask, param)

                if self.step % reset_interval == self.config.refine_every:
                    reset_value = self.config.cull_alpha_thresh * 0.8
                    self.opacities.data = torch.full_like(
                        self.opacities.data, torch.logit(torch.tensor(reset_value)).item()
                    )
                    # reset the exp of optimizer
                    optim = optimizers.optimizers["opacity"]
                    param = optim.param_groups[0]["params"][0]
                    param_state = optim.state[param]
                    param_state["exp_avg"] = torch.zeros_like(param_state["exp_avg"])
                    param_state["exp_avg_sq"] = torch.zeros_like(param_state["exp_avg_sq"])
                self.xys_grad_norm = None
                self.vis_counts = None
                self.max_2Dsize = None

    def cull_gaussians(self):
        """
        This function deletes gaussians with under a certain opacity threshold
        """
        n_bef = self.num_points
        # cull transparent ones
        culls = (torch.sigmoid(self.opacities) < self.config.cull_alpha_thresh).squeeze()
        if self.step > self.config.refine_every * self.config.reset_alpha_every:
            # cull huge ones
            toobigs = (torch.exp(self.scales).max(dim=-1).values > self.config.cull_scale_thresh).squeeze()
            culls = culls | toobigs
            if self.step < self.config.stop_screen_size_at:
                # cull big screen space
                culls = culls | (self.max_2Dsize > self.config.cull_screen_size).squeeze()
        self.means = Parameter(self.means[~culls].detach())
        self.scales = Parameter(self.scales[~culls].detach())
        self.quats = Parameter(self.quats[~culls].detach())
        self.colors_all = Parameter(self.colors_all[~culls].detach())
        self.opacities = Parameter(self.opacities[~culls].detach())

        print(f"Culled {n_bef - self.num_points} gaussians")
        return culls

    def split_gaussians(self, split_mask, samps):
        """
        This function splits gaussians that are too large. Same as 3DGS
        """

        n_splits = split_mask.sum().item()
        print(f"Splitting {split_mask.sum().item()/self.num_points} gaussians: {n_splits}/{self.num_points}")

        centered_samples = torch.randn((samps * n_splits, 3), device=self.device)  # Nx3 of axis-aligned scales
        scaled_samples = (
            torch.exp(self.scales[split_mask].repeat(samps, 1)) * centered_samples
        )  # how these scales are rotated
        # same thing is done in 3DGS.

        quats = self.quats[split_mask] / self.quats[split_mask].norm(dim=-1, keepdim=True)  # normalize them first
        rots = quat_to_rotmat(quats.repeat(samps, 1))  # how these scales are rotated
        rotated_samples = torch.bmm(rots, scaled_samples[..., None]).squeeze()
        new_means = rotated_samples + self.means[split_mask].repeat(samps, 1)  # same

        # step 2, sample new colors
        new_colors_all = self.colors_all[split_mask].repeat(samps, 1, 1)
        # step 3, sample new opacities
        new_opacities = self.opacities[split_mask].repeat(samps, 1)
        # step 4, sample new scales
        size_fac = 1.6  # 3DGS divides by 0.8. There is a 2x diff.
        new_scales = torch.log(torch.exp(self.scales[split_mask]) / size_fac).repeat(samps, 1)
        self.scales[split_mask] = torch.log(torch.exp(self.scales[split_mask]) / size_fac)
        # step 5, sample new quats
        new_quats = self.quats[split_mask].repeat(samps, 1)
        return new_means, new_colors_all, new_opacities, new_scales, new_quats

    def dup_gaussians(self, dup_mask):
        """
        This function duplicates gaussians that are too small. Same as 3DGS.
        """
        n_dups = dup_mask.sum().item()
        print(f"Duplicating {dup_mask.sum().item()/self.num_points} gaussians: {n_dups}/{self.num_points}")
        dup_means = self.means[dup_mask]
        dup_colors = self.colors_all[dup_mask]
        dup_opacities = self.opacities[dup_mask]
        dup_scales = self.scales[dup_mask]
        dup_quats = self.quats[dup_mask]
        return dup_means, dup_colors, dup_opacities, dup_scales, dup_quats

    @property
    def num_points(self):
        return self.means.shape[0]

    def get_training_callbacks(
        self, training_callback_attributes: TrainingCallbackAttributes
    ) -> List[TrainingCallback]:
        cbs = []
        cbs.append(TrainingCallback([TrainingCallbackLocation.BEFORE_TRAIN_ITERATION], self.step_cb))
        # The order of these matters
        cbs.append(
            TrainingCallback(
                [TrainingCallbackLocation.AFTER_TRAIN_ITERATION],
                self.after_train,
            )
        )
        cbs.append(
            TrainingCallback(
                [TrainingCallbackLocation.AFTER_TRAIN_ITERATION],
                self.refinement_after,
                update_every_num_iters=self.config.refine_every,
                args=[training_callback_attributes.optimizers],
            )
        )
        return cbs

    def step_cb(self, step):
        self.step = step

    def get_gaussian_param_groups(self) -> Dict[str, List[Parameter]]:
        return {
            "xyz": [self.means],
            "color": [self.colors_all],
            "opacity": [self.opacities],
            "scaling": [self.scales],
            "rotation": [self.quats],
        }

    def get_param_groups(self) -> Dict[str, List[Parameter]]:
        """Obtain the parameter groups for the optimizers

        Returns:
            Mapping of different parameter groups
        """
        gps = self.get_gaussian_param_groups()
        # add camera optimizer param groups
        self.camera_optimizer.get_param_groups(gps)
        return gps

    def _get_downscale_factor(self):
        if self.training:
            return 2 ** max((self.config.num_downscales - self.step // self.config.resolution_schedule), 0)
        else:
            return 1

    def get_outputs(self, camera: Cameras) -> Dict[str, Union[torch.Tensor, List]]:
        """Takes in a Ray Bundle and returns a dictionary of outputs.

        Args:
            ray_bundle: Input bundle of rays. This raybundle should have all the
            needed information to compute the outputs.

        Returns:
            Outputs of model. (ie. rendered colors)
        """
        if not isinstance(camera, Cameras):
            print("Called get_outputs with not a camera")
            return {}
        assert camera.shape[0] == 1, "Only one camera at a time"
        if self.training:
            # currently relies on the branch vickie/camera-grads
            self.camera_optimizer.apply_to_camera(camera)
        if self.crop_box is not None and not self.training:
            crop_ids = self.crop_box.within(self.means).squeeze()
            if crop_ids.sum() == 0:
                return {"rgb": torch.full((camera.height.item(), camera.width.item(), 3), 0.5, device=self.device)}
        else:
            crop_ids = None
        camera_downscale = self._get_downscale_factor()
        camera.rescale_output_resolution(1 / camera_downscale)
        # shift the camera to center of scene looking at center
        R = camera.camera_to_worlds[0, :3, :3]  # 3 x 3
        T = camera.camera_to_worlds[0, :3, 3:4]  # 3 x 1
        # flip the z axis to align with gsplat conventions
        R_edit = torch.tensor(vtf.SO3.from_x_radians(np.pi).as_matrix(), device=R.device, dtype=R.dtype)
        R = R @ R_edit
        # analytic matrix inverse to get world2camera matrix
        R_inv = R.T
        T_inv = -R_inv @ T
        viewmat = torch.eye(4, device=R.device, dtype=R.dtype)
        viewmat[:3, :3] = R_inv
        viewmat[:3, 3:4] = T_inv
        # calculate the FOV of the camera given fx and fy, width and height
        cx = camera.cx.item()
        cy = camera.cy.item()
        fovx = 2 * math.atan(camera.width / (2 * camera.fx))
        fovy = 2 * math.atan(camera.height / (2 * camera.fy))
        W, H = camera.width.item(), camera.height.item()
        self.last_size = (H, W)
        projmat = projection_matrix(0.001, 1000, fovx, fovy, device=self.device)
        BLOCK_X, BLOCK_Y = 16, 16
        tile_bounds = (
            (W + BLOCK_X - 1) // BLOCK_X,
            (H + BLOCK_Y - 1) // BLOCK_Y,
            1,
        )
        if self.training:
            background = torch.rand(3, device=self.device)
        else:
            background = self.back_color

        if crop_ids is not None:
            opacities_crop = self.opacities[crop_ids]
            means_crop = self.means[crop_ids]
            colors_crop = self.colors_all[crop_ids]
            scales_crop = self.scales[crop_ids]
            quats_crop = self.quats[crop_ids]
        else:
            opacities_crop = self.opacities
            means_crop = self.means
            colors_crop = self.colors_all
            scales_crop = self.scales
            quats_crop = self.quats
        self.xys, depths, self.radii, conics, num_tiles_hit, cov3d = ProjectGaussians.apply(
            means_crop,
            torch.exp(scales_crop),
            1,
            quats_crop / quats_crop.norm(dim=-1, keepdim=True),
            viewmat.squeeze()[:3, :],
            projmat.squeeze() @ viewmat.squeeze(),
            camera.fx.item(),
            camera.fy.item(),
            cx,
            cy,
            H,
            W,
            tile_bounds,
        )
        # self.xys are not zeros, as they are used to passed to the rasterizer
        # as it is passed via python, we need real values. 3DGS doesnt' have this dependency

        # Important to allow xys grads to populate properly

        if self.training:
            if self.xys.requires_grad:
                self.xys.retain_grad()
        if self.config.sh_degree > 0:
            viewdirs = means_crop.detach() - camera.camera_to_worlds.detach()[..., :3, 3]  # (N, 3)
            viewdirs = viewdirs / viewdirs.norm(dim=-1, keepdim=True)
            n = min(self.step // self.config.sh_degree_interval, self.config.sh_degree)
            # breakpoint()
            if self.config.eval_og: 
                viewdirs = viewdirs @ self.transform_matrix[:,:3].cuda()

            rgbs = SphericalHarmonics.apply(n, viewdirs, colors_crop)
            rgbs = torch.clamp(rgbs + 0.5, 0.0, 1.0)
            # rgbs = torch.clamp(SH2RGB(rgbs), 0.0, 1.0)
        else:
            rgbs = self.get_colors.squeeze()  # (N, 3)
            rgbs = torch.sigmoid(rgbs)

        rgb = RasterizeGaussians.apply(
            self.xys,
            depths,
            self.radii,
            conics,
            num_tiles_hit,
            rgbs,
            torch.sigmoid(opacities_crop),
            H,
            W,
            background,
        )
        depth_im = None
        if not self.training:
            depth_im = RasterizeGaussians.apply(
                self.xys,
                depths,
                self.radii,
                conics,
                num_tiles_hit,
                depths[:, None].repeat(1, 3),
                torch.sigmoid(opacities_crop),
                H,
                W,
                torch.ones(3, device=self.device) * 10,
            )[..., 0:1]
        # rescale the camera back to original dimensions
        camera.rescale_output_resolution(camera_downscale)
        return {"rgb": rgb, "depth": depth_im}

    def get_metrics_dict(self, outputs, batch) -> Dict[str, torch.Tensor]:
        """Compute and returns metrics.

        Args:
            outputs: the output to compute loss dict to
            batch: ground truth batch corresponding to outputs
        """
        d = self._get_downscale_factor()
        if d > 1:
            # use torchvision to resize
            import torchvision.transforms.functional as TF

            newsize = (batch["image"].shape[0] // d, batch["image"].shape[1] // d)
            gt_img = TF.resize(batch["image"].permute(2, 0, 1), newsize).permute(1, 2, 0)
        else:
            gt_img = batch["image"]
        metrics_dict = {}
        gt_rgb = gt_img.to(self.device)[...,:3]  # RGB or RGBA image

        predicted_rgb = outputs["rgb"]
        metrics_dict["psnr"] = self.psnr(predicted_rgb, gt_rgb)

        self.camera_optimizer.get_metrics_dict(metrics_dict)
        metrics_dict["gaussian_count"] = self.num_points
        return metrics_dict

    def get_loss_dict(self, outputs, batch, metrics_dict=None) -> Dict[str, torch.Tensor]:
        """Computes and returns the losses dict.

        Args:
            outputs: the output to compute loss dict to
            batch: ground truth batch corresponding to outputs
            metrics_dict: dictionary of metrics, some of which we can use for loss
        """
        d = self._get_downscale_factor()
        if d > 1:
            # use torchvision to resize
            import torchvision.transforms.functional as TF

            newsize = (batch["image"].shape[0] // d, batch["image"].shape[1] // d)
            gt_img = TF.resize(batch["image"].permute(2, 0, 1), newsize).permute(1, 2, 0)
        else:
            gt_img = batch["image"]
        
        gt_img  = gt_img [...,:3]
        Ll1 = torch.abs(gt_img - outputs["rgb"]).mean()
        simloss = 1 - self.ssim(gt_img.permute(2, 0, 1)[None, ...], outputs["rgb"].permute(2, 0, 1)[None, ...])
        if self.step % 10 == 0:
            # Before, we made split sh and colors onto different optimizer, with shs having a low learning rate
            # This is slow, instead we apply a regularization every few steps
            sh_reg = self.colors_all[:, 1:, :].norm(dim=1).mean()
        else:
            sh_reg = 0.0
        return {"main_loss": (1 - self.config.ssim_lambda) * Ll1 + self.config.ssim_lambda * simloss, "sh_reg": sh_reg}

    @torch.no_grad()
    def get_outputs_for_camera_ray_bundle(
        self, camera_ray_bundle: RayBundle, camera: Optional[Cameras] = None
    ) -> Dict[str, torch.Tensor]:
        """Takes in camera parameters and computes the output of the model.

        Args:
            camera_ray_bundle: ray bundle to calculate outputs over
        """
        assert camera is not None, "must provide camera to gaussian model"
        outs = self.get_outputs(camera.to(self.device))
        return outs

    def get_image_metrics_and_images(
        self, outputs: Dict[str, torch.Tensor], batch: Dict[str, torch.Tensor]
    ) -> Tuple[Dict[str, float], Dict[str, torch.Tensor]]:
        """Writes the test image outputs.

        Args:
            image_idx: Index of the image.
            step: Current step.
            batch: Batch of data.
            outputs: Outputs of the model.

        Returns:
            A dictionary of metrics.
        """
        gt_rgb = batch["image"].to(self.device)
        predicted_rgb = outputs["rgb"]  # Blended with background (black if random background)
        # print(gt_rgb.shape)
        gt_rgb = gt_rgb[...,:3]
        combined_rgb = torch.cat([gt_rgb, predicted_rgb], dim=1)

        # Switch images from [H, W, C] to [1, C, H, W] for metrics computations
        gt_rgb = torch.moveaxis(gt_rgb, -1, 0)[None, ...]
        predicted_rgb = torch.moveaxis(predicted_rgb, -1, 0)[None, ...]

        psnr = self.psnr(gt_rgb, predicted_rgb)
        ssim = self.ssim(gt_rgb, predicted_rgb)
        lpips = self.lpips(gt_rgb, predicted_rgb)

        # all of these metrics will be logged as scalars
        metrics_dict = {"psnr": float(psnr.item()), "ssim": float(ssim)}  # type: ignore
        metrics_dict["lpips"] = float(lpips)

        images_dict = {"img": combined_rgb}

        return metrics_dict, images_dict

    def construct_list_of_attributes(self):
        l = ["x", "y", "z", "nx", "ny", "nz"]
        # All channels except the 3 DC
        # features_dc, features_rest = SH2RGB(self.colors_all[:, :1, :]), self.colors_all[:, 1:, :]
        features_dc, features_rest = self.colors_all[:, :1, :], self.colors_all[:, 1:, :]

        for i in range(features_dc.shape[1] * features_dc.shape[2]):
            l.append("f_dc_{}".format(i))
        for i in range(features_rest.shape[1] * features_rest.shape[2]):
            l.append("f_rest_{}".format(i))
        l.append("opacity")
        for i in range(self.scales.shape[1]):
            l.append("scale_{}".format(i))
        for i in range(self.quats.shape[1]):
            l.append("rot_{}".format(i))
        return l

    @torch.no_grad()
    def save_ply(self):
        from plyfile import PlyData, PlyElement

        print("\n \n \n saving ply")
        path = "point_cloud_new_transformed.ply"

        xyz = self.means.detach().cpu().numpy()
        normals = np.zeros_like(xyz)

        print(num_sh_bases(self.config.sh_degree))

        # f_dc = SH2RGB(self.colors_all[:, :1, :]).detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
        f_dc = self.colors_all[:, :1, :].detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
        f_rest = (
            self.colors_all[:, 1:, :]
            # .permute(0, 2, 1)
            .detach()
            .transpose(1, 2)
            .flatten(start_dim=1)
            .contiguous()
            .cpu()
            .numpy()
        )

        opacities = self.opacities.detach().cpu().numpy()

        scale = self.scales.detach().cpu().numpy()
        rotation = self.quats.detach().cpu().numpy()

        dtype_full = [(attribute, "f4") for attribute in self.construct_list_of_attributes()]

        elements = np.empty(xyz.shape[0], dtype=dtype_full)

        attributes = np.concatenate((xyz, normals, f_dc, f_rest, opacities, scale, rotation), axis=1)
        elements[:] = list(map(tuple, attributes))
        el = PlyElement.describe(elements, "vertex")
        PlyData([el]).write(path)
