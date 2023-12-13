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
Evaluation utils
"""
from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Literal, Optional, Tuple, Callable, Dict

import torch
import yaml
from torch import Tensor

from nerfstudio.configs.method_configs import all_methods
from nerfstudio.engine.trainer import TrainerConfig
from nerfstudio.pipelines.base_pipeline import Pipeline
from nerfstudio.utils.rich_utils import CONSOLE
import pdb
import numpy as np
from plyfile import PlyData, PlyElement
import torch.nn.functional as F
from nerfstudio.data.dataparsers.blender_dataparser import BlenderDataParserConfig



def SH2RGB(sh):
    C0 = 0.28209479177387814
    return sh * C0 + 0.5

def RGB2SH(rgb):
    C0 = 0.28209479177387814
    return (rgb - 0.5) / C0



def _sqrt_positive_part(x):
    """
    Returns torch.sqrt(torch.max(0, x))
    but with a zero subgradient where x is 0.
    """
    ret = torch.zeros_like(x)
    positive_mask = x > 0
    ret[positive_mask] = torch.sqrt(x[positive_mask])
    return ret


def matrix_to_quaternion(matrix: Tensor) -> Tensor:
    """
    Convert rotations given as rotation matrices to quaternions.

    Args:
        matrix: Rotation matrices as tensor of shape (..., 3, 3).

    Returns:
        quaternions with real part first, as tensor of shape (..., 4).
    """
    if matrix.size(-1) != 3 or matrix.size(-2) != 3:
        raise ValueError(f"Invalid rotation matrix shape {matrix.shape}.")

    batch_dim = matrix.shape[:-2]
    m00, m01, m02, m10, m11, m12, m20, m21, m22 = torch.unbind(matrix.reshape(batch_dim + (9,)), dim=-1)

    q_abs = _sqrt_positive_part(
        torch.stack(
            [
                1.0 + m00 + m11 + m22,
                1.0 + m00 - m11 - m22,
                1.0 - m00 + m11 - m22,
                1.0 - m00 - m11 + m22,
            ],
            dim=-1,
        )
    )

    # we produce the desired quaternion multiplied by each of r, i, j, k
    quat_by_rijk = torch.stack(
        [
            # pyre-fixme[58]: `**` is not supported for operand types `Tensor` and
            #  `int`.
            torch.stack([q_abs[..., 0] ** 2, m21 - m12, m02 - m20, m10 - m01], dim=-1),
            # pyre-fixme[58]: `**` is not supported for operand types `Tensor` and
            #  `int`.
            torch.stack([m21 - m12, q_abs[..., 1] ** 2, m10 + m01, m02 + m20], dim=-1),
            # pyre-fixme[58]: `**` is not supported for operand types `Tensor` and
            #  `int`.
            torch.stack([m02 - m20, m10 + m01, q_abs[..., 2] ** 2, m12 + m21], dim=-1),
            # pyre-fixme[58]: `**` is not supported for operand types `Tensor` and
            #  `int`.
            torch.stack([m10 - m01, m20 + m02, m21 + m12, q_abs[..., 3] ** 2], dim=-1),
        ],
        dim=-2,
    )

    # We floor here at 0.1 but the exact level is not important; if q_abs is small,
    # the candidate won't be picked.
    flr = torch.tensor(0.1).to(dtype=q_abs.dtype, device=q_abs.device)
    quat_candidates = quat_by_rijk / (2.0 * q_abs[..., None].max(flr))

    # if not for numerical problems, quat_candidates[i] should be same (up to a sign),
    # forall i; we pick the best-conditioned one (with the largest denominator)

    return quat_candidates[F.one_hot(q_abs.argmax(dim=-1), num_classes=4) > 0.5, :].reshape(batch_dim + (4,))


def quaternion_to_matrix(quaternions: Tensor) -> Tensor:
    """
    Convert rotations given as quaternions to rotation matrices.

    Args:
        quaternions: quaternions with real part first,
            as tensor of shape (..., 4).

    Returns:
        Rotation matrices as tensor of shape (..., 3, 3).
    """
    r, i, j, k = torch.unbind(quaternions, -1)
    # pyre-fixme[58]: `/` is not supported for operand types `float` and `Tensor`.
    two_s = 2.0 / (quaternions * quaternions).sum(-1)

    o = torch.stack(
        (
            1 - two_s * (j * j + k * k),
            two_s * (i * j - k * r),
            two_s * (i * k + j * r),
            two_s * (i * j + k * r),
            1 - two_s * (i * i + k * k),
            two_s * (j * k - i * r),
            two_s * (i * k - j * r),
            two_s * (j * k + i * r),
            1 - two_s * (i * i + j * j),
        ),
        -1,
    )
    return o.reshape(quaternions.shape[:-1] + (3, 3))



def transform_point_cloud(points3D: Tensor, transform_matrix: Tensor) -> Tensor:

    points3D = (
        torch.cat(
            (
                points3D,
                torch.ones_like(points3D[..., :1]),
            ),
            -1,
        )
        @ transform_matrix.T
    )
    return points3D

def transform_shs(shs,transform_matrix) -> Tensor:
    # return (shs.permute(0,2,1) @ transform_matrix[:,:3].double()).permute(0,2,1)
    return shs

def transform_rotation(quats, transform_matrix) -> Tensor:
    quats = quats / quats.norm(dim=-1, keepdim=True)
    rots = quaternion_to_matrix(quats)
    rots = torch.bmm(transform_matrix[None,:,:3].repeat(rots.shape[0],1,1).double(), rots)
    quats2 = matrix_to_quaternion(rots)
    quats2 = quats2 / quats2.norm(dim=-1, keepdim=True)

    return quats2


def create_state_dict(load_path, transform_matrix):
    plydata = PlyData.read(load_path)
    xyz = np.stack(
        (
            np.asarray(plydata.elements[0]["x"]),
            np.asarray(plydata.elements[0]["y"]),
            np.asarray(plydata.elements[0]["z"]),
        ),
        axis=1,
    )

    opacities = np.asarray(plydata.elements[0]["opacity"])[..., np.newaxis]

    features_dc = np.zeros((xyz.shape[0], 3, 1))
    features_dc[:, 0, 0] = np.asarray(plydata.elements[0]["f_dc_0"])
    features_dc[:, 1, 0] = np.asarray(plydata.elements[0]["f_dc_1"])
    features_dc[:, 2, 0] = np.asarray(plydata.elements[0]["f_dc_2"])

    extra_f_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("f_rest_")]
    extra_f_names = sorted(extra_f_names, key=lambda x: int(x.split("_")[-1]))

    features_extra = np.zeros((xyz.shape[0], len(extra_f_names)))
    for idx, attr_name in enumerate(extra_f_names):
        features_extra[:, idx] = np.asarray(plydata.elements[0][attr_name])

    features_extra = features_extra.reshape((features_extra.shape[0], 3, int(features_extra.shape[1] / 3)))

    scale_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("scale_")]
    scale_names = sorted(scale_names, key=lambda x: int(x.split("_")[-1]))
    scales = np.zeros((xyz.shape[0], len(scale_names)))
    for idx, attr_name in enumerate(scale_names):
        scales[:, idx] = np.asarray(plydata.elements[0][attr_name])

    rot_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("rot")]
    rot_names = sorted(rot_names, key=lambda x: int(x.split("_")[-1]))
    rots = np.zeros((xyz.shape[0], len(rot_names)))
    for idx, attr_name in enumerate(rot_names):
        rots[:, idx] = np.asarray(plydata.elements[0][attr_name])

    load_state_dict = dict()
    load_state_dict["_model.means"] = transform_point_cloud(torch.tensor(xyz), transform_matrix)
    load_state_dict["_model.scales"] = torch.tensor(scales)
    load_state_dict["_model.quats"] = transform_rotation(torch.tensor(rots), transform_matrix)

    load_state_dict["_model.opacities"] = torch.tensor(opacities)
    load_state_dict["_model.colors_all"] = torch.cat((
        torch.tensor(features_dc), transform_shs(torch.tensor(features_extra),transform_matrix)), dim=2
    ).permute(0, 2, 1)

    return {"pipeline": load_state_dict, "step": 29999}


def eval_load_checkpoint(config: TrainerConfig, pipeline: Pipeline) -> Tuple[Path, int]:
    ## TODO: ideally eventually want to get this to be the same as whatever is used to load train checkpoint too
    """Helper function to load checkpointed pipeline

    Args:
        config (DictConfig): Configuration of pipeline to load
        pipeline (Pipeline): Pipeline instance of which to load weights
    Returns:
        A tuple of the path to the loaded checkpoint and the step at which it was saved.
    """
    assert config.load_dir is not None
    load_step = 29999 # hardcoded for now. 

    transform_matrix = pipeline.datamanager.train_dataparser_outputs.dataparser_transform
    scale_factor = pipeline.datamanager.train_dataparser_outputs.dataparser_scale
    print(transform_matrix, scale_factor)

    if Path(config.load_dir / f"step-{load_step:09d}.ckpt").exists():
        CONSOLE.log("loading ckpt file")
        load_path = config.load_dir / f"step-{load_step:09d}.ckpt"
        loaded_state = torch.load(load_path, map_location="cpu")
    elif Path(config.load_dir / f"point_cloud.ply").exists():
        CONSOLE.log("loading point cloud")
        load_path = config.load_dir / f"point_cloud.ply"
        loaded_state = create_state_dict(load_path, transform_matrix)
    else:
        CONSOLE.log(f"Checkpoint at {config.load_dir} does not exist")
        exit()

    pipeline.load_pipeline(loaded_state["pipeline"], loaded_state["step"])

    CONSOLE.print(f":white_check_mark: Done loading checkpoint from {load_path}")
    return load_path, load_step


def generate_config(config_path: Path) -> Dict: 
    CONSOLE.log(f"[yellow] Generating config and loading data from {config_path}")
    config = all_methods["gaussian-splatting"]


    config.pipeline.datamanager.dataparser.auto_scale_poses = False 
    config.pipeline.datamanager.data = config_path
    if Path(config_path / "colmap/sparse/0/").exists():
        config.pipeline.datamanager.dataparser.colmap_path = Path("colmap/sparse/0")
    elif Path(config_path / "sparse/0/").exists():
        config.pipeline.datamanager.dataparser.colmap_path = Path("sparse/0")
    else:
        CONSOLE.log("[red] Could not find valid colmap directory")
        
    config.pipeline.model.eval_og = True
    config.pipeline.datamanager.dataparser.orientation_method = "none"
    config.pipeline.datamanager.dataparser.center_method = "none"
    CONSOLE.log(config)

    return config


def eval_setup(
    config_path: Path,
    checkpoint_path: Optional[Path] = None,
    eval_num_rays_per_chunk: Optional[int] = None,
    test_mode: Literal["test", "val", "inference"] = "test",
    update_config_callback: Optional[Callable[[TrainerConfig], TrainerConfig]] = None,
) -> Tuple[TrainerConfig, Pipeline, Path, int]:
    """Shared setup for loading a saved pipeline for evaluation.

    Args:
        config_path: Path to config YAML file.
        eval_num_rays_per_chunk: Number of rays per forward pass
        test_mode:
            'val': loads train/val datasets into memory
            'test': loads train/test dataset into memory
            'inference': does not load any dataset into memory
        update_config_callback: Callback to update the config before loading the pipeline


    Returns:
        Loaded config, pipeline module, corresponding checkpoint, and step
    """
    # load save config
    if config_path.suffix == ".yml":
        config = yaml.load(config_path.read_text(), Loader=yaml.Loader)
    else:
        config = generate_config(config_path)

    assert isinstance(config, TrainerConfig)

    config.pipeline.datamanager._target = all_methods[config.method_name].pipeline.datamanager._target
    if eval_num_rays_per_chunk:
        config.pipeline.model.eval_num_rays_per_chunk = eval_num_rays_per_chunk

    if update_config_callback is not None:
        config = update_config_callback(config)

    if checkpoint_path is not None:
        config.load_dir = checkpoint_path
    else:
        config.load_dir = config.get_checkpoint_dir()

    # setup pipeline (which includes the DataManager)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pipeline = config.pipeline.setup(device=device, test_mode=test_mode)

    assert isinstance(pipeline, Pipeline)
    pipeline.eval()

    # load checkpointed information
    checkpoint_path, step = eval_load_checkpoint(config, pipeline)

    return config, pipeline, checkpoint_path, step
