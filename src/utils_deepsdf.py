"""
Utility functions and classes for the training and using the implicit model.
Besides some basic functions, this includes the dataset class and loss function used
to train the model.

torch is required for using this file.

Author: Till Schnabel (contact till.schnabel@inf.ethz.ch); parts of the code were copied, cf. further below.

MIT License

Copyright (c) 2026 ETH Zurich, Till Schnabel

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

The datasets class was initially copied from https://github.com/maurock/DeepSDF, and thus
may be subject to the following license:

MIT License

Copyright (c) 2023 Mauro

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

Other functions were copied from https://github.com/MingwuZheng/ImFace, and are subject to
the following license:

MIT License

Copyright (c) 2022 Mingwu Zheng, Haiyu Zhang

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""

import warnings
from pathlib import Path
from typing import Dict, Union, List

import torch
import numpy as np
from scipy.spatial import KDTree
from torch.autograd import grad
from torch.utils.data import Dataset

from src import utils
from src.objects.indices_and_masks import IndicesAndMasks
from src.objects.mesh import Mesh, Landmarks
from src.objects.registration_file_communicator import RegistrationFileCommunicator


def get_volume_coords(resolution: int, device=None) -> torch.Tensor:
    """
    Get 3-dimensional vector (M, N, P) according to the desired resolution.
    :param resolution: The desired resolution (int).
    :param device: The torch device to use.
    :return torch 3D coordinates
    """
    # Define grid
    grid_values = torch.linspace(-1, 1, steps=resolution)
    if device is not None:
        grid_values = grid_values.to(device)
    grid = torch.meshgrid(grid_values, grid_values, grid_values)

    # Reshape grid to (M*N*P, 3)
    coords = torch.vstack((grid[0].ravel(), grid[1].ravel(), grid[2].ravel())).transpose(1, 0)
    if device is not None:
        coords = coords.to(device)

    return coords


def extract_mesh(grad_size_axis, sdf):
    import skimage
    # Extract zero-level set with marching cubes
    grid_sdf = sdf.view(grad_size_axis, grad_size_axis, grad_size_axis).detach().cpu().numpy()
    vertices, faces, normals, _ = skimage.measure.marching_cubes(grid_sdf, level=0.00)

    # Rescale vertices extracted with marching cubes
    # (https://stackoverflow.com/questions/70834443/converting-indices-in-marching-cubes-to-original-x-y-z-space-visualizing-isosu)
    x_max = np.array([1, 1, 1])
    x_min = np.array([-1, -1, -1])
    vertices = vertices * ((x_max-x_min) / grad_size_axis) + x_min

    return vertices, faces

@torch.jit.script
def skew(w):
    """Build a skew matrix ("cross product matrix") for vector w.
    Modern Robotics Eqn 3.30.
    Args:
      w: (B,N,3) A 3-vector
    Returns:
      W: (B,N,3,3) A skew matrix such that W @ v == w x v
    """
    B, N, _ = w.size()
    W = torch.zeros(B, N, 3, 3).float().to(w.device)
    W[:, :, 0, 1] = -w[:, :, 2]
    W[:, :, 0, 2] = w[:, :, 1]
    W[:, :, 1, 0] = w[:, :, 2]
    W[:, :, 1, 2] = -w[:, :, 0]
    W[:, :, 2, 0] = -w[:, :, 1]
    W[:, :, 2, 1] = w[:, :, 0]
    return W

def rp_to_se3(R: torch.Tensor, p: torch.Tensor) -> torch.Tensor:
    """Rotation and translation to homogeneous transform.
    Args:
      R: (B,N,3,3) An orthonormal rotation matrix.
      p: (B,N,3) A 3-vector representing an offset.
    Returns:
      X: (B,N,4,4) The homogeneous transformation matrix described by rotating by R
        and translating by p.
    """
    B, N, _ = p.size()
    p = torch.reshape(p, (B, N, 3, 1))
    h = torch.tensor([[0.0, 0.0, 0.0, 1.0]]).to(R.device)  # (1,4)
    return torch.cat([torch.cat([R, p], dim=-1), h[None, None, ...].repeat(B, N, 1, 1)], dim=-2)

def exp_so3(w: torch.tensor, theta: torch.tensor) -> torch.tensor:
    """Exponential map from Lie algebra so3 to Lie group SO3.
    Modern Robotics Eqn 3.51, a.k.a. Rodrigues' formula.
    Args:
      w: (B,N,3) An axis of rotation.
      theta: (B,N) An angle of rotation.
    Returns:
      R: (B,N,3,3) An orthonormal rotation matrix representing a rotation of
        magnitude theta about axis w.
    """
    B, N, _ = w.size()
    theta = theta[..., None, None].repeat(1, 1, 3, 3)
    W = skew(w)
    # WxW = torch.einsum('bnhi,bniw->bnhw', W, W)
    return torch.eye(3)[None, None, ...].repeat(B, N, 1, 1).to(w.device) + torch.sin(theta) * W + (
            1.0 - torch.cos(theta)) * W @ W

def exp_se3(S: torch.tensor, theta: torch.tensor) -> torch.tensor:
    """Exponential map from Lie algebra so3 to Lie group SO3.
    Modern Robotics Eqn 3.88.
    Args:
      S: (B,N,6) A screw axis of motion.
      theta: (B,N) Magnitude of motion.
    Returns:
      a_X_b: (B,N,4,4) The homogeneous transformation matrix attained by integrating
        motion of magnitude theta about S for one second.
    """
    B, N, _ = S.size()
    w, v = torch.split(S, 3, dim=-1)
    W = skew(w)
    R = exp_so3(w, theta)

    # assert torch.isnan(W).sum() == 0, print(W)
    # assert torch.isnan(R).sum() == 0, print(R)

    theta = theta[..., None, None].repeat(1, 1, 3, 3)

    # assert torch.isnan(theta).sum() == 0, print(theta)

    p = (theta * (torch.eye(3)[None, None, ...].repeat(B, N, 1, 1).to(S.device)) + (1.0 - torch.cos(theta)) * W +
         (theta - torch.sin(theta)) * W @ W) @ v[..., None]
    return rp_to_se3(R, p.squeeze(-1))


def to_homogenous(v):
    return torch.cat([v, torch.ones_like(v[..., :1]).to(v.device)], dim=-1)


def from_homogenous(v):
    return v[..., :3] / v[..., -1:]



def warp(xyz, deformation, warp_type):
    """
    From IMFACE
    :param xyz: (B,N,3)
    :param deformation: (B,N,6)/(B,N,3)
    :param warp_type: 'translation' or 'se3'
    :return: warped xyz
    """
    if len(xyz.size()) == 2:
        xyz = xyz.unsqueeze(0)
    if len(deformation.size()) == 2:
        deformation = deformation.unsqueeze(0)

    B, N, _ = xyz.size()
    if warp_type == 'translation':
        return xyz + deformation
    elif warp_type == 'se3':
        w = deformation[:, :, :3]
        v = deformation[:, :, 3:6]
        theta = torch.norm(w, dim=-1)
        w = w / (theta[..., None] + 1e-8)
        v = v / (theta[..., None] + 1e-8)
        screw_axis = torch.cat([w, v], dim=-1)
        transform = exp_se3(screw_axis, theta)
        warped_points = from_homogenous(
            (transform @ (to_homogenous(xyz)[..., None])).squeeze(-1))
        return warped_points
    else:
        raise ValueError

def gradient(outputs, inputs):
    """
    compute the gradient of the provided outputs with respect to inputs.
    """
    d_points = torch.ones_like(outputs, requires_grad=False, device=outputs.device)
    if isinstance(inputs, list):
        points_grad = []
        for input_ind in inputs:
            points_grad_ind = grad(
                outputs=outputs, inputs=input_ind, grad_outputs=d_points, create_graph=True, retain_graph=True,
                only_inputs=True, allow_unused=True)[0]
            if points_grad_ind is None:  # None means the whole region is masked out, but we do also mask the loss out, so it should be fine
                points_grad.append(torch.zeros_like(input_ind))
            else:
                points_grad.append(points_grad_ind[..., -3:])
        points_grad = torch.cat(points_grad, dim=1)
    else:
        points_grad = grad(
            outputs=outputs, inputs=inputs, grad_outputs=d_points, create_graph=True, retain_graph=True,
            only_inputs=True, allow_unused=True)[0][..., -3:]
    return points_grad

def loss(net_input: Dict, net_middle: Dict, pred_sdf: torch.Tensor, weights: Dict,
         skip_weighting_deformed_landmarks: bool = False):
    """
    Compute loss to train implicit model.
    """
    landmarks_gt = net_input['key_pts']
    landmark_weights = net_input["landmark_weights"]

    gt_sdf = net_input['gt_sdf']
    gt_normals = net_input['gt_normals']
    x_latent = net_input['latent']

    # Deform-Nets
    coords = net_input['coords']  # First Input

    # Reference-Nets
    sdf_correct = net_middle['correction']  # Second Output 2

    if "normal" in weights or "eikonal" in weights:
        pred_grad = gradient(pred_sdf, coords)

    on_surface = gt_sdf == 0
    near_surface = gt_sdf < 0
    assert torch.all(~torch.logical_and(on_surface, near_surface))
    off_surface = ~torch.logical_or(on_surface, near_surface)

    loss_dict = {}
    loss_tot = 0

    for loss_kind, weight in weights.items():
        if weight == 0:
            continue

        # Surface loss: the model is tasked to predict values close to 0 for points on the surface.
        if loss_kind == "surface":
            if not torch.any(on_surface):
                continue
            loss_current = torch.mean(torch.abs(pred_sdf[on_surface]))

        # Normal loss: For points on the surface, the gradient of the model prediction
        # should be close to the respective surface normals.
        elif loss_kind == "normal":
            if not torch.any(on_surface):
                continue
            loss_current = torch.mean(1 - torch.nn.functional.cosine_similarity(pred_grad[on_surface], gt_normals.view(-1, 3), dim=-1))

        # Eikonal loss: The gradient of the model prediction should have unit norm everywhere,
        # so that it fulfills classical SDF properties.
        elif loss_kind == "eikonal":
            loss_current = torch.mean(torch.abs(torch.linalg.norm(pred_grad, dim=-1) - 1))

        # Off-surface loss: For points sampled in space, the model prediction should not be too close to 0.
        elif loss_kind == "off_surface":
            if not torch.any(off_surface):
                continue
            alpha = weights.get("off_surface_alpha", 10)
            loss_current = torch.mean(torch.exp(-alpha * torch.abs(pred_sdf[off_surface])))

        # Latent regularization: L2 norm on latent codes. This incentivizes the model to keep the
        # latents small and normal-distributed.
        elif loss_kind == "latent_reg":
            loss_current = torch.mean(torch.linalg.norm(x_latent, dim=-1) ** 2)

        # If we have additional landmark correspondence latents, they can be similarly regularized.
        elif loss_kind == "correspondence_latent_reg":
            correspondence_latent = net_input.get("correspondence_latent", None)
            if correspondence_latent is None:
                raise AssertionError("You set the correspondence_latent_reg, "
                                     "but this method did not receive the latent")

            loss_current = torch.mean(torch.linalg.norm(correspondence_latent, dim=-1) ** 2)

        # SDF correction loss: Penalize large corrections to drive the model to only add corrections
        # where really necessary, while keeping the remaining SDF in correspondence with the template.
        elif loss_kind == 'sdf_correction':
            if sdf_correct is None:
                raise ValueError("You probably chose to not use the SDF correction, "
                                 "so please also remove the sdf_correction from your loss settings.")
            loss_current = torch.mean(torch.abs(sdf_correct))

        # If we predict correspondence landmarks via the correspondence latents, we add a landmark loss here.
        elif loss_kind == "landmarks_correspondence_pred":
            landmarks_correspondence_pred = net_middle['landmarks_correspondence_pred']
            if landmarks_correspondence_pred is None or landmarks_gt is None:
                raise AssertionError("Landmark loss requires landmarks.")
            if landmarks_correspondence_pred.shape[-2] == 0:
                continue
            loss_current = torch.linalg.norm(landmarks_correspondence_pred - landmarks_gt, dim=-1) ** 2
            if landmark_weights is None:
                loss_current = torch.mean(loss_current)
            else:
                loss_current = torch.sum(loss_current * landmark_weights) / torch.sum(landmark_weights)

        # This is the default landmark loss. It forces the deformation to adhere to the average landmark positions,
        # so landmarks on the different targets should be backward-deformed toward the average template landmarks.
        elif loss_kind == 'landmarks_deform':
            landmarks_deformed = net_middle['landmarks_deformed']
            landmarks_avg_all = net_input['all_key_pts']

            if landmarks_deformed is None or landmarks_avg_all is None:
                raise AssertionError("Landmark loss requires landmarks.")
            if landmarks_deformed.shape[-2] == 0:
                continue
            loss_current = torch.linalg.norm(landmarks_deformed - landmarks_avg_all, dim=-1) ** 2
            if landmark_weights is None or skip_weighting_deformed_landmarks:
                loss_current = torch.mean(loss_current)
            else:
                loss_current = torch.sum(loss_current * landmark_weights) / torch.sum(landmark_weights)

        # Since off_surface_alpha can be specified in the weights, we need to handle it here.
        elif loss_kind == "off_surface_alpha":
            continue

        else:
            raise ValueError("Unknown loss kind: {}".format(loss_kind))
        if loss_kind in loss_dict:
            raise AssertionError(f"Duplicated loss entry: {loss_kind}")

        # In the loss_dict, we keep the original losses, whereas we add them with their respective weightings
        # to the loss_tot.
        loss_dict[loss_kind] = loss_current
        loss_tot += weight * loss_current

    return loss_tot, loss_dict

def compute_axis_aligned_bounding_box(points: np.ndarray):
    """
    Compute centroid and extents along x, y, and z
    """
    assert points.shape[1] == 3
    max_points, min_points = np.max(points, axis=0), np.min(points, axis=0)
    centroid = (max_points + min_points) / 2.0
    extents = max_points - min_points
    return centroid, extents

def uniform_ball(n_points: int, rad: float = 1.0):
    """
    Sample points within a 3D sphere by randomizing spherical coordinates.
    """
    angle1 = np.random.rand(n_points) * 2 - 1
    angle2 = np.random.rand(n_points)
    radius = np.random.rand(n_points) * rad

    r = radius ** (1/3)
    theta = np.arccos(angle1) #np.pi * angle1
    phi = 2 * np.pi * angle2
    x = r * np.sin(theta) * np.cos(phi)
    y = r * np.sin(theta) * np.sin(phi)
    z = r * np.cos(theta)

    return np.stack([x, y, z], axis=-1)

class SDFDataset(Dataset):

    def __init__(self, cfg: Dict, device):
        """
        This is the dataset class for training our DeepSDF-like model.
        It is passed a config dictionary that must specify the "dataset" with the path to the dataset folder.
        The meshes from the dataset folder are all loaded into RAM, so make sure you have enough of that.
        Inside the dataset folder, we require the meshes to adhere to a specific structure:
        input mesh: MESHNAME_input.ply (we just use ply here, but other mesh formats should work as well).
        additional input meshes from other domains (e.g. palates): MESHNAME_input_i.ply (i in [2, 3, 4, ...]
        registered mesh: MESHNAME_registered.ply (note that there may be multiple input meshes, but only one
                         registered mesh, which is assumed to have registered all input meshes together).
        input mesh exclude indices: MESHNAME_input(_i)_exclude.txt (doesn't need to exist)
        input mesh include indices: MESHNAME_input(_i)_include.txt (doesn't need to exist,
                                    overwrites any filters to exclude vertices from training)
        registered mesh exclude indices: MESHNAME_registered_exclude.txt (doesn't need to exist)
        everything also flipped: MESHNAME_flipped_REMAININGSTUFF.ply/txt

        You can further use the cfg to:
        "dataset_input_numbers": Set this e.g. to [1, 2] to train on the first two mesh domains,
                                 i.e., meshes ending with "input" or "input_2". This could be the faces and palates.
                                 By default, this is [1], so only the "input" meshes are loaded and trained on.
        "mesh_filter": Use this to filter meshes to train on from the dataset.
        "exclude_mesh_filter": Negative filter to exclude meshes from the dataset.
        "load_only_samples_with_all_inputs": Set True to only train on meshes for which all dataset_input_numbers are
                                             available. Default False means that the data samples are still used to
                                             train the model even if only one of the dataset_input_numbers is available.
                                             E.g., you chose faces and palates via [1, 2], but the model is also trained
                                             on samples of which only the face or the palate is available, but not both.
        "registered_regions_per_input": Specify a list of file names, e.g. ["outer_head", "palate"], to specify for each
                                        of the "dataset_input_numbers" domains, which region on the registered meshes
                                        they occupy. If you specify this, you need additional txt files in the dataset
                                        folder that encode these regions (vertex indices) for the registered meshes.
        "load_flipped": Default True. Set False to only train on original meshes. Note that you still need to have the
                        flipped files in the dataset folder.
        "far_sampling": Default "cube". Use "sphere" to sample far points within the unit sphere.
        "num_surface_samples": How many points to sample on the surface (and near the surface) each iteration.
                               Default 750.
        "num_far_samples": How many points to sample uniformly in space. Default num_surface_samples/8.
        "close_sample_offset": For the near surface points, range around surface points to sample them from.
                               Default 0.01.
        "transform_initial_input": Default False. Set True to upscale input meshes to fill more of the unit cube.
        "test_run": Set True to only load six meshes from the dataset to speed up the loading process for test runs.
        "include_landmarks": List of list of registered vertex indices, each list representing the landmarks for one
                             domain of meshes, where each entry represents a landmark. E.g.: [[20, 250, 30], [10]].
        "lm_class_masking": Default False. Set True to set the landmark weights to 0 for all landmarks if their
                            respective input mesh is missing (recommended if you haven't properly predicted the missing
                            regions in the registered meshes.
        "far_sample_dist_thresh": Default 0.1. How close far samples are allowed to come to excluded regions of the
                                  registered mesh. We chose 0.1, which is relatively large, but you may experiment
                                  with it.
        "use_template_for_avg_landmarks": Default False. Set True to use the template for computing the average
                                          landmarks. This requires you to have a template.ply file in the dataset
                                          folder. The "include_landmarks" indices are used to get the landmarks from
                                          that template. Otherwise, the landmarks from all loaded registered meshes
                                          are averaged.
        "num_jobs": Default 7, unless test_run is True, then 1. Define how many processors to use in parallel
                    for loading the meshes.
        """

        self.dataset_path = Path(cfg["dataset"])
        self.device = device

        # Which mesh domains to train on
        input_numbers = cfg.get("dataset_input_numbers", [1])
        self.input_numbers = input_numbers
        mesh_dict: Dict[str, List[Union[None, Path]]] = {}
        mesh_filter = cfg.get("mesh_filter", None)
        exclude_mesh_filter = cfg.get("exclude_mesh_filter", None)
        # Load all input meshes from all input domains
        for num_input, input_number in enumerate(input_numbers):
            input_number_addon = "" if input_number == 1 else f"_{input_number}"
            # We first find all flipped meshes, and assume that the unflipped meshes are the same
            # without the "_flipped" term.
            mesh_filter_base = f"*flipped_input{input_number_addon}"
            if mesh_filter is not None:
                mesh_filter_current = [mesh_filter] if not isinstance(mesh_filter, list) else mesh_filter
                mesh_filter_current = [mf + mesh_filter_base for mf in mesh_filter_current]
            else:
                mesh_filter_current = mesh_filter_base
            flipped_mesh_paths_current: List[Path] = Mesh.get_all_mesh_files_in_path(
                self.dataset_path, regex_names=mesh_filter_current,
                exclude_regex_names=exclude_mesh_filter
            )

            for path in flipped_mesh_paths_current:
                path_stem = path.stem
                if len(input_number_addon) > 0:
                    path_stem = path_stem[:-len(input_number_addon)]
                if path_stem in mesh_dict:
                    mesh_dict[path_stem].append(path)
                else:
                    # We use None for non-existing meshes
                    mesh_dict[path_stem] = [None] * num_input + [path]
            for key, value in mesh_dict.items():
                if len(value) < num_input + 1:
                    assert len(value) == num_input
                    mesh_dict[key] = value + [None]

        # Optionally discard all samples for which not all domains are available
        load_only_samples_with_all_inputs = cfg.get("load_only_samples_with_all_inputs", False)
        if load_only_samples_with_all_inputs:
            flipped_mesh_paths = [mesh_paths for mesh_paths in mesh_dict.values() if
                                  all([mp is not None for mp in mesh_paths])]
        else:
            flipped_mesh_paths = list(mesh_dict.values())

        # If registered regions are provided, we can use them for filtering
        if len(input_numbers) > 1 and "registered_regions_per_input" in cfg:
            registered_regions_per_input = [IndicesAndMasks.load(self.dataset_path.joinpath(path + ".txt"))
                                            for path in cfg["registered_regions_per_input"]]
        else:
            registered_regions_per_input = None

        # Optionally discard flipped meshes
        self.orientations = ["normal", "flipped"] if cfg.get("load_flipped", True) else ["normal"]

        # This is the attribute that saves all relevant mesh data for training.
        self.subjects = {
            orientation: {} for orientation in self.orientations
        }

        # Point sampling settings.
        self.far_sampling_strat = cfg.get("far_sampling", "cube").lower()
        self.num_surface_samples = cfg.get("num_surface_samples", 750)
        self.num_far_samples = cfg.get("num_far_samples", self.num_surface_samples//8)
        self.close_sample_offset = cfg.get("close_sample_offset", 0.01)
        self.far_sample_dist_thresh = cfg.get("far_sample_dist_thresh", 0.1)

        # Optionally upscale input meshes to unit cube.
        transform_initial_input = cfg.get("transform_initial_input", False)

        # test_run only loads six samples from the training set, so that we can test things more quickly.
        test_run = cfg.get("test_run", False)
        if test_run:
            center_ind = len(flipped_mesh_paths)//2
            flipped_mesh_paths = [*flipped_mesh_paths[:2], *flipped_mesh_paths[-center_ind:-center_ind+2], *flipped_mesh_paths[-2:]]

        # Landmark settings
        landmarks_to_include = cfg.get("include_landmarks", [[]])
        self.include_landmarks = len(landmarks_to_include) > 0
        lm_class_masking = cfg.get("lm_class_masking", False)

        # Method to load the attributes of the input mesh(es).
        # Note that the flipped_mesh_path here is still a list with the paths for the different domains.
        def load_meshes(flipped_mesh_path: List[Union[Path, None]], orientation: str):
            if orientation == "flipped":
                mesh_path = flipped_mesh_path
            # If the mesh is not flipped, find the respective path without the flip
            else:
                mesh_path = [None if path is None else path.with_name(path.name.replace("flipped_", "")) for path in flipped_mesh_path]
                for path in mesh_path:
                    if path is None:
                        continue
                    if not path.exists():
                        raise FileNotFoundError(f"{path} does not exist!")
            # From the first existing path, we get the registered mesh (it's on registered mesh for all input meshes)
            for path in mesh_path:
                if path is not None:
                    registered_mesh_path = path.with_name(path.stem.split("input")[0] + "registered" + path.suffix)
                    if not registered_mesh_path.exists():
                        raise FileNotFoundError(f"Couldn't associate {path} with registered path {registered_mesh_path}")
                    break

            # Load the registered mesh along with its excluded regions
            registered_mesh = Mesh.load(registered_mesh_path)
            registered_vertices = registered_mesh.get_vertices()
            registered_exclude_indices, _ = RegistrationFileCommunicator.get_excluded_regions_on_registered_mesh(
                registered_mesh_path)
            registered_exclude_mask = IndicesAndMasks.get_mask(registered_exclude_indices, registered_mesh)

            # This method returns for a single input mesh its filtered vertices and normals,
            def get_mesh_data(
                    current_mesh_path: Path, current_registered_exclude_mask,
                    mesh_dist_thresh: float = 0.02):
                # Load the mesh
                mesh = Mesh.load(current_mesh_path)
                if init_transform is not None:
                    mesh.transform(init_transform, in_place=True)
                vertices = np.asarray(mesh.vertices, dtype=np.float32)
                normals = np.asarray(mesh.get_vertex_normals(), dtype=np.float32)
                include_vertex_mask = np.ones(len(vertices), dtype=bool)

                # Load the exclude labels for the original mesh and exclude the respective labeled vertices.
                exclude_indices_path = current_mesh_path.with_name(f"{current_mesh_path.stem}_exclude.txt")
                exclude_indices_path_alt = current_mesh_path.with_name(f"{current_mesh_path.stem.replace('input', 'exclude')}.txt")
                if exclude_indices_path.exists() or exclude_indices_path_alt.exists():
                    exclude_indices = IndicesAndMasks.load(exclude_indices_path if exclude_indices_path.exists() else exclude_indices_path_alt)
                    try:
                        include_vertex_mask[exclude_indices] = False
                    except IndexError:
                        warnings.warn(f"Problem with indices of {mesh_path}. Skipping for now.")

                # We also remove all vertices from being sampled if they're outside of the bounding box defined
                # by the registered mesh.
                vertex_out_of_box = np.any((vertices > 1) | (vertices < -1), axis=1)
                include_vertex_mask[vertex_out_of_box] = False

                # Find closest points on the registered mesh
                _, closest_registered_vertex_indices = registered_mesh.get_closest_vertices(vertices)
                closest_registered_triangle_points, _ = registered_mesh.get_closest_triangle_points(vertices)

                vertex_triangle_connection_vector = closest_registered_triangle_points - vertices
                # We also exclude mesh vertices of which the closest triangle point on the
                # registered mesh is relatively far away
                mesh_distance = np.linalg.norm(vertex_triangle_connection_vector, axis=1)
                registered_far_mask = mesh_distance > mesh_dist_thresh
                # We exclude mesh vertices of which the closest registered vertex couldn't find a correspondence
                is_closest_vertex_excluded = current_registered_exclude_mask[closest_registered_vertex_indices]
                include_vertex_mask[IndicesAndMasks.join(is_closest_vertex_excluded, registered_far_mask)] = False

                # If there are some vertices explicitly marked to be included, we always set the mask for those to True
                include_indices_path = current_mesh_path.with_name(f"{current_mesh_path.stem}_include.txt")
                if include_indices_path.exists():
                    include_vertex_mask[IndicesAndMasks.load(include_indices_path)] = True

                # Return the filtered vertices and normals
                return vertices[include_vertex_mask], normals[include_vertex_mask]

            # Save all instance attributes in these lists
            vertices_tot, normals_tot, landmarks_tot, landmark_weights_tot, forbidden_registered_kd_tree = [], [], [], [], []
            # Iterate over the domains
            for num_path, path in enumerate(mesh_path):
                # We use the registered mesh with defined domain regions (if available) for filtering.
                if registered_regions_per_input is not None:
                    current_region = registered_regions_per_input[num_path]
                    other_regions = IndicesAndMasks.invert(current_region, registered_mesh)
                    exclude_region = IndicesAndMasks.join(registered_exclude_indices, other_regions)
                    exclude_mask = IndicesAndMasks.get_mask(exclude_region, registered_mesh)
                    forbidden_registered_kd_tree.append(KDTree(registered_mesh.vertices[exclude_region]))
                else:
                    current_region = list(range(len(registered_vertices)))
                    exclude_mask = registered_exclude_mask

                # By default, we don't compute any transform on the meshes.
                init_transform = None
                # Only if the user chose an initial transform do we upscale the mesh.
                # We only do this for the first domain.
                if num_path == 0 and transform_initial_input:
                    center, extents = compute_axis_aligned_bounding_box(registered_vertices[current_region])
                    transform_center_mat = np.eye(4)
                    transform_center_mat[:3, -1] = -center
                    extents = extents * 1.005
                    init_transform = utils.get_scaling_matrix(2 / np.max(extents), homogeneous=True) @ transform_center_mat
                    registered_mesh.transform(init_transform, in_place=True)
                    registered_vertices = registered_mesh.get_vertices()

                current_landmarks_to_include = landmarks_to_include[num_path] if num_path  < len(landmarks_to_include) else None

                landmarks_add, landmark_weights_add = None, None
                if current_landmarks_to_include is not None and len(current_landmarks_to_include) > 0:
                    # We theoretically allow the user to specify a list of vertex indices per landmark,
                    # so that they could be averaged. We don't really use this in practice, though.
                    landmarks_add = np.asarray([np.mean(
                        registered_vertices[lm_idx if isinstance(lm_idx, list) else [lm_idx]], axis=0)
                        for lm_idx in current_landmarks_to_include])
                    # One could exclude landmarks based on whether they lie in a registered region or not.
                    # Currently, we only put them to 0 for missing domain instances (if the user chose lm_class_masking)
                    landmark_weights_add = np.ones(len(landmarks_add))

                if path is None:
                    vertices_tot.append(None)
                    normals_tot.append(None)
                    # Here, we put the landmark weights to 0 if the user chose lm_class_masking
                    if lm_class_masking:
                        landmark_weights_add = np.zeros(len(landmarks_add))
                else:
                    # This is a fully hard-coded mesh distance threshold that defines how close
                    # mesh vertices need to be to the closest registered point in order to be included.
                    # This could be made dependent on the mesh scale.
                    vertices_add, normals_add = get_mesh_data(
                        path, exclude_mask, mesh_dist_thresh=0.02 if num_path == 0 else 0.005)
                    vertices_tot.append(vertices_add)
                    normals_tot.append(normals_add)
                # these can still be None here (handled in getitem function)
                landmarks_tot.append(landmarks_add)
                landmark_weights_tot.append(landmark_weights_add)

            # We return all relevant attributes in a dictionary to be saved inside self.subjects
            return_dict = {
                "vertices": vertices_tot,
                "normals": normals_tot,
                "name": registered_mesh_path.stem.replace("_flipped", "").replace("_registered", ""),
                "forbidden_registered_kd_tree": forbidden_registered_kd_tree                }
            if landmarks_tot is not None:
                return_dict["landmarks"] = landmarks_tot
            if landmark_weights_tot is not None:
                return_dict["landmark_weights"] = landmark_weights_tot

            return return_dict

        # The load_meshes function is run in parallel, unless test_run, because then we can also debug more easily
        # into the function. The results are saved into the self.subjects attribute.
        for orientation in self.orientations:
            self.subjects[orientation] = utils.parallel_method(
                load_meshes, flipped_mesh_paths, num_jobs=1 if test_run else cfg.get("num_jobs", 7),
                orientation=orientation,
                tqdm_message=f"Loading {orientation} meshes.")

        # If we include landmarks, we need to define the average landmarks here,
        # either from the template or by averaging over all individual instances.
        if self.include_landmarks:
            if cfg.get("use_template_for_avg_landmarks", False):
                template_mesh = Mesh.load(self.dataset_path.joinpath("template.ply"))
                template_vertices = template_mesh.get_vertices()
                if transform_initial_input:
                    # This is duplicated stuff. It should be made less hard-coded,
                    # so the init_transform defined within the load_meshes function should ideally be used.
                    warnings.warn("Currently maybe hardcoded for palate-only case. Adjust if needed.")
                    center, extents = compute_axis_aligned_bounding_box(template_vertices[registered_regions_per_input[0]])
                    transform_center_mat = np.eye(4)
                    transform_center_mat[:3, -1] = -center
                    extents = extents * 1.005
                    init_transform = utils.get_scaling_matrix(
                        2 / np.max(extents), homogeneous=True) @ transform_center_mat
                    template_mesh.transform(init_transform, in_place=True)
                    template_vertices = template_mesh.get_vertices()
                self.avg_landmarks = np.concatenate([template_vertices[indices] for indices in landmarks_to_include])
            else:
                self.avg_landmarks = np.concatenate([np.mean([
                    subject_ind["landmarks"][i] for orientation in self.orientations
                    for subject_ind in self.subjects[orientation] if subject_ind["landmarks"][i] is not None], axis=0)
                    for i in range(len(input_numbers))])

    def __len__(self):
        return len(self.subjects["normal"])

    # This function is called during training to retrieve the data from one instance.
    def __getitem__(self, idx):
        # The orientation (normal or flipped) is randomized.
        orientation = np.random.choice(self.orientations)
        # These are all vertices and normals, from which we subsample.
        vertices_all, normals_all = self.subjects[orientation][idx]["vertices"], self.subjects[orientation][idx]["normals"]

        # This is the function that returns the sampled points and normals.
        def get_samples(vertices, normals, forbidden_registered_kd_tree,
                        num_surface_samples=None, num_far_samples=None):
            if num_surface_samples is None:
                num_surface_samples = self.num_surface_samples
            if num_far_samples is None:
                num_far_samples = self.num_far_samples

            vertex_indices = torch.randperm(vertices.shape[0])
            if len(vertex_indices) < num_surface_samples:
                warnings.warn(f"Mesh {self.subjects[orientation][idx]['name']} (idx {idx}) doesn't even have "
                              f"{self.num_surface_samples} vertices. Check why that's the case.")
                vertex_indices = vertex_indices.repeat(int(np.ceil(num_surface_samples / len(vertex_indices))))[:num_surface_samples]
            else:
                vertex_indices = vertex_indices[:num_surface_samples]
            vertices_current_np = vertices[vertex_indices]
            vertices_current = torch.asarray(vertices_current_np, device=self.device, requires_grad=False)
            normals_current = torch.asarray(normals[vertex_indices], device=self.device, requires_grad=False)

            if num_far_samples > 0:
                p_far_current = []
                while True:
                    num_samples_current = int(num_far_samples * 1.5)
                    if "cube" in self.far_sampling_strat:
                        # Generate random points in the predefined volume that surrounds all the shapes.
                        # NOTE: shapes must be normalized within [-1, 1]^3
                        p_far_ind = np.random.rand(num_samples_current, 3) * 2 - 1
                    else:
                        # This is a more shape-specific sampler that assumes that the shapes are kinda roundish, so
                        # they lie within a uniform ball (the cube is probably generally safer, but for some
                        # specific domains, this here might be more efficient).
                        p_far_ind = uniform_ball(num_samples_current)
                    if forbidden_registered_kd_tree is not None:
                        p_far_ind_distances, _ = forbidden_registered_kd_tree.query(p_far_ind, k=1)
                        # We exclude samples that are too close to unregistered vertices
                        allowed_samples = np.where(p_far_ind_distances > self.far_sample_dist_thresh)[0]
                    else:
                        allowed_samples = np.arange(len(p_far_ind))
                    p_far_current.extend(p_far_ind[allowed_samples[:min(len(allowed_samples), num_far_samples - len(p_far_current))]])
                    if len(p_far_current) == num_far_samples:
                        break
                p_far_current = torch.tensor(np.array(p_far_current), device=self.device,
                                             requires_grad=False, dtype=torch.float)
            else:
                p_far_current = None

            if np.any(self.close_sample_offset > 0):
                p_close_current = vertices_current_np + np.random.randn(*vertices_current_np.shape) * self.close_sample_offset
                p_close_current = torch.tensor(np.array(p_close_current), device=self.device,
                                               requires_grad=False, dtype=torch.float)
            else:
                p_close_current = None
            return vertices_current, normals_current, p_close_current, p_far_current

        # Surface points, surface normals, near surface points, off-surface points.
        vertices, normals, p_close, p_far = [], [], [], []
        # If we only train on a single data domain, we simply sample from those
        if len(self.input_numbers) == 1:
            num_surface_samples_per_input = [self.num_surface_samples] * len(self.input_numbers)
            num_far_samples_per_input = [self.num_far_samples] * len(self.input_numbers)
        # If we have several data domains, we sample evenly from each domain for which the instances exist
        else:
            non_nan_entries = [entry for entry in vertices_all if entry is not None]
            count = len(non_nan_entries)
            def get_evenly_spread_list(num_samples_current: int):
                base, rem = divmod(num_samples_current, count)
                extras = (1 if i < rem else 0 for i in range(count))
                return [base + next(extras) if x is not None else 0 for x in vertices_all]
            num_surface_samples_per_input = get_evenly_spread_list(self.num_surface_samples)
            num_far_samples_per_input = get_evenly_spread_list(self.num_far_samples)

        # Loop over the data domains.
        for num_data in range(len(self.input_numbers)):
            # Only if the current data domain is defined do we add the respective points to the lists.
            if vertices_all[num_data] is not None:
                vertices_ind, normals_ind = vertices_all[num_data], normals_all[num_data]
                if len(self.subjects[orientation][idx]["forbidden_registered_kd_tree"]) > 0:
                    forbidden_registered_kd_tree = self.subjects[orientation][idx]["forbidden_registered_kd_tree"][num_data]
                else:
                    forbidden_registered_kd_tree = None
                # Here we get the samples
                samples_out = get_samples(
                    vertices_ind, normals_ind, forbidden_registered_kd_tree,
                    num_surface_samples_per_input[num_data], num_far_samples_per_input[num_data])
                vertices.append(samples_out[0])
                normals.append(samples_out[1])
                p_close.append(samples_out[2])
                p_far.append(samples_out[3])

        # The dictionary we return.
        ret_dict = {
            "surface": torch.cat(vertices, dim=0),
            "normals": torch.cat(normals, dim=0),
            "idx": idx,
            "orientation": orientation,
        }

        # If near and far surface points are available, we also add them here.
        if any([p is not None for p in p_close]):
            ret_dict["close"] = torch.cat([p for p in p_close if p is not None], dim=0)
        if any([p is not None for p in p_far]):
            ret_dict["far"] = torch.cat([p for p in p_far if p is not None], dim=0)

        # If we include landmarks, we also add them here.
        if self.include_landmarks:
            # If we have zero landmarks, we still return a 0x3 array here.
            if len(self.avg_landmarks) == 0:
                ret_dict["landmarks"] = torch.zeros((0, 3), device=self.device, requires_grad=False, dtype=torch.float)
            else:
                ret_dict["landmarks"] = torch.cat([torch.asarray(
                    torch.zeros(self.avg_landmarks[i].shape) if lm is None else lm,
                    device=self.device, requires_grad=False, dtype=torch.float)
                    for i, lm in enumerate(self.subjects[orientation][idx]["landmarks"])], dim=0)
            if "landmark_weights" in self.subjects[orientation][idx]:
                if len(self.avg_landmarks) == 0:
                    ret_dict["landmark_weights"] = torch.ones((0, 3), device=self.device,
                                                              requires_grad=False, dtype=torch.float)
                else:
                    ret_dict["landmark_weights"] = torch.cat([torch.asarray(
                        torch.zeros(len(self.avg_landmarks[i])) if lmw is None else lmw,
                        device=self.device, requires_grad=False, dtype=torch.float)
                        for i, lmw in enumerate(self.subjects[orientation][idx]["landmark_weights"])], dim=0)

        return ret_dict
