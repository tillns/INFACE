"""
utils file containing methods used for our adaptation of DiffusionNet.

Author: Till Schnabel, Vera Schubert (contact till.schnabel@inf.ethz.ch);
        parts of the code were copied, cf. further below.


MIT License

Copyright (c) 2026 ETH Zurich, Till Schnabel; potentially also other copyright, cf. further below.

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



Three methods were developed mostly by Vera Schubert as part of here Bachelor's thesis supervised by Till Schnabel.
The remaining methods were mostly copied from https://github.com/nmwsharp/diffusion-net, which has the following
license:

MIT License

Copyright (c) 2020-2021 Nicholas Sharp and coauthors

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

import os
import warnings
from typing import List, Tuple

import torch
import hashlib
import numpy as np
import scipy

from src.objects.mesh import Landmarks

# Partial credit for the first three methods goes to Vera Schubert

#
# Rescaling
#

def normalize_positions(verts_np: np.ndarray):
    """
    Mean-centers and scales vertices to unit max-radius.
    :param verts_np: Numpy array of nx3 vertex positions.
    :return normed vertices, also nx3, cast to float32 (for torch compatibility).
    """
    center  = verts_np.mean(axis=0)
    centered = verts_np - center
    max_rad = float(np.max(np.linalg.norm(centered, axis=1)))
    if max_rad < 1e-12:
        max_rad = 1.0
    return (centered / max_rad).astype(np.float32)

# ---------------------------------------------------------------------------
# Inter-ocular rescaling
# ---------------------------------------------------------------------------

def compute_rescale_params(
        lm_norm: Landmarks, align_pairs: List[Tuple[str, str, float]], confidence_lower_threshold: float = 0.5) \
        -> Tuple[float, np.ndarray, Tuple[str, str]]:
    """
    Determines the scale factor and pivot point to normalize the scan size.
    align_pairs specifies which landmarks to compute the scaling and pivot and to which size to rescale.
    :param lm_norm: Landmarks to extract relevant entries for distance computation from.
                    The landmarks should also contain names, such that the provided names of landmarks
                    pairs can be matched to actual points.
    :param confidence_lower_threshold: Landmark points with confidence below this threshold are skipped.
    :param align_pairs: List of landmark name pairs and respective normative distance.
                        The first entry is the highest priority.
                        E.g., we used the interocular distance normalized to 0.3 via the
                        ("right_eye_outer_corner", "left_eye_outer_corner", 0.3) entry in align_pairs.

    :return scale (float), pivot (pair center 3D numpy vector)
    :raise ValueError if no landmark pair with sufficiently high confidence could be found
    """
    if lm_norm.names is None:
        warnings.warn("Provided landmarks do not contain any names, so this method won't work.")
    for num_pair, align_pair in enumerate(align_pairs):
        try:
            pair1_idx, pair2_idx = lm_norm.get_point_index(align_pair[0]), lm_norm.get_point_index(align_pair[1])
        except (ValueError, AssertionError):
            continue
        if all([lm_norm.confidence[idx] >= confidence_lower_threshold for idx in [pair1_idx, pair2_idx]]):
            points = lm_norm.points[[pair1_idx, pair2_idx]]
            dist = np.linalg.norm(points[1] - points[0])
            if dist < 1e-6:
                continue
            pivot = np.mean(points, axis=0)
            pair = (align_pair[0], align_pair[1])
            scale = align_pair[2] / dist
            return scale, pivot, pair
    raise ValueError("Couldn't find any pair above confidence threshold for rescaling.")



def apply_rescale(pts: np.ndarray, scale: float, pivot: np.ndarray) -> np.ndarray:
    """Scales points around pivot by scale factor."""
    return (pts - pivot) * scale + pivot

# From here, code was taken from https://github.com/nmwsharp/diffusion-net

# == Pytorch things

def toNP(x):
    """
    Really, definitely convert a torch tensor to a numpy array
    """
    return x.detach().to(torch.device('cpu')).numpy()

def label_smoothing_log_loss(pred, labels, smoothing=0.0):
    n_class = pred.shape[-1]
    one_hot = torch.zeros_like(pred)
    one_hot[labels] = 1.
    one_hot = one_hot * (1 - smoothing) + (1 - one_hot) * smoothing / (n_class - 1)
    loss = -(one_hot * pred).sum(dim=-1).mean()
    return loss


# Randomly rotate points.
# Torch in, torch out
# Note fornow, builds rotation matrix on CPU. 
def random_rotate_points(pts, randgen=None):
    R = random_rotation_matrix(randgen) 
    R = torch.from_numpy(R).to(device=pts.device, dtype=pts.dtype)
    return torch.matmul(pts, R) 

def random_rotate_points_y(pts):
    angles = torch.rand(1, device=pts.device, dtype=pts.dtype) * (2. * np.pi)
    rot_mats = torch.zeros(3, 3, device=pts.device, dtype=pts.dtype)
    rot_mats[0,0] = torch.cos(angles)
    rot_mats[0,2] = torch.sin(angles)
    rot_mats[2,0] = -torch.sin(angles)
    rot_mats[2,2] = torch.cos(angles)
    rot_mats[1,1] = 1.

    pts = torch.matmul(pts, rot_mats)
    return pts


# Numpy sparse matrix to pytorch
def sparse_np_to_torch(A):
    Acoo = A.tocoo()

    # Use as_tensor to avoid unnecessary copies and ensure proper dtype
    values = torch.as_tensor(Acoo.data, dtype=torch.float32)
    indices = torch.as_tensor(np.vstack((Acoo.row, Acoo.col)), dtype=torch.long)
    shape = torch.Size(Acoo.shape)

    # Modern constructor: torch.sparse_coo_tensor
    return torch.sparse_coo_tensor(indices, values, shape).coalesce()


# Pytorch sparse to numpy csc matrix
def sparse_torch_to_np(A):
    if len(A.shape) != 2:
        raise RuntimeError("should be a matrix-shaped type; dim is : " + str(A.shape))

    indices = toNP(A.indices())
    values = toNP(A.values())

    mat = scipy.sparse.coo_matrix((values, indices), shape=A.shape).tocsc()

    return mat


# Hash a list of numpy arrays
def hash_arrays(arrs):
    running_hash = hashlib.sha1()
    for arr in arrs:
        binarr = arr.view(np.uint8)
        running_hash.update(binarr)
    return running_hash.hexdigest()

def random_rotation_matrix(randgen=None):
    """
    Creates a random rotation matrix.
    randgen: if given, a np.random.RandomState instance used for random numbers (for reproducibility)
    """
    # adapted from http://www.realtimerendering.com/resources/GraphicsGems/gemsiii/rand_rotation.c
    
    if randgen is None:
        randgen = np.random.RandomState()
        
    theta, phi, z = tuple(randgen.rand(3).tolist())
    
    theta = theta * 2.0*np.pi  # Rotation about the pole (Z).
    phi = phi * 2.0*np.pi  # For direction of pole deflection.
    z = z * 2.0 # For magnitude of pole deflection.
    
    # Compute a vector V used for distributing points over the sphere
    # via the reflection I - V Transpose(V).  This formulation of V
    # will guarantee that if x[1] and x[2] are uniformly distributed,
    # the reflected points will be uniform on the sphere.  Note that V
    # has length sqrt(2) to eliminate the 2 in the Householder matrix.
    
    r = np.sqrt(z)
    Vx, Vy, Vz = V = (
        np.sin(phi) * r,
        np.cos(phi) * r,
        np.sqrt(2.0 - z)
        )
    
    st = np.sin(theta)
    ct = np.cos(theta)
    
    R = np.array(((ct, st, 0), (-st, ct, 0), (0, 0, 1)))
    # Construct the rotation matrix  ( V Transpose(V) - I ) R.

    M = (np.outer(V, V) - np.eye(3)).dot(R)
    return M

# Python string/file utilities
def ensure_dir_exists(d):
    if not os.path.exists(d):
        os.makedirs(d)
