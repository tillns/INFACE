"""
Script to register scans using a pre-trained model.
We define three distinct registration modes:
1) model-based registration: Optimize the model parameters to minimize the distance between the model-reconstructed
                             mesh and the target scan, as well as landmark-based distances. The size of the model
                             parameters serves as regularization.
2) Going out of model space: Optimize the template mesh instead of the model parameters with the same objective,
                             regularized via the Laplacian and a coupling to the model space.
3) Non-rigid iterative closest point (NICP): This one works without the model (the code still requires you to provide
                                             a model, though). The objective stays the same, but the regularization
                                             is based on a stiffness term between vertex neighbors, and the
                                             optimization works without torch.

The first two parts a majorly based on the FLAME paper, the last one on the NICP paper. We use a custom filter
mechanism of the correspondences, which is an extension of the mechanism described in the NICP paper.
Additionally, the registration can use so-called curvilinear features in addition to conventional landmarks.
These curvilinear features are combinations of several 3D points. During each iteration, the respective
points on the template look for the closest point on the linear connections between these points on the target.
Optionally, the features on the target can be made less linear through cubic spline interpolation.
The script was tested on baby scans using the trained baby models described in INFACE and INCRAN.
The custom filtering of correspondences should make the registration more robust against artifacts in these scans.

To use this script, call it and provide at least a config file. Add --help to understand all command-line arguments.
We provide some sample configurations inside the registration_configs folder.

Author: Till Schnabel (contact till.schnabel@inf.ethz.ch); parts of the code were copied, cf. further below.

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



Those parts of the code that use the Non-rigid iterative closest point (NICP) algorithm were copied from menpo3d:
https://github.com/menpo/menpo3d/blob/9a19872484f2b529b2c22b6e46d3243f9d895904/menpo3d/correspond/nicp.py
A lot of adjustments were done to the original code. Nevertheless, the code was originally developed by



Institutions
------------
Imperial College London

Individuals
-----------
James Booth             <jabooth@gmail.com> <james.booth@imperial.ac.uk>
Patrick Snape           <patricksnape@gmail.com> <p.snape@imperial.ac.uk>
Joan Alabort-i-Medina   <joan.alabort@gmail.com> <ja310@imperial.ac.uk>
Epameinondas Antonakos  <antonakosn@gmail.com> <e.antonakos@imperial.ac.uk>
Stefanos Zafeiriou      <s.zafeiriou@imperial.ac.uk>




and may be subject to the following separate license:



Copyright (c) 2014, Imperial College London and others. All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

    * Redistributions of source code must retain the above copyright
      notice, this list of conditions and the following disclaimer.
    * Redistributions in binary form must reproduce the above copyright
      notice, this list of conditions and the following disclaimer in the
      documentation and/or other materials provided with the distribution.
    * The name of Imperial College London or that of other
      contributors may not be used to endorse or promote products
      derived from this software without specific prior written
      permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTERS
''AS IS'' AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
HOLDERS OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS
OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR
TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE
USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH
DAMAGE.
"""

import re
import warnings
from argparse import ArgumentParser
from contextlib import nullcontext
from pathlib import Path
from typing import Union, Tuple, List, Dict, Callable
import torch
import yaml
from torch import optim
from tqdm import tqdm
import scipy.sparse as sp
import os
import sys
from io import UnsupportedOperation
from contextlib import contextmanager
import numpy as np

from src import utils
from src.objects.PCA_morphable_model import PCAMorphableModel
from src.objects.morphable_model import MorphableModel
from src.objects.mesh import Mesh, Landmarks, CurvilinearFeatures
from src.objects.indices_and_masks import IndicesAndMasks
from src.objects.visualizer import Visualizer
from registration_configs import registration_configs_path
from src.objects.registration_file_communicator import RegistrationFileCommunicator


# This method was fully copied from menpo3d.
@contextmanager
def stdout_redirected(to=os.devnull):
    r"""
    import os

    with stdout_redirected(to=filename):
        print("from Python")
        os.system("echo non-Python applications are also supported")
    """
    try:
        fd = sys.stdout.fileno()
    except UnsupportedOperation:
        # It's possible this is being run in an interpreter like an IPython
        # notebook where stdout doesn't behave the same as in a "normal" python
        # interpreter and in this case we cannot treat stdout like a file
        # descriptor
        warnings.warn('Unable to duplicate stdout file descriptor, likely due '
                      'to stdout having been replaced (e.g. a notebook)')
        yield
    else:
        # assert that Python and C stdio write using the same file descriptor
        # assert libc.fileno(ctypes.c_void_p.in_dll(libc, "stdout")) == fd == 1

        def redirect_stdout(to):
            sys.stdout.close()  # + implicit flush()
            os.dup2(to.fileno(), fd)  # fd writes to 'to' file
            sys.stdout = os.fdopen(fd, 'w')  # Python writes to fd

        with os.fdopen(os.dup(fd), 'w') as old_stdout:
            with open(to, 'w') as file:
                redirect_stdout(to=file)
            try:
                yield  # allow code to be run with the redirected stdout
            finally:
                # restore stdout.
                # buffering and flags such as CLOEXEC may be different
                redirect_stdout(to=old_stdout)

# This import was partly copied from menpo3d, and extended with pypardiso, which
# seems to be a good alternative solver on Windows.
try:
    try:
        # First try the newer scikit-sparse namespace
        from sksparse.cholmod import cholesky_AAt
    except ImportError:
        # Fall back to the older scikits.sparse namespace
        from scikits.sparse.cholmod import cholesky_AAt

    #print("Using scikit solver")
    # user has cholesky available - provide a fast solve
    # The default menpo implementation doesn't specify a mode. We default here to "simplicial",
    # instead of "auto", since "auto" and "supernodal" produced completely wrong results
    # on one of our machines, and "simplicial" doesn't seem to be much slower.
    # This might have been a local building issue, so feel free to change this.
    def spsolve(sparse_A, dense_b, mode="simplicial"):
        # wrap the cholesky call in a context manager that swallows the
        # low-level std-out to stop it from swamping our stdout (these low-level
        # prints come from METIS, but the solution behaves as normal)
        with stdout_redirected():
            factor = cholesky_AAt(sparse_A.T, mode=mode)
        return factor(sparse_A.T.dot(dense_b)).toarray()

except ImportError:
    # pyPardiso
    try:
        import pypardiso

        def spsolve(sparse_X, dense_b):
            return pypardiso.spsolve(sparse_X.T.dot(sparse_X), sparse_X.T.dot(dense_b).toarray())
    except ImportError:
        # Fallback to (much slower) scipy solve
        warnings.warn("suitesparse and pypardiso are not installed - NICP will run "
                      "considerably (~5-10x) slower. If possible install "
                      "suitesparse and scikit-sparse.")
        from scipy.sparse.linalg import spsolve as scipy_spsolve

        def spsolve(sparse_A, dense_b):
            return scipy_spsolve(sparse_A.T.dot(sparse_A),
                                 sparse_A.T.dot(dense_b)).toarray()


def get_rot_matrix(rot_angle: Union[float, torch.Tensor], rot_axis: str, device: Union[str, torch.device],
                   dtype=torch.float32):
    """
    Get a 4x4 torch rotation matrix from rotation axis, angle, moved to specified device.
    :param rot_angle: Rotation angle in radian.
    :param rot_axis: "x", "y", or "z" string, specifying around which of the three axes to rotate.
    :param device: torch device
    :param dtype: torch dtype
    :return: torch.Tensor 4x4
    """
    rot_matrix = torch.eye(4, device=device, dtype=dtype)
    cos_angle = torch.cos(rot_angle)
    sin_angle = torch.sin(rot_angle)
    if rot_axis == 'x':
        rot_matrix[1, 1] = cos_angle
        rot_matrix[2, 2] = cos_angle
        rot_matrix[2, 1] = sin_angle
        rot_matrix[1, 2] = -sin_angle
    elif rot_axis == 'y':
        rot_matrix[0, 0] = cos_angle
        rot_matrix[2, 2] = cos_angle
        rot_matrix[2, 0] = -sin_angle
        rot_matrix[0, 2] = sin_angle
    elif rot_axis == 'z':
        rot_matrix[0, 0] = cos_angle
        rot_matrix[1, 1] = cos_angle
        rot_matrix[1, 0] = sin_angle
        rot_matrix[0, 1] = -sin_angle
    else:
        raise ValueError(f"Unknown rot axis {rot_axis}")
    return rot_matrix


def get_transf_mat(translation, rotation, convert_to_numpy: bool = False, dtype=torch.float32):
    """
    Get a 4x4 homogeneous transformation torch matrix from translation and rotation, optionally return as numpy.
    """
    translation_matrix = torch.eye(4, device=translation.device, dtype=dtype)
    translation_matrix[[0, 1, 2], -1] = translation
    rotx = get_rot_matrix(rotation[0], "x", device=rotation.device, dtype=dtype)
    roty = get_rot_matrix(rotation[1], "y", device=rotation.device, dtype=dtype)
    rotz = get_rot_matrix(rotation[2], "z", device=rotation.device, dtype=dtype)
    rot_mat = rotz @ roty @ rotx
    full_mat = translation_matrix @ rot_mat
    if convert_to_numpy:
        full_mat = full_mat.detach().cpu().numpy()
    return full_mat

def build_laplacian(
    source_vertex_neighbors: List[Union[List[int], np.ndarray]],
    num_vertices: int,
    device: str = "cpu",
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """
    This method was generated by ChatGPT, and tested on correctness, but the inner workings were not analyzed.

    Build a sparse random-walk Laplacian L = I - D^{-1} A

    Args:
        source_vertex_neighbors: list of neighbor index lists, length = num_vertices
        num_vertices: total number of vertices
        device: device for the resulting tensor
        dtype: dtype for the values

    Returns:
        L: sparse (num_vertices, num_vertices) tensor
    """
    row_indices = []
    col_indices = []
    values = []

    for i, neighbors in enumerate(source_vertex_neighbors):
        # Diagonal entry L_ii = 1
        row_indices.append(i)
        col_indices.append(i)
        values.append(1.0)

        deg = len(neighbors)
        if deg == 0:
            # Isolated vertex: no off-diagonal terms, diagonal stays 1
            continue

        weight = -1.0 / deg  # off-diagonal L_ij

        for j in neighbors:
            row_indices.append(i)
            col_indices.append(j)
            values.append(weight)

    indices = torch.tensor(
        [row_indices, col_indices],
        dtype=torch.long,
        device=device,
    )
    values = torch.tensor(values, dtype=dtype, device=device)

    L = torch.sparse_coo_tensor(
        indices,
        values,
        size=(num_vertices, num_vertices),
        device=device,
        dtype=dtype,
    )

    # Ensure canonical form (sums duplicates if any)
    L = L.coalesce()
    return L

# This method was copied from menpo3d and adjusted to use torch and our Mesh class.
def get_node_arc_incidence_matrix(mesh: Mesh, device: str, dtype=torch.float32) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Get mesh edge connections as a sparse matrix that has one row per edge and one column per vertex.
    Each row has two entries: 1 at the end vertex and -1 at the beginning vertex.
    When you multiply this matrix with the mesh vertices, you get the actual edge vectors.
    :return: Tuple:
        1) torch coo_matrix (sparse) of shape [num_edges, num_vertices]
        2) unique edge pairs (from which the sparse matrix was built)
    """
    unique_edges = torch.asarray(mesh.get_edges_unique(), device=device)
    num_edges = len(unique_edges)

    row = torch.hstack((torch.arange(num_edges, device=device), torch.arange(num_edges, device=device)))
    col = unique_edges.T.ravel()
    data = torch.hstack((-1 * torch.ones(num_edges, device=device, dtype=dtype),
                         torch.ones(num_edges, device=device, dtype=dtype)))
    matrix_shape = (num_edges, mesh.get_num_vertices())

    # Generate a "node-arc" (i.e. vertex-edge) incidence matrix.
    node_arc_incidence_matrix = torch.sparse_coo_tensor(torch.vstack((row, col)), data, size=matrix_shape)

    return node_arc_incidence_matrix, unique_edges

def sparse_kron(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    """
    This method was generated by ChatGPT, and tested on correctness, but the inner workings were not analyzed.

    :param A: sparse COO (m,n)
    :param B: dense (p,q)
    returns: sparse COO (m*p, n*q)
    """
    assert A.is_sparse
    A = A.coalesce()
    Ai = A.indices()   # (2, nnzA)
    Av = A.values()    # (nnzA,)

    m, n = A.shape
    p, q = B.shape

    # indices for all entries of B (dense)
    Bi = torch.cartesian_prod(
        torch.arange(p, device=B.device),
        torch.arange(q, device=B.device)
    ).T  # (2, p*q)
    Bv = B.reshape(-1)  # (p*q,)

    # combine: for each nnz in A, replicate all indices in B
    nnzA = Av.numel()
    nnzB = Bv.numel()

    # Row indices: (Ai_row * p + Bi_row)
    row = (Ai[0].repeat_interleave(nnzB) * p) + Bi[0].repeat(nnzA)
    col = (Ai[1].repeat_interleave(nnzB) * q) + Bi[1].repeat(nnzA)

    vals = Av.repeat_interleave(nnzB) * Bv.repeat(nnzA)

    indices = torch.stack([row, col], dim=0).to(torch.long)
    out = torch.sparse_coo_tensor(indices, vals, size=(m * p, n * q))
    return out.coalesce()


def sparse_diag(diag_data: torch.Tensor) -> torch.Tensor:
    """
    Build a sparse diagonal coo torch tensor from diagonal data.
    """
    num_el = len(diag_data)
    idx = torch.arange(num_el, device=diag_data.device)
    indices = torch.stack([idx, idx])  # (2, n)

    return torch.sparse_coo_tensor(indices, diag_data, (num_el, num_el), device=diag_data.device)


def torch_csr_to_scipy_csr(t_csr: torch.Tensor) -> sp.csr_matrix:
    """
    This method was generated by ChatGPT, and tested on correctness, but the inner workings were not analyzed.

    Convert a torch csr tensor to scipy sparse csr matrix.
    """
    assert t_csr.layout == torch.sparse_csr

    # Move to CPU
    t_csr = t_csr.cpu()

    # Extract CSR components
    crow_indices = t_csr.crow_indices().numpy()
    col_indices = t_csr.col_indices().numpy()
    values = t_csr.values().numpy()

    shape = t_csr.shape

    return sp.csr_matrix((values, col_indices, crow_indices), shape=shape)


def reweight_loss(current_loss: torch.Tensor, weights: Union[torch.Tensor, np.ndarray, List]) -> torch.Tensor:
    """
    Reweight the loss according to the weights, such that its sum stays the same.
    """
    if not isinstance(weights, torch.Tensor):
        weights = torch.asarray(weights, dtype=current_loss.dtype, device=current_loss.device)
    weight_sum = torch.sum(weights)
    if weight_sum < 0.001:
        warnings.warn("Seems there are no correspondences left. Check why this happens.")
        return current_loss * weights
    else:
        return current_loss * weights * len(current_loss) / torch.sum(weights)


def main():
    parser = ArgumentParser()
    parser.add_argument('--config', type=str, required=True,
                        help="Path to registration configuration yaml file. "
                             "A relative path/name is assumed to lie inside the registration_configs directory.")
    parser.add_argument('--input_file_or_dir', type=str, default=None,
                        help="Directory of dataset to register, or path to a specific file to register. "
                             "You can also specify this in the config file; CLI is prioritized over config.")
    parser.add_argument('--output_dir', type=str, required=True,
                        help="Directory in which to save the registered meshes. "
                             "You can also specify this in the config file; CLI is prioritized over config.")
    parser.add_argument('--mesh_regex', default=None, type=str, nargs="+",
                        help="Filter the meshes you want to register inside the input directory. "
                             "You can specify one or several regexes, e.g. '0001_*' to only "
                             "load meshes that start with '0001_'.")
    parser.add_argument('--mesh_exclude_regex', default=None, type=str, nargs="+",
                        help="Negative filtering of meshes to register from the input directory. "
                             "All files that match the regexes you supply here will not be registered.")
    parser.add_argument('--overwrite', default=False, action="store_true",
                        help="Overwrite existing registration, so every mesh will "
                             "be registered again if you choose this option.")
    parser.add_argument('--lm_creators', default=None, type=str, nargs="+",
                        help="Specify one or more allowed landmark creators. If none specified, "
                             "the first fitting landmark file will be used.")
    parser.add_argument('--verbose', type=int, default=0,
                        help="Verbosity level. Default is 0, which is minimum verbosity. "
                             "Choose 1 for printed feedback, "
                             "and 2 to visualize progress (will slow down registration).")
    parser.add_argument('--log_tensorboard', default=False, action="store_true",
                        help="Log losses and metrics from the registration to tensorboard.")
    parser.add_argument('--register_flipped', default=False, action="store_true",
                        help="In addition to registering the normal targets, also register a flipped version, "
                             "which will be saved in a separate folder.")
    parser.add_argument('--save_aligned_target', default=False, action="store_true",
                        help="Choose this to also save the target/input scan in the same folder "
                             "as the registered mesh after aligning it with the registered mesh.")
    parser.add_argument('--use_lbfgs', default=False, action="store_true",
                        help="Optionally choose this to use LBFGS as optimizer instead of ADAM. "
                             "Not thoroughly tested. Has potential to converge faster, but our "
                             "current setting doesn't really seem to do that..")
    args = parser.parse_args()

    config_path = Path(args.config).with_suffix(".yaml")
    if not config_path.is_absolute():
        config_path = registration_configs_path.joinpath(config_path)
    with open(config_path, 'rb') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    #
    # Pre-define some parameters
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float64

    verbose = args.verbose
    output_dir = Path(args.output_dir if args.output_dir is not None else config["output_dir"])
    output_dir.mkdir(exist_ok=True)

    #
    # load model
    model = MorphableModel.load_correct_morphable_model(
        config["model_path"], use_torch=True, device=device, double_precision=dtype==torch.float64)

    #
    # load template and its landmarks
    # template mesh
    template = model.get_average_mesh()

    model_landmark_indexed = np.zeros(template.get_num_vertices(), dtype=bool)

    # Find the model vertex indices corresponding to each landmark.
    # The user may choose to define the indices directly using the predefined indices in the model object.
    # Alternatively, the user can also define a landmark on their own with two options:
    # 1) they can provide explicit coordinates for which the closest point on the model
    #    template is chosen as landmark vertex index.
    # 2) they can also just provide the model mesh indices directly.
    #
    #
    # The model_landmarks that the user sets up in the config is a dictionary that includes a sub-dictionary
    # for each landmark, which contains the landmark weight (default 1), its respective target landmark
    # index (default entry number), potentially also explicit coordinates or model mesh indices
    # (the key may be the name of the indices as saved in the respective model class), potentially
    # also the info which landmark to swap it with during flipping.
    model_landmarks_config: Dict[str, Dict[str, Union[float, int, List[int]]]] = config["model_landmarks"]
    # This encodes all the mesh indices for the individual landmarks
    model_landmark_indices_all = []
    # This encodes the landmark indices in each individual landmark config. We need this, since one config
    # can contain a name that refers to more than one mesh index, so that we don't lose this info.
    landmark_indices_per_config = []
    # template landmark weights, so with how much weight the optimization
    # tries to fit the parameters to match each landmark
    landmark_weights = []
    # How to parse respective target landmarks from files
    target_landmark_indices = []
    for num_landmark, (landmark_name, landmark_config) in enumerate(model_landmarks_config.items()):
        landmark_weight = landmark_config.get("weight", 1.0)

        # The user might have provided the information which target landmark index
        # the current model landmark corresponds to. If not, the target landmarks
        # are assumed to be in correct order.
        target_landmark_index = landmark_config.get(
            "target_landmark_index", landmark_config.get("target_index", num_landmark))

        # User provided an actual landmark via its 3D coordinates, so we retrieve the respective mesh index
        # by finding the closest vertex on the average model mesh.
        if "coordinates" in landmark_config:
            _, model_landmark_indices_current = template.resnap_landmarks(
                np.asarray([landmark_config["coordinates"]]), resnap_to_closest_triangle_point=False
            )
        else:
            model_landmark_indices_current = landmark_config.get(
                "model_landmark_index", landmark_config.get("model_index", None))
            # User didn't give more information, so we retrieve the respective mesh index from the model class.
            # (this can also include more than one index if a group of landmarks was saved under the same name).
            if model_landmark_indices_current is None:
                if model.has_index(landmark_name):
                    model_landmark_indices_current = model.get_indices(landmark_name, make_unique=False)
                else:
                    raise ValueError(f"Couldn't find the landmark for the name {landmark_name}. Please provide the "
                                     f"coordinates explicitly via a 'coordinates' key, or the model index/indices via "
                                     f"a 'model_landmark_index' key.")
            # User provided the mesh index/indices explicitly
            else:
                if not isinstance(model_landmark_indices_current, list):
                    model_landmark_indices_current = [model_landmark_indices_current]

        # Sanity check for duplicates.
        if np.any(model_landmark_indexed[model_landmark_indices_current]):
            warnings.warn(f"Model indices {np.where(model_landmark_indexed[model_landmark_indices_current])[0]} "
                          f"were chosen as landmarks multiple times. Is this intentional? "
                          f"This concerns your choice of {landmark_name} as a landmark.")

        # Keep track which landmarks were already used.
        model_landmark_indexed[model_landmark_indices_current] = True

        # Add landmark mesh indices to list.
        model_landmark_indices_all.extend(model_landmark_indices_current)

        # Keep track of the index range the current landmark occupies for later back-mapping
        indices_per_config_start_index = 0 if len(landmark_indices_per_config) == 0 else landmark_indices_per_config[-1][-1] + 1
        landmark_indices_per_config.append(list(range(
            indices_per_config_start_index, indices_per_config_start_index + len(model_landmark_indices_current))))

        # Add the indices of the corresponding target landmark and the weight of the landmark to the lists.
        if isinstance(target_landmark_index, list):
            target_landmark_indices.extend(target_landmark_index)
            if isinstance(landmark_weight, list):
                # Sanity check match between weights and indices.
                if not len(landmark_weight) == len(target_landmark_index):
                    raise AssertionError(f"You provided {len(landmark_weight)} landmark weights {landmark_weight} "
                                         f"for {len(target_landmark_index)} target landmark indices "
                                         f"{target_landmark_index}. Make sure these numbers match.")
                landmark_weights.extend(landmark_weight)
            else:
                landmark_weights.extend(len(target_landmark_index) * [landmark_weight])
        else:
            if isinstance(landmark_weight, list):
                # Sanity check match between weights and indices.
                if len(landmark_weight) > 1:
                    raise AssertionError(f"You provided {len(landmark_weight)} landmark weights {landmark_weight} "
                                         f"for a single target landmark {target_landmark_index}. "
                                         f"Please provide only one weight.")
                landmark_weight = landmark_weight[0]
            target_landmark_indices.append(target_landmark_index)
            landmark_weights.append(landmark_weight)
    model_landmark_indices_all = np.array(model_landmark_indices_all)
    # Sanity check match between model and target landmarks and the landmark weights.
    assert len(model_landmark_indices_all) == len(landmark_weights) == len(target_landmark_indices)
    # Individual landmark weights are later multiplied with the landmark distance, yielding the loss.
    landmark_weights_torch = torch.tensor(landmark_weights, device=device, dtype=dtype, requires_grad=False)

    #
    #
    # In addition to landmarks, we also check if the user wants to use curvilinear features for registration.
    # Unlike, the landmarks, curvilinear features are optional for the registration.
    #
    #
    # This config is similar to the landmarks config. One difference is that for each individual setting,
    # if the user provides explicit coordinates, then this needs to be several 3d points.
    # Another difference is that while for the landmarks, it is optional to provide the
    # respective target landmark index, it is required for the curvilinear features,
    # since one feature consists of several points.
    model_curvilinear_features_config: Dict[str, Dict[str, Union[float, int, List[int]]]] = config.get(
        "model_curvilinear_features", None)

    if model_curvilinear_features_config is None:
        # Indicates to the file loader that no curvilinear features should be extracted from the landmark file.
        target_curvilinear_feature_indices = None
        # Indicator whether to use curvilinear features during optimization
        use_curvilinear_features = False
    else:
        use_curvilinear_features = True
        # Similar processing as for the landmarks
        model_curvilinear_feature_indices_all = []
        curvilinear_feature_weights = []
        target_curvilinear_feature_indices = []
        # Loop over individual line set configs.
        for feature_name, feature_config in model_curvilinear_features_config.items():

            # User provided the actual points of the line set, in which case we check for validity before
            # resnapping the points to the average mesh.
            if "coordinates" in feature_config:
                model_line_set = np.asarray(feature_config["coordinates"])
                if not model_line_set.ndim == 2 and model_line_set.shape[1] == 3 and np.issubdtype(model_line_set.dtype, np.float):
                    raise AssertionError(
                        f"The coordinates you provided for {feature_name} are not a list of 3D points. "
                        f"Given {model_line_set}.")
                _, model_curvilinear_feature_indices_current = template.resnap_landmarks(
                    model_line_set, resnap_to_closest_triangle_point=False)
            else:
                model_curvilinear_feature_indices_current = feature_config.get(
                    "model_feature_indices", feature_config.get("model_indices", None))
                # User didn't give more information, so we retrieve the respective mesh indices from the model class.
                if model_curvilinear_feature_indices_current is None:
                    if model.has_index(feature_name):
                        model_curvilinear_feature_indices_current = model.get_indices(feature_name, make_unique=False)
                    else:
                        raise ValueError(
                            f"Couldn't find the curvilinear feature for the name {feature_name}. Please provide the "
                            f"coordinates explicitly via a 'coordinates' key, or the model indices via "
                            f"a 'model_feature_indices' key.")
                # User provided the mesh indices explicitly
                else:
                    if not isinstance(model_curvilinear_feature_indices_current, list):
                        raise AssertionError(
                            f"You provided the curvilinear feature indices explicitly for {feature_name}, "
                            f"but it should be a list of indices.")
            model_curvilinear_feature_indices_all.append(model_curvilinear_feature_indices_current)

            # The user can again choose a single weight for every point within the line set,
            # or they can choose individual weights.
            feature_weight = feature_config.get("weight", 1.0)
            if isinstance(feature_weight, list):
                feature_weight_expanded = np.asarray(feature_weight)
            else:
                feature_weight_expanded = np.ones(len(model_curvilinear_feature_indices_current)) * feature_weight
            # We set the weights of points within a curvilinear feature to 0 if these points are also landmarks,
            # such that we don't put too much weight on them. E.g., if a corner landmark also marks the beginning
            # of several line sets, it would get an additional pull for each of these line sets.
            feature_weight_expanded[model_landmark_indexed[model_curvilinear_feature_indices_current]] = 0
            curvilinear_feature_weights.append(feature_weight_expanded)

            # The target indices are always a list
            target_curvilinear_feature_indices.append(feature_config.get("target_landmark_indices") or feature_config["target_indices"])
        assert len(model_curvilinear_feature_indices_all) == len(curvilinear_feature_weights) == len(target_curvilinear_feature_indices)

    # Define rules for flipping. We need to rearrange the target landmarks (and curvilinear features),
    # such that the model landmarks are still compatible (we only flip the target mesh, not the model
    # mesh, so the correspondences across left and right features need to be swapped).
    register_flipped: bool = getattr(args, "register_flipped", config.get("register_flipped", False))
    flip_landmarks_order, flip_curvilinear_features_order, reverse_curvilinear_features = None, None, None
    if register_flipped:
        def get_flipped_index(config_index, current_config) -> int:
            index_names = list(current_config.keys())
            model_index_name = index_names[config_index]
            if "flip_with" in current_config[model_index_name]:
                return index_names.index(current_config[model_index_name]["flip_with"])
            if "left" not in model_index_name and "right" not in model_index_name:
                return config_index
            else:
                # Swap all "left" and "right" entries (do it simultaneously, so we don't overwrite already entries
                # that were already swapped)
                mapping = {"left": "right", "right": "left"}
                pattern = re.compile("|".join(mapping.keys()))
                flipped_name = pattern.sub(lambda m: mapping[m.group(0)], model_index_name)
                return index_names.index(flipped_name)

        # We go over the individual landmarks and check if the user provided information about
        # the landmark's mirrored version. If not, we use the information given by the landmark names,
        # i.e., we look for names with "left" and "right" in them, and swap those.
        flip_landmarks_order = np.concatenate([
            landmark_indices_per_config[get_flipped_index(i, model_landmarks_config)]
            for i in range(len(model_landmarks_config))])
        if not IndicesAndMasks.is_unique(flip_landmarks_order):
            raise AssertionError(f"The computed landmark flipping order {flip_landmarks_order} is not unique, "
                                 f"meaning that you create duplicates through the flipping. "
                                 f"Make sure you provide the 'flip_with' argument symmetrically.")

        # Same procedure for curvilinear features
        if model_curvilinear_features_config is not None:
            flip_curvilinear_features_order = [get_flipped_index(i, model_curvilinear_features_config)
                                               for i in range(len(model_curvilinear_features_config))]
            reverse_curvilinear_features = [feature_config.get("reverse_on_flip", False)
                                            for feature_config in model_curvilinear_features_config.values()]
            if not IndicesAndMasks.is_unique(flip_curvilinear_features_order):
                raise AssertionError(
                    f"The computed curvilinear feature flipping order {flip_curvilinear_features_order} is not unique, "
                    f"meaning that you create duplicates through the flipping. "
                    f"Make sure you provide the 'flip_with' argument symmetrically.")


    normalization_matrix = utils.get_transformation_matrix(
        translate=-template.get_center(), scale=1.0 / np.sqrt(np.sum(template.get_range() ** 2)))
    normalization_matrix_torch = torch.tensor(normalization_matrix, device=device, dtype=dtype, requires_grad=False)

    template.transform(normalization_matrix)

    def get_regional_weights(
            weight_name, default: Union[float, None, str] = None,
            current_config: Union[Dict, None] = None) -> Union[None, torch.Tensor, np.ndarray]:
        """
        This method gives the user the option to choose for some parameters/weights regional values.
        In the config file, the respective parameter/weight must be structured as follows:
        weight_name:
            index1_name_or_path: float_value1
            index2_name_or_path: float_value2
            default: float_value_default (optional)
        If a name for an index selection is provided, that index selection must already exist in the model's indices.
        Otherwise, the user can also provide a path to another index selection.
        :param weight_name: str representing the weight name. The usage of them is hard-coded in this
                            registration file, cf. e.g., the curvature thresholds.
        :param default: float value representing the value all unspecified regions will take. Note that if you
                        provide a default value in the config, it will overwrite this default argument even
                        if it is not None, since the default argument is hard-coded in this file, while a user
                        can change the default value via the config.
        :param current_config: Optionally provide a different config than the globally used one, e.g.,
                               the current optimization setting.
        :return None if weight name is not contained in config;
                otw. torch tensor with the regional values of length the number of model vertices.
        """
        if current_config is None:
            current_config = config
        if weight_name not in current_config:
            return None
        weights = torch.ones(model.get_number_of_vertices(), device=device, dtype=dtype, requires_grad=False)
        if isinstance(default, str):
            if "req" not in default.lower():
                raise ValueError(f"Provided {default=} is a string that doesn't say 'required'")
            if not "default" in current_config[weight_name]:
                raise ValueError(f"'default' entry required for {weight_name}")
        default_value = current_config[weight_name].get("default", 0 if default is None else default)
        weights *= default_value
        for key, value in current_config[weight_name].items():
            if key == "default":
                continue

            if model.has_index(key):
                indices = model.get_indices(key)
            else:
                indices = IndicesAndMasks.load(key)
            weights[indices] = value

        return weights

    # Load curvature thresholds if available.
    # The user can choose to provide upper and/or lower limits for the principal curvature
    # on the target when matched with specific regions.
    curvature_thresholds = {
        "min_lower": get_regional_weights("min_curvature_lower_bound", default=-float("inf")),
        "min_upper": get_regional_weights("min_curvature_upper_bound", default=float("inf")),
        "max_lower": get_regional_weights("max_curvature_lower_bound", default=-float("inf")),
        "max_upper": get_regional_weights("max_curvature_upper_bound", default=float("inf")),
    }

    # Ignored vertices are weighted 0 during registration.
    # Importantly, they are removed entirely from the optimization of NICP, unlike the excluded vertices,
    # which are just always weighted 0.
    vertex_regions_to_ignore = config.get("vertex_regions_to_ignore", [])
    if len(vertex_regions_to_ignore) > 0:
        ignore_source_vertices = model.get_indices(*vertex_regions_to_ignore)
    else:
        ignore_source_vertices = []

    vertex_regions_to_exclude = config.get("vertex_regions_to_exclude", [])
    if len(vertex_regions_to_exclude) > 0:
        exclude_source_vertices = model.get_indices(*vertex_regions_to_exclude)
    else:
        exclude_source_vertices = []

    model_triangles = model.get_triangles()

    source_weights = get_regional_weights("source_weights", default=1)
    # By default, we assign a higher weight to vertices that are part of larger triangles.
    # This kinda makes sure that we pull evenly at the template everywhere instead of focusing more
    # on regions with higher vertex density while neglecting those with fewer
    # (e.g., for the head model, the scalp region contains fewer vertices than the face region, and
    #  especially since there are more landmarks in the face, this creates an uneven pull)
    if source_weights is None:
        source_weights = torch.tensor(
            template.get_triangle_area_sum_per_vertex(), device=device, dtype=dtype, requires_grad=False)
    # The vertices part of ignore_source_vertices and exclude_source_vertices are weighted 0,
    # so they are not registered to the target. During model-based optimization, all the vertices
    # still move along. However, during NICP registration, only the ignore vertices move along;
    # the exclude vertices do not change.
    source_weights[ignore_source_vertices] = 0
    source_weights[exclude_source_vertices] = 0

    # With this, we allow the user to choose the triangle-area-weighted default source weights,
    # yet still adjust the weights in certain regions by multiplying the default ones with
    # a multiplier.
    source_weights_multiplier = get_regional_weights("source_weights_multiplier", default=1)
    if source_weights_multiplier is not None:
        source_weights *= source_weights_multiplier

    # normalization is not necessary, but it makes sure the cond_weights variable in the dist_loss
    # function always stays between 0 and 1
    source_weights /= torch.max(source_weights)

    # If we want to go out of the model space during the optimization,
    # we need the vertex neighbors for the laplacian smoothing loss
    # and the edges for the coupling loss between template and model
    source_vertex_neighbors = template.get_vertex_neighbors()
    # We build a sparse torch matrix out of the vertex neighbors for efficient computation during runtime
    laplacian_matrix = build_laplacian(source_vertex_neighbors, num_vertices=template.get_num_vertices(),
                                       device=device, dtype=dtype)
    source_edges_unique = template.get_edges_unique()

    # Load all meshes in the provided directory, or just the one file if the path is a file.
    # Note that even if it's only one file, target_mesh_paths is still a list.
    input_file_or_dir = args.input_file_or_dir if args.input_file_or_dir is not None else config["input_file_or_dir"]
    target_mesh_paths = Mesh.get_all_mesh_files_in_path(
        input_file_or_dir, regex_names=args.mesh_regex, exclude_regex_names=args.mesh_exclude_regex,
        remove_suffix_duplicates=True)

    for target_mesh_path in tqdm(target_mesh_paths, desc="Registering meshes"):
        for flip in [False, True] if register_flipped else [False]:
            # Define all the file outputs. If they all exist, we skip the registration of this mesh,
            # unless the user chose to --overwrite
            target_mesh_stem = target_mesh_path.stem
            if flip:
                target_mesh_stem += "_flipped"
            output_folder = output_dir.joinpath(target_mesh_stem)
            output_mesh_path = output_folder.joinpath(target_mesh_stem).with_suffix(".ply")
            output_transform_path = RegistrationFileCommunicator.get_transform(output_mesh_path, return_path=True)
            vertex_exclude_path, texture_exclude_path = RegistrationFileCommunicator.get_excluded_regions_on_registered_mesh(
                output_mesh_path, return_paths=True)
            if not args.overwrite and (output_mesh_path.exists() and output_transform_path.exists() and vertex_exclude_path.exists()):
                continue

            # Import mesh and its landmarks. We skip the mesh if no landmarks exist.
            target_mesh = RegistrationFileCommunicator.get_scan_with_landmarks(
                scan_path=target_mesh_path, lm_creators=args.lm_creators, landmark_indices=target_landmark_indices,
                curvilinear_feature_indices=target_curvilinear_feature_indices
            )
            if target_mesh is None or not target_mesh.has_landmarks():
                if verbose:
                    warnings.warn(f"No corresponding landmarks found for {target_mesh_path}")
                continue

            if flip:
                target_mesh.flip(
                    in_place=True, flip_landmarks_order=flip_landmarks_order,
                    flip_curvilinear_features_order=flip_curvilinear_features_order,
                    reverse_curvilinear_features=reverse_curvilinear_features)

            target_mesh_normed = target_mesh.transform(normalization_matrix, in_place=False)

            # Create output directory and optionally log the registration via tensorboard
            output_folder.mkdir(exist_ok=True)
            if args.log_tensorboard:
                try:
                    from torch.utils.tensorboard import SummaryWriter
                except ImportError:
                    warnings.warn("Tensorboard is not installed, so logging is disabled")
                    args.log_tensorboard = False
                else:
                    writer = SummaryWriter(log_dir=str(output_folder))

            # We remove undefined landmarks from the target. non_nan_rows contains the indices of the landmarks that
            # are defined, which we use to also adjust the indices of the model's landmarks and landmark weights
            # to match with the target landmarks.
            target_landmarks, non_nan_rows = Landmarks.strip_landmark_rows(
                target_mesh_normed.get_landmarks(), return_non_nan_row_indices=True)
            target_landmark_points = target_landmarks.get_points()
            model_landmark_indices = model_landmark_indices_all[non_nan_rows]
            # The landmark loss is adjusted proportionally to the removed landmarks.
            landmark_weights_torch_current = reweight_loss(
                landmark_weights_torch, IndicesAndMasks.get_mask(non_nan_rows, len(landmark_weights_torch)))[non_nan_rows]
            # If the target landmarks contain confidence values, we reweight the landmark loss by this confidence.
            # Note that 0 confidence landmarks were already excluded.
            if target_landmarks.confidence is not None:
                landmark_weights_torch_current = reweight_loss(
                    landmark_weights_torch_current,
                    torch.asarray(target_landmarks.confidence, dtype=dtype, device=device))
            # Move landmarks to torch
            target_landmarks_torch = torch.tensor(
                target_landmark_points, device=device, dtype=dtype, requires_grad=False)

            # Similar processing for curvilinear features
            if target_mesh_normed.has_curvilinear_features() and use_curvilinear_features:
                target_line_sets, model_line_set_indices, line_set_weights_current = CurvilinearFeatures.strip_landmark_line_sets(
                    target_mesh_normed.get_curvilinear_features().get_line_sets(), model_curvilinear_feature_indices_all, curvilinear_feature_weights)
                if len(target_line_sets) == 0:
                    target_mesh_normed.remove_curvilinear_features()
                else:
                    line_set_weights_current_flattened, _ = CurvilinearFeatures.get_flattened_line_sets(line_set_weights_current)
                    model_line_set_indices_flattened, _ = CurvilinearFeatures.get_flattened_line_sets(model_line_set_indices)
                    curvilinear_feature_weights_current_torch = torch.tensor(
                        line_set_weights_current_flattened, device=device, dtype=dtype, requires_grad=False
                    )
                    target_mesh_normed.set_curvilinear_features(target_line_sets)
                    # Optionally increase the resolution of the target curvilinear features via cubic splines,
                    # which can yield a smoother interpolation if the line sets, potentially resulting in a
                    # slightly better registration.
                    curvilinear_feature_refine_factor = config.get("curvilinear_feature_refine_factor", 1)
                    if curvilinear_feature_refine_factor > 1:
                        target_mesh_normed.get_curvilinear_features().refine_with_cubic_splines(
                            curvilinear_feature_refine_factor)

            # Define a 0/1 mask that encodes which target vertices should be ignored.
            # This stands in contrast to the source ignore indices, as these define which
            # model vertices should not be registered. However, those model vertices that are
            # indeed registered find a correspondence on the target during each iteration,
            # and if this correspondence is among the target vertices to be ignored, the vertex
            # is assigned a zero weight as well, but only for that iteration.
            target_vertex_ignore_mask = torch.ones(
                target_mesh_normed.get_num_vertices(), device=device, dtype=dtype, requires_grad=False)
            # Target vertices to be ignored can be defined by the user
            target_ignore_indices = RegistrationFileCommunicator.get_scan_ignore_indices(
                target_mesh_path, scan=target_mesh, flip=flip)
            if target_ignore_indices is not None:
                target_vertex_ignore_mask[target_ignore_indices] = 0

            # We also exclude vertices that are part of non-manifold edges; we also exclude boundary vertices,
            # since boundaries are most often unreliable correspondences.
            non_manifold_edges = target_mesh_normed.get_non_manifold_edges(allow_boundary_edges=False)
            # There might be some special meshes that have a lot of non-manifold edges; we do a sanity check here.
            if len(non_manifold_edges) / target_mesh_normed.get_num_edges() > 0.9:
                warnings.warn("Your target has over 90 percent non manifold edges. "
                              "This will probably lead to errors, so they won't be added to ignored target vertices.")
            elif len(non_manifold_edges) > 0:
                target_vertex_ignore_mask[np.concatenate(non_manifold_edges)] = 0

            # We compute the principal curvatures on the mesh, if the user has chosen to use them during
            # correspondence matching, but only if the mesh is not too big,
            # since this can cause a lot of computation overhead otherwise.
            target_vertex_min_curvatures, target_vertex_max_curvatures = None, None
            if target_mesh_normed.get_num_vertices() < 150000 and any([ct is not None for ct in curvature_thresholds.values()]):
                try:
                    _, _, target_vertex_min_curvatures, target_vertex_max_curvatures = target_mesh_normed.compute_principal_curvatures()
                except (ImportError, ModuleNotFoundError):
                    warnings.warn("You chose curvature thresholds in your config, but igl is not installed, "
                                  "so we cannot include them as constraint. Try to install it via "
                                  "`python -m pip install libigl`.")
                else:
                    target_vertex_min_curvatures = torch.tensor(target_vertex_min_curvatures, device=device,
                                                                dtype=dtype, requires_grad=False)
                    target_vertex_max_curvatures = torch.tensor(target_vertex_max_curvatures, device=device,
                                                                dtype=dtype, requires_grad=False)
                    target_vertex_ignore_mask[torch.logical_or(torch.isnan(target_vertex_min_curvatures),
                                                        torch.isnan(target_vertex_max_curvatures))] = 0

            # Target triangles and normals required for correspondence matching and barycentric vertex interpolation
            target_triangles = target_mesh_normed.get_triangles(copy=True)
            target_triangle_normals_torch = torch.tensor(
                target_mesh_normed.get_triangle_normals(), device=device, dtype=dtype)

            # This is a specific optimization of the registration that assumes that we normally only want to
            # register the most outer layer of a scan. Any vertices that are inside the mesh might not
            # be there on purpose. We detect inner layers by raycasting from the mesh center
            # through each vertex and check for each ray if the mesh is hit after the vertex was hit.
            # Those vertices, for which this is True, are assumed to be internal vertices.
            # The user can choose during each iteration whether to include such vertices as possible
            # correspondences or not via "remove_internal_structures".
            internal_structures = target_mesh_normed.get_internal_vertices()
            external_structures = IndicesAndMasks.invert_mask(internal_structures)
            # We create a separate mesh instance that has the internal vertices removed.
            target_mesh_normed_internal_removed = target_mesh_normed.remove_vertices(internal_structures)
            # We create also separate instances of indices and other target-related variables for the
            # target mesh with the removed internal structures.
            target_vertex_ignore_mask_internal_removed = target_vertex_ignore_mask[external_structures]
            # non-manifold edge vertices are again ignored. We recompute them, because by removing
            # certain vertices, we might have created new boundaries, and we generally don't want to
            # allow boundaries as valid correspondences, closest point finding can easily snap to a boundary
            # even if the source vertex is far away.
            non_manifold_edges_internal_removed = target_mesh_normed_internal_removed.get_non_manifold_edges(
                allow_boundary_edges=False)
            if len(non_manifold_edges_internal_removed) / target_mesh_normed.get_num_edges() < 0.9 and len(non_manifold_edges_internal_removed) > 0:
                target_vertex_ignore_mask_internal_removed[np.concatenate(non_manifold_edges_internal_removed)] = 0
            # Curvatures don't need to be recomputed, just indexed correctly.
            if target_vertex_min_curvatures is None:
                target_vertex_min_curvatures_internal_removed = None
            else:
                target_vertex_min_curvatures_internal_removed = target_vertex_min_curvatures[external_structures]
            if target_vertex_max_curvatures is None:
                target_vertex_max_curvatures_internal_removed = None
            else:
                target_vertex_max_curvatures_internal_removed = target_vertex_max_curvatures[external_structures]
            # Triangles are re-fetched from the new mesh.
            target_triangles_internal_removed = target_mesh_normed_internal_removed.get_triangles(copy=True)
            target_triangle_normals_torch_internal_removed = torch.tensor(
                target_mesh_normed_internal_removed.get_triangle_normals(), device=device, dtype=dtype)

            def get_model_weight_loss() -> torch.Tensor:
                """
                Return average of the squared model parameters.
                """
                return torch.mean(torch.square(model_params_torch))

            def get_current_transf_matrix(convert_to_numpy: bool = False) -> Union[np.ndarray, torch.Tensor]:
                """
                Get the transformation matrix from the current state of the optimized translation and rotation
                torch tensors. This transformation is multiplied with the initial transformation matrix, and
                may optionally be converted to a numpy array.
                :param convert_to_numpy: Set True to convert the returned transformation matrix to a numpy array.
                :return: 4x4 transformation matrix as torch tensor or numpy array.
                """
                additional_transform_matrix = get_transf_mat(translation, rotation, dtype=dtype)
                full_transform_matrix = additional_transform_matrix @ transf_mat_init_torch @ normalization_matrix_torch
                if convert_to_numpy:
                    full_transform_matrix = full_transform_matrix.detach().cpu().numpy()
                return full_transform_matrix

            def transform_vertices(current_vertices: torch.Tensor, invert_transform: bool = False) -> torch.Tensor:
                """
                Transform current vertices with the optimized transformation matrix. Optionally invert the transformation,
                which is useful when the current vertices are in the already transformed frame,
                """
                vertices_hom = torch.cat([
                    current_vertices,
                    torch.ones((current_vertices.shape[0], 1), device=current_vertices.device,
                               dtype=current_vertices.dtype)], dim=1)
                transf_matrix = get_current_transf_matrix()
                if invert_transform:
                    transf_matrix = torch.linalg.inv(transf_matrix)
                return torch.einsum("ab,cb->ca", transf_matrix, vertices_hom)[:, :3]

            def get_model_vertices(convert_to_numpy=False, apply_transform: bool = True,
                                   disable_gradients: bool = False) -> Union[np.ndarray, torch.Tensor]:
                """
                Get the mesh vertices, decoded from the current model parameters.
                :param convert_to_numpy: Set True to convert the vertices from a tracked pytorch tensor to a numpy array.
                :param apply_transform: Set False to get the vertices in the model's own frame instead of the frame
                                        matched to the target through the optimized transformation.
                :param disable_gradients: Set True if you don't want to optimize the model or transformation here.
                :return: numpy array or torch tensor nx3 of the vertices.
                """
                # torch.no_grad() is used if no gradients need to be tracked.
                # Otherwise, nullcontext() is used, which doesn't do anything.
                with (torch.no_grad() if disable_gradients or convert_to_numpy else nullcontext()):
                    # reconstruct vertices from the model space.
                    model_vertices_unnormed = model.decode(model.unnormalize_weights(model_params_torch))
                    # The transformation is optimized jointly with the model space, so the vertices need to be transformed
                    # to match with the arbitrary orientation of the input scan
                    if apply_transform:
                        model_vertices_transformed = transform_vertices(model_vertices_unnormed, invert_transform=False)
                    else:
                        model_vertices_transformed = model_vertices_unnormed
                if convert_to_numpy:
                    model_vertices_transformed = model_vertices_transformed.detach().cpu().numpy()
                return model_vertices_transformed

            def get_current_model_vertex_normals(model_vertices: Union[torch.Tensor, None] = None,
                                                 current_triangles: Union[torch.Tensor, None] = None) -> torch.Tensor:
                """
                Compute the vertex normals based on the current vertices.
                They are computed as area-weighted averages over the surrounding triangle normals.
                Note that if you don't supply the current vertices, they are retrieved from :func:`get_model_vertices()`
                with default arguments.
                :param model_vertices: Provide the model vertices to avoid recomputation.
                :param current_triangles: In case you don't work on all vertices, provide the respective triangles.
                :return torch tensor nx3 of the vertex normals.
                """
                if model_vertices is None:
                    model_vertices = get_model_vertices(disable_gradients=True)
                if current_triangles is None:
                    current_triangles = model_triangles
                # verts: (V, 3), faces: (F, 3) long
                v0 = model_vertices[current_triangles[:, 0], :]
                v1 = model_vertices[current_triangles[:, 1], :]
                v2 = model_vertices[current_triangles[:, 2], :]
                # (F, 3), unnormalized face normals (area-weighted)
                current_face_normals = torch.cross(v1 - v0, v2 - v0, dim=-1)
                # scatter-add to vertices (area-weighted)
                vn = torch.zeros_like(model_vertices)
                vn.index_add_(0, current_triangles[:, 0], current_face_normals)
                vn.index_add_(0, current_triangles[:, 1], current_face_normals)
                vn.index_add_(0, current_triangles[:, 2], current_face_normals)
                return torch.nn.functional.normalize(vn, dim=-1, eps=1e-8)  # (V, 3)

            def get_current_mesh(current_vertices: Union[torch.Tensor, None] = None,
                                 apply_transform: bool = True, mark_landmark_indices: bool = False) -> Mesh:
                """
                Get the current mesh based on the current vertices, decoded from the current model parameters.
                :param current_vertices: Optionally provide the vertices; if not provided, the vertices are decoded
                                         from the current model latent.
                :param apply_transform: Set False to get the mesh in the model's own frame instead of the frame
                                        of the target mesh, cf. also :func:`get_model_vertices()`.
                :param mark_landmark_indices: Set True to mark the vertex indices on the mesh that correspond to
                                              the specified landmarks. This is meant for visualization purposes.
                :return Mesh with current vertices.
                """
                if current_vertices is None:
                    current_vertices = get_model_vertices(convert_to_numpy=True, apply_transform=apply_transform)
                else:
                    current_vertices = current_vertices.detach().cpu().numpy()
                current_mesh = model.get_mesh_from_vertices(current_vertices)
                if mark_landmark_indices:
                    current_mesh.mark_indices(model_landmark_indices, in_place=True)
                return current_mesh

            def get_dist_loss(
                    current_vertices: torch.Tensor, remove_internal_structures: bool = False,
                    normal_thresh: float = None, normal_connect_thresh: float = None,
                    source_weights_adjustment: Union[torch.Tensor, None] = None,
                    return_closest_points: bool = False,
                    vertex_indices_or_mask: Union[torch.Tensor, np.ndarray, None] = None,
                    current_triangles: Union[torch.Tensor, None] = None) -> Tuple[torch.Tensor, torch.Tensor]:
                """
                We find the correspondences for each vertex on the target using closest mesh point search
                (so not only the vertices, but the closest point on any triangle).
                We then perform several checks to exclude invalid correspondences.
                We finally return the average distance between the current_vertices and the valid correspondences.
                We also return the result of the checks, so which vertices were accepted as correspondences and which not.
                The checks ar mostly 0 or 1, but the source_weights can also be continuous, so the result is [0,1]
                """
                # Check if we work on all or a subset of the vertices
                if len(current_vertices) != template.get_num_vertices():
                    assert vertex_indices_or_mask is not None and current_triangles is not None
                else:
                    vertex_indices_or_mask = torch.ones(len(current_vertices), dtype=torch.bool)

                # Choose the correct target mesh
                if remove_internal_structures:
                    current_target_mesh = target_mesh_normed_internal_removed
                    current_target_vertex_ignore_mask = target_vertex_ignore_mask_internal_removed
                    current_target_triangle_normals_torch = target_triangle_normals_torch_internal_removed
                    current_target_triangles = target_triangles_internal_removed
                    current_target_vertex_min_curvatures = target_vertex_min_curvatures_internal_removed
                    current_target_vertex_max_curvatures = target_vertex_max_curvatures_internal_removed
                else:
                    current_target_mesh = target_mesh_normed
                    current_target_vertex_ignore_mask = target_vertex_ignore_mask
                    current_target_triangle_normals_torch = target_triangle_normals_torch
                    current_target_triangles = target_triangles
                    current_target_vertex_min_curvatures = target_vertex_min_curvatures
                    current_target_vertex_max_curvatures = target_vertex_max_curvatures

                # Find the correspondences
                with torch.no_grad():
                    # Find closest triangle points and the respective triangle indices
                    closest_points_np, closest_triangle_indices = current_target_mesh.get_closest_triangle_points(
                        current_vertices.detach().cpu().numpy())
                    # The target normals then correspond to the respective triangle normals
                    closest_normals = current_target_triangle_normals_torch[closest_triangle_indices]
                    # Compute barycentric coordinates of closest triangle points to get meaningful
                    # interpolation of vertex-based weights.
                    closest_points_barycentric_np = current_target_mesh.convert_points_to_barycentric_coordinates(
                        closest_points_np, triangle_indices=closest_triangle_indices)
                    # Move to torch
                    closest_points = torch.tensor(closest_points_np, dtype=current_vertices.dtype, device=device)
                    closest_points_barycentric = torch.tensor(
                        closest_points_barycentric_np, dtype=current_vertices.dtype, device=device)
                closest_dists = torch.linalg.norm(closest_points - current_vertices, dim=1)

                def get_barycentric_interpolated_weight(vertex_weight: Union[np.ndarray, torch.Tensor]) -> Union[np.ndarray, torch.Tensor]:
                    """
                    Interpolate vertex-based weights based on the barycentric interpolation of the closest points.
                    E.g., target ignore regions are defined over the vertices, which is why we need to convert
                    them to triangles here.
                    """
                    triangle_vertex_weight = vertex_weight[current_target_triangles[closest_triangle_indices]]
                    if isinstance(vertex_weight, torch.Tensor):
                        return torch.sum(triangle_vertex_weight * closest_points_barycentric, dim=1)
                    else:
                        return np.sum(triangle_vertex_weight * closest_points_barycentric_np, axis=1)

                # We check our conditions and reweight according to them without backpropagating through these
                # conditions. Otherwise, the model might be tempted to try to misalign the source with the target in
                # some regions, so as to make these conditions fail and thus lower the weight in these regions.
                with torch.no_grad():
                    def show_condition_weight(cond_weight_current, thresh: float = 0.5, indices=None):
                        """
                        Method for debugging. Call it with any of the intermediate cond weights to check
                        which source vertices have valid correspondences according to the condition.
                        """
                        if indices is not None:
                            cond_weight_to_visualize = torch.zeros(len(current_vertices), device=device, dtype=dtype)
                            cond_weight_to_visualize[indices] += cond_weight_current
                        else:
                            cond_weight_to_visualize = cond_weight_current
                        if len(current_vertices) == template.get_num_vertices():
                            mesh = get_current_mesh(current_vertices, apply_transform=False)
                        else:
                            mesh = Mesh(vertices=current_vertices.cpu().numpy(), triangles=current_triangles.cpu().numpy())
                        Mesh.show_multiple_meshes(
                            current_target_mesh,
                            mesh.mark_indices(cond_weight_to_visualize.detach().cpu().numpy() > thresh))

                    cond_weight = torch.ones_like(closest_dists)

                    # Ignored target vertices are zeroed out here
                    cond_weight *= get_barycentric_interpolated_weight(current_target_vertex_ignore_mask)

                    # Source vertices can have different weights (this is the global weighting)
                    cond_weight *= source_weights[vertex_indices_or_mask]

                    # potential adjustments of the source weights (can be changed each iteration)
                    if source_weights_adjustment is not None:
                        cond_weight *= source_weights_adjustment[vertex_indices_or_mask]

                    # that's the normal condition, i.e., do the normals of the source and corresponding
                    # target points match well
                    if normal_thresh is not None:
                        model_vertex_normals = get_current_model_vertex_normals(current_vertices, current_triangles=current_triangles)
                        normal_dot_prod = torch.einsum("ab,ab->a", model_vertex_normals, closest_normals)
                        normal_cond = normal_dot_prod > normal_thresh
                        cond_weight *= normal_cond

                    # that's the normal connection condition, i.e., do the target normals match with the
                    # respective connection vector between source and target point
                    if normal_connect_thresh is not None:
                        connect_vector = torch.nn.functional.normalize(closest_points - current_vertices, dim=1, eps=1e-8)
                        connect_normal_dot_prod = torch.einsum("ab,ab->a", connect_vector, closest_normals)
                        normal_connect_cond = torch.abs(connect_normal_dot_prod) > normal_connect_thresh
                        cond_weight *= normal_connect_cond

                    # Curvature min and max bound condition
                    if target_vertex_max_curvatures is not None:
                        target_min_curvatures = get_barycentric_interpolated_weight(current_target_vertex_min_curvatures)
                        target_max_curvatures = get_barycentric_interpolated_weight(current_target_vertex_max_curvatures)
                        # Even though target_min_curvatures and target_max_curvatures may still contain
                        # nan values, these operations are still safe, torch evaluates nan comparisons
                        # to False, so we basically just set the cond_weight to 0 for correspondences
                        # with undefined curvature.
                        curvature_cond  = target_min_curvatures > curvature_thresholds["min_lower"][vertex_indices_or_mask]
                        curvature_cond *= target_min_curvatures < curvature_thresholds["min_upper"][vertex_indices_or_mask]
                        curvature_cond *= target_max_curvatures > curvature_thresholds["max_lower"][vertex_indices_or_mask]
                        curvature_cond *= target_max_curvatures < curvature_thresholds["max_upper"][vertex_indices_or_mask]
                        cond_weight *= curvature_cond

                    # That's the inverse correspondence condition that tests whether the distance between the original source
                    # and the closest source point to the corresponding target point is smaller or equal to the distance
                    # between the original source and corresponding target point. We only do this check for those points
                    # that are still valid given all the previous conditions (to save some computation time)
                    still_valid_indices = cond_weight > 0
                    if len(torch.where(still_valid_indices)[0]) > 0:
                        # We find the closest source points to the closest target points by checking
                        # all source points for each closest target point and then taking the one with minimal distance.
                        # Note that this is O(n*n) with n being the number of template vertices, but we assume
                        # the template is more lowres than the target, and we also only check for those points
                        # that are still considered valid correspondences after the previous checks.
                        dists_inv = torch.cdist(closest_points[still_valid_indices], current_vertices[still_valid_indices], p=2)
                        closest_points_inv = current_vertices[still_valid_indices][torch.argmin(dists_inv, dim=1)]
                        closest_dists_inv_corr = torch.linalg.norm(
                            closest_points[still_valid_indices] - closest_points_inv, dim=1)
                        closest_dists_inv_src = torch.linalg.norm(
                            current_vertices[still_valid_indices] - closest_points_inv, dim=1)
                        dist_inv_cond = closest_dists_inv_src <= closest_dists_inv_corr
                        cond_weight[still_valid_indices] *= dist_inv_cond
                    else:
                        raise AssertionError("Couldn't find any correspondences. Check why.")

                if return_closest_points:
                    return closest_points, cond_weight

                else:
                    # here we apply the conditions to the distance loss, and increase the weighting of the remaining
                    # points proportionally, since torch is a bit sensitive to weighting.
                    closest_dists = reweight_loss(closest_dists, cond_weight)

                    return torch.mean(closest_dists), cond_weight

            def get_landmark_loss(current_vertices: torch.Tensor) -> torch.Tensor:
                # just the distance between source and target landmarks.
                lm_dists = torch.linalg.norm(current_vertices[model_landmark_indices] - target_landmarks_torch, dim=-1)

                # Also use curvilinear features if available.
                # Unlike with the landmarks, we first need to determine the closest points on the target line sets
                # based on the current state of the model vertices, before we can compute the distance to these points.
                if target_mesh_normed.has_curvilinear_features() and use_curvilinear_features:
                    model_line_sets_current = CurvilinearFeatures.parse_points_to_line_sets(
                        current_vertices.detach().cpu().numpy(), model_line_set_indices)
                    # Based on these line set points, we find the closest points on the fixed
                    # target curvilinear features.
                    target_closest_curv_points, _ = CurvilinearFeatures(
                        model_line_sets_current).get_closest_points(
                        target_mesh_normed.get_curvilinear_features())

                    # Flatten the closest points
                    target_closest_curv_points_flattened, _ = CurvilinearFeatures.get_flattened_line_sets(
                        target_closest_curv_points)

                    # Convert them to torch tensor
                    target_closest_curv_points_flattened_torch = torch.tensor(
                        target_closest_curv_points_flattened, device=device, dtype=dtype)

                    # Compute the curvilinear feature loss as the distance between the source line sets
                    # and their closest points on the target lines.
                    curv_feat_dists = torch.linalg.norm(
                        current_vertices[model_line_set_indices_flattened] -
                        target_closest_curv_points_flattened_torch, dim=-1)

                    # We reweight the landmark loss and curvilinear feature loss together.
                    lm_dists = reweight_loss(
                        torch.cat([lm_dists, curv_feat_dists]),
                        torch.cat([landmark_weights_torch_current, curvilinear_feature_weights_current_torch]))

                else:
                    # Without curvilinear features, we only need to reweight the landmark loss
                    lm_dists = reweight_loss(lm_dists, landmark_weights_torch_current)

                return torch.mean(lm_dists)

            def get_laplacian_loss(current_vertices: torch.Tensor) -> torch.Tensor:
                """
                Compute the difference between each vertex and all its neighbors, average that difference, compute
                the length of this average vector, then average over all these averaged difference vectors.
                """
                diff = torch.sparse.mm(laplacian_matrix, current_vertices)
                return (diff ** 2).sum() / current_vertices.shape[0]


            def get_coupling_loss(current_vertices: torch.Tensor) -> torch.Tensor:
                """
                Compute the difference between the edge vectors corresponding to the current vertices
                and the ones from the model.
                """
                def get_source_edge_vectors(vertices_local: torch.Tensor) -> torch.Tensor:
                    return vertices_local[source_edges_unique[:, 1]] - vertices_local[source_edges_unique[:, 0]]
                current_edges = get_source_edge_vectors(current_vertices)
                model_edges = get_source_edge_vectors(get_model_vertices(apply_transform=True, disable_gradients=True))
                return torch.mean(torch.linalg.norm(current_edges - model_edges, dim=1))

            # We define an initial transformation based on classical procrustes analysis applied to the landmarks
            transf_mat_init, _ = Landmarks(template.get_vertices()[model_landmark_indices]).procrustes(
                target_landmarks, translation=True, scale=False, reflection=False)
            transf_mat_init_torch = torch.tensor(transf_mat_init, device=device, requires_grad=False, dtype=dtype)
            # the learned translation and rotation are applied on top of the initial transformation
            translation = torch.zeros(3, requires_grad=True, device=device, dtype=dtype)
            rotation = torch.zeros(3, requires_grad=True, device=device, dtype=dtype)

            # We initialize the model parameters by fitting it to the landmarks in a classical way,
            # which helps with convergence during optimization.
            if isinstance(model, PCAMorphableModel):
                # If it's a PCA model, we regularize the fitting
                lm_fit_kwargs = dict(sigma=0.05, num_components=min(32, model.get_number_of_components()))
            else:
                lm_fit_kwargs = dict()
            _, model_latent_lm_fit = model.fit_to_landmarks(
                Landmarks(Landmarks.transform_points(target_landmark_points, np.linalg.inv(transf_mat_init @ normalization_matrix))),
                source_vertex_indices=model_landmark_indices, **lm_fit_kwargs)

            # Here we define the torch tensor for the model parameters that needs to be optimized.
            model_latent_lm_fit_normed = model.normalize_weights(model_latent_lm_fit)
            if len(model_latent_lm_fit_normed) < model.get_number_of_components():
                model_latent_lm_fit_normed = torch.cat([
                    model_latent_lm_fit_normed,
                    torch.zeros(model.get_number_of_components() - len(model_latent_lm_fit_normed),
                                device=device, dtype=dtype)])
            model_params_torch = torch.tensor(model_latent_lm_fit_normed, requires_grad=True, device=device, dtype=dtype)

            # This is the actual optimization method that loops over the iteration setting, and for each,
            # It optimizes the model parameters (and transformations) until convergence on the losses.
            # Convergence is measured with a learning rate scheduler that reduces the learning rate on plateaus.
            # Optional logging to tensorboard of all individual losses and the total training loss is available.
            def optimize(visualize_geometries: Union[Callable, None] = None):
                # We can have several iterations of optimizations using different weights for the
                # model, landmark, distance, coupling, and laplacian losses.
                # Ideally, one first starts with more rigid setting (high model/smoothing loss)
                # to get a good alignment first, before allowing more fine-grained registration.
                iteration_settings: List[Dict] = config["iteration_settings"]
                dist_condition = None
                template_vertices = None
                # If the user wants to use NICP, we initialize all variables in the first iteration it's used.
                nicp_init = False
                # This is the stop criterion threshold of NICP
                eps = config.get("eps", 1e-3)

                # We loop over each iteration setting until convergence given by the learning rate scheduler
                epoch_tot = 0
                for setting in iteration_settings:
                    if verbose:
                        print(f"Optimizing for: {setting}")

                    # Optionally only optimize the vertices in a specific region.
                    # This is basically a per-iteration adjustment of the source_weights config setting.
                    source_weights_adjustment = get_regional_weights(
                        "source_weights_adjustment", current_config=setting, default=1)

                    use_nicp = setting.get("use_nicp", False)
                    # This part uses NICP. It was copied from menpo3d and heavily adjusted, cf. top of page
                    # for the license.
                    if use_nicp:
                        # We use torch for compatibility and maybe faster array handling due to GPU usage,
                        # but we don't optimize via torch in NICP; we use classical optimization instead.
                        with torch.no_grad():
                            if not nicp_init:
                                # We start NICP with the current model state of the vertices.
                                if template_vertices is None:
                                    template_vertices = get_model_vertices(apply_transform=True, disable_gradients=True)
                                # We keep track of which vertices are excluded from NICP and which are kept
                                nicp_remove_mask = IndicesAndMasks.get_mask(exclude_source_vertices, template)
                                nicp_keep_mask = IndicesAndMasks.invert(nicp_remove_mask, template)
                                # A separate template instance with excluded vertices removed is created.
                                # Results are later back-matched to the original template.
                                template_removed = template.remove_vertices(nicp_remove_mask)
                                # Adjust triangle and vertices variables after removing excluded regions
                                triangles_removed = torch.tensor(
                                    template_removed.get_triangles(),
                                    device=device, dtype=model_triangles.dtype, requires_grad=False)
                                template_vertices_removed = template_vertices[nicp_keep_mask]

                                # Keep track of the previous transformation stage to decide when to stop optimizing.
                                X_prev = torch.tile(torch.zeros((3, 4), device=device, dtype=dtype),
                                                    dims=(len(template_vertices_removed),)).T

                                # Row and column indices for the source vertices are later used to fill sparse
                                # matrices with source indices and target correspondences.
                                row = torch.hstack(
                                    (torch.repeat_interleave(torch.arange(len(template_vertices_removed),
                                                                          device=device)[:, None], 3, axis=1).ravel(),
                                     torch.arange(len(template_vertices_removed), device=device))
                                )
                                x = torch.arange(len(template_vertices_removed) * 4, device=device).reshape((len(template_vertices_removed), 4))
                                col = torch.hstack((x[:, :3].ravel(), x[:, 3]))
                                # Landmark indices also need to be adjusted after removing excluded regions.
                                model_landmark_indices_removed = IndicesAndMasks.get_new_indices_after_index_removal(
                                    model_landmark_indices, nicp_remove_mask)
                                if len(model_landmark_indices_removed) < len(model_landmark_indices):
                                    raise NotImplementedError(
                                        f"We lost {len(model_landmark_indices) - len(model_landmark_indices_removed)} "
                                        f"landmarks, because they were positioned on a region you defined to be ignored. "
                                        f"We currently have no mechanism implemented to adjust for this, so please either "
                                        f"include this region or reposition your landmarks.")

                                # Here we create separate row and column indices for the landmarks,
                                # since those are added separately to the sparse matrices.
                                model_landmark_mask = IndicesAndMasks.get_mask(
                                    model_landmark_indices_removed, num_indices_or_mesh=template_removed)
                                vertices_hom_stacked_lm_mask = model_landmark_mask[row.detach().cpu().numpy()]
                                col_lm = col[vertices_hom_stacked_lm_mask]
                                # pull out the rows for the landmarks and map them back to the order of the landmarks
                                row_lm_to_fix = row[vertices_hom_stacked_lm_mask]
                                model_landmark_indices_list = list(model_landmark_indices_removed)
                                row_lm = torch.tensor([model_landmark_indices_list.index(r) for r in row_lm_to_fix], device=device)
                                # Required for landmarks individual weighting
                                landmark_weights_torch_current_diag = sparse_diag(landmark_weights_torch_current)

                                # We do the same for the curvilinear features, after flattening them
                                if target_mesh_normed.has_curvilinear_features() and use_curvilinear_features:
                                    model_curvilinear_feature_indices_flattened, _ = CurvilinearFeatures.get_flattened_line_sets(model_line_set_indices)
                                    model_curvilinear_feature_mask = IndicesAndMasks.get_mask(
                                        model_curvilinear_feature_indices_flattened, num_indices_or_mesh=template_removed)
                                    vertices_hom_stacked_feat_mask = model_curvilinear_feature_mask[row.detach().cpu().numpy()]
                                    col_curv_feat = col[vertices_hom_stacked_feat_mask]
                                    # pull out the rows for the landmarks and map them back to the order of the landmarks
                                    row_curv_feat_to_fix = row[vertices_hom_stacked_feat_mask]
                                    model_curvilinear_feature_indices_flattened_list = list(
                                        model_curvilinear_feature_indices_flattened)
                                    row_curv_feat = torch.tensor([model_curvilinear_feature_indices_flattened_list.index(r)
                                                           for r in row_curv_feat_to_fix], device=device)
                                    curv_feat_weights_torch_current_diag = sparse_diag(curvilinear_feature_weights_current_torch)


                                # M_s is the node-arc incidence matrix, which is defined for directed graphs.
                                # It contains one row for each arc (edge) of the graph and one column per node (vertex).
                                # If edge r connects the vertices (i,j), the nonzero entries of M in row r are M_{ri}=-1 and M_{rj}=1.
                                # The stiffness term is the E_s(X) = ||(M kron G)X||², where G=diag(1,1,1,gamma),
                                # but gamma was set to 1 experimentally, since the data was scaled to the unit cube.
                                M_s, unique_edges = get_node_arc_incidence_matrix(template_removed, device=device, dtype=dtype)
                                G = torch.eye(4, device=device, dtype=dtype)



                                # Variables have been initialized, no need to do it again
                                nicp_init = True

                            # Handle uniform stiffness
                            if isinstance(setting["stiffness"], (float, int)):
                                alpha_M_kron_G_s = setting["stiffness"] * sparse_kron(M_s, G)
                            # Handle per-vertex stiffness, mapping it to per-edge stiffness by averaging
                            # the stiffness for each edge over the stiffnesses of its two vertices.
                            else:
                                stiffness = get_regional_weights(
                                    "stiffness", current_config=setting, default="required")
                                stiffness_per_edge = torch.mean(stiffness[nicp_keep_mask][unique_edges], dim=1)
                                alpha_M_s = torch.sparse.mm(sparse_diag(stiffness_per_edge), M_s)
                                alpha_M_kron_G_s = sparse_kron(alpha_M_s, G)

                            # The stop criterion threshold may be adjusted per iteration.
                            eps_current = setting.get("eps", eps)

                            # Loop until convergence
                            while True:

                                # We use the same correspondence search as during model-based optimization
                                closest_points, dist_condition = get_dist_loss(
                                    template_vertices_removed,
                                    remove_internal_structures=setting.get("remove_internal_structures", False),
                                    normal_thresh=setting.get("normal_thresh", None),
                                    normal_connect_thresh=setting.get("normal_connect_thresh", None),
                                    source_weights_adjustment=source_weights_adjustment,
                                    return_closest_points=True,
                                    vertex_indices_or_mask=nicp_keep_mask, current_triangles=triangles_removed
                                )
                                dist_condition = dist_condition.clip(0, 1)

                                # Build the sparse diagonal weight matrix from the correspondence conditions.
                                # The diagonal entries are between 0 (not registered towards target) and 1 (highest
                                # registration weight).
                                W_s = sparse_diag(dist_condition)

                                # data represents the homogeneous vertex coordinates flattened.
                                data = torch.hstack([template_vertices_removed.ravel(),
                                                     torch.ones(len(template_vertices_removed), device=device, dtype=dtype)])

                                # D_s is the diagonalized vertex matrix (built from data) of dimensions Nx4N.
                                D_s = torch.sparse_coo_tensor(torch.vstack((row, col)), data)

                                # to_stack_A is the array of matrices to stack to build the left-hand side matrix A from.
                                # to_stack_B is the array of matrix to stack to build the right-hand side matrix B from.
                                #
                                # alpha_M_kron_G_s represent the stiffness term E_s(X) = ||(M kron G) X||,
                                # so in the solver, we have (M kron G) as part of the left-hand side matrix A,
                                # and zeros as the corresponding part of the right-hand side matrix B,
                                # since there are no correspondences here;
                                # it's just a regularization on the vertices; there is no right-hand side
                                #
                                # W_s * D_s represents the distance term E_d(X) = ||W(DX - U)||=||WDX - WU||,
                                # so in the solver, we have WD as one part of the left-hand side matrix A,
                                # and WU as the corresponding part of the right-hand side matrix B.
                                to_stack_A = [alpha_M_kron_G_s, torch.sparse.mm(W_s, D_s)]
                                to_stack_B = [
                                    torch.zeros((alpha_M_kron_G_s.shape[0], 3), device=device, dtype=dtype),
                                    closest_points * dist_condition[:, None]]

                                #
                                # The landmark terms E_l(X) = ||D_L X - U_L|| are added.
                                # We take the rows out of D that correspond to the landmark vertices on the template,
                                # thus giving us D_L as part of the left-hand side matrix A.
                                # The points on the target corresponding to these landmarks are U_L,
                                # which is the part of the right-hand side matrix B.
                                if setting.get("lm_weight", 0) > 0:
                                    # landmarks
                                    D_L = torch.sparse_coo_tensor(
                                        torch.vstack((row_lm, col_lm)), data[vertices_hom_stacked_lm_mask],
                                        size=(len(target_landmarks_torch), D_s.shape[1])
                                    )
                                    to_stack_A.append(setting["lm_weight"] * torch.sparse.mm(
                                        landmark_weights_torch_current_diag, D_L))
                                    to_stack_B.append(setting["lm_weight"] * torch.unsqueeze(
                                        landmark_weights_torch_current, 1) * target_landmarks_torch)

                                    # We also add curvilinear features in a similar way as the landmarks.
                                    # They key difference is that we need to determine the target points
                                    # dynamically, using the closest points on the target line sets.
                                    # Afterward, we flatten the arrays and add them to the optimization
                                    # the same way as the landmarks.
                                    if target_mesh_normed.has_curvilinear_features() and use_curvilinear_features:
                                        D_CF = torch.sparse_coo_tensor(
                                            torch.vstack((row_curv_feat, col_curv_feat)),
                                            data[vertices_hom_stacked_feat_mask],
                                            size=(len(model_curvilinear_feature_indices_flattened), D_s.shape[1])
                                        )
                                        to_stack_A.append(setting["lm_weight"] * torch.sparse.mm(
                                            curv_feat_weights_torch_current_diag, D_CF))
                                        # Template vertices change each iteration, so we have to redefine
                                        # its line set points everytime.
                                        template_line_sets_current = CurvilinearFeatures.parse_points_to_line_sets(
                                                template_vertices.detach().cpu().numpy(), model_line_set_indices)
                                        # Based on these line set points, we find the closest points on the fixed
                                        # target curvilinear features.
                                        target_closest_curv_points, _ = CurvilinearFeatures(
                                            template_line_sets_current).get_closest_points(
                                            target_mesh_normed.get_curvilinear_features())
                                        # Flatten the closest points and add them (weighted) to right-hand side.
                                        target_closest_curv_points_flattened, _ = CurvilinearFeatures.get_flattened_line_sets(
                                            target_closest_curv_points)
                                        to_stack_B.append(setting["lm_weight"] * torch.unsqueeze(
                                            curvilinear_feature_weights_current_torch, 1) * torch.tensor(
                                            target_closest_curv_points_flattened, device=device, dtype=dtype))

                                # We stack the lists into left- and right-hand side sparse matrices
                                A_s = torch.vstack(to_stack_A).to_sparse_csr()
                                B_s = torch.vstack(to_stack_B).to_sparse_csr()

                                # We solve the system of linear equations. We don't use pytorch for this.
                                # We sometimes get errors here when the linear system is not solvable. This can be caused
                                # by obviously impossible settings caused by e.g. wrongly set landmarks.
                                # Also watch out for creating small disconnected components with the exclude_vertices
                                # variable, and for landmarks positioned on mesh boundaries.
                                X_np = spsolve(torch_csr_to_scipy_csr(A_s), torch_csr_to_scipy_csr(B_s))
                                X = torch.asarray(X_np, device=device, dtype=dtype)

                                # Apply transformation to the template vertices
                                template_vertices_removed = D_s @ X
                                # Back-match to the original template vertices including the excluded regions.
                                # todo: One could apply ARAP adjustment here to deform the excluded vertices, such
                                #       that they still kind of move along with the other vertices.
                                template_vertices[nicp_keep_mask] = template_vertices_removed

                                # Compute difference between the transformation of last and current iteration
                                transform_diff = torch.linalg.norm(X_prev - X, ord="fro") / np.sqrt(X_prev.numel())

                                X_prev = X

                                if visualize_geometries is not None:
                                    visualize_geometries(
                                        target_mesh_normed, get_current_mesh(template_vertices, apply_transform=True))

                                # If the difference is below our threshold, we assume the optimization has converged,
                                # so we break the loop.
                                if transform_diff < eps_current:
                                    # Back-match dist_condition to include excluded regions.
                                    dist_condition_full = torch.zeros(
                                        template.get_num_vertices(), device=device, dtype=dtype, requires_grad=False)
                                    dist_condition_full[nicp_keep_mask] = dist_condition
                                    dist_condition = dist_condition_full
                                    break

                    # If no NICP is used, we perform model-based registration,
                    # or we go out of model-space via torch-based optimization based
                    # on FLAME.
                    else:
                        #
                        # Define the parameters to optimize based on the user settings.
                        #
                        # This option optimizes the rigid transformation applied
                        # to the model output used for alignment with the target
                        optimize_transform = setting.get("optimize_transf", False)
                        # This option allows the mesh to leave model space by optimizing the
                        # vertices directly.
                        optimize_template = setting.get("optimize_template", False)
                        # This option optimizes the model parameters. It's turned on by default, unless
                        # the setting wants to go out of model space by optimizing the template.
                        optimize_model = setting.get("optimize_model", not optimize_template)

                        # Define the parameters to be given to the optimizer
                        params_to_optimize = []
                        if optimize_model:
                            params_to_optimize.append(model_params_torch)
                        if optimize_transform:
                            params_to_optimize.extend([translation, rotation])
                        if optimize_template:
                            if optimize_transform:
                                warnings.warn("Not sure if the optimization holds if the transformation is still optimized "
                                              "during template optimization. Good luck!")
                            if optimize_model:
                                warnings.warn("Not sure if it's a good idea to keep optimizing the model parameters "
                                              "while you go out of model space. Good luck!")
                            # The first time we optimize the template, we initialize it to the model vertices.
                            # Afterwards, they stay distinct from the model vertices,
                            # even if a new optimization iteration is started.
                            if template_vertices is None:
                                template_vertices = get_model_vertices(
                                    apply_transform=True, disable_gradients=True).clone().detach().requires_grad_(True)
                            params_to_optimize.append(template_vertices)
                        if len(params_to_optimize) == 0:
                            raise AssertionError("You need to optimized something.")

                        # We haven't thoroughly tested these optimization settings. They seem to produce decent
                        # results, but I think convergence could be detected better than with the current
                        # scheduler setting. LBFGS can theoretically help with faster convergence, but the
                        # current settings usually doesn't converge faster. Feedback is appreciated here.

                        # Start and stop conditions for the learning rate
                        init_lr = 0.01
                        min_lr = 5e-5
                        lr_adjust_factor = 0.3
                        # Define the optimizer, which starts with the initial learning rate
                        use_lbfgs = args.use_lbfgs
                        if use_lbfgs:
                            optimizer = optim.LBFGS(params_to_optimize, lr=init_lr)
                        else:
                            optimizer = optim.Adam(params_to_optimize, lr=init_lr)

                        # Scheduler for fast initial and later more slow and accurate convergence
                        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                            optimizer, mode='min', factor=lr_adjust_factor, patience=5,
                            cooldown=0, min_lr=min_lr * lr_adjust_factor, threshold=1e-2)

                        epoch = 0
                        while True:
                            loss_dict = {}
                            def closure():
                                optimizer.zero_grad()
                                loss_tot = torch.tensor(0.0, device=device, dtype=dtype)


                                # Compute individual losses, and log all of them to tensorboard (even if they're weighted 0)

                                if optimize_template:
                                    current_vertices = template_vertices
                                else:
                                    current_vertices = get_model_vertices(
                                        apply_transform=True, disable_gradients=False, convert_to_numpy=False)

                                    # model parameter regularization
                                    loss_dict['model_weight'] = get_model_weight_loss()
                                    if setting["model_weight"] > 0:
                                        if args.log_tensorboard:
                                            writer.add_scalar("model loss", loss_dict['model_weight'].item(), epoch_tot)
                                        loss_tot += setting["model_weight"] * loss_dict['model_weight']

                                # landmark term
                                loss_dict["landmark"] = get_landmark_loss(current_vertices)
                                if setting["lm_weight"] > 0:
                                    if args.log_tensorboard:
                                        writer.add_scalar("landmark loss", loss_dict['landmark'].item(), epoch_tot)
                                    loss_tot += setting["lm_weight"] * loss_dict['landmark']

                                # distance term
                                loss_dict['distance'], dist_condition = get_dist_loss(
                                    current_vertices,
                                    remove_internal_structures=setting.get("remove_internal_structures", False),
                                    normal_thresh=setting.get("normal_thresh", None),
                                    normal_connect_thresh=setting.get("normal_connect_thresh", None),
                                    source_weights_adjustment=source_weights_adjustment
                                )
                                if setting["dist_weight"] > 0:
                                    if args.log_tensorboard:
                                        writer.add_scalar("distance loss", loss_dict['distance'].item(), epoch_tot)
                                    loss_tot += setting["dist_weight"] * loss_dict['distance']

                                # The coupling and smoothing loss are only used when the template is optimized.
                                # This objective was adapted from the FLAME paper.
                                if optimize_template:
                                    # The coupling penalizes differences between the template and the model edges;
                                    # this was taken from FLAME.
                                    loss_dict["coupling"] = get_coupling_loss(current_vertices)
                                    if setting["coupling_weight"] > 0:
                                        if args.log_tensorboard:
                                            writer.add_scalar("coupling loss", loss_dict['coupling'].item(), epoch_tot)
                                        loss_tot += setting["coupling_weight"] * loss_dict['coupling']

                                    # The Laplacian term penalizes unsmoothness of the template;
                                    # this was taken from FLAME.
                                    loss_dict["laplacian"] = get_laplacian_loss(current_vertices)
                                    if setting.get("laplacian_weight", 0) > 0:
                                        if args.log_tensorboard:
                                            writer.add_scalar("laplacian loss", loss_dict['laplacian'].item(), epoch_tot)
                                        loss_tot += setting["laplacian_weight"] * loss_dict['laplacian']

                                    # The Laplacian difference term penalizes differences in the deviation of the smoothness
                                    # between the template and the model. This can avoid general template shrinkage problems.
                                    # It's more similar to the coupling loss, because a high weight will basically force
                                    # the template to stay very close to the model-based mesh.
                                    loss_dict["laplacian_diff"] = get_laplacian_loss(
                                        current_vertices - get_model_vertices(disable_gradients=True))
                                    if setting.get("laplacian_diff_weight", 0) > 0:
                                        if args.log_tensorboard:
                                            writer.add_scalar("laplacian diff loss", loss_dict['laplacian_diff'].item(), epoch_tot)
                                        loss_tot += setting["laplacian_diff_weight"] * loss_dict['laplacian_diff']

                                # Also log total training loss
                                if args.log_tensorboard:
                                    writer.add_scalar("loss tot", loss_tot.item(), epoch_tot)

                                # back prop and optimization
                                loss_tot.backward()
                                return loss_tot

                            if use_lbfgs:
                                loss_tot = optimizer.step(closure)
                            else:
                                loss_tot = closure()
                                optimizer.step()
                            scheduler.step(loss_tot)

                            # Print feedback on the optimization
                            if verbose and epoch > 0 and epoch % 25 == 0:
                                print(f"Epoch {epoch}. Loss: {'; '.join([f'{name}: {loss_val:.3f}' for name, loss_val in loss_dict.items()])}" +
                                      f"...model params stats: abs mean {np.mean(np.abs(model_params_torch.detach().cpu().numpy())):.3f}, "
                                      f"abs std {np.std(np.abs(model_params_torch.detach().cpu().numpy())):.3f}  "
                                      f"abs max {np.max(np.abs(model_params_torch.detach().cpu().numpy())):.3f}" +
                                      f"...current translation {translation.detach().cpu().numpy().tolist()}, "
                                      f"rotation {rotation.detach().cpu().numpy()}")

                            if visualize_geometries is not None:
                                #current_template_vertices = get_model_vertices(apply_transform=True, disable_gradients=True) if template_vertices is None else template_vertices
                                visualize_geometries(
                                    target_mesh_normed, get_current_mesh(apply_transform=True))

                            # Break condition always happens after 10k iterations or when min_lr has been passed, i.e.,
                            # after the loss has converged with the minimum learning rate.
                            if epoch > 10000 or optimizer.param_groups[0]['lr'] < min_lr:
                                break

                            epoch += 1
                            epoch_tot += 1
                if template_vertices is None:
                    final_vertices = get_model_vertices(apply_transform=True, disable_gradients=True)
                else:
                    final_vertices = template_vertices
                return (get_current_mesh(transform_vertices(final_vertices, invert_transform=True), apply_transform=False),
                        get_current_transf_matrix(convert_to_numpy=True), dist_condition.detach().cpu().numpy())

            # Call the optimization (optionally running parallel to a visualization thread that shows the progress
            # of the registration)
            if verbose >= 2:
                optimize_return = Visualizer.visualization_thread(optimize, close_automatically=True)
            else:
                optimize_return = optimize()
            registered_mesh, transformation, vertex_dist_weights = optimize_return

            # Save the registered mesh
            registered_mesh.export(output_mesh_path)
            # Also save its transformation (excluding normalization) to allow for back-matching it with the input scans
            RegistrationFileCommunicator.write_transform(
                np.linalg.inv(transformation) @ normalization_matrix, output_transform_path)
            # Export the indices which were excluded from the registration
            exclude_registered_vertices = IndicesAndMasks.get_indices(vertex_dist_weights == 0)
            IndicesAndMasks.export(exclude_registered_vertices, vertex_exclude_path)
            # Optionally also export the target mesh aligned with the registered mesh
            if getattr(args, "save_aligned_target", config.get("save_aligned_target", False)):
                target_mesh_normed.transform(transformation, in_place=False, invert=True).export(
                    output_mesh_path.with_name(f"{output_mesh_path.stem}_target_aligned{target_mesh_path.suffix}"))


if __name__ == '__main__':
    main()