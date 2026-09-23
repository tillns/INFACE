"""
PCA morphable model class. An instance can be created by providing the constructor with path to PCA model,
saved as an HDF5 file. This file can be run as a script by providing the path to said HDF5 file as CL argument.
The model visualizer from the abstract MorphableModel class is then called.

Visualization requires open3d.
Reconstruction with back matching requires scipy, fbpca, and sksparse.

Author: Till Schnabel (contact till.schnabel@inf.ethz.ch)

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
"""
from pathlib import Path

from numpy.linalg import LinAlgError

from src.objects.morphable_model import MorphableModel, main
from typing import Union, TYPE_CHECKING, List, Tuple, Dict

import warnings
import numpy as np

try:
    import torch
except ImportError:
    torch = None
if TYPE_CHECKING:
    from torch import Tensor
ArrayLike = Union[np.ndarray, "Tensor"]

class PCAMorphableModel(MorphableModel):
    def __init__(self, path_to_hdf5_file: Union[str, Path], use_torch: bool = False, device: str = "cpu",
                 double_precision: bool = False):
        params = self.load_params_from_hdf5_to_dict(path_to_hdf5_file)

        self.initialize_torch(use_torch=use_torch, device=device, double_precision=double_precision)

        self.components = self.get_array(params["components"], copy=False, dtype=self.float_dtype)
        self.eigenvalues = self.get_array(params["eigenvalues"], copy=False, dtype=self.float_dtype)

        self.latent_mean = self.get_array(np.zeros(len(self.eigenvalues)), copy=False, dtype=self.float_dtype)
        self.latent_std = self.eigenvalues ** 0.5
        self.mean_vertices = self.get_array(params["mean_vertices"], copy=False, dtype=self.float_dtype)
        self.triangles = self.get_array(params["triangles"], copy=False, dtype=self.int_dtype)

        self.indices = params["indices"] if "indices" in params else None
        colors = params.get("colors", None)
        if colors is None:
            self.colors = None
        else:
            self.colors = self.get_array(colors, copy=False, dtype=self.float_dtype)

        super().__init__()

    def encode(self, vertices: ArrayLike, unknown_vertex_mask: np.ndarray = None, invert_mask: bool = False,
               sigma: float = 0, num_components: Union[int, None] = None) -> ArrayLike:
        """
        Currently no torch support if unknown_vertex_mask is provided because of dependency on external libraries.
        """
        data = self.get_array(vertices, copy=False, dtype=self.float_dtype).flatten()
        simple_projection = self.components @ (data - self.mean_vertices.flatten())
        unknown_vertex_mask = self.get_unknown_vertex_mask(
            unknown_vertex_mask=unknown_vertex_mask, vertices=vertices, invert_mask=invert_mask)
        # If no unknown vertices, we just do simple projection here without requiring additional libraries

        if not np.any(unknown_vertex_mask):
            return simple_projection

        # else we require those libraries, because we optimize the PCA latent to match only the known vertices
        try:
            import fbpca
            import scipy.sparse as spsp
            from scipy.sparse import diags
        except ImportError:
            warnings.warn("To support PCA encoding fitted to only known vertices, you must install scipy and fbpca. "
                          "Returning latent computed via simple PCA projection.")
            return simple_projection

        unknown_vertex_mask_flattened = unknown_vertex_mask.repeat(3)
        #
        #
        # Credit goes to Xiaojing Xia for implementing the following method based on the work
        # "Statistically Motivated 3D Faces Reconstruction" from Basso and Vetter (2006)

        d_star = self.get_array(data, copy=True, convert_numpy=True)
        L_tmp = np.ones_like(d_star)
        L_tmp[unknown_vertex_mask_flattened] = 0
        L_flat = L_tmp.ravel()
        L = spsp.csr_matrix(diags(L_flat))
        pca_components = self.get_array(self.components, copy=True, convert_numpy=True)
        if num_components is not None:
            pca_components = pca_components[:num_components]
        C = pca_components.T
        Q = L @ C

        # Linalg decomposition occasionally failed randomly,
        # so we try multiple times in case it's really just random.
        while True:
            counter = 0
            try:
                U, w, V_t = fbpca.pca(Q, C.shape[1])
            except LinAlgError as e:
                counter += 1
                if counter > 3:
                    raise LinAlgError("Linalg decomposition failed three times in a row. "
                                      f"This could be due to RAM issues. Full error: {e}")
            else:
                break

        W = spsp.csr_matrix(np.diag(w))

        d_star[unknown_vertex_mask_flattened] = 0

        sigma_squared_id = sigma * sigma * np.identity(W.shape[0])

        V_t_t = V_t.T
        alpha = V_t_t @ spsp.csr_matrix(np.linalg.inv(W @ W + sigma_squared_id)) @ W @ U.T @ L @ (
                    d_star.flatten() - self.get_array(self.mean_vertices.flatten(), copy=False, convert_numpy=True))
        return alpha
        #
        #
        # End credit

    def decode(self, latent: ArrayLike) -> ArrayLike:
        latent_array = self.get_array(latent, copy=False, dtype=self.float_dtype)
        decoded_flat = latent_array @ self.components[:latent_array.shape[-1]]
        return decoded_flat.reshape((*decoded_flat.shape[:-1], -1, 3)) + self.mean_vertices

    def get_covariance_matrix(self) -> ArrayLike:
        diag_func = torch.diag if self.use_torch else np.diag
        return self.components.transpose() @ diag_func(1/self.eigenvalues) @ self.components

    def save(self, path_to_hdf5_file: Union[Path, str]) -> None:
        params = {
            "components": self.components,
            "eigenvalues": self.eigenvalues,
            "triangles": self.triangles,
            "mean_vertices": self.mean_vertices,
            "indices": self.indices,
        }
        self.save_params_to_hdf5_dict(params, path_to_hdf5_file)

    @classmethod
    def train(cls, mesh_folders: Union[str, Path, List[Union[str, Path]]],
              path_to_hdf5_file: Union[Path, str],
              mesh_regex_filter: Union[str, List[str], None] = None,
              exclude_mesh_regex_filter: Union[str, List[str], None] = None,
              retain_components: Union[float, int, None] = None,
              indices: Union[Dict, None] = None, include_colors: bool = False) -> None:
        """
        Train a PCA morphable model on registered meshes and save the output as hdf5 file.
        :param mesh_folders: One or more folders containing the registered meshes.
        :param path_to_hdf5_file: Output path to save the model.
        :param mesh_regex_filter: Optionally filter the meshes to train the model on with this argument.
        :param exclude_mesh_regex_filter: Optionally use this argument to filter meshes to be excluded from training.
        :param retain_components: Optionally specify how many components to keep in the final morphable model.
                                  A number >=1 is interpreted as integer that defines how many components to keep.
                                  A number between 0 and 1 is interpreted as percentage of variance to keep.
        :param indices: Index selections made on the template topology to save in the morphable model object.
        :param include_colors: Include average colors in morphable model
                               (not a separate PCA model, really just the average).
        :return: None
        """

        # Load and flatten all mesh vertices from the meshes inside the provided folders
        from src.objects.mesh import Mesh
        mesh_paths = Mesh.get_all_mesh_files_in_path(
            mesh_folders, regex_names=mesh_regex_filter, exclude_regex_names=exclude_mesh_regex_filter)
        if len(mesh_paths) < 2:
            raise AssertionError(f"Need at least two meshes to train a PCA morphable model. "
                                 f"Given are {len(mesh_paths)}.")
        meshes = [Mesh.load(mesh_path) for mesh_path in mesh_paths]
        if not all([meshes[0].get_num_vertices() == mesh.get_num_vertices() for mesh in meshes[1:]]):
            raise AssertionError("The meshes do not all have the same number of vertices. Make sure you only have "
                                 "registered meshes in the folder you provided, or use the filter options "
                                 "to remove the other ones.")
        mesh_vertices_flattened = np.asarray([mesh.get_vertices(copy=True).flatten() for mesh in meshes])

        # We now compute PCA over the flattened mesh vertices.
        # The code for this was copied from
        # https://github.com/menpo/menpo/blob/bf08b913bb84ba5f5323e9b7d372a8349b9d1fa8/menpo/model/pca.py and
        # https://github.com/menpo/menpo/blob/bf08b913bb84ba5f5323e9b7d372a8349b9d1fa8/menpo/math/decomposition.py
        # and adjusted. It may be subject to the following separate license:
        #
        #
        #
        # Copyright 2014 Menpo Developers
        #
        # Redistribution and use in source and binary forms, with or without modification,
        # are permitted provided that the following conditions are met:
        #
        # 1. Redistributions of source code must retain the above copyright notice,
        #    this list of conditions and the following disclaimer.
        #
        # 2. Redistributions in binary form must reproduce the above copyright notice, this list of conditions and
        #    the following disclaimer in the documentation and/or other materials provided with the distribution.
        #
        # 3. Neither the name of the copyright holder nor the names of its contributors may be used to endorse or
        #    promote products derived from this software without specific prior written permission.
        #
        # THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR
        # IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND
        # FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS
        # BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
        # (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA,
        # OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER
        # IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY
        # OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

        # Define dimensions for computing PCA
        num_meshes, vertex_dim = mesh_vertices_flattened.shape

        # average
        mean_vertices = np.mean(mesh_vertices_flattened, axis=0)

        # Center the flattened mesh vertices by subtracting the mean
        mesh_vertices_centered = mesh_vertices_flattened - mean_vertices

        def eigenvalue_decomposition(cov_mat: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
            eigenvalues, eigenvectors = np.linalg.eigh(cov_mat)

            # sort eigenvalues from largest to smallest
            index = np.argsort(eigenvalues)[::-1]
            eigenvalues = eigenvalues[index]
            eigenvectors = eigenvectors[:, index]

            # set tolerance limit
            limit = np.max(np.abs(eigenvalues)) * 1e-10

            # select positive eigenvalues
            pos_index = eigenvalues > 0.0
            pos_eigenvalues = eigenvalues[pos_index]
            pos_eigenvectors = eigenvectors[:, pos_index]
            # check that they are within the expected tolerance
            index = pos_eigenvalues > limit
            pos_eigenvalues = pos_eigenvalues[index]
            pos_eigenvectors = pos_eigenvectors[:, index]

            return pos_eigenvectors, pos_eigenvalues

        if vertex_dim < num_meshes:
            # compute covariance matrix d x d
            covariance = np.dot(mesh_vertices_centered.conj().T, mesh_vertices_centered) / (num_meshes - 1)
            # covariance should be perfectly symmetrical, but numerical error can creep in.
            # Enforce symmetry here to avoid creating complex eigenvectors
            covariance = (covariance + covariance.conj().T) / 2.0

            # perform eigenvalue decomposition
            # eigenvectors: d x (n - 1)
            # eigenvalues:  n - 1
            eigenvectors, eigenvalues = eigenvalue_decomposition(covariance)

            # transpose eigenvectors -> (n - 1) x d
            eigenvectors = eigenvectors.T

        else:
            # compute small covariance matrix n x n
            covariance = np.dot(mesh_vertices_centered, mesh_vertices_centered.conj().T) / (num_meshes - 1)
            # covariance should be perfectly symmetrical, but numerical error can creep in.
            # Enforce symmetry here to avoid creating complex eigenvectors
            covariance = (covariance + covariance.conj().T) / 2.0

            # perform eigenvalue decomposition
            # eigenvectors: n x (n-1)
            # eigenvalues:  n-1
            eigen_vectors_init, eigenvalues = eigenvalue_decomposition(covariance)

            # compute final eigenvectors (n - 1) x d
            w = np.sqrt(1.0 / ((num_meshes - 1) * eigenvalues))
            eigenvectors = np.dot(eigen_vectors_init.conj().T, mesh_vertices_centered)
            eigenvectors *= w[:, None]

        # Menpo-copied code ends here.

        if retain_components is not None:
            if isinstance(retain_components, float) and 0 < retain_components < 1.0:
                eigen_values_cumulative_sum = np.cumsum(eigenvalues) / np.sum(eigenvalues)
                retain_components = np.where(eigen_values_cumulative_sum > retain_components)[0][0]
            else:
                retain_components = int(retain_components)
            eigenvectors = eigenvectors[:retain_components]
            eigenvalues = eigenvalues[:retain_components]

        params = {
            "components": eigenvectors,
            "eigenvalues": eigenvalues,
            "triangles": meshes[0].get_triangles(),
            "mean_vertices": np.reshape(mean_vertices, (-1, 3)),
        }
        if indices is not None:
            params["indices"] = indices
        if include_colors:
            params["colors"] = np.mean([mesh.get_colors(copy=False) for mesh in meshes], axis=0)

        cls.save_params_to_hdf5_dict(params, path_to_hdf5_file)


if __name__ == '__main__':
    main()