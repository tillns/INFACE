"""
PCA morphable model class. An instance can be created by providing the constructor with path to PCA model,
saved as an HDF5 file. This file can be run as a script by providing the path to said HDF5 file as CL argument.
The model visualizer from the abstract MorphableModel class is then called.

Visualization requires open3d.
Reconstruction with back matching requires scipy, fbpca, and sksparse.

Author: Till Schnabel (contact till.schnabel@inf.ethz.ch)

MIT License

Copyright (c) 2025 ETH Zurich, Till Schnabel

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

from numpy.linalg import LinAlgError
from morphable_model import MorphableModel, main

import warnings
import numpy as np


def average_dist_to_mean(vertices):
    mean_vertex = np.mean(vertices, axis=0)
    dist = np.linalg.norm(vertices - mean_vertex, axis=1)
    return np.mean(dist)

class PCAMorphableModel(MorphableModel):
    def __init__(self, path_to_hdf5_file: str):
        params = self.load_params_from_hdf5_to_dict(path_to_hdf5_file)

        self.components = np.asarray(params["components"])
        self.eigenvalues = np.asarray(params["eigenvalues"])

        self.latent_mean = np.zeros_like(self.eigenvalues)
        self.latent_std = self.eigenvalues ** 0.5
        self.mean_vertices = np.asarray(params["mean_vertices"])
        self.triangles = np.asarray(params["triangles"])

        super().__init__()

    def encode(self, vertices: np.ndarray, unknown_vertex_mask: np.ndarray = None, invert_mask: bool = False) -> np.ndarray:
        data = vertices.flatten()
        simple_projection = np.dot(self.components, data - self.mean_vertices.flatten())
        unknown_vertex_mask = self.get_unknown_vertex_mask(
            unknown_vertex_mask=unknown_vertex_mask, vertices=vertices, invert_mask=invert_mask)
        # If no unknown vertices, we just do simple projection here without requiring additional libraries

        if not np.any(unknown_vertex_mask):
            return simple_projection

        unknown_vertex_mask_flattened = unknown_vertex_mask.repeat(3)

        # else we require those libraries, because we optimize the PCA latent to match only the known vertices
        try:
            import fbpca
            import scipy.sparse as spsp
            from scipy.sparse import diags
        except ImportError:
            warnings.warn("To support PCA encoding fitted to only known vertices, you must install scipy and fbpca. "
                          "Returning latent computed via simple PCA projection.")
            return simple_projection

        #
        #
        # Credit goes to Xiaojing Xia for implementing the following method based on the work
        # "Statistically Motivated 3D Faces Reconstruction" from Basso and Vetter (2006)

        L_tmp = np.ones_like(data)
        L_tmp[unknown_vertex_mask_flattened] = 0
        L_flat = L_tmp.ravel()
        L = spsp.csr_matrix(diags(L_flat))
        pca_components = self.components
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
                    raise LinAlgError("Could be RAM issues, but I don't know.")
            else:
                break

        W = spsp.csr_matrix(np.diag(w))

        d_star = np.array(data)
        d_star[unknown_vertex_mask_flattened] = 0
        V_t_t = V_t.T
        alpha = V_t_t @ spsp.csr_matrix(np.linalg.inv(W @ W + 0 * np.identity(W.shape[0]))) @ W @ U.T @ L @ (
                    d_star.flatten() - self.mean_vertices.flatten())
        return alpha
        #
        #
        # End credit

    def decode(self, latent: np.ndarray) -> np.ndarray:
        decoded_flat = np.dot(latent, self.components)
        return np.reshape(decoded_flat, (*decoded_flat.shape[:-1], -1, 3)) + self.mean_vertices

    def get_covariance_matrix(self):
        return np.transpose(self.components).dot(np.diag(1/self.eigenvalues)).dot(self.components)


if __name__ == '__main__':
    main()