"""
Autoencoder morphable model class. An instance can be created by providing the constructor with path to AE model,
saved as an HDF5 file. This file can be run as a script by providing the path to said HDF5 file as CL argument.
The model visualizer from the abstract MorphableModel class is then called.

Visualization requires open3d.
Reconstruction with back matching requires scipy, and sksparse.

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

from morphable_model import MorphableModel, main
from typing import List, Tuple
from mesh import Mesh

import numpy as np

class Encoder:
    def __init__(self, matrix_mults: List[np.ndarray], biases: List[np.ndarray], relu_factor: float = 0.2):
        self.matrix_mults = matrix_mults
        self.biases = biases
        self.relu_factor = relu_factor

    def encode(self, x: np.ndarray) -> np.ndarray:
        x_encoded = x.flatten()
        for matrix_mult, bias in zip(self.matrix_mults, self.biases):
            # fully-connected layer
            x_encoded = matrix_mult @ x_encoded + bias
            # leaky ReLU activation
            if self.relu_factor < 1:
                x_encoded[x_encoded < 0] *= self.relu_factor
        return x_encoded

    def get_latent_dimension(self):
        return len(self.biases[-1])


class Decoder:
    def __init__(self, matrix_mults: List[np.ndarray], biases: List[np.ndarray], relu_factor: float = 0.2):
        self.matrix_mults = matrix_mults
        self.biases = biases
        self.relu_factor = relu_factor

    def decode(self, latent: np.ndarray) -> np.ndarray:
        x_decoded = np.array(latent)
        for num_layer, (matrix_mult, bias) in enumerate(zip(self.matrix_mults, self.biases)):
            # fully-connected layer
            x_decoded = matrix_mult @ x_decoded + bias
            # leaky ReLU activation (not applied to output)
            if num_layer < len(self.matrix_mults) - 1 and self.relu_factor < 1:
                x_decoded[x_decoded < 0] *= self.relu_factor
        return x_decoded

    def get_latent_dimension(self):
        return self.matrix_mults[0].shape[1]



class AEMorphableModel(MorphableModel):
    def __init__(self, path_to_hdf5_file: str):
        params = self.load_params_from_hdf5_to_dict(path_to_hdf5_file)

        self.latent_mean = np.asarray(params['latent_mean'])
        self.latent_std = np.asarray(params['latent_std'])

        # Load encoder(s)
        if "encoder" in params:
            self.encoders = [Encoder(params["encoder"]["matrix_mults"], params["encoder"]["biases"],
                                     relu_factor=params["encoder"].get("relu_factor", 0.2))]
        else:
            self.encoders = [Encoder(params[encoder_type]["matrix_mults"], params[encoder_type]["biases"],
                                     relu_factor=params[encoder_type].get("relu_factor", 0.2))
                             for encoder_type in ["id_encoder", "exp_encoder", "age_encoder"]]
        # sanity check dimensions
        self.dims_per_encoder = [encoder.get_latent_dimension() for encoder in self.encoders]
        if not sum(self.dims_per_encoder) == len(self.latent_mean):
            raise AssertionError("The size of the latent vector should match "
                                 "the sum of the individual encoder output sizes.")
        self.is_disentangled = len(self.encoders) > 1

        # Load decoder
        self.decoder = Decoder(params["decoder"]["matrix_mults"], params["decoder"]["biases"],
                               relu_factor=params["decoder"].get("relu_factor", 0.2))
        if not self.decoder.get_latent_dimension() == len(self.latent_mean):
            raise AssertionError("The size of the latent vector should match "
                                 "the decoder's input size.")

        self.mean_vertices = np.asarray(params["mean_vertices"])
        self.triangles = np.asarray(params["triangles"])

        super().__init__()

    def encode(self, vertices: np.ndarray, unknown_vertex_mask: np.ndarray = None, invert_mask: bool = False) -> np.ndarray:
        vertices_flattened = (vertices - self.mean_vertices).flatten()
        return np.concatenate([encoder.encode(vertices_flattened) for encoder in self.encoders])

    def decode(self, latent: np.ndarray) -> np.ndarray:
        vertices_recon_flattened = self.decoder.decode(latent)
        return np.reshape(vertices_recon_flattened, (-1, 3)) + self.mean_vertices

    def get_exp_dims(self) -> Tuple[int, int]:
        if not self.is_disentangled:
            raise AssertionError("The current autoencoder does not have an expression dimension")
        exp_start = self.dims_per_encoder[0]
        exp_end = self.dims_per_encoder[0] + self.dims_per_encoder[1]
        return exp_start, exp_end

    def set_expression(self, mesh_to_set_expression_for: Mesh, latent_exp_code: np.ndarray) -> Mesh:
        if not self.is_disentangled:
            raise AssertionError("The current autoencoder does not support disentanglement.")
        mesh_latent = self.encode(mesh_to_set_expression_for.get_vertices())
        exp_start, exp_end = self.get_exp_dims()
        if len(latent_exp_code) == self.get_number_of_components():
            latent_exp_code = latent_exp_code[exp_start:exp_end]
        mesh_latent[exp_start:exp_end] = latent_exp_code
        return Mesh(self.decode(mesh_latent), self.triangles)

    def transfer_expression(self, source_mesh: Mesh, target_mesh: Mesh) -> Mesh:
        """
        Transfer expression from source mesh to target mesh. Return target mesh with transferred expression.
        """
        if not self.is_disentangled:
            raise AssertionError("The current autoencoder does not support disentanglement.")
        source_mesh_latent = self.encode(source_mesh.get_vertices())
        exp_start, exp_end = self.get_exp_dims()
        target_mesh_transferred_expression = self.set_expression(target_mesh, source_mesh_latent[exp_start:exp_end])
        return target_mesh_transferred_expression

    def adjust_age(self, mesh_to_adjust_age_for: Mesh, age_adjust: float) -> Mesh:
        if not self.is_disentangled:
            raise AssertionError("The current autoencoder does not support disentanglement.")
        mesh_latent = self.encode(mesh_to_adjust_age_for.get_vertices())
        mesh_latent[-1] = age_adjust
        return Mesh(self.decode(mesh_latent), self.triangles)

    def visualize(self, min_val: float = -3, max_val: float = 3,
                  num_components_per_disentangled_latent: int = 5, **kwargs):
        """
        Supports disentangled visualization by providing sliders for each of the disentangled spaces.
        Cf. docs for visualize() method in MorphableModel superclass.
        """
        shape_components = []
        start_index = 0
        for num_encoder, encoder in enumerate(self.encoders):
            latent_dim = encoder.get_latent_dimension()
            shape_components.extend(list(range(start_index, start_index + min(num_components_per_disentangled_latent, latent_dim))))
            start_index += latent_dim

        super().visualize(min_val=min_val, max_val=max_val, shape_components=shape_components, **kwargs)


if __name__ == '__main__':
    main()
