"""
Autoencoder morphable model class. An instance can be created by providing the constructor with path to AE model,
saved as an HDF5 file. This file can be run as a script by providing the path to said HDF5 file as CL argument.
The model visualizer from the abstract MorphableModel class is then called.

Visualization requires open3d.
Reconstruction with back matching requires scipy, and sksparse.

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
import warnings
import numpy as np
from typing import List, Tuple, Union, TYPE_CHECKING
from pathlib import Path
from src.objects.mesh import Mesh
from src.objects.morphable_model import MorphableModel, main
try:
    import torch
except ImportError:
    torch = None
if TYPE_CHECKING:
    from torch import Tensor
ArrayLike = Union[np.ndarray, "Tensor"]


class EncoderDecoder:
    def __init__(self, matrix_mults: List[np.ndarray], biases: List[np.ndarray], relu_factor: float = 0.2,
                 use_torch: bool = False, device: str = "cpu"):
        self.use_torch = use_torch
        self.device = device
        if self.use_torch:
            if torch is None:
                raise RuntimeError(
                    "use_torch=True but PyTorch is not installed. "
                    "Please install torch or set use_torch=False."
                )

            self.dtype = torch.float32
            self.matrix_mults = [torch.tensor(matrix_mult, requires_grad=False, device=self.device, dtype=self.dtype)
                                 for matrix_mult in matrix_mults]
            self.biases = [torch.tensor(bias, requires_grad=False, device=self.device, dtype=self.dtype)
                           for bias in biases]
        else:
            self.dtype = np.float32
            self.matrix_mults = matrix_mults
            self.biases = biases
        self.relu_factor = relu_factor

    def forward(self, input_vector: ArrayLike, activate_last_layer: bool = True) -> ArrayLike:
        if self.use_torch:
            output_vector = torch.as_tensor(input_vector, dtype=self.dtype, device=self.device)
        else:
            output_vector = np.asarray(input_vector)
        for num_layer, (matrix_mult, bias) in enumerate(zip(self.matrix_mults, self.biases)):
            # fully-connected layer
            if output_vector.ndim == 2:
                output_vector = (matrix_mult @ output_vector.T).T
            else:
                output_vector = matrix_mult @ output_vector
            output_vector += bias
            # leaky ReLU activation
            if (activate_last_layer or num_layer < len(self.matrix_mults) - 1) and self.relu_factor < 1:
                output_vector[output_vector < 0] *= self.relu_factor
        return output_vector

    def encode(self, x: ArrayLike) -> ArrayLike:
        return self.forward(x, activate_last_layer=True)

    def decode(self, latent: ArrayLike) -> ArrayLike:
        # leaky ReLU is not applied to the output
        return self.forward(latent, activate_last_layer=False)

    def get_latent_dimension(self, is_encoder: bool):
        if is_encoder:
            return self.biases[-1].shape[0]
        else:
            return self.matrix_mults[0].shape[1]



class AEMorphableModel(MorphableModel):
    def __init__(self, path_to_hdf5_file: Union[str, Path], use_torch: bool = False, device: str = "cpu",
                 double_precision: bool = False):
        params = self.load_params_from_hdf5_to_dict(path_to_hdf5_file)
        self.initialize_torch(use_torch=use_torch, device=device, double_precision=double_precision)

        self.latent_mean = self.get_array(params['latent_mean'], copy=False, dtype=self.float_dtype)
        self.latent_std = self.get_array(params['latent_std'], copy=False, dtype=self.float_dtype)
        colors = params.get("colors", None)
        if colors is None:
            self.colors = None
        else:
            self.colors = self.get_array(colors, copy=False, dtype=self.float_dtype)

        # Load encoder(s)
        if "encoder" in params:
            self.encoders = [EncoderDecoder(
                params["encoder"]["matrix_mults"], params["encoder"]["biases"],
                relu_factor=params["encoder"].get("relu_factor", 0.2), use_torch=use_torch, device=device)]
        else:
            self.encoders = [EncoderDecoder(
                params[encoder_type]["matrix_mults"], params[encoder_type]["biases"],
                relu_factor=params[encoder_type].get("relu_factor", 0.2), use_torch=use_torch, device=device)
                for encoder_type in ["id_encoder", "exp_encoder", "age_encoder"]]
        # sanity check dimensions
        self.dims_per_encoder = [encoder.get_latent_dimension(is_encoder=True) for encoder in self.encoders]
        if not sum(self.dims_per_encoder) == len(self.latent_mean):
            raise AssertionError("The size of the latent vector should match "
                                 "the sum of the individual encoder output sizes.")
        self.is_disentangled = len(self.encoders) > 1

        # Load decoder
        self.decoder = EncoderDecoder(
            params["decoder"]["matrix_mults"], params["decoder"]["biases"],
            relu_factor=params["decoder"].get("relu_factor", 0.2), use_torch=use_torch, device=device)
        if not self.decoder.get_latent_dimension(is_encoder=False) == len(self.latent_mean):
            raise AssertionError("The size of the latent vector should match "
                                 "the decoder's input size.")

        self.mean_vertices = self.get_array(params["mean_vertices"], copy=False, dtype=self.float_dtype)
        self.triangles = self.get_array(params["triangles"], copy=False, dtype=self.int_dtype)

        self.indices = params["indices"] if "indices" in params else None

        super().__init__()

    def encode(self, vertices: ArrayLike, unknown_vertex_mask: np.ndarray = None, invert_mask: bool = False,
               sigma: float = 0, num_components: Union[int, None] = None) -> ArrayLike:
        if sigma > 0:
            warnings.warn("Autoencoder does not support making unknown vertex filling more average."
                          "This was probably meant for a PCA model?")
        if num_components is not None:
            warnings.warn("Autoencoder does not support encoding with only some components. "
                          "This was probably meant for a PCA model?")
        vertices_flattened = (self.get_array(vertices, copy=False, dtype=self.float_dtype) - self.mean_vertices).flatten()
        encoded_list = [encoder.encode(vertices_flattened) for encoder in self.encoders]
        if self.use_torch:
            return torch.cat(encoded_list)
        else:
            return np.concatenate(encoded_list)

    def decode(self, latent: ArrayLike) -> ArrayLike:
        decoded_flat = self.decoder.decode(latent)
        return decoded_flat.reshape((*decoded_flat.shape[:-1], -1, 3)) + self.mean_vertices

    def get_exp_dims(self) -> Tuple[int, int]:
        if not self.is_disentangled:
            raise AssertionError("The current autoencoder does not have an expression dimension")
        exp_start = self.dims_per_encoder[0]
        exp_end = self.dims_per_encoder[0] + self.dims_per_encoder[1]
        return exp_start, exp_end

    def set_expression(self, mesh_to_set_expression_for: Mesh, latent_exp_code: ArrayLike) -> Mesh:
        if not self.is_disentangled:
            raise AssertionError("The current autoencoder does not support disentanglement.")
        mesh_latent = self.encode(mesh_to_set_expression_for.get_vertices())
        exp_start, exp_end = self.get_exp_dims()
        if len(latent_exp_code) == self.get_number_of_components():
            latent_exp_code = latent_exp_code[exp_start:exp_end]
        mesh_latent[exp_start:exp_end] = latent_exp_code
        return self.get_mesh_from_vertices(self.decode(mesh_latent))

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
        return self.get_mesh_from_vertices(self.decode(mesh_latent))

    def visualize(self, min_val: float = -3, max_val: float = 3,
                  num_components_per_disentangled_latent: int = 5, **kwargs):
        """
        Supports disentangled visualization by providing sliders for each of the disentangled spaces.
        Cf. docs for visualize() method in MorphableModel superclass.
        """
        shape_components = []
        start_index = 0
        for num_encoder, encoder in enumerate(self.encoders):
            latent_dim = encoder.get_latent_dimension(is_encoder=True)
            shape_components.extend(list(range(start_index, start_index + min(num_components_per_disentangled_latent, latent_dim))))
            start_index += latent_dim

        super().visualize(min_val=min_val, max_val=max_val, shape_components=shape_components, **kwargs)


if __name__ == '__main__':
    main()
