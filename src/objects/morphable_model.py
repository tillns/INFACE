"""
Abstract morphable model class, to be inherited by PCA and autoencoder classes.
Basic functionality includes:
1) encode an input array into the corresponding model's latent space (the input array is the flattened array of
   vertices that are already in the correct correspondence). Unknown vertices to be replaced can be marked with a mask.
2) decode a latent to an output array (the output array is the flattened array of vertices, which form a mesh
   together with the triangle attribute.
3) Reconstruct a mesh that is possibly incomplete. Optionally only replace the missing regions while keeping the
   other regions unchanged. The mesh needs to be in correct correspondence.
4) Visualize the model by providing a set of sliders to change the model's individual components,
   yielding various output meshes. Each mesh instance can be viewed in the visualizer and optionally saved.

Visualization requires open3d.
Reconstruction with back matching requires scipy, and sksparse, potentially also fbpca.

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
from argparse import ArgumentParser
from functools import partial
from pathlib import Path
from typing import Union, Tuple, List, Dict, TYPE_CHECKING
import h5py
import numpy as np
from abc import ABC, abstractmethod

from src.objects.indices_and_masks import IndicesAndMasks
from src.objects.linear_regressor import LinearRegressor
from src.objects.mesh import Mesh, Landmarks
from models import get_model_path

try:
    import torch
except ImportError:
    torch = None
if TYPE_CHECKING:
    from torch import Tensor
ArrayLike = Union[np.ndarray, "Tensor"]


class MorphableModel(ABC):
    def __init__(self):
        """
        Inheriting class should accept some path to load its weights, and additionally
        initialize the latent_mean, latent_std, and triangles attributes (possibly also from file)
        """
        if not hasattr(self, "latent_mean") or not hasattr(self, "latent_std") or not hasattr(self, "triangles"):
            raise TypeError("Please define in your inheriting class the attributes latent_mean and latent_std "
                            "(numpy arrays of the same size defining the range of the trained morphable model's"
                            "latent space), as well as the numpy triangle array that defines the connectivity"
                            "between the vertices.")
        use_torch = getattr(self, "use_torch", False)
        device = getattr(self, "device", "cpu")
        double_precision = use_torch and hasattr(self, "float_dtype") and self.float_dtype == torch.float64
        self.initialize_torch(use_torch=use_torch, device=device, double_precision=double_precision)

        # Define attributes again here (this is purely for IDE recommendation purposes; they are already defined)
        self.latent_mean: ArrayLike = self.get_array(getattr(self, "latent_mean"))
        self.latent_std: ArrayLike = self.get_array(getattr(self, "latent_std"))
        if not np.array_equal(self.latent_mean.shape, self.latent_std.shape):
            raise TypeError("Latent space dimensions for mean and standard deviation do not match.")
        self.triangles: ArrayLike = self.get_array(getattr(self, "triangles"))
        # Indices are kept as numpy arrays. because torch also supports indexing via lists and numpy arrays
        self.indices: Union[Dict[str, np.ndarray], None] = getattr(self, "indices", None)
        self.colors: Union[ArrayLike, None] = getattr(self, "colors", None)

    def initialize_torch(self, use_torch: bool = False, device: str = "cpu", double_precision: bool = False) -> None:
        self.use_torch = use_torch
        self.device = device
        if self.use_torch:
            if torch is None:
                raise RuntimeError(
                    "use_torch=True but PyTorch is not installed. "
                    "Please install torch or set use_torch=False."
                )
            self.float_dtype = torch.float64 if double_precision else torch.float32
            self.int_dtype = torch.int64
        else:
            self.float_dtype = float
            self.int_dtype = int

    def get_array(self, some_list_or_array: Union[List, ArrayLike], copy: bool = False, convert_numpy: bool = False,
                  dtype=None) -> ArrayLike:

        # ---- TORCH OUTPUT PATH ----
        if self.use_torch and not convert_numpy:
            # as_tensor: avoids copy when possible
            t = torch.as_tensor(some_list_or_array, device=self.device, dtype=dtype)
            # clone if caller explicitly wants a copy
            return t.clone() if copy else t

        if torch is not None and isinstance(some_list_or_array, torch.Tensor):
            return some_list_or_array.detach().cpu().numpy()

        return np.array(some_list_or_array) if copy else np.asarray(some_list_or_array)

    def normalize_weights(self, latent_weights: ArrayLike) -> ArrayLike:
        """
        Normalize the given weights, such that they are scaled relative to the latent distribution of the
        morphable model. The entries then represent how many standard deviations each value is away from the mean.
        :param latent_weights: Unnormalized weights. A vector that's shorter than the model's components is supported;
                               we simply shorten the latent mean and stdev vectors by cutting of the end. Note that
                               this probably doesn't make sense for an autoencoder. If the vector is too long, it will
                               raise an assertion error.
        :return: Numpy array with normalized weights. Same length as latent_weights.
        """
        latent_weights_array = self.get_array(latent_weights, copy=False, dtype=self.float_dtype)
        latent_weights_dim = latent_weights_array.shape[-1]
        assert latent_weights_dim <= len(self.latent_mean)
        return (latent_weights_array - self.latent_mean[:latent_weights_dim]) / self.latent_std[:latent_weights_dim]

    def unnormalize_weights(self, latent_weights: ArrayLike) -> ArrayLike:
        """
        Unnormalize the given weights, such that they are absolute, not relatively scaled. This method scales the
        weights by the standard deviation and adds the mean.
        :param latent_weights: Normalized weights, representing how many standard deviations each value is away from
                               the mean. A vector that's shorter than the model's components is supported;
                               we simply shorten the latent mean and stdev vectors by cutting of the end. Note that
                               this probably doesn't make sense for an autoencoder. If the vector is too long, it will
                               raise an assertion error.
        :return: Numpy array with unnormalized weights. Same length as latent_weights.
        """
        latent_weights_array = self.get_array(latent_weights, copy=False, dtype=self.float_dtype)
        latent_weights_dim = latent_weights_array.shape[-1]
        assert latent_weights_dim <= len(self.latent_mean)
        return latent_weights_array * self.latent_std[:latent_weights_dim] + self.latent_mean[:latent_weights_dim]

    def get_triangles(self) -> ArrayLike:
        return self.triangles

    def get_faces(self) -> ArrayLike:
        return self.get_triangles()

    def has_colors(self) -> bool:
        return self.colors is not None

    def get_colors(self) -> Union[ArrayLike, None]:
        return self.colors


    def fit_to_landmarks(self, landmarks_or_point_array_or_path: Union[Landmarks, np.ndarray, Union[str, Path]],
                         source_vertex_indices: np.ndarray, **kwargs) -> Tuple[Mesh, np.ndarray]:
        """
        Fit morphable model to target landmarks. This is basically a vertex reconstruction from a relatively
        sparse set of points. Currently no torch support.
        :param landmarks_or_point_array_or_path: Landmarks to fit model to (or point array or path to landmarks).
        :param source_vertex_indices: Vertex indices of model that correspond to the landmarks. Don't provide a mask,
                                      as this does not preserve the order! (An assertion error is raised in that case).
        :param kwargs: Any additional arguments passed to the reconstruct_data method
                       (e.g. sigma or num_components for pca model)
        :return: Tuple: 1) Reconstructed mesh of type Mesh 2) latent representation (numpy array)
        """
        if IndicesAndMasks.is_mask(source_vertex_indices):
            raise AssertionError("You provided the indices corresponding to the landmarks via a boolean mask. "
                          "Note that boolean masks do not preserve the order, so I would really guess that you made "
                          "a mistake here. Provide an index array, where for each landmark, you have the corresponding "
                          "source vertex index (multiple possible, even though it's probably not ideal ^^)")
        if IndicesAndMasks.contains_duplicates(source_vertex_indices):
            warnings.warn("You provided source vertex indices that are not unique. "
                          "If this is intentional, you can ignore this warning. ")
        landmarks = Landmarks(landmarks_or_point_array_or_path)
        assert IndicesAndMasks.get_num_indexed(source_vertex_indices, count_only_unique=False) == landmarks.get_num_points()
        # Sample average vertices
        average_mesh = self.get_average_mesh()
        # Replace indexed vertices with landmark points
        average_mesh.vertices[source_vertex_indices] = landmarks.get_points()
        # Reconstruct all vertices from these
        reconstructed_mesh, latent_representation = self.reconstruct_mesh(
            average_mesh, unknown_vertex_mask=source_vertex_indices, invert_mask=True,
            return_latent=True, **kwargs)
        return reconstructed_mesh, latent_representation


    @staticmethod
    def load_correct_morphable_model(
            path_to_hdf5_file: Union[str, Path], use_torch: bool = False, device: str = "cpu",
            double_precision: bool = False) -> "MorphableModel":
        # local imports to avoid circular imports at top level
        from src.objects.AE_morphable_model import AEMorphableModel
        from src.objects.PCA_morphable_model import PCAMorphableModel
        params = MorphableModel.load_params_from_hdf5_to_dict(path_to_hdf5_file)
        if "eigenvalues" in params:
            return PCAMorphableModel(path_to_hdf5_file, use_torch=use_torch, device=device,
                                     double_precision=double_precision)
        else:
            return AEMorphableModel(path_to_hdf5_file, use_torch=use_torch, device=device,
                                    double_precision=double_precision)

    @staticmethod
    def load_params_from_hdf5_to_dict(path_to_hdf5_file: Union[str, Path]) -> dict:
        def load_hdf5_to_dict(hdf5_group) -> dict:
            """
            Recursively load an HDF5 group into a dictionary.
            """
            data = {}
            for key, item in hdf5_group.items():
                if isinstance(item, h5py.Group):
                    # If it's a group, recurse into it
                    if all(subkey.startswith("item_") for subkey in item.keys()):
                        # If all keys in the group are like "item_0", treat it as a list
                        data[key] = [item[f"item_{i}"][:] for i in range(len(item))]
                    else:
                        # Otherwise, treat it as a nested dictionary
                        data[key] = load_hdf5_to_dict(item)
                else:
                    # If it's a dataset, read the data (array or scalar)
                    data[key] = item[()]
            return data

        with h5py.File(str(get_model_path(path_to_hdf5_file)), "r") as f:
            params = load_hdf5_to_dict(f)
        return params

    @staticmethod
    def save_params_to_hdf5_dict(params: Dict, path_to_hdf5_file: Union[str, Path]) -> None:

        def get_numpy_array(some_list_or_array):
            """
            Convert list, torch tensor or numpy array to numpy array (copied).
            """
            if torch is not None and isinstance(some_list_or_array, torch.Tensor):
                return some_list_or_array.detach().cpu().numpy()
            else:
                return np.array(some_list_or_array)


        def save_dict_to_hdf5(data, hdf5_group):
            """
            Recursively save a dictionary to an HDF5 group.
            """
            for key, value in data.items():
                if isinstance(value, dict):
                    # Create a new group for dictionaries
                    subgroup = hdf5_group.create_group(key)
                    save_dict_to_hdf5(value, subgroup)
                elif isinstance(value, list):
                    # Handle lists by creating datasets within a group
                    list_group = hdf5_group.create_group(key)
                    for i, item in enumerate(value):
                        list_group.create_dataset(f"item_{i}", data=get_numpy_array(item))
                else:
                    # Save arrays directly
                    hdf5_group.create_dataset(key, data=get_numpy_array(value))

        with h5py.File(path_to_hdf5_file, "w") as f:
            save_dict_to_hdf5(params, f)

    def convert_mesh_with_linear_regressor(
            self, linear_regressor: Union[LinearRegressor, str, Path], input_mesh: Mesh,
            output_morphable_model: "MorphableModel" = None,
            unknown_vertex_mask: np.ndarray = None, invert_mask: bool = False,
            include_colors_if_available: bool = False) -> Union[np.ndarray, Mesh]:
        """
        Convert mesh from current morphable model to a mesh from another morphable model using a linear regressor.
        Currently no torch support.
        :param linear_regressor: The linear regressor to convert the mesh latent code with (can also be path to json)
        :param input_mesh: The mesh from the current morphable model to be converted.
        :param output_morphable_model: (Optional) The other morphable model to go from mesh latent to actual mesh.
                                       If not provided, mesh latent is returned.
        :param unknown_vertex_mask: (Optional) cf. encode() function.
        :param invert_mask: (Optional) cf. encode() function.
        :param include_colors_if_available: (Optional) Set True to include colors in the translated mesh
                                            if the respective morphable model has colors.
        :return: If no output_morphable_model was provided, the translated latent code is returned as numpy array.
                 If it is, the actual translated mesh is returned.

        """
        if not isinstance(linear_regressor, LinearRegressor):
            linear_regressor = LinearRegressor.from_json(linear_regressor)
        mesh_latent = self.encode(input_mesh.get_vertices(), unknown_vertex_mask=unknown_vertex_mask,
                                  invert_mask=invert_mask)
        translated_latent = linear_regressor.translate_latent_code(mesh_latent)
        if isinstance(output_morphable_model, MorphableModel):
            translated_vertices = output_morphable_model.decode(translated_latent)
            return output_morphable_model.get_mesh_from_vertices(
                translated_vertices, include_colors_if_available=include_colors_if_available)
        else:
            if not output_morphable_model is None:
                raise TypeError(f"Unknown type for output_morphable_model: {output_morphable_model}")
            return translated_latent

    @staticmethod
    def get_unknown_vertex_mask(unknown_vertex_mask: np.ndarray, vertices: np.ndarray,
                                invert_mask: bool = False) -> np.ndarray:
        """
        Converts the unknown vertex mask into a boolean format, optionally inverting it.
        """
        if unknown_vertex_mask is None:
            if invert_mask:
                raise AssertionError("You provided an empty mask to be inverted?")
            unknown_vertex_mask = np.zeros(len(vertices)).astype(bool)
        unknown_vertex_mask = np.asarray(unknown_vertex_mask)
        if np.issubdtype(unknown_vertex_mask.dtype, np.integer):
            unknown_vertex_mask_copy = np.zeros(len(vertices)).astype(bool)
            unknown_vertex_mask_copy[unknown_vertex_mask] = True
            unknown_vertex_mask = unknown_vertex_mask_copy
        if invert_mask:
            unknown_vertex_mask = np.invert(unknown_vertex_mask)
        if np.all(unknown_vertex_mask):
            raise AssertionError("The unknown vertex mask contains all vertices.")
        return unknown_vertex_mask


    @abstractmethod
    def encode(self, vertices: ArrayLike, unknown_vertex_mask: np.ndarray = None, invert_mask: bool = False,
               sigma: float = 0, num_components: Union[int, None] = None) -> ArrayLike:
        """
        Return latent code corresponding to the vertices given as input to this method.
        To be overridden by inheriting classes (use get_unknown_vertex_mask() method to convert vertex mask correctly)
        Note that currently, this method does not accept more than one set of vertices due to compatibility with
        the PCA encode method. Batch processing may be added upon request.
        :param vertices: nx3 numpy/torch array of vertices.
        :param unknown_vertex_mask: Optionally provide a numpy mask array indicating which of the vertices
                                    inside the provided array should not be considered for encoding.
        :param invert_mask: If your array for unknown_vertex_mask actually contains the vertices that should be
                            considered, use this argument to invert the array.
        :param sigma: (Optional) Parameter to control how average unknown vertices will be filled in.
                      Default is 0, which is as individual as possible, so without additional averaging.
        :param num_components: (Optional) Only use a specified number of components to encode the mesh with.
                               This is mainly targeted for PCA models, since they have sorted directions,
                               so you can avoid overfitting to an input by only using a limited number of
                               components to encode it.
        :return: latent code representing the input vertices in the morphable model's compact space.
                 Should be a 1D numpy/torch array of the same shape as latent_mean and latent_std.
        """
        pass

    @abstractmethod
    def decode(self, latent: ArrayLike) -> ArrayLike:
        """
        Given one or multiple latent vector reconstruct the vertices using the morphable model.
        Note that if you normalized the weights, you need to unnormalize them before
        giving them to this method.
        :param latent: m or bxm numpy array with m equal to the length of latent_mean and latent_std.
        :return: nx3 or bxnx3 numpy array of vertices.
        """
        pass

    def get_projected_vertices(self, *registered_meshes: Mesh, normalize_projection: bool = False) -> np.ndarray:
        """
        Project multiple registered meshes into 3DMM space.
        :param registered_meshes: list of registered meshes (Mesh type)
        :param normalize_projection: (Optional) Set True to normalize the projections
                                     (0 mean and unit stdev, cf. normalize_weights() function)
        :return np array of projected vertices [num_meshes x 3DMM_latent_size], optionally normalized
        """
        registered_vertices = np.asarray([
            registered_mesh.get_vertices() for registered_mesh in registered_meshes])

        projections = np.asarray([
            self.encode(registered_vertices_ind) for registered_vertices_ind in registered_vertices])
        if normalize_projection:
            projections = self.normalize_weights(projections)
        return projections

    def get_mesh_from_vertices(
            self, vertices: ArrayLike, include_colors_if_available: bool = False) -> Mesh:
        """
        Return triangle mesh given vertices (basically just create a Mesh instance with the provided vertices
        and the model's triangles, optionally also with colors).
        :param vertices: nx3 numpy/torch array of vertices.
        :param include_colors_if_available: (Optional) Set True to include colors in the translated mesh
                                    if the respective morphable model has colors.
        :return: Triangle mesh using provided vertices, optionally also with colors (if available).
        """
        mesh = Mesh(vertices=self.get_array(vertices, copy=False, convert_numpy=True),
                    triangles=self.get_array(self.triangles, copy=False, convert_numpy=True))
        if self.has_colors() and include_colors_if_available:
            mesh.set_colors(self.get_colors())
        return mesh

    def get_number_of_components(self) -> int:
        return len(self.latent_mean)

    def get_number_of_vertices(self) -> int:
        if self.use_torch:
            return int(torch.max(self.triangles)) + 1
        else:
            return int(np.max(self.triangles)) + 1

    def sample(self, deviation_factor: float = 1) -> ArrayLike:
        """
        Sample model output (vertices).
        :param deviation_factor: How far to deviate from average during sampling relative to standard deviation.
        :return: A set of vertices, randomly sampled by the morphable model.
        """
        if self.use_torch:
            normal_func = torch.normal
        else:
            normal_func = np.random.normal
        return self.decode(normal_func(self.latent_mean, deviation_factor*self.latent_std))

    def sample_multiple(self, num_samples: int, deviation_factor: float = 1) -> List[ArrayLike]:
        """
        Sample multiple model outputs (vertices).
        :param num_samples: Number of vertex sets to be randomly sampled (int).
        :param deviation_factor: How far to deviate from average during sampling relative to standard deviation.
        :return: list of randomly sampled vertices (each entry has nx3 vertices).
        """
        return [self.sample(deviation_factor=deviation_factor) for _ in range(num_samples)]

    def sample_meshes(self, num_samples: int, deviation_factor: float = 1,
                      include_colors_if_available: bool = False) -> List[Mesh]:
        """
        Sample a list of meshes with the morphable model (unlike the sample() and sample_multiple() methods,
        which only return the vertices, this method returns the actual meshes).
        Currently no torch support, since the Mesh class only works with numpy.
        :param num_samples: Number of meshes to be randomly sampled (int).
        :param deviation_factor: How far to deviate from average during sampling relative to standard deviation.
        :param include_colors_if_available: (Optional) Set True to include colors in the meshes
                                            (cf. get_mesh_from_vertices())
        :return: list of randomly sampled meshes.
        """
        sampled_vertex_sets = self.sample_multiple(num_samples, deviation_factor=deviation_factor)
        return [self.get_mesh_from_vertices(sampled_vertices, include_colors_if_available=include_colors_if_available)
                for sampled_vertices in sampled_vertex_sets]

    def get_average_mesh(self, include_colors_if_available: bool = False) -> Mesh:
        """
        Returns the average mesh of the morphable model.
        """
        average_vertices = self.decode(self.latent_mean)
        return self.get_mesh_from_vertices(average_vertices, include_colors_if_available=include_colors_if_available)

    def reconstruct_mesh(
            self, mesh: Mesh, unknown_vertex_mask: Union[None, np.ndarray] = None, invert_mask: bool = False,
            back_match_known_vertices: bool = False, repeat: int = 1, return_latent: bool = False,
            sigma: float = 0, num_components: Union[int, None] = None) -> Union[Mesh, Tuple[Mesh, np.ndarray]]:
        """
        Basically encode and decode a mesh again, optionally only using parts of the vertices,
        back-matching reconstructed mesh to these vertices also supported.
        Currently no torch support.
        :param mesh: Mesh to be reconstructed by morphable model.
        :param unknown_vertex_mask: Optionally provide a mask of vertices that are unknown. The model will reconstruct
                                    the mesh based only on the information of the remaining vertices.
        :param invert_mask: If the mask you provided specifies the known vertices instead of the unknown, set this True.
        :param back_match_known_vertices: Set True to treat the known vertices as hard constraints. The reconstructed
                                          mesh will then be back-registered to the known vertices.
        :param repeat: How many times to repeat the reconstruction (since the unknown vertices cannot always be
                       fully ignored, you could do multiple iterations of reconstructing and then back-matching.
        :param return_latent: Set True to return not only the reconstructed mesh, but also its latent projection.
        :param sigma: Regularizer on the reconstruction. 0 means no regularization. Higher values will make the
                      reconstruction more average.
        :param num_components: Number of components to use for reconstruction. None means use all components.
                               Mainly targeted at PCA models, since they have sorted components, so similar to sigma,
                               this can be used as a kind of regularizer.
        :return Either only the reconstructed Mesh, or a tuple with the reconstructed Mesh and the latent
                if return_latent=True.
        """
        mesh_vertices = mesh.get_vertices()
        unknown_vertex_mask = self.get_unknown_vertex_mask(
            unknown_vertex_mask=unknown_vertex_mask, vertices=mesh_vertices, invert_mask=invert_mask)

        def reconstruct_current_vertices(current_vertices_local):
            latent_local = self.get_array(self.encode(
                current_vertices_local, unknown_vertex_mask=unknown_vertex_mask, invert_mask=False,
                sigma=sigma, num_components=num_components), copy=False, convert_numpy=True)
            reconstructed_vertices = self.get_array(self.decode(latent_local), copy=False, convert_numpy=True)
            if back_match_known_vertices:
                reconstructed_vertices = self.back_match_data(
                    current_vertices_local, reconstructed_vertices, unknown_vertex_mask=unknown_vertex_mask)
            return reconstructed_vertices, latent_local

        current_vertices = np.array(mesh_vertices)
        latent = np.zeros(self.get_number_of_components())
        for _ in range(max(repeat, 1)):
            current_vertices, latent = reconstruct_current_vertices(current_vertices)

        mesh = self.get_mesh_from_vertices(current_vertices)
        if return_latent:
            return mesh, latent
        return mesh

    def has_indices(self) -> bool:
        return self.indices is not None

    def has_index(self, index_name: str) -> bool:
        return self.has_indices() and index_name in self.indices

    def get_indices(self, *index_names: str, join_indices: bool = True,
                    make_unique: bool = True) -> Union[np.ndarray, List[np.ndarray], Dict[str, np.ndarray]]:
        """
        Get either all or specific indices.
        :param index_names: Names of indices to return. If none provided, all indices will be returned as a dictionary.
                            Any index name not defined in the model's indices attribute is simply skipped.
        :param join_indices: Set to False to return individual index arrays per provided index_name.
        :param make_unique: If join_indices is True, whether to make the indices unique and sorted.
        :return: If no index_name provided, a dictionary of indices is returned with their respective names as keys.
                 If multiple index_names are provided and join_indices is False,,
                 a list of numpy arrays is returned; one for each index_name
                 Otherwise, a 1D numpy array is returned containing the (joined) indices for all provided index_names.
        """
        if len(index_names) == 0:
            return self.indices
        else:
            if not self.has_indices():
                raise AssertionError("Current model doesn't have any defined indices.")
            indices = [self.indices[idx_name] for idx_name in index_names if self.has_index(idx_name)]
            if any([indices_ind.ndim > 1 for indices_ind in indices]):
                # IndicesAndMasks only works with 1D arrays, so we handle this case specifically here.
                if join_indices and len(index_names) > 1:
                    if all([indices[0].ndim == indices_ind.ndim for indices_ind in indices[1:]]):
                        indices = np.concatenate(indices, axis=0)
                    else:
                        warnings.warn("You chose multiple index names, but the respective index arrays have "
                                      "different dimensionality, so they cannot be joined. Please choose "
                                      "either only indices of the same dimensionality or set join_indices to False.")
                join_indices = False
            if join_indices:
                indices = IndicesAndMasks.join(*indices, make_unique=make_unique)
            elif len(index_names) == 1:
                indices = np.asarray(indices[0])
            return indices

    #
    #
    # Partial credit for implementation of this method goes to Defne Kurtulus
    def visualize(self, min_val: float = -3, max_val: float = 3,
                  shape_components: Union[List[int], int] = 5,
                  linear_regressors: Union[LinearRegressor, List[LinearRegressor]] = None,
                  translated_morphable_models: Union["MorphableModel", List["MorphableModel"]] = None,):
        """
        Visualize a morphable model via sliders that each represent an adjustable component of the morphable model.
        The visualized mesh represents the model's generation based on the sliders' state.
        Note that this visualizer is generally more meaningful for the PCA model, since adjustment of individual
        components is less interpretable in the autoencoder.
        :param min_val: Maximum negative deviation from the model's mean for each
                        component relative to the standard deviation.
        :param max_val: Maximum positive deviation from the model's mean for each
                        component relative to the standard deviation.
        :param shape_components: Which components to show. Can be an int, then the first so many components
                                 can be adjusted via sliders, but can also be a list of component indices,
                                 e.g., [0, 5] means that the model's first and sixth component can be adjusted via
                                 indices.
        :param linear_regressors: Optionally provide one or more linear regressors to translate the
                                  morphable model latent to that of other morphable models.
                                  This requires you to also provide translated morphable models.
        :param translated_morphable_models: Other morphable models that, in combination with the linear regressors,
                                            also give you the meshes from the models' correlated spaces.

        :return: None
        """
        try:
            import open3d
        except ImportError:
            print("Install open3d for visualization.")
            return

        # Number of shape components
        n_components = self.get_number_of_components()

        current_weights = np.zeros(n_components)


        def get_current_vertices() -> np.ndarray:
            return self.decode(self.latent_mean + current_weights*self.latent_std)

        def get_current_mesh() -> Mesh:
            return self.get_mesh_from_vertices(
                get_current_vertices(), include_colors_if_available=True)

        open3d.visualization.gui.Application.instance.initialize()
        w = open3d.visualization.gui.Application.instance.create_window("3DMM GUI", 1920, 1080)
        em = w.theme.font_size
        spacing = int(np.round(0.5 * em))

        layout = open3d.visualization.gui.Vert(spacing, open3d.visualization.gui.Margins(
            0.5 * em, 0.5 * em, 0.5 * em, 0.5 * em))
        layout.frame = open3d.visualization.gui.Rect(
            w.content_rect.x, w.content_rect.y, 500, w.content_rect.height)

        _widget3d = open3d.visualization.gui.SceneWidget()
        _widget3d.scene = open3d.visualization.rendering.Open3DScene(w.renderer)
        _widget3d.set_view_controls(open3d.visualization.gui.SceneWidget.Controls.ROTATE_CAMERA)
        _widget3d.frame = open3d.visualization.gui.Rect(
            500, w.content_rect.y, w.content_rect.width, w.content_rect.height)
        _widget3d.scene.set_background([0, 0, 0, 0])

        material = open3d.visualization.rendering.MaterialRecord()
        material.shader = "defaultLit"
        material.base_color = [0.5, 0.5, 0.5, 1.0]

        if not isinstance(linear_regressors, list):
            linear_regressors = [] if linear_regressors is None else [linear_regressors]
        if not isinstance(translated_morphable_models, list):
            translated_morphable_models = [] if translated_morphable_models is None else [translated_morphable_models]

        assert len(linear_regressors) == len(translated_morphable_models)

        def update_mesh():
            mesh = get_current_mesh()
            _widget3d.scene.clear_geometry()
            _widget3d.scene.add_geometry(f'mesh', mesh.get_open3d_mesh(compute_normals=True), material)
            for linear_regressor, translated_morphable_model in zip(linear_regressors, translated_morphable_models):
                translated_mesh = self.convert_mesh_with_linear_regressor(
                    linear_regressor=linear_regressor, input_mesh=mesh,
                    output_morphable_model=translated_morphable_model,
                    include_colors_if_available=True)
                _widget3d.scene.add_geometry(f'translated_mesh', translated_mesh.get_open3d_mesh(compute_normals=True),
                                             material)

        def _on_mouse_widget3d(event):
            return open3d.visualization.gui.Widget.EventCallbackResult.IGNORED

        update_mesh()

        # look_at(center, eye, up): sets the camera view so that the camera is located at ‘eye’,
        # pointing towards ‘center’, and oriented so that the up vector is ‘up’
        current_vertices = get_current_vertices()
        vertex_range = np.max(current_vertices) - np.min(current_vertices)
        mesh_center = np.mean(current_vertices, axis=0)
        _widget3d.scene.camera.look_at(mesh_center, mesh_center + np.asarray([0, 0, 2*vertex_range]), [0, 1, 0])
        _widget3d.set_on_mouse(_on_mouse_widget3d)

        def update_mesh_weights(new_weights):
            current_weights[:] = np.asarray(new_weights)
            update_mesh()

        def update_mesh_component(n, val):
            current_weights[n] = val
            update_mesh()

        def export_current_mesh():
            try:
                import tkinter.filedialog
                import tkinter as tk
            except ImportError:
                print("Install tkinter to choose path for exporting mesh.")
                return
            root = tk.Tk()
            root.withdraw()  # Hide the main window
            file_path = tk.filedialog.asksaveasfilename(
                filetypes=[("OBJ files", "*.obj"), ("PLY files", "*.ply"), ("STL files", "*.stl"),
                           ("JSON files", "*.json")])
            if file_path:
                get_current_mesh().export(file_path)

        export_button = open3d.visualization.gui.Button(f"Export Mesh")
        export_button.set_on_clicked(export_current_mesh)
        layout.add_child(export_button)

        shape_components = list(range(shape_components)) if isinstance(shape_components, int) else shape_components
        sliders = []

        def reset_sliders():
            for _, slider in sliders:
                slider.double_value = 0
            update_mesh_weights(0)


        reset_button = open3d.visualization.gui.Button(f"Reset shape components")
        reset_button.set_on_clicked(reset_sliders)
        layout.add_child(reset_button)

        # Loop over components to add slider for
        for comp in shape_components:
            if comp >= len(current_weights):
                continue
            slider = open3d.visualization.gui.Slider(open3d.visualization.gui.Slider.DOUBLE)
            current_val = current_weights[comp]
            slider.set_limits(min(min_val, current_val), max(max_val, current_val))
            slider.double_value = current_val
            slider.set_on_value_changed(partial(update_mesh_component, comp))
            layout.add_child(open3d.visualization.gui.Label(f"{comp + 1}:"))
            layout.add_child(slider)
            sliders.append((comp, slider))

        # Start visualization
        w.add_child(layout)
        w.add_child(_widget3d)
        open3d.visualization.gui.Application.instance.run()

    def back_match_data(self, source_vertices: np.ndarray, reconstructed_vertices: np.ndarray,
                        unknown_vertex_mask: np.ndarray) -> np.ndarray:
        """
        Match reconstructed vertices back to original source vertices in known regions while deforming
        unknown regions along as-rigid-as-possibly.
        Currently no torch support
        :param source_vertices: Source vertices to back-match (num_el x 3)
        :param reconstructed_vertices: Reconstruction of the source_data vector (must be same shape).
        :param unknown_vertex_mask: Specify a mask of vertices that are unknown and, thus, require statistical
                                    reconstruction.
        :return: back-matched vertices (np array of shape (num_el x 3))
        """
        try:
            import scipy.sparse as spsp
            from scipy.sparse import diags
            from sksparse.cholmod import cholesky_AAt
        except ImportError:
            warnings.warn("Some more libraries are required to use the back matching feature. Install scipy, "
                          "and sksparse.")
            return reconstructed_vertices

        if not np.any(unknown_vertex_mask):
            warnings.warn("No vertices are marked as unknown, meaning you will just get the original vertices "
                          "out of this method. Reconsider the usage of this method.")
            return source_vertices

        #
        #
        # Credit goes to Xiaojing Xia for implementing the following method based on the work
        # "Statistically Motivated 3D Faces Reconstruction" from Basso and Vetter (2006)

        # Connection matrix (which vertex is connected to which as defined by triangles)
        adjacency_matrix = self.get_average_mesh().get_adjacency_matrix()
        adj_count = np.sum(adjacency_matrix, axis=1)
        K = adjacency_matrix.multiply(spsp.csc_matrix(1 / adj_count.reshape((-1, 1))))
        # Convert to LIL format for efficient diagonal modification
        K_lil = K.tolil()
        K_lil.setdiag(-1)

        # Convert back to CSC format
        K = K_lil.tocsc()

        # Construct Lambda matrix
        lambda_matrix = np.ones(len(source_vertices))
        lambda_matrix[unknown_vertex_mask] = 0
        D = spsp.csc_matrix(diags(lambda_matrix))

        # Construct P_star matrix
        P_star = np.array(source_vertices)
        P_star[unknown_vertex_mask] = 0

        # Get H_star matrix to solve for H
        H_star = P_star - reconstructed_vertices

        # Solve a sparse linear system
        I = spsp.identity(D.shape[0])
        A = D + (I - D) @ K
        b = D @ H_star

        factor = cholesky_AAt(A.T.tocsc())
        H = factor(A.T.dot(b))

        # Return reconstructed data
        reconstruction_adjusted = reconstructed_vertices + H
        output = np.array(source_vertices)
        output[unknown_vertex_mask] = reconstruction_adjusted[unknown_vertex_mask]

        return output
        #
        #
        # End credit


def main():
    parser = ArgumentParser()
    parser.add_argument('--path_to_hdf5_file', type=str, required=True,
                        help="Absolute path to autoencoder or PCA model file with .h5 ending.")
    parser.add_argument('--path_to_linear_regressor', type=str, default=None,
                        help="Absolute path to linear regressor file with .json ending.")
    parser.add_argument('--path_to_other_hdf5_file', type=str, default=None,
                        help="Absolute path to another model to translate to via linear regressor.")
    parser.add_argument("--visualize_with_blender", default=False, action="store_true",
                        help="By default, Open3D's visualizer will be used. Choose this to use Blender instead.")
    args = parser.parse_args()
    if args.visualize_with_blender:
        from src.blender.python_interface import visualize_morphable_model
        visualize_morphable_model(args.path_to_hdf5_file, linear_regressors=args.path_to_linear_regressor,
                                  translated_morphable_models=args.path_to_other_hdf5_file)
    else:
        model = MorphableModel.load_correct_morphable_model(args.path_to_hdf5_file)
        if args.path_to_other_hdf5_file is not None and args.path_to_linear_regressor is not None:
            model.visualize(linear_regressors=LinearRegressor.from_json(args.path_to_linear_regressor),
                            translated_morphable_models=model.load_correct_morphable_model(args.path_to_other_hdf5_file))
        else:
            model.visualize()


if __name__ == '__main__':
    main()
