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
Reconstruction with back matching requires scipy, igl, and sksparse, potentially also fbpca.

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

import warnings
from functools import partial
import h5py
import numpy as np
from abc import ABC, abstractmethod
from mesh import Mesh
from typing import Union, Tuple, List


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

        # Define attributes again here (this is purely for IDE recommendation purposes; they are already defined)
        self.latent_mean = np.asarray(getattr(self, "latent_mean"))
        self.latent_std = np.asarray(getattr(self, "latent_std"))
        if not np.array_equal(self.latent_mean.shape, self.latent_std.shape):
            raise TypeError("Latent space dimensions for mean and standard deviation do not match.")
        self.triangles = getattr(self, "triangles")

    @staticmethod
    def load_params_from_hdf5_to_dict(path_to_hdf5_file: str) -> dict:
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
                    # If it's a dataset, read the data
                    data[key] = item[:]
            return data

        with h5py.File(path_to_hdf5_file, "r") as f:
            params = load_hdf5_to_dict(f)
        return params

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
    def encode(self, vertices: np.ndarray, unknown_vertex_mask: np.ndarray = None, invert_mask: bool = False) -> np.ndarray:
        """
        Return latent code corresponding to the vertices given as input to this method.
        To be overridden by inheriting classes (use get_unknown_vertex_mask() method to convert vertex mask correctly)
        :param vertices: nx3 numpy array of vertices.
        :param unknown_vertex_mask: Optionally provide a numpy mask array indicating which of the vertices
                                    inside the provided array should not be considered for encoding.
        :param invert_mask: If your array for unknown_vertex_mask actually contains the vertices that should be
                            considered, use this argument to invert the array.
        :return: latent code representing the input vertices in the morphable model's compact space.
                 Should be a 1D numpy array of the same shape as latent_mean and latent_std.
        """
        pass

    @abstractmethod
    def decode(self, latent: np.ndarray) -> np.ndarray:
        """
        Given 1D latent vector reconstruct the vertices using the morphable model.
        :param latent: 1D numpy array of the same shape as latent_mean and latent_std.
        :return: nx3 numpy array of vertices.
        """
        pass

    def get_number_of_components(self) -> int:
        return len(self.latent_mean)

    def get_number_of_vertices(self) -> int:
        return np.max(self.triangles) + 1

    def sample(self, deviation_factor: float = 1) -> np.ndarray:
        """
        Sample model output (vertices).
        :param deviation_factor: How far to deviate from average during sampling relative to standard deviation.
        :return: A set of vertices, randomly sampled by the morphable model.
        """
        return self.decode(np.random.normal(self.latent_mean, deviation_factor*self.latent_std))

    def sample_multiple(self, num_samples: int, deviation_factor: float = 1) -> List[np.ndarray]:
        """
        Sample multiple model outputs (vertices).
        :param num_samples: Number of vertex sets to be randomly sampled (int).
        :param deviation_factor: How far to deviate from average during sampling relative to standard deviation.
        :return: list of randomly sampled vertices (each entry has nx3 vertices).
        """
        return [self.sample(deviation_factor=deviation_factor) for _ in range(num_samples)]

    def sample_meshes(self, num_samples: int, deviation_factor: float = 1) -> List[Mesh]:
        """
        Sample a list of meshes with the morphable model (unlike the sample() and sample_multiple() methods,
        which only return the vertices, this method returns the actual meshes).
        :param num_samples: Number of meshes to be randomly sampled (int).
        :param deviation_factor: How far to deviate from average during sampling relative to standard deviation.
        :return: list of randomly sampled meshes.
        """
        sampled_vertex_sets = self.sample_multiple(num_samples, deviation_factor=deviation_factor)
        return [Mesh(sampled_vertices, self.triangles) for sampled_vertices in sampled_vertex_sets]

    def get_average_mesh(self):
        average_vertices = self.decode(self.latent_mean)
        return Mesh(average_vertices, self.triangles)

    def reconstruct_mesh(self, mesh: Mesh, unknown_vertex_mask: np.ndarray = None, invert_mask: bool = False,
                         back_match_known_vertices: bool = False, repeat: int = 1, return_latent: bool = False) \
            -> Union[Mesh, Tuple[Mesh, np.ndarray]]:
        mesh_vertices = mesh.get_vertices()
        unknown_vertex_mask = self.get_unknown_vertex_mask(
            unknown_vertex_mask=unknown_vertex_mask, vertices=mesh_vertices, invert_mask=invert_mask)

        def reconstruct_current_vertices(current_vertices):
            latent = self.encode(mesh_vertices, unknown_vertex_mask=unknown_vertex_mask, invert_mask=False)
            reconstructed_vertices = self.decode(latent)
            if back_match_known_vertices:
                reconstructed_vertices = self.back_match_data(
                    current_vertices, reconstructed_vertices, unknown_vertex_mask=unknown_vertex_mask)
            return reconstructed_vertices, latent

        current_vertices = np.array(mesh_vertices)
        latent = np.zeros(self.get_number_of_components())
        for _ in range(max(repeat, 1)):
            current_vertices, latent = reconstruct_current_vertices(current_vertices)

        mesh = Mesh(current_vertices, mesh.get_triangles())
        if return_latent:
            return mesh, latent
        return mesh

    #
    #
    # Partial credit for implementation of this method goes to Defne Kurtulus
    def visualize(self, min_val: float = -3, max_val: float = 3,
                  shape_components: Union[List[int], int] = 5):
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
            return Mesh(get_current_vertices(), self.triangles)

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

        def update_mesh():
            mesh = get_current_mesh()
            _widget3d.scene.clear_geometry()
            _widget3d.scene.add_geometry(f'mesh', mesh.get_open3d_mesh(compute_normals=True), material)

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

    def back_match_data(self, source_vertices, reconstructed_vertices, unknown_vertex_mask) -> np.ndarray:
        """
        Match reconstructed vertices back to original source vertices in known regions while deforming
        unknown regions along as-rigid-as-possibly.
        :param source_vertices: Source vertices to back-match (num_el x 3)
        :param reconstructed_vertices: Reconstruction of the source_data vector (must be same shape).
        :param unknown_vertex_mask: Specify a mask of vertices that are unknown and, thus, require statistical
                                    reconstruction.
        :return: back-matched vertices (np array of shape (num_el x 3))
        """
        try:
            import scipy.sparse as spsp
            from scipy.sparse import diags
            import igl
            from sksparse.cholmod import cholesky_AAt
        except ImportError:
            warnings.warn("Some more libraries are required to use the back matching feature. Install scipy, igl, "
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
        adjacency_matrix = spsp.csc_matrix(igl.adjacency_matrix(self.triangles))
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
