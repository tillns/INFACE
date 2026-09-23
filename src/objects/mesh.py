"""
Mesh object class to support basic triangle mesh handling without requiring many libraries.
This class also contains the Landmarks and CurvilinearFeatures classes, of which instances
can be attributes of a Mesh object.

Open3D must be installed for certain import, export, and visualization functions.

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

from __future__ import annotations
import os
from collections import defaultdict, deque
from copy import deepcopy
from itertools import accumulate
from pathlib import Path
from typing import Union, Dict, List, Tuple

import numpy as np
import json
import warnings

from src.objects.indices_and_masks import IndicesAndMasks
from src import utils

# We make open3d an optional dependency. The user only gets an ImportError
# when calling a method that requires open3d.
try:
    import open3d
except ImportError:
    class _Open3DFallback:
        def __getattr__(self, name):
            raise ImportError(
                f"open3d could not be imported. Tried to access attribute '{name}'. "
                "Check out the Readme for installation instructions.")

        def __call__(self, *args, **kwargs):
            raise ImportError(
                "open3d could not be imported. Check out the Readme for installation instructions.")

    open3d = _Open3DFallback()


class Mesh:
    """
    Mesh object class that contains at least the vertices and triangles as attributes.
    It can further contain Landmarks, CurvilinearFeatures, vertex colors,
    textures (along with UVs and material IDs), and virtual connections as attributes.
    A mesh can be loaded from a json file that contains the vertices and triangles; usual mesh files, i.e.,
    ply, stl, obj, are also supported, but open3d must be installed. If open3d is installed.
    A mesh can be visualized with open3D using the show() and show_multiple_meshes() functions.
    A mesh can also be transformed, in which case its attributes (including landmarks and curvilinear features)
    are transformed along. We offer further processing methods like edge retrieval/sorting, procrustes alignment,
    smoothing, closest point search (requires open3d for efficient hashing),
    principal curvatures (requires IGL), getting disconnected components and mesh boundaries,
    volume computation (requires trimesh), vertex edge distance computation,
    combining meshes, and marking indices.
    Check out the individual methods with their documentation for more information.
    """
    def __init__(
            self, vertices: np.ndarray, triangles: np.ndarray, colors: Union[np.ndarray, None] = None,
            landmarks_or_points_array: Union[Landmarks, np.ndarray, None] = None,
            curvilinear_features_or_line_sets: Union[CurvilinearFeatures, List[np.ndarray], None] = None,
            uvs: Union[np.ndarray, None] = None, material_ids: Union[np.ndarray, None] = None,
            textures: Union[List[np.ndarray], None] = None,
            virtual_connections: Union[List, np.ndarray, None] = None
    ):
        self.vertices = np.asarray(vertices)
        self.triangles = np.asarray(triangles)

        assert self.vertices.ndim == 2
        assert self.triangles.ndim == 2

        # Optional arguments
        # colors
        self.colors = None
        if colors is not None:
            self.set_colors(colors)
        # landmarks
        self.landmarks: Union[Landmarks, None] = None
        if landmarks_or_points_array is not None:
            self.set_landmarks(landmarks_or_points_array)
        self.curvilinear_features: Union[CurvilinearFeatures, None] = None
        if curvilinear_features_or_line_sets is not None:
            self.set_curvilinear_features(curvilinear_features_or_line_sets)
        # UVs
        self.uvs = None
        if uvs is not None:
            self.set_uvs(uvs)
        # textures
        self.material_ids, self.textures = None, None
        if material_ids is not None or textures is not None:
            assert material_ids is not None and textures is not None
            self.set_textures(material_ids=material_ids, textures=textures)
        self.open3d_scene = None

        self.virtual_connections = None
        if virtual_connections is not None:
            self.set_virtual_connections(virtual_connections)

    def set_from_mesh(self, mesh: Mesh, copy: bool = True):
        """
        Adjust current mesh instance to match another mesh instance.
        """
        self.vertices = mesh.get_vertices(copy=copy)
        self.triangles = mesh.get_triangles(copy=copy)
        self.colors = mesh.get_colors(copy=copy)
        self.landmarks = mesh.get_landmarks(copy=copy)
        self.open3d_scene = mesh.open3d_scene

    def get_vertices(self, copy: bool = True) -> np.ndarray:
        if copy:
            return self.vertices.copy()
        else:
            return self.vertices

    def get_triangles(self, copy: bool = False) -> np.ndarray:
        if copy:
            return self.triangles.copy()
        else:
            return self.triangles

    def get_faces(self, copy: bool = False) -> np.ndarray:
        return self.get_triangles(copy=copy)

    def get_num_vertices(self) -> int:
        return len(self.vertices)

    def get_num_triangles(self) -> int:
        return len(self.triangles)

    def get_num_edges(self) -> int:
        return len(self.get_edges_unique())

    def get_colors(self, copy: bool = False) -> Union[np.ndarray, None]:
        if not self.has_colors():
            return None
        if copy:
            return self.colors.copy()
        else:
            return self.colors

    def set_colors(self, colors: np.ndarray) -> None:
        colors_np = np.asarray(colors)
        if colors_np.ndim == 1:
            colors_np = np.expand_dims(colors_np, axis=0)
        if colors_np.shape[0] == 1:
            colors_np = np.repeat(colors_np, axis=0, repeats=self.get_num_vertices())
        if not np.array_equal(self.vertices.shape, colors_np.shape):
            raise AssertionError(f"Vertex colors must have the same shape as the vertices: {np.shape(self.vertices)}. "
                                 f"Given color shape: {np.shape(colors)}. It's also okay to give a single color, "
                                 f"but this doesn't work.")
        self.colors = colors_np

    def remove_colors(self) -> None:
        self.colors = None

    def has_colors(self) -> bool:
        return self.colors is not None

    def has_landmarks(self) -> bool:
        return self.landmarks is not None

    def has_curvilinear_features(self) -> bool:
        return self.curvilinear_features is not None

    def get_landmarks(self, copy: bool = False) -> Union[Landmarks, None]:
        if not self.has_landmarks():
            return None
        if copy:
            return self.landmarks.get_copy()
        else:
            return self.landmarks

    def get_curvilinear_features(self, copy: bool = False) -> Union[CurvilinearFeatures, None]:
        if not self.has_curvilinear_features():
            return None
        if copy:
            return self.curvilinear_features.get_copy()
        else:
            return self.curvilinear_features

    def set_landmarks(self, landmarks_or_point_array_or_path: Union[Landmarks, np.ndarray, Union[str, Path]],
                      parse_order: Union[List, np.ndarray, None] = None, strip_landmark_rows: bool = False) -> None:
        """
        Set the landmarks for the current Mesh object.
        :param landmarks_or_point_array_or_path: Landmarks object, point array, or path to a Landmarks file.
        :param parse_order: Optionally provide an index array to filter/reorder the landmark points by.
        :param strip_landmark_rows: Set True to remove undefined points from the Landmarks.
        :return: None (landmarks are set in-place)
        """
        landmarks = Landmarks(landmarks_or_point_array_or_path, parse_order=parse_order,
                              strip_landmark_rows=strip_landmark_rows)
        if not landmarks.dim == 3:
            raise AssertionError("Mesh landmarks must be 3D!")
        self.landmarks = landmarks

    def set_curvilinear_features(
            self,
            curvilinear_features_or_line_sets_or_path: Union[str, Path, np.ndarray, List[np.ndarray], CurvilinearFeatures],
            parse_order: Union[List[Union[List, np.ndarray, int]], None] = None, strip: bool = False) -> None:
        """
        Set the curvilinear features for the current Mesh object.
        :param curvilinear_features_or_line_sets_or_path: CurvilinearFeatures object, line sets, or path to file
                                                          from which to load the curvilinear features, optionally
                                                          rearranged via parse_order.
        :param parse_order: Optionally rearrange the line_sets by providing a list of index arrays.
        :param strip: Set True to strip curvilinear features of any undefined points, potentially altering the length.
        """
        curvilinear_features = CurvilinearFeatures(
            curvilinear_features_or_line_sets_or_path, parse_order=parse_order, strip=strip)
        if not curvilinear_features.dim == 3:
            raise AssertionError("Mesh curvilinear features must be 3D!")
        self.curvilinear_features = curvilinear_features

    @staticmethod
    def load_landmarks_and_curvilinear_features_from_single_file(
            file_path: Union[str, Path],
            point_indices: Union[List, np.ndarray], line_set_indices: List[Union[List, np.ndarray]],
            strip: bool = False) -> Tuple[Landmarks, CurvilinearFeatures]:
        """
        Load landmarks and curvilinear features from a single file, parsing them with separate parsers.
        We assume the file you provide contains a single 3D point per line, e.g., [p1, p2, p3, p4, p5, p6, p7, p8, p9].
        You can then provide an index array for the landmarks, e.g., [0, 4, 5], giving you landmark points [p1, p5, p6].
        For the curvilinear features, you provide a List of index arrays, e.g., [[1, 2, 3], [5, 6, 7, 8]], yielding
        [[p2, p3, p4], [p6, p7, p8, p9]] as line sets. As shown in this example, there may be overlaps. Index order
        is preserved.
        :param file_path: Path to Landmarks-like file.
        :param point_indices: Index array for the landmarks.
        :param line_set_indices: List of index arrays for the curvilinear features.
        :param strip: Set True to strip landmarks and curvilinear features of undefined points.
        :return: Tuple
            1) Landmarks containing index points.
            2) Curvilinear features containing indexed line sets.
        """
        points = Landmarks.load(file_path).get_points()
        landmarks = Landmarks(points[point_indices], strip_landmark_rows=strip)
        line_sets = CurvilinearFeatures.parse_points_to_line_sets(points, parse_order=line_set_indices)
        return landmarks, CurvilinearFeatures(line_sets, strip=strip)

    def set_landmarks_and_curvilinear_features_from_single_file(
            self, file_path: Union[str, Path], point_indices: Union[List, np.ndarray],
            line_set_indices: List[Union[List, np.ndarray]], strip: bool = False) -> None:
        """
        Non-static version of load_landmarks_and_curvilinear_features_from_single_file(). Cf. its docs.
        """
        landmarks, curvilinear_features = self.load_landmarks_and_curvilinear_features_from_single_file(
            file_path, point_indices=point_indices, line_set_indices=line_set_indices, strip=strip)
        self.set_landmarks(landmarks)
        self.set_curvilinear_features(curvilinear_features)

    def remove_landmarks(self):
        self.landmarks = None

    def remove_curvilinear_features(self):
        self.curvilinear_features = None

    def set_uvs(self, uvs: Union[List, np.ndarray]) -> None:
        uvs_np = np.asarray(uvs)
        expected_shape = [self.get_num_triangles(), 3, 2]
        if not np.array_equal(uvs_np.shape, expected_shape):
            raise AssertionError(f"UV shape expected to be {expected_shape} (num_triangles, 3, 2), "
                                 f"but given array has shape {uvs_np.shape}.")
        self.uvs = np.asarray(uvs)

    def get_uvs(self, copy: bool = False) -> Union[np.ndarray, None]:
        if not self.has_uvs():
            return None
        if copy:
            return self.uvs.copy()
        else:
            return self.uvs

    def has_uvs(self):
        return self.uvs is not None and len(self.uvs) > 0

    def set_textures(self, material_ids: Union[List, np.ndarray], textures: List[np.ndarray]) -> None:
        assert len(material_ids) == self.get_num_triangles()
        assert max(material_ids) <= len(textures) - 1
        self.material_ids = np.asarray(material_ids)
        self.textures = [None if texture is None else np.asarray(texture) for texture in textures]

    def has_textures(self):
        return self.material_ids is not None and len(self.material_ids) > 0 and self.textures is not None and len(self.textures) > 0

    def get_textures(self, copy: bool = False) -> Union[Tuple[np.ndarray, List[np.ndarray]], Tuple[None, None]]:
        if not self.has_textures():
            return None, None
        if copy:
            return self.material_ids.copy(), [None if texture is None else texture.copy() for texture in self.textures]
        else:
            return self.material_ids, self.textures

    def set_virtual_connections(self, virtual_connections: Union[List, np.ndarray]) -> None:
        """
        Virtual connections are edges between vertices that are not defined by the mesh triangles.
        This gives you a way to connect vertices in a virtual manner.
        :param virtual_connections: List or numpy array of vertex pairs to be virtually connected,
                                    e.g., [[0, 1], [2, 3]]
        :return: None
        """
        virtual_connections_np = np.asarray(virtual_connections)
        # Sort vertex indices in each edge so that [a, b] and [b, a] are considered the same
        virtual_connections_sorted = np.sort(virtual_connections_np, axis=1)

        # Use numpy unique to get only unique edges
        self.virtual_connections = np.unique(virtual_connections_sorted, axis=0)

    def get_virtual_connections(self) -> Union[np.ndarray, None]:
        """
        :return: Numpy array of virtual connection if the mesh has virtual connections defined, else None
                 (not an empty list)
        """
        return self.virtual_connections

    def remove_virtual_connections(self) -> None:
        """
        Remove the virtual connections from the mesh (set them to None).
        :return: None
        """
        self.virtual_connections = None

    def has_virtual_connections(self) -> bool:
        """
        :return: True if the mesh has virtual connections, False otherwise
        """
        return self.virtual_connections is not None

    def add_virtual_connections(self, virtual_connections: Union[List, np.ndarray]) -> None:
        """
        Append more virtual connections to the mesh. If the mesh currently doesn't have any virtual connections,
        this method works the same as set_virtual_connections(), but if it does, we don't replace the current
        virtual connections, but we add these new ones to the current ones.
        :param virtual_connections: List or numpy array of virtual connections to be appended to the current ones.
        :return: None
        """
        if not self.has_virtual_connections():
            self.set_virtual_connections(virtual_connections)
        else:
            self.set_virtual_connections(np.concatenate([self.virtual_connections, virtual_connections], axis=0))

    def transform(self, transform_mat: np.ndarray, in_place: bool = True, invert: bool = False) -> Mesh:
        """
        Transform a mesh with a general linear transformation matrix (3x3 or 4x4). The matrix is applied to all
        vertices. The triangle order is switched if the matrix contains a flip, and potential landmarks
        and curvilinear features are transformed accordingly.
        :param transform_mat: transformation matrix to be applied to the vertices np.ndarray (3x3 or 4x4)
        :param in_place: whether to perform the transformation in-place to the current mesh instance (default)
                         or to create a new mesh instance.
        :param invert: set True to invert the transformation matrix before applying it to the mesh.
        :return: transformed mesh instance (also returned if in_place is True)
        """
        if in_place:
            mesh_to_transform = self
        else:
            mesh_to_transform = self.get_copy()

        transformation_matrix = np.asarray(transform_mat)
        if invert:
            transformation_matrix = np.linalg.inv(transformation_matrix)

        mesh_to_transform.vertices = Landmarks.transform_points(self.vertices, transformation_matrix)
        if utils.matrix_contains_flip(transformation_matrix):
            mesh_to_transform.triangles = np.stack(
                [self.triangles[:, 0], self.triangles[:, 2], self.triangles[:, 1]], axis=1)
            if self.has_uvs():
                mesh_to_transform.uvs = np.stack([self.uvs[:, 0], self.uvs[:, 2], self.uvs[:, 1]], axis=1)

        # Also transform landmarks and curvilinear features if the mesh has them
        if mesh_to_transform.has_landmarks():
            mesh_to_transform.landmarks.transform(transformation_matrix)
        if mesh_to_transform.has_curvilinear_features():
            mesh_to_transform.curvilinear_features.transform(transformation_matrix)

        mesh_to_transform.open3d_scene = None

        return mesh_to_transform

    def translate(self, translate: np.ndarray, in_place: bool = True) -> Mesh:
        return self.transform(utils.get_translation_matrix(translate), in_place=in_place)

    def scale(self, scale: np.ndarray, in_place: bool = True) -> Mesh:
        return self.transform(utils.get_scaling_matrix(scale), in_place=in_place)

    def get_center(self) -> np.ndarray:
        return np.mean(self.vertices, axis=0)

    def center(self, in_place: bool = True) -> Mesh:
        return self.translate(-self.get_center(), in_place=in_place)

    def flip(self, in_place: bool = True, flip_landmarks_order: Union[np.ndarray, List, None] = None,
             flip_curvilinear_features_order: Union[np.ndarray, List, None] = None,
             reverse_curvilinear_features: Union[np.ndarray, List, None] = None) -> Mesh:
        """
        Flip the mesh along the x-axis, cf. also transform() method. The method additionally offers options
        to reorder the landmarks and curvilinear features, since left and right labels are swapped after
        flipping.
        :param in_place: Set False to create a new mesh instance to be flipped.
                         True by default, so current object is changed.
        :param flip_landmarks_order: Index array that specifies how to reorder landmarks.
        :param flip_curvilinear_features_order: Index array that specifies how to reorder curvilinear features.
                                                Those features that are changed in position are also reversed.
        :param reverse_curvilinear_features: Optionally provide an index or mask array to specify which curvilinear
                                             features to reverse due to the flipping.
                                             The indexing in this array works on the original line sets
                                             array before it is reordered. Note that if you defined the line sets
                                             in a mirrored way, you don't need to reverse the order
                                             of left and right switched line sets.
        :return: Mesh instance (either the current one with applied changes or a new one)
        """
        flip_matrix = utils.get_flip_matrix(flip_axis=0, dim=3, homogeneous=True)
        flipped_mesh = self.transform(transform_mat=flip_matrix, in_place=in_place)
        if flip_landmarks_order is not None:
            if not self.has_landmarks():
                raise AssertionError("Mesh doesn't have landmarks that could be reordered during flipping.")
            flipped_mesh.get_landmarks(copy=False).reorder(flip_landmarks_order)
        if self.has_curvilinear_features():
            reverse_curvilinear_features = IndicesAndMasks.get_mask(
                reverse_curvilinear_features, self.curvilinear_features.get_num_line_sets())
            flipped_mesh.curvilinear_features.reverse(reverse_curvilinear_features)

            if flip_curvilinear_features_order is not None:
               flipped_mesh.curvilinear_features.reorder(flip_curvilinear_features_order)
        elif flip_curvilinear_features_order is not None:
            raise AssertionError("Mesh doesn't have any curvilinear features to be reordered during flipping.")

        return flipped_mesh

    def get_range(self) -> np.ndarray:
        """
        :return: 3D numpy vector, with first entry indicating the range of the vertices along the x-axis, i.e.,
                 the difference between the vertices' minimum and maximum x-coordinate; analogous for y and z
        """
        return np.max(self.vertices, axis=0) - np.min(self.vertices, axis=0)

    def rotate(self, axis, angle) -> None:
        """
        Rotate mesh, cf. get_rotation_matrix_from_axis_and_angle() in utils for an explanation of the arguments.
        """
        rot_mat = utils.get_rotation_matrix_from_axis_and_angle(axis=axis, angle=angle)
        self.transform(rot_mat)

    def convert_vertex_to_triangle_indices(
            self, vertex_indices_or_path: Union[Union[List, np.ndarray], Union[str, Path]],
            num_vertices_per_triangle: int = 1) -> np.ndarray:
        """
        Given a vertex index/mask array, convert to a corresponding selection of triangles, so all triangles that include
        a choosable number of indexed vertices are selected.
        :param vertex_indices_or_path: Vertex indices/mask or path to them.
        :param num_vertices_per_triangle: Number of vertices in a triangle that need to be indexed for the
                                          triangle to be included in the conversion, so if 1 (default),
                                          then each triangle is indexed that includes at least 1 indexed vertex.
                                          If 3, only those triangle are selected, of which all 3 vertices are indexed.
        :return: Corresponding triangle indices or mask (fitting input type).
        """
        vertex_mask = IndicesAndMasks.get_mask(vertex_indices_or_path, num_indices_or_mesh=self)
        triangles = self.get_triangles()
        triangle_indices = np.asarray([
            num_triangle for num_triangle, triangle in enumerate(triangles)
            if len(np.where(vertex_mask[triangle])[0]) >= num_vertices_per_triangle])

        if IndicesAndMasks.is_mask(vertex_indices_or_path):
            return IndicesAndMasks.get_mask(triangle_indices, num_indices_or_mesh=len(triangles))
        else:
            return triangle_indices

    def convert_triangle_to_vertex_indices(self, triangle_indices_mask_or_path) -> np.ndarray:
        """
        Given indices or mask of mesh triangles, convert them to the corresponding vertex indices or mask,
        which includes all vertices that belong to an indexed triangle.
        :param triangle_indices_mask_or_path: triangle indices, mask, or path to them.
                                              Needs to be compatible with the current mesh instance.
        :return: Corresponding vertex indices (unique) or mask, same as input.
        """
        triangles, num_triangles = self.get_triangles(), self.get_num_triangles()
        triangle_indices_or_mask = IndicesAndMasks.get_mask_or_indices(
            triangle_indices_mask_or_path, num_indices_or_mesh=num_triangles)
        vertex_indices = np.unique(triangles[triangle_indices_or_mask].flatten())
        if IndicesAndMasks.is_indices(triangle_indices_or_mask):
            return vertex_indices
        else:
            return IndicesAndMasks.get_mask(vertex_indices, num_indices_or_mesh=self.get_num_vertices())

    def convert_vertex_to_edge_indices(
            self, vertex_indices_or_path, num_vertices_per_edge: int = 1) -> Tuple[np.ndarray, np.ndarray]:
        """
        Given a vertex index/mask array, convert to a corresponding selection of edges, so all edges that include
        a choosable number of indexed vertices (1 or 2 recommended :)) are selected. Since edges are not a primary
        attribute, they are also returned (computed using get_edges_unique())
        :param vertex_indices_or_path: Vertex indices/mask or path to them.
        :param num_vertices_per_edge: Number of vertices in an edge that need to be indexed for the
                                      edge to be included in the conversion, so if 1 (default),
                                      then each edge is indexed that includes at least 1 indexed vertex.
                                      If 2, only those edges are selected, of which all 2 vertices are indexed.
        :return: Tuple: 1) edges
                        2) Corresponding edge indices or mask (fitting input type).
        """
        vertex_mask = IndicesAndMasks.get_mask(vertex_indices_or_path, num_indices_or_mesh=self)
        edges = self.get_edges_unique()
        edge_indices = np.asarray([num_edge for num_edge, edge in enumerate(edges) if
                                   IndicesAndMasks.get_num_indexed(vertex_mask[edge]) >= num_vertices_per_edge])
        if IndicesAndMasks.is_indices(vertex_indices_or_path):
            edge_indices_or_mask = edge_indices
        else:
            edge_indices_or_mask = IndicesAndMasks.get_mask(edge_indices, num_indices_or_mesh=len(edges))
        return edges, edge_indices_or_mask

    @staticmethod
    def convert_edge_to_vertex_indices(edges: np.ndarray) -> np.ndarray:
        """
        Convert edges to vertex indices. Since edges are not a primary attribute, you must provide the actual edges,
        not the edge indices! Since the actual edges already index vertices, it's basically just flattening the array
        and making it unique.
        :param edges: Edge array of size nx2 containing pair if vertex indices for each of the n edges.
        :return Converted vertex indices.
        """
        return IndicesAndMasks.get_indices(np.asarray(edges).flatten(), make_unique=True)

    @staticmethod
    def _get_points_from_points_or_landmarks_or_mesh(points_or_landmarks_or_mesh: Union[np.ndarray, Landmarks, Mesh]):
        # Check point validity
        if isinstance(points_or_landmarks_or_mesh, Landmarks):
            assert points_or_landmarks_or_mesh.landmarks_contain_no_nan()
            return points_or_landmarks_or_mesh.get_points()
        elif isinstance(points_or_landmarks_or_mesh, Mesh):
            return points_or_landmarks_or_mesh.get_vertices()
        else:
            return np.asarray(points_or_landmarks_or_mesh)

    def get_closest_triangle_points(
            self, points_or_landmarks_or_mesh: Union[np.ndarray, Landmarks, Mesh],
            vertex_indices_or_mask: Union[np.ndarray, List, None] = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        This method finds the closest points on the mesh on any part of the triangle. This stands in contrast to
        the methods get_closest_vertices(), as this finds the closest vertices. This method requires open3d's
        RaycastingScene for efficient spatial hashing.
        :param points_or_landmarks_or_mesh: You can provide the points for which you want to find the closest triangle
                                            points on the mesh as a normal [n,3] numpy point array,
                                            as a Landmarks object, or as a Mesh object. In the last case, we take the
                                            mesh's vertices as the point array. Please note that Landmarks should be
                                            stripped. Use the resnap_landmarks() method for better Landmark handling.
        :param vertex_indices_or_mask: Vertex indices for self mesh to be considered for closest point computation
                                       (returned triangle indices are mapped back to original mesh)
        :return: Tuple:
                        1) 3D point array, so for each input point one 3D point on top of the geometry that's
                           closest to that input point. Note that even if you provide a Landmarks object,
                           this method still returns a 3D points array, cf. resnap_landmarks() for better Landmarks
                           handling.
                        2) Index array containing the triangle indices corresponding to the closest points.
        """
        # Get points in valid format
        points = self._get_points_from_points_or_landmarks_or_mesh(points_or_landmarks_or_mesh)

        if vertex_indices_or_mask is not None:
            mesh_removed, _, triangles_kept = self.remove_vertices(vertex_indices_or_mask, invert=True)
        else:
            mesh_removed = self
            triangles_kept = None

        # Compute closest points on geom and the corresponding triangle indices.
        closest_points_dict = mesh_removed.get_open3d_scene().compute_closest_points(open3d.core.Tensor(
            points, dtype=open3d.core.Dtype.Float32))
        closest_points = closest_points_dict["points"].numpy()
        tri_indices = np.asarray(closest_points_dict["primitive_ids"].numpy(), dtype=np.int64)

        # Map back to original mesh
        if triangles_kept is not None:
            tri_indices = triangles_kept[tri_indices]

        return closest_points, tri_indices

    def get_closest_vertices(self, points_or_landmarks_or_mesh: Union[np.ndarray, Landmarks, Mesh],
            vertex_indices_or_mask: Union[np.ndarray, List, None] = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        This method finds the closest vertices on the mesh from some provided points.
        :param points_or_landmarks_or_mesh: You can provide the points for which you want to find the closest triangle
                                            points on the mesh as a normal [n,3] numpy point array,
                                            as a Landmarks object, or as a Mesh object. In the last case, we take the
                                            mesh's vertices as the point array. Please note that Landmarks should be
                                            stripped. Use the resnap_landmarks() method for better Landmark handling.
        :param vertex_indices_or_mask: Vertex indices for self mesh to be considered for closest point computation
                                       (returned vertex indices are mapped back to original mesh)
        :return: Tuple:
                        1) 3D point array that includes the closest vertices on the mesh (actual positions)
                        2) Integer index array that specifies the indices of the closest vertices on the mesh
        """

        points = self._get_points_from_points_or_landmarks_or_mesh(points_or_landmarks_or_mesh)

        # We compute the closest vertex indices by taking the closest triangle and finding the closest vertex to the
        # input point among the three vertices indexed by the closest triangle. This avoids that vertices that are
        # at the same position, but not part of the same triangle, interfere with each other.
        closest_triangle_points, closest_triangle_indices = self.get_closest_triangle_points(
            points_or_landmarks_or_mesh=points, vertex_indices_or_mask=vertex_indices_or_mask)

        # (t x 3)
        triangles = self.get_triangles(copy=False)
        # (v x 3)
        vertices = self.get_vertices(copy=False)
        # p x 3 (three possible vertices per point, given by the vertices corresponding
        # to the triangle closest to the input point)
        closest_triangle_vertex_indices = triangles[closest_triangle_indices]
        # p x 3 x 3 (these are the actual positions, not the indices, of the three possible vertices)
        closest_triangle_vertices = vertices[closest_triangle_vertex_indices]
        # l x 3 (three distances per landmark for the three vertices on the closest triangle)
        closest_triangle_vertex_distances = np.linalg.norm(
            np.expand_dims(closest_triangle_points, axis=1) - closest_triangle_vertices, axis=-1)
        # p (per point, the index of the closest vertex, but within the frame of the triangle,
        # so the index is 0, 1, or 2 for triangles.
        closest_triangle_vertex_distances_argmin = np.argmin(closest_triangle_vertex_distances, axis=-1)
        # p (per point, the actual index of the closest vertex, computed by indexing the
        # closest triangle vertex indices via the distance argmin)
        closest_vertex_indices = closest_triangle_vertex_indices[
            np.arange(len(points)), closest_triangle_vertex_distances_argmin]
        # p x 3 (the actual closest points)
        closest_vertices = vertices[closest_vertex_indices]

        return closest_vertices, closest_vertex_indices

    def get_closest_vertex(self, point: Union[np.ndarray, List],
                           vertex_indices_or_mask: Union[np.ndarray, List, None] = None) -> Union[np.ndarray, int]:
        """
        For a single point, this method returns the closest vertex and its index on the mesh instance.
        It's a special case of the more general :func:`Mesh.get_closest_vertices()` method.
        :param point: single 3D point, so List or numpy float array of length 3 with one axis.
        :param vertex_indices_or_mask: Vertex indices for self mesh to be considered for closest point computation
                                       (returned vertex index is mapped back to original mesh)
        :return 3D numpy float vector for the position, and vertex index as integer
        """
        vertices, indices = self.get_closest_vertices(np.asarray([point]), vertex_indices_or_mask=vertex_indices_or_mask)
        return vertices[0], indices[0]

    def get_closest_triangle_point(
            self, point: Union[np.ndarray, List],
            vertex_indices_or_mask: Union[np.ndarray, List, None] = None) -> Union[np.ndarray, int]:
        """
        For a single point, this method returns the closest point on any triangle on the mesh and the respective
        triangle index. It's a special case of the more general :func:`Mesh.get_closest_triangle_points()` method.
        :param point: single 3D point, so List or numpy float array of length 3 with one axis.
        :param vertex_indices_or_mask: Vertex indices for self mesh to be considered for closest point computation
                                       (returned triangle index is mapped back to original mesh)
        :return 3D numpy float vector for the position, and triangle index as integer
        """
        triangle_points, triangle_indices = self.get_closest_triangle_points(np.asarray([point]), vertex_indices_or_mask=vertex_indices_or_mask)
        return triangle_points[0], triangle_indices[0]

    def get_internal_vertices(self, use_center_for_direction: bool = True):
        """
        Structures inside a mesh (possibly from a volumetric scan, such as CT or MRI) can be annoying,
        as they're difficult to select manually and they can disturb processing methods,
        such as the registration. This automatic algorithm estimates which vertices don't lie
        on the outermost surface via ray casting. From each vertex, a ray is cast outwards,
        and those rays that still intersect with the surface are estimated to belong to vertices that lie
        inside the mesh.
        :param use_center_for_direction: By default the ray direction per vertex is based on the connection between the
                                         mesh center and that vertex. This is a good estimate for meshes that are kind
                                         of spherical. Set this to False to use the vertex normal instead,
                                         which might be more appropriate for differently shaped meshes.
        :return: bool mask array with True entries for vertices that are estimated to be inside the mesh.
        """
        # Ray directions are either the connections from the center to each vertex
        if use_center_for_direction:
            center = self.get_center()
            ray_directions = self.vertices - center
        # or the vertex normals
        else:
            ray_directions = self.get_vertex_normals()

        # normalize directions
        ray_directions /= np.linalg.norm(ray_directions, axis=1, keepdims=True)

        # Define origins as just slightly outside each vertex to avoid intersecting the surface right at the vertex
        ray_origins = self.vertices + ray_directions * 1e-3

        # Define rays as 6D vectors (origin and direction)
        rays = open3d.core.Tensor(np.concatenate([ray_origins, ray_directions], axis=1),
                                  dtype=open3d.core.Dtype.Float32)

        # Count the number of ray intersections
        intersection_counts = self.get_open3d_scene().count_intersections(rays).numpy()

        # Return bool mask array where all vertices with corresponding rays
        # that have more than 0 intersections are marked
        return intersection_counts > 0

    def transfer_vertex_indices_to_new_mesh(
            self, indices_mask_or_path: Union[List, np.ndarray], new_mesh: Mesh,
            criterion: str="triangle_edge_point") -> np.ndarray:
        """
        Given a mesh (source) with corresponding vertex indices, try to transfer the indices to a new mesh (target).
        There are three different strategies to transfer these indices. Check criterion.
        :param indices_mask_or_path: Indices, mask or path to one of them.
        :param new_mesh: New (target) mesh to transfer the indices to.
        :param criterion: "closest_point": This will simply find the closest point on the mesh for each indexed vertex
                                           on the input mesh. Note that the resulting index selection may be
                                           very disconnected.
                          "triangle_edge_point": For each point on the new mesh, check if it's inside any triangle
                                                 that is indexed on the old mesh, (i.e., all three vertices of
                                                 that triangle are indexed), on top of an edge, or right at an
                                                 indexed point. This is the default, as it's more versatile than
                                                 simple closest point search.
                          "equal_vertices": Stricter version of "closest_point". Will only index those
                                            closest points that are at the same position (up to some tolerance)
                                            as the originally indexed vertices.
        :return: Unordered, unique list of indices, or mask if indices_mask_or_path is a mask.
        """
        mask = IndicesAndMasks.get_mask(indices_mask_or_path, num_indices_or_mesh=self)
        if IndicesAndMasks.get_num_indexed(mask) == 0:
            return []

        if criterion == "triangle_edge_point":
            dist_thresh = 1e-4  # precision tolerance
            new_mesh_vertices = new_mesh.get_vertices()
            old_mesh_triangles = self.get_triangles()
            old_mesh_vertices = self.get_vertices()

            # Check which triangles are fully indexed, i.e., of which triangles all their vertices are indexed
            old_mesh_selected_triangle_indices = self.convert_vertex_to_triangle_indices(
                mask, num_vertices_per_triangle=3)

            # Also check which edges are fully indexed (so by both their vertices)
            old_mesh_edges, old_mesh_selected_edge_indices = self.convert_vertex_to_edge_indices(
                mask, num_vertices_per_edge=2)
            is_edge_selected = {tuple(sorted(old_mesh_edge)): old_mesh_selected_edge_indices[num_edge]
                                for num_edge, old_mesh_edge in enumerate(old_mesh_edges)}

            # A point on the new mesh is considered to be indexed if it's either inside a fully indexed triangle,
            # on top of a fully indexed edge, or closest to an indexed point, as long as it's generally not too
            # far away from the original mesh. That way, we can fill any triangles selections on the new mesh, but we can
            # also transfer line selections, where no full triangle is selected, but there's still some
            # connected vertices selected. Lastly, single point selections are also transferred. All the measurements must
            # be within the dist_thresh tolerance.
            new_mesh_corresponding_indices = []
            closest_points, closest_triangles = self.get_closest_triangle_points(new_mesh.get_vertices())
            for new_mesh_vertex_index, (new_mesh_vertex, closest_point, closest_triangle_index) in (
                    enumerate(zip(new_mesh_vertices, closest_points, closest_triangles))):
                closest_triangle = old_mesh_triangles[closest_triangle_index]

                # If the vertex is further away from the mesh than the max edge length of the closest triangle,
                # we skip the vertex. This ensures that vertices that are far away from the mesh don't get included
                # just because their closest triangle is indexed.
                max_edge_length = max(np.linalg.norm(
                    old_mesh_vertices[closest_triangle[idx]] - old_mesh_vertices[closest_triangle[(idx + 1) % 3]])
                                      for idx in range(3))
                if np.linalg.norm(new_mesh_vertex - closest_point) > max_edge_length:
                    continue

                # If the closest triangle is among the selected ones, we add the point
                if old_mesh_selected_triangle_indices[closest_triangle_index]:
                    new_mesh_corresponding_indices.append(new_mesh_vertex_index)
                    continue

                # If the closest edge of the closest triangle is among the indexed edges, we add the point
                _, edge_dist, closest_edge_triangle_index = CurvilinearFeatures.get_closest_points_to_line_set(
                    points=closest_point, line_set=old_mesh_vertices[closest_triangle], connected_last_to_first=True)
                closest_edge = closest_triangle[[closest_edge_triangle_index, (closest_edge_triangle_index + 1) % 3]]
                if edge_dist < dist_thresh and is_edge_selected[tuple(sorted(closest_edge))]:
                    new_mesh_corresponding_indices.append(new_mesh_vertex_index)
                    continue

                # If the closest vertex of the closest triangle is among the indexed vertices, we add the point
                vertex_dists = np.linalg.norm(old_mesh_vertices[closest_triangle] - closest_point, axis=-1)
                if np.any(vertex_dists < dist_thresh):
                    closest_vertex_ind = closest_triangle[np.argmin(vertex_dists)]
                    if mask[closest_vertex_ind]:
                        new_mesh_corresponding_indices.append(new_mesh_vertex_index)

        elif criterion == "closest_point":
            _, new_mesh_corresponding_indices = self.get_closest_vertices(new_mesh, vertex_indices_or_mask=mask)

        elif criterion == "equal_vertices":
            closest_points, new_mesh_corresponding_indices = self.get_closest_vertices(
                new_mesh, vertex_indices_or_mask=mask)
            new_mesh_corresponding_indices = new_mesh_corresponding_indices[np.linalg.norm(
                closest_points - self.get_vertices(copy=False)[mask], axis=1) < 1e-6]

        else:
            raise ValueError(f"Unknown criterion: {criterion}. Cf. docs for allowed criteria.")

        if IndicesAndMasks.is_mask(indices_mask_or_path):
            return IndicesAndMasks.get_mask(new_mesh_corresponding_indices, num_indices_or_mesh=new_mesh)
        else:
            return new_mesh_corresponding_indices


    def dijkstra_lengths(self, start_vertex_index: int, indices_or_mask: Union[np.ndarray, List, None] = None,
                         return_as_dict: bool = False) -> Union[dict, np.ndarray]:
        """
        Get the mesh distance of each vertex to one start vertex. Distances only go along the edges.
        :param start_vertex_index: Index of the start vertex, so the vertex with distance 0 from which to measure the
                                   distance of all other vertices to.
        :param indices_or_mask: Optionally only compute the distances for a subset of the vertices to save computation.
        :param return_as_dict: Set True to get a dictionary only with the reachable vertices.
        :return If return_as_dict is False, a numpy array with a distance for each vertex is returned, those not
                reached or excluded by the indices_or_mask have distance np.inf.
                If return_as_dict is True, then it's a dictionary without the vertices with np.inf.
        """
        import heapq
        vertex_neighbors = self.get_vertex_neighbors()
        num_vertices = self.get_num_vertices()

        # Compute mask
        if indices_or_mask is None:
            mask = np.ones(num_vertices, dtype=bool)
        else:
            mask = IndicesAndMasks.get_mask(indices_or_mask, num_indices_or_mesh=num_vertices)

        # Distances initialized all with np.inf
        distances = np.full(num_vertices, np.inf, dtype=np.float64)

        # We start with the start vertex
        distances[start_vertex_index] = 0.0
        pq = [(0.0, start_vertex_index)]

        # We put all neighbors to the heap along with their distances and go further from there
        # (breadth first search)
        while pq:
            dist, vertex_index = heapq.heappop(pq)
            if dist != distances[vertex_index] or not mask[vertex_index]:
                continue
            for vertex_neighbor_index in vertex_neighbors[vertex_index]:
                neighbor_dist = dist + np.linalg.norm(self.vertices[vertex_index] - self.vertices[vertex_neighbor_index])
                if neighbor_dist < distances[vertex_neighbor_index]:
                    distances[vertex_neighbor_index] = neighbor_dist
                    heapq.heappush(pq, (neighbor_dist, vertex_neighbor_index))

        if return_as_dict:
            # Return only finite distances in a dictionary
            reached = IndicesAndMasks.intersect(np.isfinite(distances), mask)
            return {int(i): float(distances[i]) for i in np.nonzero(reached)[0]}
        else:
            return distances

    def compute_principal_curvatures(
            self, vertex_indices: np.ndarray = None, relative_radius: float = None, absolute_radius: float = None) -> Tuple[float, float, float, float]:
        """
        Compute principal curvatures for the mesh, optionally only for selected vertices, using IGL's
        principal_curvature() method. For some small disconnected components, the directions and values of the
        principal curvatures are defined as np.nan, since IGL's method doesn't work on them.
        :param vertex_indices: Optionally provide indices of those vertices you want to compute the principal curvature for.
                               Currently, this does not make the code any more efficient, as I can't just delete all the
                               other vertices, because then we'd have problems at the boundary of the vertex selection.
                               Ideally, though, one could delete a lot of vertices that are nowhere close to the boundary
                               of the selection, but that would require a more sophisticated algorithm...
        :param relative_radius: This equals IGL's radius arg. It's relative to the average edge length.
                                They had a default value of 5 last time I checked, so need to set this.
                                You may only set this or absolute_radius.
        :param absolute_radius: Unlike relative_radius, this value is not relative to the average edge length,
                                but absolute in the mesh units. It might be helpful to use this if you compute the
                                curvature on multiple meshes that you normalized in size, but that can potentially
                                have drastically different mesh resolutions.
        :return: min_dir, max_dir, min_val, max_val (see also IGL's docs for principal_curvature)
        """
        import igl
        num_vertices = self.get_num_vertices()
        # Principal curvature directions and values are initialized as np.nan
        min_dir, max_dir = np.ones((num_vertices, 3)) * np.nan, np.ones((num_vertices, 3)) * np.nan
        min_val, max_val = np.ones(num_vertices) * np.nan, np.ones(num_vertices) * np.nan

        additional_args = {}
        # Radius is optionally added as an additional argument to the IGL method
        if relative_radius is not None or absolute_radius is not None:
            if relative_radius is not None and absolute_radius is not None:
                raise ValueError("Choose either relative or absolute radius.")
            if relative_radius is not None:
                additional_args["radius"] = relative_radius
            else:
                # I'm not entirely sure that IGL computes the absolute radius like this,
                # but it should be hopefully comparable.
                average_edge_length = self.get_average_edge_length()
                additional_args["radius"] = average_edge_length / absolute_radius

        # We compute the curvatures for each mesh component separately.
        _, _, disconnected_components = self.get_disconnected_components()
        for component in disconnected_components:
            # We exclude components with fewer than six vertices, since the code otherwise crashes.
            # For those vertices, the directions and values stay at np.nan
            # Recent versions of IGL return five instead of four variables, including a bad_variable variable as fifth.
            # We allow both versions here by using *_
            if len(component) > 5:
                mesh_component = self.remove_vertices(component, invert=True)
                max_dir[component], min_dir[component], max_val[component], min_val[component], *_ = igl.principal_curvature(
                    mesh_component.get_vertices(), mesh_component.get_triangles(), **additional_args)

        # Sanity-check IGL's output order for the values.
        if not np.all(max_val[~np.isnan(max_val)] >= min_val[~np.isnan(min_val)]):
            if not np.all(max_val[~np.isnan(max_val)] <= min_val[~np.isnan(min_val)]):
                raise AssertionError("max curvature is not bigger than or equal to the min curvature everywhere. "
                                     "Also not vice versa, so it's not an issue with the order.")

            warnings.warn("You probably have an old libigl version installed, where max and min principal directions "
                          "and values were in the wrong order (compared to what docs said). We'll switch the order "
                          "here, but make sure you check this.")

            max_val, min_val = min_val, max_val
            max_dir, min_dir = min_dir, max_dir

        # Optionally filter the output by the user-provided vertex indices
        if vertex_indices is not None:
            min_dir, max_dir = min_dir[vertex_indices], max_dir[vertex_indices]
            min_val, max_val = min_val[vertex_indices], max_val[vertex_indices]

        return min_dir, max_dir, min_val, max_val

    def resnap_landmarks(
            self, other_landmarks: Union[Landmarks, np.ndarray, None] = None,
            resnap_to_closest_triangle_point: bool = False, strip_landmarks: bool = True,
            return_non_nan_rows: bool = False, in_place: bool = False):
        """
        After the mesh was somehow changed from its original, or simply if you're not sure the landmarks are
        sticking to the mesh, or if you want to find the closest points on the mesh to its landmarks or any other
        landmarks, this method resnaps the landmarks to the mesh, also returning the vertex/triangle indices the
        landmarks were resnapped to (cf. get_closest_triangle_points() and get_closest_vertices() methods).
        A warning is printed if the landmarks show some large deviation (threshold not thoroughly tested).
        @param other_landmarks: You can choose to snap other landmarks to the mesh instead of its assigned ones.
        @param resnap_to_closest_triangle_point: By default, the landmarks are resnapped to the closest vertices.
                                                 Set this param to True to snap them to the closest point on any triangle.
                                                 This may be necessary if you want to avoid losing landmark precision for
                                                 raw input scans with some potentially large triangles, where snapping the
                                                 landmarks to the closest vertices may alter their position drastically.
        @param strip_landmarks: None landmarks cannot be snapped back to a mesh, so by default, they are removed in this
                                method. However, you can set this to False to keep the Nones in there. You can also
                                choose to return the rows where there are Nones via return_non_nan_rows.
        @param return_non_nan_rows: Whether to return the row indices with Nones. Note that in this case,
                                    a tuple of three is returned.
        @param in_place: Set to True if you want to assign the resnapped landmarks directly to the mesh
                         (the landmarks are still returned). This was chosen to be False by default, since you can also
                         resnap other landmarks to the mesh, and you may not want to overwrite the current ones.
        @return: Tuple:
                        1) Landmarks object
                        2) vertex or triangle indices of the mesh corresponding to the landmark points.
                           If strip_landmarks is set to False, nan landmarks get assigned the index -1 in this array.
                        Optional 3) If return_non_nan_rows, then the row indices where there are Nones
        """
        if other_landmarks is not None:
            landmarks_to_resnap = Landmarks(other_landmarks)
        else:
            if not self.has_landmarks():
                raise AssertionError("Mesh doesn't have any landmarks defined.")
            if self.landmarks.get_num_points() == 0:
                if return_non_nan_rows:
                    return self.landmarks, [], []
                return self.landmarks, []
            landmarks_to_resnap = self.landmarks

        landmarks_stripped, non_nan_rows = Landmarks.strip_landmark_rows(
            landmarks_to_resnap, return_non_nan_row_indices=True)
        # We can either snap the landmarks to the closest triangle points,
        # so the indices are also triangle indices, not vertex indices...
        if resnap_to_closest_triangle_point:
            resnapped_landmark_points, resnapped_indices = self.get_closest_triangle_points(
            points_or_landmarks_or_mesh=landmarks_stripped)
        # ...or we resnap the landmarks to the closest vertices, in which case the indices are vertex indices.
        else:
            resnapped_landmark_points, resnapped_indices = self.get_closest_vertices(
                points_or_landmarks_or_mesh=landmarks_stripped)

        if np.linalg.norm(resnapped_landmark_points - landmarks_stripped.get_points()) > 0.1 * np.linalg.norm(landmarks_stripped.get_points()):
            warnings.warn("The resnapped landmarks are very different to the original landmarks. "
                          "It could be that the provided mesh is largely different from the mesh the "
                          "landmarks were placed on. Make sure the landmarks are still correct.")

        if not strip_landmarks and landmarks_to_resnap.landmarks_contain_nan():
            # Setup a new point array that contains the original nans and replace
            # the remaining entries with the resnapped landmarks.
            resnapped_landmark_points_with_nan = landmarks_to_resnap.get_points(copy=True)
            resnapped_landmark_points_with_nan[non_nan_rows] = resnapped_landmark_points
            resnapped_landmark_points = resnapped_landmark_points_with_nan

            # Also build a new index array that has -1 at the nan entries
            resnapped_indices_with_nan = (np.ones(len(resnapped_landmark_points)) * -1).astype(int)
            resnapped_indices_with_nan[non_nan_rows] = resnapped_indices
            resnapped_indices = resnapped_indices_with_nan

        resnapped_landmarks = Landmarks(resnapped_landmark_points)
        if in_place:
            self.set_landmarks(resnapped_landmarks)
        if return_non_nan_rows:
            return resnapped_landmarks, resnapped_indices, non_nan_rows
        return resnapped_landmarks, resnapped_indices

    def get_triangle_edge_lengths(self) -> np.ndarray:
        """
        Get edge lengths of each mesh triangle.
        :return: Numpy float array num_triangles x 3 (three edge lengths for each triangle)
        """
        vertices, triangles = self.get_vertices(), self.get_triangles()
        triangle_vertices = vertices[triangles]
        return np.linalg.norm(triangle_vertices - triangle_vertices[:, [1, 2, 0]], axis=-1)

    def get_triangle_areas(self) -> np.ndarray:
        """
        Get the area of each mesh triangle.
        :return: numpy float array with one axis of length num_triangles (one area per triangle).
        """
        triangle_edge_lengths = self.get_triangle_edge_lengths()
        a, b, c = triangle_edge_lengths[:, 0], triangle_edge_lengths[:, 1], triangle_edge_lengths[:, 2]
        s = (a + b + c) / 2.
        return np.sqrt(s * (s - a) * (s - b) * (s - c))

    def get_small_and_large_triangles(
            self, triangle_thresh_perc: float = 0.99, return_vertices=False,
            return_as_indices: bool = True) -> np.ndarray:
        """
        Get indices or mask of relatively small and big triangles, i.e., those triangles whose area falls outside
        of the normal percentage.
        :param triangle_thresh_perc: threshold that specifies the percentage of triangle areas considered normal, i.e.,
                                     the indices of those triangles are returned whose area is bigger or smaller
                                     than this percent of all triangle areas.
                                     Default is 0.99, float between 0 and 1 required.
        :param return_vertices: Set True to get indices or mask for the respective vertices, not triangles
                                (cf. convert_triangle_to_vertex_indices())
        :param return_as_indices: Set False to get mask returned.
        :return: numpy indices or mask array, indexing triangles or vertices.
        """
        assert 0 < triangle_thresh_perc < 1
        triangle_areas = self.get_triangle_areas()
        triangle_areas_sorted = sorted(triangle_areas)
        upper_thresh_size = triangle_areas_sorted[int((len(triangle_areas_sorted) - 1) * triangle_thresh_perc)]
        lower_thresh_size = triangle_areas_sorted[int((len(triangle_areas_sorted) - 1) * (1 - triangle_thresh_perc))]
        above_thresh = IndicesAndMasks.join(triangle_areas >= upper_thresh_size, triangle_areas <= lower_thresh_size)
        if return_vertices:
             above_thresh = self.convert_triangle_to_vertex_indices(above_thresh)
        return IndicesAndMasks.get_mask_or_indices(above_thresh,  return_as_indices=return_as_indices)

    def get_triangle_area_sum_per_vertex(self) -> np.ndarray:
        """
        For each mesh vertex, get the sum of the areas of all triangles the respective vertex is connected to.
        :return: Numpy float array with one axis of length num_vertices (one area sum per vertex).
        """
        triangles = self.get_triangles()
        triangle_areas = self.get_triangle_areas()
        triangle_vertex_indices_flat = triangles.ravel()
        triangle_areas_repeated = np.repeat(triangle_areas, 3)
        return np.bincount(triangle_vertex_indices_flat, weights=triangle_areas_repeated,
                           minlength=self.get_num_vertices())

    @staticmethod
    def combine_meshes(mesh_list: List[Mesh]) -> Mesh:
        """
        Combine a list of meshes into a single mesh. Vertex colors are also integrated.
        If no mesh has colors, the returned meshes won't have colors. If only some meshes do, the combined mesh
        will have gray colors ([0.5, 0.5, 0.5]) at those vertices, where the respective mesh doesn't have colors.
        :mesh_list: List of meshes (each of type Mesh)
        :return combined mesh of type Mesh
        """
        vertices_per_mesh = [mesh.get_vertices() for mesh in mesh_list]
        vertex_cum_sum = np.cumsum([len(vertices_ind) for vertices_ind in vertices_per_mesh])
        # Vertices are simply concatenated
        combined_vertices = np.concatenate(vertices_per_mesh, axis=0)
        # Triangles are raised by the number of all previous vertices for each mesh
        # (so no raise for first mesh, raise by first mesh's num vertices for second mesh, etc.)
        combined_triangles = np.concatenate(
            [mesh.get_triangles() + vertex_cum_sum_ind for mesh, vertex_cum_sum_ind in
             zip(mesh_list, [0, *vertex_cum_sum[:-1]])], axis=0)
        combined_mesh = Mesh(vertices=combined_vertices, triangles=combined_triangles)

        # Colors are concatenated, and -1 is used for those meshes without colors
        combined_colors = np.concatenate(
            [mesh.get_colors() if mesh.has_colors() else np.repeat(np.asarray([[-1, -1, -1]]), len(vertices_ind), axis=0)
             for mesh, vertices_ind in zip(mesh_list, vertices_per_mesh)])
        # don't add colors to combined mesh if no input mesh has colors
        if len(np.where(combined_colors != -1)[0]) > 0:
            # Replace -1 colors with 0.5 (gray)
            combined_colors[combined_colors == -1] = 0.5
            combined_mesh.set_colors(combined_colors)

        return combined_mesh

    def get_copy(self) -> "Mesh":
        """
        Return a copy of the mesh. A new Mesh instance is created with both vertices and triangles arrays
        being copied.
        :return a new Mesh instance with identical vertices and triangles as attributes.
        """
        return Mesh(
            self.vertices.copy(), self.triangles.copy(), colors=self.colors,
            landmarks_or_points_array=self.landmarks, curvilinear_features_or_line_sets=self.curvilinear_features,
            uvs=self.uvs, material_ids=self.material_ids,
            textures=self.textures, virtual_connections=self.virtual_connections,)

    def get_open3d_mesh(self, compute_normals: bool = False, use_legacy: bool = False, copy: bool = False):
        """
        Convert self to an open3d triangle mesh. Optionally use legacy type.
        :param compute_normals: Whether to compute the normals of the triangle mesh (useful for visualization)
        :param use_legacy: Whether to use open3d's legacy triangle mesh class, which supports some other methods.
        :param copy: Set True to copy attributes (vertices, triangles, etc.)
        :return: triangle mesh of type open3d.geometry.TriangleMesh or open3d.t.geometry.TriangleMesh
        """
        open3d_mesh = open3d.geometry.TriangleMesh(
            vertices=open3d.utility.Vector3dVector(self.get_vertices(copy=copy)),
            triangles=open3d.utility.Vector3iVector(self.get_triangles(copy=copy))
        )
        if compute_normals:
            open3d_mesh.compute_vertex_normals()
        if self.has_colors():
            open3d_mesh.vertex_colors = open3d.utility.Vector3dVector(self.get_colors(copy=copy))
        if self.has_uvs():
            open3d_mesh.triangle_uvs = open3d.utility.Vector2dVector(np.reshape(self.get_uvs(copy=copy), (-1, 2)))
        if self.has_textures():
            triangle_mat_ids, textures = self.get_textures(copy=copy)
            open3d_mesh.triangle_material_ids = open3d.utility.IntVector(triangle_mat_ids)
            open3d_mesh.textures = [open3d.geometry.Image(np.ascontiguousarray(
                np.zeros((0,0,0), dtype=np.uint8) if texture is None else texture)) for texture in textures]
        if use_legacy:
            open3d_mesh = open3d.t.geometry.TriangleMesh.from_legacy(open3d_mesh)
        return open3d_mesh

    @staticmethod
    def get_from_open3d_mesh(open3d_mesh) -> Mesh:
        """
        Convert open3d Mesh to this Mesh class.
        :param open3d_mesh: open3d triangle mesh of type open3d.geometry.TriangleMesh or open3d.t.geometry.TriangleMesh
        :return corresponding mesh of type Mesh
        """
        if isinstance(open3d_mesh, open3d.geometry.TriangleMesh):
            mesh = Mesh(vertices=np.asarray(open3d_mesh.vertices), triangles=np.asarray(open3d_mesh.triangles))
            if len(open3d_mesh.vertex_colors) > 0:
                mesh.set_colors(np.asarray(open3d_mesh.vertex_colors))
            if len(open3d_mesh.triangle_uvs) > 0:
                triangle_uvs_flat = np.asarray(open3d_mesh.triangle_uvs)
                triangle_uvs = np.reshape(triangle_uvs_flat, (-1, 3, 2))
                if len(triangle_uvs) == mesh.get_num_triangles():
                    mesh.set_uvs(triangle_uvs)
                else:
                    warnings.warn("Mesh has triangle UVs that are not compatible with the triangles. "
                                  "Sometimes, Meshlab can help fix such cases.")
            if len(open3d_mesh.triangle_material_ids) > 0:
                assert len(open3d_mesh.textures) > 0
                mesh.set_textures(
                    material_ids=np.asarray(open3d_mesh.triangle_material_ids),
                    textures=[None if texture.is_empty() else np.asarray(texture) for texture in open3d_mesh.textures]
                )

            return mesh
        elif isinstance(open3d_mesh, open3d.t.geometry.TriangleMesh):
            mesh = Mesh(open3d_mesh.vertex.positions.numpy(), open3d_mesh.triangle.indices.numpy())
            if hasattr(open3d_mesh.vertex, "colors") and len(open3d_mesh.vertex.colors) > 0:
                mesh.set_colors(open3d_mesh.vertex.colors.numpy())
            return mesh
        else:
            raise TypeError(f"Unknown input for open3d_mesh: {open3d_mesh}")

    def get_boundaries(self, indices_or_mask: Union[np.ndarray, None] = None):
        """
        Get indices of points on boundary/ies of a mesh.
        Optionally, the boundary/ies of a subset of vertices can be computed by providing the index array.
        This implementation does not require additional libraries. It might be slower than using Open3D's
        HalfEdgeTriangleMesh, but it does not hash the points in a way that a mesh that has multiple vertices
        at the same location breaks everything.
        Credit for the implementation of this method goes to Ruben Schenk, who wrote it as part of his master's thesis.
        @param indices: Optional, index or mask array specifying for which vertices to compute the boundaries.
        @return: List of vertex boundary indices (so one index array per boundary, indices are always in the frame
                 of the original mesh, not in the one from the indices_or_mask if that was provided). The list
                 is sorted by size, so the longest boundaries come first.
        """

        def compute_boundary_edges(triangles):
            """
            Given an (N,3) numpy array of triangle indices (local to a submesh),
            return a list of edges (as sorted tuples) that appear in exactly one triangle.
            """
            edge_count = defaultdict(int)
            for tri in triangles:
                # Create the three edges of the triangle in a canonical (sorted) order.
                edges = [tuple(sorted((tri[i], tri[(i + 1) % 3]))) for i in range(3)]
                for edge in edges:
                    edge_count[edge] += 1
            # Boundary edges are those that appear only once.
            boundary_edges = [edge for edge, count in edge_count.items() if count == 1]
            return boundary_edges

        def order_boundary_loops(boundary_edges):
            """
            Given a list of boundary edges (each a tuple of vertex indices),
            build an undirected graph and extract each connected boundary as an ordered list.
            For open boundaries (endpoints with degree 1), the walk starts at an endpoint;
            for closed loops, the starting vertex is chosen arbitrarily.
            Returns a list of boundaries (each is an ordered list of vertex indices).
            """
            # Build a simple graph: vertex -> list of connected vertices.
            graph = defaultdict(list)
            for u, v in boundary_edges:
                graph[u].append(v)
                graph[v].append(u)

            visited_nodes = set()
            current_boundaries = []

            # Process each connected component in the boundary graph.
            for start in graph:
                if start in visited_nodes:
                    continue
                # Extract the full connected component via BFS/DFS.
                comp = set()
                queue = deque([start])
                while queue:
                    curr = queue.popleft()
                    if curr in comp:
                        continue
                    comp.add(curr)
                    for nb in graph[curr]:
                        if nb not in comp:
                            queue.append(nb)
                visited_nodes.update(comp)

                # Order the vertices in this component.
                comp_list = list(comp)
                # Determine if the boundary is open (i.e. an endpoint exists).
                endpoints = [v for v in comp_list if len(graph[v]) == 1]
                if endpoints:
                    current = endpoints[0]
                else:
                    # For closed loops, choose an arbitrary starting vertex.
                    current = comp_list[0]
                ordered = [current]
                prev = None
                while True:
                    neighbors = graph[current]
                    if not neighbors:
                        break
                    # Choose the neighbor that isn’t the one we just came from.
                    if len(neighbors) == 1:
                        nxt = neighbors[0]
                    else:
                        nxt = neighbors[0] if neighbors[0] != prev else neighbors[1]
                    # For a closed loop, stop if we have looped back to the start.
                    if nxt == ordered[0]:
                        break
                    ordered.append(nxt)
                    prev, current = current, nxt
                    # For open boundaries, stop when reaching a dead end.
                    if len(graph[current]) == 1:
                        break
                current_boundaries.append(ordered)
            return current_boundaries

        # If a subset of vertices is specified, remove the others.
        removed_vertex_indices = None
        if indices_or_mask is not None:
            removed_vertex_indices = IndicesAndMasks.invert(indices_or_mask, self)
            mesh_removed = self.remove_vertices(removed_vertex_indices)
        else:
            mesh_removed = self

        # Compute boundaries using a graph-based algorithm.

        # Cluster triangles by connectivity.
        triangle_components, _ = mesh_removed.get_disconnected_components(
            return_vertex_components=False)

        boundaries_components = []

        # Get the triangles from the original mesh.
        all_triangles = mesh_removed.get_triangles(copy=True)

        for triangle_component in triangle_components:
            # Extract triangles belonging to the current component.
            comp_triangles = all_triangles[triangle_component]

            # Build a submesh: get unique vertices used by these triangles and a mapping.
            unique_vertices, inv_indices = np.unique(comp_triangles, return_inverse=True)
            sub_triangles = inv_indices.reshape(-1, 3)

            # Compute the boundary edges on the submesh.
            sub_boundary_edges = compute_boundary_edges(sub_triangles)
            # Order the boundary loops on the submesh (in its local vertex index space).
            sub_boundaries = order_boundary_loops(sub_boundary_edges)
            # Map submesh vertex indices back to original mesh indices.
            comp_boundaries = []
            for boundary in sub_boundaries:
                mapped = [int(unique_vertices[v]) for v in boundary]
                comp_boundaries.append(mapped)

            boundaries_components.append(comp_boundaries)

        # Flatten all boundaries (across all components) into a single list.
        boundaries = [np.asarray(boundary) for comp_boundaries in boundaries_components for boundary in comp_boundaries]

        # We sort the boundaries by size, such that we always get the biggest boundary first
        boundaries = sorted(boundaries, key=lambda i: len(i), reverse=True)

        # If vertices were removed, map the indices back to the original mesh.
        if removed_vertex_indices is not None:
            boundaries = [IndicesAndMasks.get_old_indices_before_deletion(
                boundary, deleted_indices_or_mask=removed_vertex_indices)
                for boundary in boundaries]

        return boundaries

    def mark_indices(
            self, indices_or_mask: Union[np.ndarray, List], in_place: bool = False,
            marking_colors: Union[np.ndarray, List] = None, remaining_colors: Union[np.ndarray, List] = None,
            keep_current_colors: bool = False) -> Mesh:
        """
        Mark vertices on a mesh via an index or mask array. By default, the mesh will be made gray and the indexed
        vertices colored red, but the colors are fully adjustable. This method is mainly meant for visualization
        purposes.
        :param indices_or_mask: indices or mask array, indicating which vertices to mark.
        :param in_place: By default, a new mesh instance is created and returned.
                         Set in_place to True to assign the marking colors to the current Mesh instance.
        :param marking_colors: Which colors to assign to the indexed region (default [1, 0, 0], can also be
                               individual colors for each indexed vertex, but make sure dimensionality matches).
        :param remaining_colors: Same as for marking_colors, but concerning the vertices not indexed.
                                 Default is gray ([0.7, 0.7, 0.7]). You can also choose keep_current_colors
                                 to keep the mesh's previous colors for the unindexed vertices.
        :param keep_current_colors: Set to True to keep the mesh's previous colors for the unindexed vertices.
        :return: Mesh instance with marked colors, potentially the same instance if in_place is True.
        """
        if keep_current_colors and self.has_colors():
            if remaining_colors is not None:
                raise AssertionError("You chose that the current colors should be kept in the unindexed region, "
                                     "but you also chose to assign another color to the unindexed region. "
                                     "Choose one of the two.")
            new_colors = self.get_colors(copy=True)
        else:
            new_colors = np.ones_like(self.get_vertices()) * 0.7
        marking_colors = np.asarray([1, 0, 0]) if marking_colors is None else np.asarray(marking_colors)
        new_colors[indices_or_mask] = marking_colors
        if remaining_colors is not None:
            new_colors[IndicesAndMasks.invert(indices_or_mask, num_indices_or_mesh=self)] = np.asarray(remaining_colors)

        if in_place:
            self.set_colors(new_colors)
            return self
        else:
            new_mesh = self.get_copy()
            new_mesh.set_colors(new_colors)
            return new_mesh

    def mark_several_indices(
            self, list_of_indices_or_masks: List[Union[np.ndarray, List]], in_place: bool = False,
            marking_colors: Union[List[Union[np.ndarray, List]], None] = None,
            remaining_colors: Union[np.ndarray, List] = None, keep_current_colors: bool = False) -> Mesh:
        """
        Mark several sets of index or mask arrays on the mesh, each with a different color.
        :param list_of_indices_or_masks: List of index or mask arrays, each specifying which vertices to mark.
        :param in_place: By default, a new mesh instance is created and returned. Set True to apply in-place
                         (mesh is still returned)
        :param marking_colors: Optionally provide the colors to mark for each index/mask array. By default, the
                               Landmark color scheme is used (gradient from blue over green to red).
        :param remaining_colors: Optionally specify which colors to color all vertices in that are not indexed by any
                                 of the provided index/mask arrays.
        :param keep_current_colors: Set True to keep the colors for all vertices that are not indexed by any of the
                                    provided index/mask arrays.
        :return: Mesh instance with all indices/masks marked.
        """
        if in_place:
            mesh_to_mark = self
        else:
            mesh_to_mark = self.get_copy()

        if marking_colors is None:
            marking_colors = Landmarks.get_landmark_colors_static(len(list_of_indices_or_masks))

        for num_indices, indices_or_mask in enumerate(list_of_indices_or_masks):
            marking_colors_current = marking_colors[num_indices]
            if num_indices == 0:
                mesh_to_mark.mark_indices(
                    indices_or_mask, marking_colors=marking_colors_current, in_place=True,
                    remaining_colors=remaining_colors, keep_current_colors=keep_current_colors)
            else:
                mesh_to_mark.mark_indices(indices_or_mask, marking_colors=marking_colors_current,
                                          in_place=True, keep_current_colors=True)

        return mesh_to_mark

    def mark_boundaries(self, indices_or_mask: Union[np.ndarray, List, None] = None,
                        in_place: bool = False, **kwargs) -> Mesh:
        boundaries = self.get_boundaries(indices_or_mask=indices_or_mask)
        return self.mark_several_indices(boundaries, in_place=in_place, **kwargs)

    def mark_disconnected_components(self, indices_or_mask: Union[np.ndarray, List, None] = None,
                                     in_place: bool = False, **kwargs) -> Mesh:
        _, _, vertex_components = self.get_disconnected_components(
            indices=indices_or_mask, return_vertex_components=True)
        return self.mark_several_indices(vertex_components, in_place=in_place, **kwargs)

    def get_trimesh_mesh(self):
        import trimesh
        return trimesh.Trimesh(vertices=self.vertices, faces=self.triangles, process=False, vertex_colors=self.colors)

    @staticmethod
    def get_from_trimesh_mesh(trimesh_mesh):
        mesh = Mesh(vertices=trimesh_mesh.vertices, triangles=trimesh_mesh.faces)
        if (not hasattr(trimesh_mesh.visual, "defined") or trimesh_mesh.visual.defined) \
               and hasattr(trimesh_mesh.visual, "vertex_colors") and len(trimesh_mesh.visual.vertex_colors) > 0:
            mesh.set_colors(np.asarray(trimesh_mesh.visual.vertex_colors)[:, :3].astype(float) / 255)
        return mesh

    @staticmethod
    def get_sphere(sphere_radius: float, sphere_resolution: int = 8, position: np.ndarray = None,
                   color: Union[np.ndarray, None] = None) -> Mesh:
        """
        Return a sphere mesh with a choosable radius.
        By default, the sphere is centered at the origin, but the position can be chosen, as can the resolution
        and a color. Note that this method currently requires open3d for the creation of the sphere mesh.
        :param sphere_radius: Float value for sphere radius, must be specified.
        :param sphere_resolution: Number of circles to form the sphere, 8 by default.
        :param position: 3D location of sphere center. Default [0, 0, 0].
        :param color: Optionally assign a color (3D array) to the sphere.
        :return: sphere mesh of type Mesh.
        """
        sphere = Mesh.get_from_open3d_mesh(open3d.geometry.TriangleMesh.create_sphere(
            radius=sphere_radius, resolution=sphere_resolution))
        if position is not None:
            sphere.translate(position)
        if color is not None:
            sphere.set_colors(color)
        return sphere

    @staticmethod
    def get_cylinder(
            end_point_or_length: Union[np.ndarray, float], radius: float, start_point: Union[None, np.ndarray] = None,
            resolution: int = 5, color: Union[None, np.ndarray] = None) -> Mesh:
        if isinstance(end_point_or_length, (list, np.ndarray)):
            assert start_point is not None
            length = np.linalg.norm(np.asarray(end_point_or_length) - np.asarray(start_point))
        else:
            length = end_point_or_length
        cylinder = Mesh.get_from_open3d_mesh(open3d.geometry.TriangleMesh.create_cylinder(
            radius, length, resolution=resolution))
        if start_point is not None:
            assert isinstance(end_point_or_length, (list, np.ndarray))
            connection_vector = np.asarray(end_point_or_length) - np.asarray(start_point)
            cylinder.rotate(*utils.get_axis_angle_rotation_between_vectors([0, 0, 1], connection_vector))
            cylinder.translate(start_point + connection_vector/2)
        if color is not None:
            cylinder.set_colors(color)
        return cylinder



    def remove_vertices(self, indices_or_mask: Union[np.ndarray, List], invert: bool = False,
                        return_kept_indices: bool = False) -> Union[Mesh, Tuple[Mesh, np.ndarray, np.ndarray]]:
        """
        Remove indexed vertices from mesh. This does not happen in-place, i.e.,
        a new Mesh instance is created and returned. Optionally, you can choose to also return the indices of the
        vertices/colors and triangles that were kept, which can be useful for later back-matching.
        The method for removing goes as follows: For vertices and colors, we simply only keep the indexed ones.
        For triangles, we keep only those that have all vertices still defined (vertex indices adjusted accordingly).
        Landmarks are all kept, even if the closest vertex is removed.
        Currently, textures are ignored in this method.
        :param indices_or_mask: Indices or mask of vertices to remove.
        :param invert: Set to True if the indices_or_mask actually represents the vertices you want to keep.
        :param return_kept_indices: Set to True to return indices of vertices and indices of triangles that were kept
                                    in addition to the new Mesh instance.
        :return: new Mesh instance with removed vertices. If return_kept_indices is True, then a Tuple is returned with
                 (Mesh, vertex_indices_kept, triangle_indices_kept)
        """
        mask_to_remove = IndicesAndMasks.get_mask(indices_or_mask, num_indices_or_mesh=self)
        if invert:
            mask_to_remove = IndicesAndMasks.invert(mask_to_remove, num_indices_or_mesh=self)

        mask_to_keep = IndicesAndMasks.invert(mask_to_remove)
        indices_to_keep = IndicesAndMasks.get_indices(mask_to_keep)

        new_vertices = self.get_vertices(copy=True)[mask_to_keep]
        if self.has_colors():
            new_colors = self.get_colors(copy=True)[mask_to_keep]
        else:
            new_colors = None

        # Only triangles are kept which have all three vertices defined.
        # The vertex indices within the triangles also need to be adjusted,
        # since some vertices were removed.
        # This is an efficient, fully vector-based implementation to achieve this.
        current_triangles = self.get_triangles(copy=True)
        index_map = np.full(mask_to_keep.shape[0], -1, dtype=np.int64)
        index_map[indices_to_keep] = np.arange(indices_to_keep.shape[0], dtype=np.int64)

        mapped_triangles = index_map[current_triangles]
        triangle_mask = np.all(mapped_triangles != -1, axis=1)
        triangle_indices_to_keep = np.nonzero(triangle_mask)[0]
        new_triangles = mapped_triangles[triangle_mask]


        new_mesh = Mesh(vertices=new_vertices, triangles=new_triangles, colors=new_colors,
                    landmarks_or_points_array=self.landmarks)

        if return_kept_indices:
            return new_mesh, indices_to_keep, triangle_indices_to_keep
        else:
            return new_mesh

    def get_edges(self, exclude_virtual_connections: bool = False) -> np.ndarray:
        """
        Return all triangle mesh edges (including directional, so [0, 1] and [1, 0] might be in the returned array.
        Duplicates are also not handled.
        Note that if you defined any virtual connections for this mesh, they are included here,
        unless you explicitly choose differently.
        :param exclude_virtual_connections: Set True to only return the edges defined by the triangles.
        :return: numpy int array of shape [num_edges, 2]
        """
        # Collect all edges from the triangles
        edges = np.vstack([
            self.triangles[:, [0, 1]],
            self.triangles[:, [1, 2]],
            self.triangles[:, [2, 0]]
        ])

        if not exclude_virtual_connections and self.has_virtual_connections():
            edges = np.concatenate([edges, self.virtual_connections], axis=0)

        return edges

    def get_edges_unique(self, exclude_virtual_connections: bool = False) -> np.ndarray:
        """
        Return all unique triangle mesh edges (not directional, so [0, 1] and [1, 0] are the same edge).
        Note that if you defined any virtual connections for this mesh, they are included here,
        unless you explicitly choose differently.
        :param exclude_virtual_connections: Set True to only return the edges defined by the triangles.
        :return: numpy int array of shape [num_unique_edges, 2]
        """

        edges = self.get_edges(exclude_virtual_connections=exclude_virtual_connections)

        # Sort vertex indices in each edge so that [a, b] and [b, a] are considered the same
        edges = np.sort(edges, axis=1)

        # Use numpy unique to get only unique edges
        return np.unique(edges, axis=0)

    def get_average_edge_length(self, exclude_virtual_connections: bool = False) -> float:
        """
        Return the average edge length of the mesh.
        The unique edges are retrieved first, then the length for each is computed,
        which is finally averaged and returned.
        """
        unique_edges = self.get_edges_unique(exclude_virtual_connections=exclude_virtual_connections)
        unique_edge_lengths = np.linalg.norm(self.vertices[unique_edges], axis=1)
        return np.mean(unique_edge_lengths)

    def get_edges_per_vertex(self, return_dict: bool = True, exclude_virtual_connections: bool = False) -> Union[Dict, List[np.ndarray]]:
        """
        Return for each mesh vertex all edges that are attached to that vertex.
        :param return_dict: Whether to return dictionary or list.
        :param exclude_virtual_connections: Set True to only include the edges defined by the triangles.
        :return dictionary {vertex_ind: [connected_vertex_ind0, connected_vertex_ind1,...]
                or list [[connected_vertex_ind0, connected_vertex_ind1,...], [connected_vertex_ind0, connected_vertex_ind1,...], ..]
        """
        edges = self.get_edges_unique(exclude_virtual_connections=exclude_virtual_connections)
        num_vertices = self.get_num_vertices()
        if return_dict:
            edges_per_vertex = {vertex_ind: [] for vertex_ind in range(num_vertices)}
        else:
            edges_per_vertex = [[] for _ in range(num_vertices)]
        for edge in edges:
            edges_per_vertex[edge[0]].append(edge[1])
            edges_per_vertex[edge[1]].append(edge[0])
        return edges_per_vertex

    def get_triangles_per_edge(self) -> Dict[Tuple[int], List[int]]:
        """
        Get a dictionary that includes for each edge the indices of the triangles that contain this edge
        :return: Dictionary with edge as key (Edge is represented as a sorted tuple of its two vertex indices), and
                 list of triangle indices as value, the indices referring to this Mesh object's triangles attribute.
        """
        triangles = self.get_triangles(copy=False)
        edge_to_tris = defaultdict(list)
        for num_triangle, triangle in enumerate(triangles):
            for edge_beg, edge_end in zip(triangle, [*triangle[1:], triangle[0]]):
                edge_to_tris[tuple(sorted([edge_beg, edge_end]))].append(num_triangle)
        return edge_to_tris


    def get_vertex_normals(self, normalize: bool = True, use_area_weighting: bool = True) -> np.ndarray:
        """
        Get the mesh's vertex normals. By default, area weighting over the surrounding triangles
        is used, and the output normals are normalized.
        :param normalize: Set False to get the unnormalized vertex normals (probably not much use for this).
        :param use_area_weighting: Set False to perform uniform averaging for each vertex over
                                   its surrounding triangles' normals.
        :return numpy array of shape [num_vertices, 3] containing the normalized/unnormalized vertex normals.
        """
        # Area weighting uses the unnormalized triangle normals, which are the cross product
        # of two edges. Since the lengths of these normals are all equally proportional to the triangle
        # areas (factor 2), summing all of a vertex' surrounding triangle normals puts more emphasis on the
        # longer vectors, thus having this area-weighting.
        # If we normalize the triangle normals first, we basically perform uniform averaging over the triangle normals.
        triangle_normals = self.get_triangle_normals(normalize=not use_area_weighting)
        triangles = self.get_triangles(copy=False)
        vertex_normals = np.zeros_like(self.get_vertices(copy=False))
        for i in range(3):
            np.add.at(vertex_normals, triangles[:, i], triangle_normals)
        if normalize:
            vertex_normals = utils.normalize_vector(vertex_normals)
        return vertex_normals

    def get_triangle_normals(self, normalize: bool = True) -> np.ndarray:
        """
        Get the mesh's triangle normals, which are computed as the cross product between two of its edges, using
        common winding conventions. By default, the normals are normalized, but you can choose not to to get
        the raw cross product, which still contains information about the triangle size (the length of the vector
        is double the triangle's area).
        :param normalize: Set False to get raw cross product.
        :return: numpy array of shape [num_triangles, 3] containing the normalized/unnormalized triangle normals.
        """
        vertices = self.get_vertices(copy=False)
        triangles = self.get_triangles(copy=False)
        v0 = vertices[triangles[:, 0], :]
        v1 = vertices[triangles[:, 1], :]
        v2 = vertices[triangles[:, 2], :]
        # (F, 3), unnormalized triangle normals
        triangle_normals = np.cross(v1 - v0, v2 - v0, axis=-1)
        if normalize:
            triangle_normals = utils.normalize_vector(triangle_normals)
        return triangle_normals

    def get_vertex_neighbors(self) -> List[np.ndarray]:
        """
        Returns each vertex' neighbor as a list, cf. docs of get_edges_per_vertex()
        """
        return self.get_edges_per_vertex(return_dict=False)

    def get_adjacency_matrix(self, normalize_num_neighbors: bool = False):
        """
        Build a sparse adjacency matrix for the mesh. Optionally don't put 1s for the neighbors,
        but the inverse of the number of neighbors, which can be useful when you want to average over the
        vertex neighbors.
        :return scipy csc_matrix that contains all vertex neighbors
        """
        import scipy.sparse as spsp
        neighbors = self.get_vertex_neighbors()
        row_idx = []
        col_idx = []
        data = []

        for v, nbrs in enumerate(neighbors):
            deg = len(nbrs)
            if deg == 0:
                continue
            row_idx.extend([v] * deg)
            col_idx.extend(nbrs)
            if normalize_num_neighbors:
                data.extend([1/deg] * deg)
            else:
                data.extend(np.asarray([1] * deg, dtype=np.uint8))

        num_vertices = self.get_num_vertices()
        adj = spsp.csc_matrix((data, (row_idx, col_idx)), shape=(num_vertices, num_vertices))

        return adj

    def convert_points_to_barycentric_coordinates(
            self, points_or_landmarks: Union[np.ndarray, List, Landmarks],
            triangle_indices: Union[List, np.ndarray, None] = None) -> np.ndarray:
        """
        Convert array of points into mesh-relative barycentric coordinates.
        :param points_or_landmarks: Points or Landmarks nx3
        :param triangle_indices: Optionally specify relative to which triangles to compute the
                                 barycentric coordinates for the provided points. If not provided,
                                 the closest triangles to the points will be used.
        :return: numpy array of barycentric coordinates nx3 containing normed alpha, beta, gamma values for each
                 point, each of which a normalized weight that represents how close the point is to the triangles
                 respective vertex.
        """
        points_np = Landmarks(points_or_landmarks).get_points()
        if triangle_indices is None:
            _, triangle_indices = self.get_closest_triangle_points(points_np)

        triangle_points = self.get_vertices(copy=False)[self.get_triangles(copy=False)[triangle_indices]]
        # Vectors from triangle vertices to point
        v0 = triangle_points[:, 2] - triangle_points[:, 0]
        v1 = triangle_points[:, 1] - triangle_points[:, 0]
        v2 = points_np - triangle_points[:, 0]

        # Compute dot products
        dot00 = np.einsum('ij,ij->i', v0, v0)
        dot01 = np.einsum('ij,ij->i', v0, v1)
        dot02 = np.einsum('ij,ij->i', v0, v2)
        dot11 = np.einsum('ij,ij->i', v1, v1)
        dot12 = np.einsum('ij,ij->i', v1, v2)

        # Compute barycentric coordinates
        inv_denom = 1 / (dot00 * dot11 - dot01 * dot01)
        gamma = (dot11 * dot02 - dot01 * dot12) * inv_denom
        beta = (dot00 * dot12 - dot01 * dot02) * inv_denom
        alpha = 1 - beta - gamma

        # return barycentric coordinates (Corresponds to [alpha, beta, gamma])
        return np.stack([alpha, beta, gamma], axis=1)


    def align_procrustes(
            self, target_mesh_or_landmarks: Union[Mesh, Landmarks, np.ndarray, Path],
            landmark_mask_or_indices: Union[List, np.ndarray, None] = None, reflection: bool = True,
            translation: bool = True, scale: bool = True, in_place: bool = False,
            return_transform: bool = False) -> Union[Mesh, Tuple[Mesh, np.ndarray]]:
        """
        Uses procrustes method to get the best alignment from this mesh object to another,
        using only the landmarks with procrustes alignment (no ICP).
        :param target_mesh_or_landmarks: Mesh to align this mesh with, must have the same number of
                                         landmarks as this mesh (at least three). It can also just be the landmarks.
        :param landmark_mask_or_indices: Optionally provide a mask or indices to indicate which landmarks
                                         to use for alignment.
        :param reflection: Set False to rule out the possibility of flipping/reflecting the mesh.
        :param translation: Set False to rule out any translation (not so common).
        :param scale: Set False to rule out any rescaling (common when you want preserve the mesh size).
        :param in_place: Set True to transform this mesh. Doesn't change the return.
        :param return_transform: Set True to return the transformation matrix as well.
        :return: If return_transform is False, transformed Mesh object is returned.
                 If return_transform is True, a tuple with the transformed Mesh object
                                              and the numpy transformation matrix is returned.
        """
        assert self.has_landmarks()
        source_landmarks = self.get_landmarks()
        if isinstance(target_mesh_or_landmarks, Mesh):
            assert target_mesh_or_landmarks.has_landmarks()
            target_landmarks = target_mesh_or_landmarks.get_landmarks()
        else:
            target_landmarks = Landmarks(target_mesh_or_landmarks)
        transformation_matrix, _ = source_landmarks.procrustes(
            target_landmarks, landmark_mask_or_indices=landmark_mask_or_indices, in_place=False,
            reflection=reflection, translation=translation, scale=scale)
        aligned_mesh = self.transform(transformation_matrix, in_place=in_place)
        if return_transform:
            return aligned_mesh, transformation_matrix
        else:
            return aligned_mesh


    def smooth_laplacian(self, num_iterations: int, lambda_fact: float = 0.5, indices: np.ndarray = None) -> "Mesh":
        """
        Apply Laplacian smoothing with a simple forward Euler step v <- v + lambda*(avg(N) - v).
        Not in-place, i.e., a new Mesh instance is created and returned.
        This method first builds a sparse matrix for averaging over the vertex neighbors before computing
        the update steps as described above.
        :param num_iterations: Number of vertex update steps
        :param lambda_fact: Laplacian smoothing factor. Choose between 0 and 1.
        :param indices: Optionally apply the smoothing only to a subset of the vertices.
                        The remaining vertices will keep the smoothed vertices from fully collapsing
                        after many iterations.
        :return new Mesh instance with smoothed vertices
        """
        if lambda_fact <= 0:
            raise ValueError("Lambda smoothing factor must be positive.")
        if lambda_fact > 1:
            warnings.warn("The forward Euler step is very unstable when setting lambda above 1.")

        # Get the mesh vertices and their number
        new_vertices = self.get_vertices(copy=True)
        num_vertices = self.get_num_vertices()

        # Define the vertex indices over which to apply the smoothing
        indices = np.arange(num_vertices) if indices is None else np.asarray(indices)

        # Compute the adjacency matrix for efficient averaging over the vertex neighbors
        adj_mat = self.get_adjacency_matrix(normalize_num_neighbors=True)
        adj_mat = adj_mat[indices]

        for _ in range(num_iterations):
            # Compute average over the vertex neighbors
            averages = adj_mat @ new_vertices
            # Compute weighted average between each vertex and the average of its vertex neighbors
            # with lambda defining the weighting of the averaging.
            new_vertices[indices] = (1 - lambda_fact) * new_vertices[indices] + lambda_fact * averages

        return Mesh(vertices=new_vertices, triangles=self.triangles)

    def get_edge_sorted_selection(self, indices_or_mask, cut_big_circles: bool = False) -> List[List[int]]:
        """
        Given a selection of one or multiple distinct lines on a mesh (one selected line consists of multiple selected
        vertices connected via single edges, two edges possible if line is circularly closed), this method returns a list,
        where each element is one of these lines, so the vertex indices belonging to one line,
        sorted so that one vertex comes after another connected one. Make sure there's only at most one big circle
        in any of your selected lines. This is a simple greedy algorithm that cannot handle things otherwise.
        The returned line indices are always unique unless there's one big circle, in which case the beginning
        and end index are the same; you can choose to remove this duplicated last index via cut_big_circles.
        :param indices_or_mask: Index selection (bool or int).
        :param cut_big_circles: For every circular line selection, remove the duplicated index entry at the end.
        :return: List of sorted index selections, could look like this: [[0, 1, 2], [3, 4, 5, 6], [7]]
        """
        edges_per_vertex = self.get_edges_per_vertex()
        indices = IndicesAndMasks.get_indices(indices_or_mask, make_unique=True)
        mask = IndicesAndMasks.get_mask(indices_or_mask, num_indices_or_mesh=self)
        added_indices = set()
        sorted_index_sets = []
        while True:
            # We start at the first vertex from the index selection that hasn't been added yet
            possible_current_indices = [idx for idx in indices if idx not in added_indices]
            if len(possible_current_indices) == 0:
                break
            else:
                current_index = possible_current_indices[0]
            # We define a new list of indices, starting with this first vertex
            sorted_index_set = [current_index]
            added_indices.add(current_index)
            next_indices = [vertex_idx for vertex_idx in edges_per_vertex[current_index]
                            if vertex_idx not in added_indices and mask[vertex_idx]]
            if not len(next_indices) <= 2:
                raise AssertionError("You provided an index selection, where a vertex is connected to more than "
                                     "two other selected vertices, meaning the index selection cannot represent a line.")
            # For each selected vertex that's connected to the first vertex, we iterate over the line selection
            # and add all other selected and connected vertices in that direction to our list of indices
            for num_next_index, next_index in enumerate(next_indices):
                current_index = next_index
                added_indices.add(current_index)
                sub_sorted_index_set = [current_index]
                while True:
                    next_next_indices = [vertex_idx for vertex_idx in edges_per_vertex[current_index]
                                         if vertex_idx not in added_indices and mask[vertex_idx]]
                    if not len(next_next_indices) <= 1:
                        raise AssertionError(
                            "You provided an index selection, where a vertex is connected to more than "
                            "two other selected vertices, meaning the index selection cannot represent a "
                            "line.")
                    if len(next_next_indices) == 0:
                        break
                    current_index = next_next_indices[0]
                    sub_sorted_index_set.append(current_index)
                    added_indices.add(current_index)
                # We add the new selected and connected vertices to our list. In the first round, we simply append the
                # them to the first vertex. In the second round, we reverse the order and put the vertices in front of
                # the other vertices. This considers the fact that the first vertex we find is very probably not the first
                # or last vertex of the selected line, so there's two directions to go from that vertex, but we want our
                # final sorted index set to start at the beginning or the end of the selected line, not somewhere in the
                # middle.
                if num_next_index == 0:
                    sorted_index_set = [*sorted_index_set, *sub_sorted_index_set]
                else:
                    sorted_index_set = [*sub_sorted_index_set[::-1], *sorted_index_set]
            if sorted_index_set[0] == sorted_index_set[-1] and cut_big_circles:
                sorted_index_set = sorted_index_set[:-1]
            sorted_index_sets.append(sorted_index_set)
        return sorted_index_sets

    def get_shortest_path_between_vertices(self, vertex_indices, use_edge_length_weight: bool = True):
        """
        Get the shortest path (Dijkstra based on edge lengths) between a set of vertices on a mesh.
        :param vertex_indices: List of vertex indices. Must contain at least two vertices.
                               Do not provide a mask here (they are unsorted)
        :param use_edge_length_weight: Set False to weigh each edge evenly.
                                       By default, the edge length is incorporated when finding the shortest path.
        :return: List of connected vertex indices starting at first provided vertex index and ending at last.
        """
        if IndicesAndMasks.is_mask(vertex_indices):
            raise ValueError("You must provide the vertices as indices here. "
                             "Masks do not make sense here, since they are unsorted.")
        if len(vertex_indices) < 2:
            raise ValueError("You must provide at least two vertices.")
        # edges without duplication
        edges = self.get_edges_unique()
        import networkx

        if use_edge_length_weight:
            # the actual length of each unique edge
            edge_lengths = np.linalg.norm(self.vertices[edges], axis=1)

            # Create graph from edge data
            ga = networkx.from_edgelist([(e[0], e[1], {'length': L}) for e, L in zip(edges, edge_lengths)])
            weight = "length"
        else:
            ga = networkx.from_edgelist(edges)
            weight = None

        path = [vertex_indices[0]]
        for num_vertex, vertex_index in enumerate(vertex_indices[:-1]):
            next_vertex_index = vertex_indices[num_vertex + 1]

            # run the shortest path query using length for edge weight
            path_tmp = networkx.shortest_path(ga, source=vertex_index, target=next_vertex_index, weight=weight)
            path += path_tmp[1:]

        return path

    def get_mesh_size(self, use_pca: bool = False) -> float:
        """
        Get the mesh extent along its longest axis (so vertex max minus vertex min along x, y, or z, whichever gives
        the highest value). Choose use_pca to find the longest extent not in direction of the three default axis,
        but really the longest extent in any direction.
        :param use_pca: Set True to find longest mesh extent in any direction.
        :return: float representing mesh extent in direction of longest axis.
        """
        vertices = self.get_vertices(copy=False)
        if use_pca:
            # Center vertices
            centered = vertices - np.mean(vertices, axis=0)

            # PCA via SVD: principal axes are rows of Vt
            _, _, Vt = np.linalg.svd(centered, full_matrices=False)

            # Project points onto principal axes
            projected = centered @ Vt.T  # shape: (N, 3)

            # Extent along each PCA axis, then take the largest
            extents = np.max(projected, axis=0) - np.min(projected, axis=0)
            return np.max(extents)
        else:
            return np.max(np.max(vertices, axis=0) - np.min(vertices, axis=0))

    def get_landmark_mesh(self, sphere_proportion: float = 0.005, **kwargs) -> Union[Mesh, List[Mesh]]:
        """
        Get the mesh's landmarks as a mesh of spheres, each sphere centered at it's corresponding landmark's location.
        The size of the spheres is chosen proportional to the actual mesh size (longest axis).
        :param sphere_proportion: Sphere radius relative to the mesh's longest axis. Default 0.5%.
        :param kwargs: Cf. :func:`~Landmarks.get_mesh()` method for choosing sphere colors, resolution, etc.
        :return Mesh containing all spheres.
        """
        if not self.has_landmarks():
            raise AssertionError("Current mesh doesn't have any landmarks assigned, so no mesh can be generated.")

        return self.landmarks.get_mesh(sphere_radius=self.get_mesh_size() * sphere_proportion, **kwargs)

    def get_curvilinear_feature_mesh(self, cylinder_proportion: float = 0.0025, **kwargs) -> Union[Mesh, List[Mesh]]:
        """
        Get the mesh's curvilinear features as a mesh of cylinders, each cylinder connecting
        two neighboring points in a line set.
        The size of the cylinders is chosen proportional to the actual mesh size.
        :param cylinder_proportion: Cylinder radius relative to the mesh's longest axis. Default 0.25%.
        :param kwargs: Cf. :func:`~CurvilinearFeatures.get_mesh()` method for choosing cylinder colors, resolution, etc.
        :return Mesh containing all cylinders.
        """
        if not self.has_curvilinear_features():
            raise AssertionError("Current mesh doesn't have any curvilinear features assigned, "
                                 "so no mesh can be generated.")

        return self.curvilinear_features.get_mesh(cylinder_radius=self.get_mesh_size() * cylinder_proportion, **kwargs)

    def show(self, show_landmarks: bool = True, indices_or_mask_to_mark: Union[np.ndarray, List] = None,
             **render_kwargs) -> None:
        """
        Show the current mesh instance in a visualizer. By default, the landmarks are also shown (as a separate mesh).
        :param show_landmarks: Set this to False if you don't want the mesh's landmarks and curvilinear features
                               added to the visualizer (note that you can always hide them by pressing respective
                               number keys)
        :param indices_or_mask_to_mark: Optionally provide an index or mask array, such that the mesh will be
                                        visualized with the array marked on it.
        :param render_kwargs: Forwarded to :func:`~Mesh.show_multiple_meshes()`
        :return: None
        """
        if indices_or_mask_to_mark is not None:
            meshes_to_show = [self.mark_indices(indices_or_mask_to_mark, in_place=False)]
        else:
            meshes_to_show = [self]
        if show_landmarks:
            render_kwargs["mesh_show_wireframe"] = render_kwargs.get("mesh_show_wireframe", False)
            if self.has_landmarks():
                meshes_to_show.append(self.get_landmark_mesh(combine_into_single_mesh=True))
            if self.has_curvilinear_features():
                meshes_to_show.append(self.get_curvilinear_feature_mesh(combine_into_single_mesh=True))
        if self.has_textures() and self.has_uvs():
            render_kwargs["light_on"] = render_kwargs.get("light_on", False)
        self.show_multiple_meshes(*meshes_to_show, **render_kwargs)

    @staticmethod
    def show_multiple_meshes(*meshes: Mesh, switch_meshes: Union[List[Mesh], List[List[Mesh]], None] = None,
                             **render_kwargs) -> None:
        """
        Show multiple meshes in an Open3D visualizer. This is a static method, so even if you call it from an instance,
        you still need to provide this instance as argument to include it in the visualizer.
        - First 10 meshes can be toggled with number keys:
            "1" -> mesh 1, "2" -> mesh 2, ..., "0" -> mesh 10
        - Any meshes beyond the tenth are always visible.
        :param meshes: All meshes (type Mesh) to show
        :param switch_meshes: Also meshes you want to show, but you can switch between them via the key arrows.
                              This is a list, and each element can either be one mesh or also a list of meshes.
                              All meshes within the currently active element of this list are added to the meshes
                              provided via "meshes" arg, so you can still toggle their visibility via the number keys.
        :param render_kwargs: Includes some render arguments. Cf. the possible arguments of Open3D's render options.
        """
        from src.objects.visualizer import Visualizer
        vis = Visualizer(meshes, switch_geometries=switch_meshes, **render_kwargs)
        vis.run()



    def get_volume(self) -> float:
        """
        Compute and return scalar volume of a mesh. The mesh should be watertight.
        We use trimesh for the computation, since it seems to be less strict with the watertightness than Open3D.
        :return volume (float)
        """
        try:
            return self.get_trimesh_mesh().volume
        except ImportError:
            raise ImportError("To get the volume of a mesh, please install trimesh.")

    def get_disconnected_components(
            self, indices: Union[List[int], np.ndarray] = None, return_vertex_components: bool = True) -> Union[Tuple[List, np.ndarray], Tuple[List, np.ndarray, List]]:
        """
        Get disconnected components of the mesh, sorted by size descending,
        optionally only for certain vertices on that mesh.
        :param indices: Optional vertex index selection. Note that choosing indices may result
                        in a lossy conversion to original triangles, cf. code. It can also significantly increase the
                        computational cost, especially for large meshes, since the vertex_components need to be computed,
                        cf. code.
        :param return_vertex_components: Next to returning triangle components and their indices, also return
                                         vertex_components (third return element). Put this to False if vertex_components
                                         are not needed, especially for large meshes, since the conversion can take a lot
                                         of time, cf. code.
        :return: - List of disconnected components. Each component is a list of triangle indices belonging component.
                 - List of component indices for each triangle.
                 - List of disconnected components. Each component is a list of vertex indices belonging component.
        """
        do_indexing = indices is not None
        # remove vertices outside of index selection if provided
        if do_indexing:
            assert IndicesAndMasks.get_num_indexed(indices) > 0
            mesh_removed_indices, kept_vertex_indices, _ = self.remove_vertices(indices, invert=True, return_kept_indices=True)
        else:
            mesh_removed_indices = self

        #
        # Compute the disconnected components on the mesh with optionally removed vertices
        #

        num_triangles = mesh_removed_indices.get_num_triangles()
        edge_to_tris = mesh_removed_indices.get_triangles_per_edge()

        # For each edge shared by multiple triangles, connect all triangles incident to that edge.
        # In manifold meshes, an edge usually has 1 or 2 incident triangles.
        adjacency = [[] for _ in range(num_triangles)]
        for tris_on_edge in edge_to_tris.values():
            if len(tris_on_edge) < 2:
                continue
            # Connect all triangles on this edge.
            # Most commonly len==2, so this is just a single pair.
            # If non-manifold edge, len>2, we connect everyone to everyone.
            for i in range(len(tris_on_edge)):
                t_i = tris_on_edge[i]
                for j in range(i + 1, len(tris_on_edge)):
                    t_j = tris_on_edge[j]
                    adjacency[t_i].append(t_j)
                    adjacency[t_j].append(t_i)

        # BFS/DFS over triangles to label connected components
        triangle_component_indices = [-1] * num_triangles
        component_num_triangles: List[int] = []

        current_cluster = 0
        for start in range(num_triangles):
            if triangle_component_indices[start] != -1:
                continue

            # New component
            q = deque([start])
            triangle_component_indices[start] = current_cluster
            count = 0

            while q:
                t = q.popleft()
                count += 1

                for nbr in adjacency[t]:
                    if triangle_component_indices[nbr] == -1:
                        triangle_component_indices[nbr] = current_cluster
                        q.append(nbr)

            component_num_triangles.append(count)
            current_cluster += 1


        triangle_component_indices = np.asarray(triangle_component_indices)
        # create a list of lists where each sublist contains all the triangles that belong to that component
        triangle_components = [np.where(triangle_component_indices == i)[0] for i in
                               range(len(component_num_triangles))]

        # We sort the disconnected components by size, such that we always get the biggest components first
        triangle_components = sorted(triangle_components, key=lambda i: len(i), reverse=True)

        # Convert triangle to vertex components by converting each component's triangle to vertex indices.
        if do_indexing or return_vertex_components:
            triangles = mesh_removed_indices.get_triangles()
            vertex_components = [np.unique(triangles[triangle_component].flatten())
                                 for triangle_component in triangle_components]

            # If we computed the disconnected components on a mesh with removed vertices,
            # we have to translate the indices back to the original mesh.
            if do_indexing:
                vertex_components = [kept_vertex_indices[vertex_component] for vertex_component in vertex_components]
                # We compute the triangle components on the original mesh based on the vertex components on the original
                # mesh, i.e., all triangles that have all three vertices in a component are added to that component.
                # Not sure if this conversion is lossless, but oh well.
                triangle_components = [self.convert_vertex_to_triangle_indices(
                    vertex_component, num_vertices_per_triangle=3) for vertex_component in vertex_components]
                # Last, we compute the component index per triangle by using the indices stored in triangle_components.
                # Triangles that don't belong to a component (i.e., triangles that were removed for the computation)
                # get assigned the index -1.
                triangle_component_indices = np.zeros(self.get_num_triangles()) - 1
                for num_triangle_component, triangle_component in enumerate(triangle_components):
                    triangle_component_indices[triangle_component] = num_triangle_component
            if return_vertex_components:
                return triangle_components, triangle_component_indices, vertex_components

        return triangle_components, triangle_component_indices

    def remove_duplicate_vertices(
            self, remove_per_component: bool = False, also_remove_degenerate_triangles: bool = True,
            in_place: bool = False) -> Mesh:
        """
        Remove vertices that are at the same location. Note that this changes the topology,
        and I can't give the index mapping, since I'm using Open3D's remove_duplicated_vertices() function.
        @param remove_per_component: If your mesh has multiple components, choose this to only remove the duplicated
                                     vertices per component (so if one component has a vertex at the same location as
                                     another component's vertex, then these are not removed)
        @param also_remove_degenerate_triangles: Would set this to True if you don't have a specific reason not to.
                                                 Open3D doesn't automatically remove the degenerate triangles after calling
                                                 remove_duplicated_vertices().
        @param in_place: Set this to True if you want to change the current mesh instance.
        @return: Same mesh or pcd with duplicate vertices removed.
        """

        def do_cleaning(mesh_instance: Mesh):
            open3d_mesh_instance = mesh_instance.get_open3d_mesh(copy=not in_place)
            open3d_mesh_instance.remove_duplicated_vertices()
            if also_remove_degenerate_triangles:
                open3d_mesh_instance.remove_degenerate_triangles()
            return self.get_from_open3d_mesh(open3d_mesh_instance)

        if remove_per_component:
            _, _, vertex_components = self.get_disconnected_components()
            component_meshes = []
            for vertex_component in vertex_components:
                mesh_component = self.remove_vertices(vertex_component, invert=True)
                component_meshes.append(do_cleaning(mesh_component))
            final_mesh = self.combine_meshes(component_meshes)
        else:
            final_mesh = do_cleaning(self)
        if in_place:
            self.set_from_mesh(final_mesh)
        return final_mesh

    def simplify_quadric_decimation(self, target_number_of_triangles: int) -> Mesh:
        """
        Basic wrapper around Open3D's implementation of simplify_quadric_decimation().
        This does not happen in-place, i.e., it always returns a new mesh instance.
        :param target_number_of_triangles: Number of triangles the simplified mesh should contain.
        :return simplified mesh.
        """
        target_number_of_triangles = int(target_number_of_triangles)
        if not 0 < target_number_of_triangles < self.get_num_triangles():
            raise ValueError(f"Please set the target number of triangles bigger than 0 and smaller than the "
                             f"current number of triangles, which is {self.get_num_triangles()} in this case.")
        simplified_open3d_mesh = self.get_open3d_mesh(copy=False).simplify_quadric_decimation(
            target_number_of_triangles)
        return self.get_from_open3d_mesh(simplified_open3d_mesh)


    def get_non_manifold_edges(self, allow_boundary_edges: bool = False) -> np.ndarray:
        """
        Cf. open3d's method of the same name. A non-manifold edge is not shared by exactly two triangles.
        :param allow_boundary_edges: Whether boundary edges should also be returned or not.
                                     So if True, only those edges with more than two shared triangles are returned.
        :return: list of non-manifold edges (numpy array), each given via start and end vertex index
                 (so shape is num_edges x 2)
        """
        return np.asarray(self.get_open3d_mesh().get_non_manifold_edges(
            allow_boundary_edges=allow_boundary_edges))

    @staticmethod
    def get_valid_mesh_suffixes():
        """
        We normally only count ".obj", ".ply", ".stl" as valid mesh formats.
        The load() function also supports ".json", since this allows some code
        to also work without open3d, but this method is mainly used for filtering of files,
        and ".json" is not a usual mesh format, so we only count these three here.
        This method also defines the order in which mesh files with the same stem will be prioritized.
        """
        return [".obj", ".ply", ".stl"]


    @staticmethod
    def load(path_to_mesh: Union[str, Path], enable_post_processing: bool = False,
             remove_duplicate_vertices: bool = False, max_num_triangles: Union[int, None] = None) -> Mesh:
        """
        Load mesh from path. We recommend common formats like ply, stl, or obj, but since loading
        from them requires open3d to be installed, we also support json as an alternative.
        :param path_to_mesh: Path from which to load the mesh (string).
        :param enable_post_processing: Allow open3d to post-process mesh (probably good if it contains textures)
        :param remove_duplicate_vertices: Choose this to remove duplicate vertices from the mesh after loading.
        :param max_num_triangles: Optionally provide an upper threshold on the allowed number of triangles.
                                  If the mesh has more triangles than this threshold, it will be simplified.
        :return: Import mesh of type Mesh.
        """
        path_to_mesh = Path(path_to_mesh)
        if path_to_mesh.suffix == '.json':
            with open(path_to_mesh) as file:
                data = json.load(file)
            vertices = np.asarray(data['vertices'])
            triangles = np.asarray(data['triangles'])
            colors = np.asarray(data['colors']) if "colors" in data else None
            landmarks = Landmarks(data['landmarks']) if "landmarks" in data else None
            mesh = Mesh(vertices=vertices, triangles=triangles, colors=colors, landmarks_or_points_array=landmarks)
        elif any([path_to_mesh.suffix == suffix for suffix in Mesh.get_valid_mesh_suffixes()]):
            open3d_mesh = open3d.io.read_triangle_mesh(str(path_to_mesh), enable_post_processing=enable_post_processing)
            mesh = Mesh.get_from_open3d_mesh(open3d_mesh)
        else:
            raise ImportError(f"Unknown file type: {path_to_mesh}")

        if remove_duplicate_vertices:
            mesh.remove_duplicate_vertices(remove_per_component=False, in_place=True)
        if max_num_triangles is not None and mesh.get_num_triangles() > max_num_triangles:
            mesh.simplify_quadric_decimation(target_number_of_triangles=max_num_triangles)
        return mesh

    def export(self, path_to_mesh: Union[str, Path]):
        """
        Export mesh to a file. We recommend common formats like ply, stl, or obj, but since saving
        to them requires open3d to be installed, we also support json as an alternative.
        :param path_to_mesh: Path to which to save the mesh (string).
        """
        path_to_mesh = Path(path_to_mesh)
        if path_to_mesh.suffix == '.json':
            mesh_dict = {"vertices": self.vertices.tolist(), "triangles": self.triangles.tolist()}
            if self.has_colors():
                mesh_dict["colors"] = self.colors.tolist()
            if self.has_landmarks():
                mesh_dict["landmarks"] = self.landmarks.tolist()
            with open(path_to_mesh, 'w') as file:
                json.dump(mesh_dict, file, indent=4)
        elif any([path_to_mesh.suffix == suffix for suffix in [".ply", ".stl", ".obj"]]):
            open3d.io.write_triangle_mesh(
                str(path_to_mesh), self.get_open3d_mesh(), write_ascii=True, write_vertex_normals=False)
        else:
            raise ImportError(f"Unknown file ending for mesh export: {path_to_mesh}")

    def compute_open3d_scene(self):
        mesh_t = self.get_open3d_mesh(compute_normals=True, use_legacy=True)
        self.open3d_scene = open3d.t.geometry.RaycastingScene()
        self.open3d_scene.add_triangles(mesh_t)

    def get_open3d_scene(self):
        if self.open3d_scene is None:
            self.compute_open3d_scene()
        return self.open3d_scene

    @staticmethod
    def get_all_mesh_files_in_path(
            path: Union[List[Union[str, Path]], Union[str, Path]], file_suffixes: Union[str, List[str], None] = None,
            nested: bool = True, regex_names: Union[str, List[str], None] = "*",
            exclude_regex_names: Union[str, List[str], None] = None,
            remove_suffix_duplicates: bool = False, priority_suffix_list: Union[str, List[str], None] = None,
            added_since: Union[str, None] = None, added_before: Union[str, None] = None) -> List[Path]:
        """
        Return all file paths within a directory that are meshes.
        Cf. get_all_files_in_path() in utils.py documentation. for more details.
        By default, this method will simply filter for "obj", "ply", and "stl" files
        (that order also defines the priority in case remove_suffix_duplicates=True),
        but you can also overwrite this behavior to get other file types.
        """
        file_suffixes = Mesh.get_valid_mesh_suffixes() if file_suffixes is None else file_suffixes
        priority_suffix_list = file_suffixes if priority_suffix_list is None else priority_suffix_list
        return utils.get_all_files_in_path(
            path, file_suffixes=file_suffixes, nested=nested, regex_names=regex_names,
            exclude_regex_names=exclude_regex_names,
            remove_suffix_duplicates=remove_suffix_duplicates,
            priority_suffix_list=priority_suffix_list, added_since=added_since, added_before=added_before)

class Landmarks:
    """
    Class for landmarks, which can be an optional attribute of the Mesh class.
    This is basically a wrapper around a numpy array of points, with some
    further landmark-specific processing, such as handling (detecting, removing...)
    undefined landmarks (a.k.a., nan entries in the numpy points array),
    transforming, aligning, loading, saving, and returning a mesh of the landmarks
    that has a sphere centered at each defined point.
    This class mostly also works with 2D points, but note that the mesh class only works in 3D,
    so any landmark attribute for a mesh must be 3D.
    """
    def __init__(self, landmarks_or_point_array_or_path: Union[Landmarks, np.ndarray, Union[str, Path]],
                 parse_order: Union[List, np.ndarray, None] = None, strip_landmark_rows: bool = False,
                 names: Union[List[str], None] = None, confidence: Union[List[float], np.ndarray, None] = None) -> None:
        """
        :param landmarks_or_point_array_or_path: Can already be a Landmarks object, can also be a 2D/3D numpy
                                                 points array, can also be a path (cf load() function).
        :param parse_order: Optionally provide an index array to filter/reorder the landmark points by.
        :param strip_landmark_rows: Set True to remove undefined entries.
        :param names: Optionally provide names for the individual landmarks. Can also be encoded in the provided
                      Landmarks object, but this argument would overwrite the names in the Landmarks object if provided.
        :param confidence: Same as for names, you can also provide a confidence per landmark, which overwrites the
                           confidence of the provided Landmarks object if it has one.
        :return: None
        """
        if isinstance(landmarks_or_point_array_or_path, (str, Path)):
            landmarks_or_point_array_or_path = Landmarks.load(landmarks_or_point_array_or_path)
        if isinstance(landmarks_or_point_array_or_path, Landmarks):
            points = landmarks_or_point_array_or_path.get_points()
            if names is None:
                names = landmarks_or_point_array_or_path.names
            if confidence is None:
                confidence = landmarks_or_point_array_or_path.confidence
        else:
            points = landmarks_or_point_array_or_path

        if parse_order is not None:
            points = points[parse_order]
            if names is not None:
                names = [names[idx] for idx in parse_order]
            if confidence is not None:
                confidence = [confidence[idx] for idx in parse_order]

        self.set_points(points, strip_landmark_rows=strip_landmark_rows, names=names, confidence=confidence)

    @staticmethod
    def get_empty_landmarks(dim: int = 3) -> Landmarks:
        """
        Return a Landmarks object with zero points. The object will still have a points attribute,
        but it's a 0x2/3 numpy array. However, the dim doesn't really matter for empty landmarks,
        since the dim automatically adjusts when new points are added.
        :param dim: Optionally set landmark dimension to 2 (it's 3 by default)
        """
        if dim not in [2, 3]:
            raise ValueError("dim must be 2 or 3.")
        return Landmarks(np.empty((0, dim), dtype=np.float32), strip_landmark_rows=False)

    def get_points(self, indices_or_mask: Union[None, np.ndarray] = None, strip_nan_rows: bool = False,
                   copy: bool = False) -> np.ndarray:
        """
        Get the landmark points, optionally only the defined ones, or filtered by an index or mask array.
        :param indices_or_mask: Optionally provide an index or mask array to filter the returned points by.
        :param strip_nan_rows: Set True to remove the undefined entries from the returned points.
        :param copy: Set True to copy the points array if you don't want to accidentally change them in this object.
        :return: 2D/3D numpy array of points.
        """
        if indices_or_mask is not None:
            points = self.points[indices_or_mask]
        else:
            points = self.points

        if strip_nan_rows:
            points = self.strip_landmark_rows(points, return_non_nan_row_indices=False).get_points(
                strip_nan_rows=False)
        if copy:
            return points.copy()
        else:
            return points

    def get_point_index(self, name: str) -> int:
        """
        Get point index from point name.
        :param name: Point name as string
        :return: Point index (integer)
        :raise AssertionError: If Landmarks object doesn't have names defined.
        :raise ValueError: If the provided name doesn't exist or is duplicate.
        """
        assert self.names is not None
        possible_indices = [idx for idx, name_ind in enumerate(self.names) if name_ind == name]
        if len(possible_indices) == 1:
            return possible_indices[0]
        else:
            if len(possible_indices) == 0:
                raise ValueError(f"Name {name} not found in names: {self.names}.")
            else:
                raise ValueError(f"Several name matches found for {name} in names: {self.names}.")

    def get_point(self, point_index_or_name: Union[int, str], return_other_attributes: bool = False) \
            -> Union[np.ndarray, Tuple[np.ndarray, Union[str, None], Union[float, None]]]:
        """
        Get specific point from the point array.
        :param point_index_or_name: Integer specifying which point to get or the name of the point.
        :param return_other_attributes: Optionally also return other attributes corresponding to the point
        :return: 2D/3D numpy vector.
        """
        if isinstance(point_index_or_name, str):
            point_index = self.get_point_index(point_index_or_name)
        else:
            point_index = point_index_or_name
        point = self.points[point_index].copy()
        if return_other_attributes:
            name = self.names[point_index] if self.names is not None else None
            confidence = self.confidence[point_index] if self.confidence is not None else None
            return point, name, confidence
        else:
            return point

    def get_dist(self, *other_landmarks: Landmarks) -> Union[float, Tuple[float, ...]]:
        """
        Get Euclidean distance of these landmarks' points to one or more other sets of other landmarks' points.
        :param other_landmarks: Other Landmarks-type objects. Note that all objects need
                                to have the same number of points.
        :return: Single distance, if only one other Landmarks object was given, otherwise tuple of distances.
        """
        distances = [np.linalg.norm(self.get_points() - other_landmarks_ind.get_points())
                     for other_landmarks_ind in other_landmarks]
        if len(distances) == 1:
            return distances[0]
        else:
            return tuple(distances)

    def set_points(self, point_array: np.ndarray, strip_landmark_rows: bool = False,
                   names: Union[List[str], None] = None,
                   confidence: Union[List[float], np.ndarray, None] = None) -> None:
        """
        Set/overwrite the points of this object, i.e., the central attribute of this class.
        Note that since this overwrites the current points, names and confidence also need to be reset.
        If you don't provide them, they will be none after calling this method.
        :param point_array: Numpy array of 2D/3D points to be assigned to this object.
        :param strip_landmark_rows: Set True to remove any undefined entries in the points array.
        :param names: Optionally provide names for the individual landmarks. Can also be encoded in the provided
              Landmarks object, but this argument would overwrite the names in the Landmarks object if provided.
        :param confidence: Same as for names, you can also provide a confidence per landmark, which overwrites the
                           confidence of the provided Landmarks object if it has one.
        """
        points_np = np.array(point_array)
        if points_np.ndim == 1:
            points_np = np.expand_dims(points_np, axis=0)
        assert points_np.ndim == 2
        if points_np.shape[1] not in [2, 3]:
            raise ValueError("dim must be 2 or 3.")
        if strip_landmark_rows:
            points_np, non_nan_rows = self.strip_landmark_rows(points_np, return_non_nan_row_indices=True)
        self.points = points_np
        self.dim = points_np.shape[1]
        self.names = names
        if self.names is not None:
            if strip_landmark_rows:
                self.names = [self.names[idx] for idx in non_nan_rows]
            assert len(self.names) == self.get_num_points()
        self.confidence = confidence
        if self.confidence is not None:
            self.confidence = np.clip(np.asarray(confidence), 0, 1)
            if strip_landmark_rows:
                self.confidence = np.asarray([self.confidence[idx] for idx in non_nan_rows])
            assert len(self.confidence) == self.get_num_points()

    def add_points(self, point_array: np.ndarray, index: Union[int, None] = None, names: Union[List[str], None] = None,
                   confidence: Union[List[float], np.ndarray, None] = None) -> None:
        """
        Append points to end, or add them at any other place.
        :param point_array: Numpy array of 2D/3D points to be appended/added to current point array
        :param index: Optionally provide an index where to add the points. 0 will add the points to the very beginning.
                      Note that this is a single index, so the newly added points are always added together,
                      not at separate places.
        :param names: Optionally provide names. If you don't provide any, and the current landmark object already has
                      names, "undefined" is assigned as name to the added points. If you provide names, but the object
                      doesn't have names yet, "undefined" is added as name for all already existing points.
        :param confidence: Optionally provide confidence values for each added point. If you don't provide them, but
                           the current object already has confidence values defined, 1 is assigned as confidence to the
                           new points. Same as with names, if the object doesn't have confidence, but you provide here
                           some, 1 is added as confidence for all existing points.
        :return: None (in-place)
        """
        points_np = np.array(point_array)
        if points_np.ndim == 1:
            points_np = np.expand_dims(points_np, axis=0)
        assert points_np.ndim == 2
        if len(self.points) == 0:
            dim = points_np.shape[-1]
            if not dim in [2, 3]:
                raise ValueError("dim must be 2 or 3.")
            self.dim = dim
            self.points = np.empty((0, self.dim), dtype=np.float32)
        assert points_np.shape[1] == self.points.shape[1]
        num_points = self.get_num_points()
        if index is None:
            index = num_points
        if not 0 <= index <= num_points:
            raise IndexError(f"Index must be between 0 and {num_points} (both bounds included).")
        points_cat = np.concatenate([self.points[:index], points_np, self.points[index:]])
        if names is not None or self.names is not None:
            default_name = "undefined"
            if names is None:
                names = [default_name]*len(points_np)
            current_names = [default_name]*num_points if self.names is None else self.names
            new_names = [*current_names[:index], *names, *current_names[index:]]
        else:
            new_names = None
        if confidence is not None or self.confidence is not None:
            default_confidence = 1
            if confidence is None:
                confidence = [default_confidence]*len(points_np)
            current_confidence = [default_confidence]*num_points if self.confidence is None else self.confidence
            new_confidence = np.concatenate([current_confidence[:index], confidence, current_confidence[index:]])
        else:
            new_confidence = None

        self.set_points(points_cat, names=new_names, confidence=new_confidence)

    def add_point(self, point: Union[np.ndarray, List], index: Union[int, None] = None, name: Union[str, None] = None,
                  confidence: Union[float, None] = None) -> None:
        """
        Append/add single point, cf. add_points().
        """
        self.add_points(np.asarray([point]), index=index, names=None if name is None else [name],
                        confidence=None if confidence is None else [confidence])

    def get_num_points(self) -> int:
        """
        :return: Integer specifying how many points this object has.
        """
        return len(self.points)

    def get_num_defined_points(self) -> int:
        """
        :return: Integer specifying how many defined points this object has.
        """
        return len(self.get_non_nan_rows(return_as_mask=False))

    def filter(self, indices_or_mask: np.ndarray) -> None:
        """
        Filter the points of this object via some indices or mask, i.e., remove the remaining points.
        :param indices_or_mask: Index or mask array, specifying which points to keep.
        :return: None (in-place)
        """
        indices = IndicesAndMasks.get_indices(indices_or_mask, make_unique=False)
        new_names = None if self.names is None else [self.names[idx] for idx in indices]
        new_confidence = None if self.confidence is None else self.confidence[indices]
        self.set_points(self.get_points(indices_or_mask=indices), names=new_names, confidence=new_confidence)

    def get_filtered(self, indices_or_mask: np.ndarray) -> Landmarks:
        """
        Get a new Landmark instance with a subset of the points, specified by the argument filter array.
        :param indices_or_mask: Index or mask array, specifying which points to keep in the new instance.
        :return: New Landmarks object with the filtered points as attribute.
        """
        landmarks = self.get_copy()
        landmarks.filter(indices_or_mask=indices_or_mask)
        return landmarks

    def get_nan_rows(self, invert: bool = False, return_as_mask: bool = False,
                     count_zero_confidence_as_nan: bool = True) -> np.ndarray:
        """
        Get the indices/mask of the undefined points. An undefined point as nan or inf positional entries
        or has confidence 0 (the last you can disable by setting include_zero_confidence=False).
        :param invert: Invert the indices/mask, so that the defined points are indexed.
        :param return_as_mask: Set True to get a mask array returned, otw. an index array is returned.
        :param count_zero_confidence_as_nan: Set False to really only get the rows of landmarks that
                                             have no position encoded. Since confidence 0 means the position
                                             can be completely random, we count those as nan by default.
        :return: Numpy index (default) of mask (if return_as_mask) array.
        """
        nan_entries = np.logical_or(np.isnan(self.points), np.isinf(self.points))
        nan_rows_mask = np.any(nan_entries, axis=tuple(range(1, nan_entries.ndim)))
        if self.confidence is not None and count_zero_confidence_as_nan:
            nan_rows_mask = IndicesAndMasks.join(nan_rows_mask, self.confidence <= 0)
        if invert:
            nan_rows_mask = IndicesAndMasks.invert_mask(nan_rows_mask)
        if return_as_mask:
            return nan_rows_mask
        else:
            return IndicesAndMasks.convert_mask_to_indices(nan_rows_mask)

    @staticmethod
    def is_point_undefined(point: np.ndarray) -> bool:
        """
        Check whether a point is defined (has a position) or not (contains nan or inf entries).
        :param point: numpy vector. Could theoretically also be multiple points, but then you'd just check
                      whether any of them is undefined.
        :return boolean that's True if the point is undefined.
        """
        return np.any(np.logical_or(np.isnan(point), np.isinf(point)))

    def has_unique_names(self):
        """
        Check if the names of this Landmarks object are unique. This can be important for defining correspondences.
        Returns False if there's at least one duplicated entry in self-names.
        Also returns False if no names are defined.
        """
        if self.names is None:
            return False
        return len(list(set(self.names))) == len(self.names)

    def get_non_nan_rows(self, return_as_mask: bool = False, count_zero_confidence_as_nan: bool = True) -> np.ndarray:
        """
        Get the indices/mask of defined points, cf. get_nan_rows().
        """
        return self.get_nan_rows(invert=True, return_as_mask=return_as_mask,
                                 count_zero_confidence_as_nan=count_zero_confidence_as_nan)

    def landmarks_contain_nan(self, count_zero_confidence_as_nan: bool = True) -> bool:
        """
        :return: True if there are undefined points in this object (or with confidence 0),
                 False only if all points are defined.
        """
        return len(self.get_nan_rows(count_zero_confidence_as_nan=count_zero_confidence_as_nan)) > 0

    def landmarks_contain_no_nan(self, count_zero_confidence_as_nan: bool = True) -> bool:
        """
        :return: True if there are no undefined landmarks in this object (so also none with confidence 0),
                 False if there's at least one.
        """
        return len(self.get_nan_rows(count_zero_confidence_as_nan=count_zero_confidence_as_nan)) == 0

    def is_point_defined(self, point_name_or_index: Union[int, str]) -> bool:
        """
        Check if the indexed point is defined. If you provide an index < 0, an IndexError is raised,
        but an index too high will just yield False as return.
        """
        if isinstance(point_name_or_index, str):
            index = self.get_point_index(point_name_or_index)
        else:
            index = point_name_or_index
        if index < 0:
            raise IndexError("Index must be bigger than 0.")
        if index >= self.get_num_points():
            return False
        return self.get_nan_rows(return_as_mask=True)[index]

    @staticmethod
    def strip_landmark_rows(
            *landmark_sets_or_paths: Union[Landmarks, np.ndarray, str],
            return_non_nan_row_indices: bool = False) -> Union[Tuple[Landmarks, ...], Tuple[Landmarks, ..., np.ndarray]]:
        """
        Given one or multiple landmark sets, remove all rows that contain any nan or inf value.
        This is more strict than the above function strip_landmark_cols() and also goes along the other axis.
        If multiple sets are provided, they are assumed to be coupled, which means that if one set
        contains a row with nan or inf values, the same row is removed from all other landmarks sets as well.
        :param landmark_sets_or_paths: One or multiple landmark sets or path to them.
        :param return_non_nan_row_indices: Set to True to have an additional numpy array returned that specifies the
                                           indices of the non nan rows, so those row indices that were kept and not
                                           stripped.
        :return: A single stripped landmark set if only one provided as input,
                 a list of stripped landmark sets otherwise. If return_non_nan_row_indices,
                 non_nan_row index array is returned as well.
        """
        landmark_sets = [Landmarks(landmarks_or_path) for landmarks_or_path in landmark_sets_or_paths]
        num_points = [landmarks.get_num_points() for landmarks in landmark_sets]
        if len(num_points) > 1:
            if not all(num_lm_current == num_lm_next for num_lm_current, num_lm_next in zip(num_points[:-1], num_points[1:])):
                raise AssertionError("Provided landmark sets don't all equal in length")
        nan_row_mask = IndicesAndMasks.join(*[landmarks.get_nan_rows(return_as_mask=True)
                                              for landmarks in landmark_sets])
        non_nan_row_mask = IndicesAndMasks.invert(nan_row_mask)
        landmarks_stripped = [landmarks.get_filtered(non_nan_row_mask) for landmarks in landmark_sets]

        if return_non_nan_row_indices:
            return *landmarks_stripped, IndicesAndMasks.get_indices(non_nan_row_mask)
        else:
            return landmarks_stripped[0] if len(landmark_sets) == 1 else landmarks_stripped

    def strip_rows(self) -> np.ndarray:
        """
        Strip the points of this object of any undefined entries (in-place).
        :return: Indices of the rows that were stripped.
        """
        _, non_nan_rows = self.strip_landmark_rows(self.points, return_non_nan_row_indices=True)
        self.filter(non_nan_rows)
        return non_nan_rows

    @staticmethod
    def get_corresponding_landmarks(first_landmarks: Landmarks, second_landmarks: Landmarks,
                                    strip_landmarks: bool = False) -> Tuple[Landmarks, Landmarks]:
        """
        Given two Landmarks object, get all points from them that correspond to each other.
        This requires the two object to have unique names defined, so that those points
        with common names across the two objects can be kept. If the landmarks don't have
        unique names, but the same number of points, we assume they're already matched. If not,
        a ValueError is raised.
        :param first_landmarks: First Landmarks object, should have names defined.
        :param second_landmarks: Second Landmarks object, should have names defined.
        :param strip_landmarks: Set True to strip the Landmarks objects of undefined entries before returning them.
        :return Tuple:
            1) first_landmarks object with points not contained in second_landmarks object removed.
            2) second_landmarks object with points not contained in first_landmarks object removed.
        :raise ValueError: If at least one of the two Landmarks objects does not have unique names defined,
                           and the objects don't have the same number of points.
        """
        # Case without names defined, so we can't really find the common landmarks.
        if not first_landmarks.has_unique_names() or not second_landmarks.has_unique_names():
            if first_landmarks.get_num_points() == second_landmarks.get_num_points():
                first_landmarks_common, second_landmarks_common = first_landmarks, second_landmarks
            else:
                raise ValueError("Two Landmarks objects don't have names defined for finding common landmarks,"
                                 "and they don't have the same number of points either.")

        # Default case: goes through names of first_landmarks, and checks for each if
        # it's also part of second_landmarks.
        else:
            second_landmarks_names_dict = {name: idx for idx, name in enumerate(second_landmarks.names)}
            first_landmarks_common = Landmarks.get_empty_landmarks(dim=first_landmarks.dim)
            second_landmarks_common = Landmarks.get_empty_landmarks(dim=second_landmarks.dim)
            for first_idx, name in enumerate(first_landmarks.names):
                if name in second_landmarks_names_dict:
                    f_point, _, f_conf = first_landmarks.get_point(first_idx, return_other_attributes=True)
                    first_landmarks_common.add_point(f_point, name=name, confidence=f_conf)
                    s_point, _, s_conf = second_landmarks.get_point(
                        second_landmarks_names_dict[name], return_other_attributes=True)
                    second_landmarks_common.add_point(s_point, name=name, confidence=s_conf)

        # Optionally strip the two objects of their undefined entries.
        if strip_landmarks:
            first_landmarks_common, second_landmarks_common = Landmarks.strip_landmark_rows(
                first_landmarks_common, second_landmarks_common)
        return first_landmarks_common, second_landmarks_common

    def get_copy(self) -> "Landmarks":
        """
        Return a copy of the landmarks. A new Landmarks instance is created with the point array
        being copied.
        :return: a new Landmarks instance with identical points (new numpy array object) and other attributes.
        """
        return deepcopy(self)

    def remove_points(self, indices_or_mask: Union[np.ndarray, List]) -> None:
        """
        Remove indexed points from current Landmark instance (in-place, no return).
        :param indices_or_mask: Indices or mask array, must fit to point array.
        :return: None (in-place change)
        """
        indices_to_keep = IndicesAndMasks.invert(IndicesAndMasks.get_indices(
            indices_or_mask, make_unique=True), num_indices_or_mesh=self.get_num_points())
        self.filter(indices_to_keep)

    def remove_point(self, index: int) -> None:
        """
        Remove indexed point from current Landmark instance  (in-place, no return).
        :param index: integer index of point to remove
        :return: None (in-place change)
        """
        self.remove_points([index])

    def truncate(self, truncate_length: int) -> None:
        """
        Truncate the number of points to the provided length, so remove all points from the back come after
        the first truncate_length points.
        :param truncate_length: integer number of points to keep.
        :return: None (in-place change)
        """
        self.remove_points(list(range(truncate_length, self.get_num_points())))

    @staticmethod
    def translate_points(points_array: np.ndarray, translate: np.ndarray) -> np.ndarray:
        """
        Basically just add a translation vector to the given points array.
        :param points_array: 2D/3D numpy array of points (can have shape 2/3 or Nx2/3) to translate
        :param translate: 2D/3D numpy translation vector
        :return: translated points (same shape as input)
        """
        utils.get_translation_matrix(translate)
        return Landmarks.transform_points(points_array, utils.get_translation_matrix(translate))

    def translate(self, translate: np.ndarray) -> None:
        """
        Translate points of this object (in-place, no return).
        :param translate: 2D/3D numpy translation vector
        :return: None (in-place operation)
        """
        self.set_points(self.translate_points(self.points, translate), confidence=self.confidence, names=self.names)

    @staticmethod
    def transform_points(points: np.ndarray, transform_matrix: np.ndarray) -> np.ndarray:
        """
        Given a set of points, this method efficiently transforms all points by the same
        provided transformation matrix, i.e., it computes the matrix-vector product between the transformation
        matrix and every point.
        @param points: np array of points (lists should also be supported).
                       If the array only has one axis, it's assumed to be a single point.
        @param transform_matrix: Numpy transformation matrix. Can be dxd (without translation)
                                 or (d+1)x(d+1) (homogeneous, supporting translation),
                                 where d is the point dimension.
        @return: Transformed points as np array, same shape as input points.
        """
        points_np = np.asarray(points)
        points_expanded = False
        if len(points_np.shape) == 1:
            points_np = np.expand_dims(points_np, axis=0)
            points_expanded = True
        if not 0 <= transform_matrix.shape[1] - points_np.shape[1] <= 1:
            raise AssertionError(f"The transformation matrix must be either {points_np.shape[1]}- or "
                                 f"{points_np.shape[1]+1}-dimensional. However, it's "
                                 f"{transform_matrix.shape[1]}-dimensional.")

        # homogeneous points have 1's at the end
        # If the transformation matrix is not homogeneous, we actually
        # don't make the points homogeneous. The multiplication still works.
        points_hom = np.concatenate([
            points_np, np.ones((points_np.shape[0], transform_matrix.shape[1] - points_np.shape[1]))], axis=1)

        # transformation matrix is applied to each individual point via einsum
        # We then truncate the output to the points' dimension (remove homogeneous coordinate if present)
        points_transformed = np.einsum("ab,cb->ca", transform_matrix, points_hom)[:, :points_np.shape[1]]

        # Shrink back to single point if only single point was given
        if points_expanded:
            points_transformed = points_transformed[0]

        return points_transformed

    @staticmethod
    def flip_points(points: np.ndarray) -> np.ndarray:
        """
        Flip the provided points array along x-axis.
        :param points: Numpy array of 2D/3D points (can have shape 2/3 or Nx(2/3))
        :return: Numpy array of flipped points (same shape as input)
        """
        dim = np.asarray(points).shape[-1]
        flip_matrix = utils.get_flip_matrix(flip_axis=0, dim=dim, homogeneous=True)
        return Landmarks.transform_points(points, flip_matrix)

    def flip(self) -> None:
        """
        Flip the points of this object along the x-axis (in-place, no return).
        :return: None (in-place operation)
        """
        self.set_points(self.flip_points(self.points), confidence=self.confidence, names=self.names)

    def transform(self, transform_matrix: np.ndarray):
        """
        Transform the points of this object (in-place, no return).
        :param transform_matrix: 3x3 or 4x4 (homogeneous) numpy transformation matrix.
        :return: None (in-place operation)
        """
        self.set_points(self.transform_points(points=self.points, transform_matrix=transform_matrix),
                        confidence=self.confidence, names=self.names)

    def reorder(self, new_order_indices: Union[List, np.ndarray]) -> None:
        """
        Reorder points of this object given the provided index array (in-place, no return).
        Quite similar to filter() method, but prints warning if a mask is provided.
        :param new_order_indices: Index array to reorder landmarks' points by. A warning is printed if this is a
                                  mask, since masks cannot preserve order.
        :return: None (in-place operation)
        """
        if IndicesAndMasks.is_mask(new_order_indices):
            warnings.warn("You called the reorder method, but provided a mask, which can only be used to "
                          "filter the points array, but cannot reorder it. Use an index array instead.")
        new_points = self.points[new_order_indices]
        new_names = None if self.names is None else [self.names[idx] for idx in new_order_indices]
        new_confidence = None if self.confidence is None else self.confidence[new_order_indices]
        self.set_points(new_points, names=new_names, confidence=new_confidence)

    @staticmethod
    def get_landmark_colors_static(landmarks_or_points_array_or_length: Union[Landmarks, np.ndarray, int],
                                   colors=None) -> np.ndarray:
        """
        By default, Landmark colors are defined as a gradient that is blue for the first landmark,
        then transitions to green for the middle landmark, and ends at red for the last landmark.
        You can also choose the colors to transition over yourself.
        :param landmarks_or_points_array_or_length: Landmarks object, or the points array, or just the number of points.
        :param colors: Optional sequence of RGB colors. Each color must have three components.
                       Values are expected to be in [0, 1].
        :return: 3D numpy array with same shape as given Landmarks' points array (one RGB color for each point).
        """
        if isinstance(landmarks_or_points_array_or_length, int):
            num_points = landmarks_or_points_array_or_length
        else:
            landmarks = Landmarks(landmarks_or_points_array_or_length)
            num_points = landmarks.get_num_points()
        if num_points > 1:
            return np.asarray([utils.get_heatmap_gradient(num_point / (num_points - 1), colors=colors)
                               for num_point in range(num_points)])
        else:
            return np.asarray([0., 0., 0.])

    def get_landmark_colors(self, colors=None):
        """
        Cf. get_landmark_colors_static()
        """
        return self.get_landmark_colors_static(self, colors=colors)

    def get_mesh(self, colorized: bool = True, combine_into_single_mesh: bool = True, sphere_radius: float = 1.5,
                 gradient_colors: Union[List, np.ndarray, None] = None,
                 custom_colors: Union[List, np.ndarray, None] = None,
                 sphere_resolution: int = 8) -> Union[Mesh, List[Mesh]]:
        """
        Return a mesh that contains a sphere for each landmark. Size, resolution, and colors of the spheres are
        adjustable. Note that the landmarks are automatically stripped.
        :param colorized: Set this to False if you don't want the landmark mesh to contain colors.
                          This also overwrites custom_colors you provide.
        :param combine_into_single_mesh: Set this to False if you want a separate Mesh object for each sphere
                                         (list of spheres is returned in that case).
        :param gradient_colors: Provide colors to transition over with gradient.
        :param custom_colors: Optionally assign any other colors to the landmarks than the default or chosen gradient.
        :param sphere_resolution: default 8, increase for smoother spheres, decrease for better performance.
        :param sphere_radius: Radius of the sphere (float), default 1.5
        :return Mesh containing all landmark spheres, or list of individual spheres if not combine_into_single_mesh.
        """
        if self.dim == 2:
            raise NotImplementedError("Landmarks' get_mesh() currently only works with 3D points.")
        landmarks_mesh_list = []
        points_stripped = self.get_points(strip_nan_rows=True)
        num_points = len(points_stripped)
        if colorized:
            if custom_colors is None:
                landmark_colors = self.get_landmark_colors(colors=gradient_colors)
            else:
                landmark_colors = custom_colors
            if landmark_colors.ndim == 1:
                landmark_colors = np.repeat(np.expand_dims(landmark_colors, axis=0), axis=0, repeats=num_points)
        else:
            landmark_colors = [None]*num_points

        for landmark, landmark_color in zip(points_stripped, landmark_colors):
            sphere = Mesh.get_sphere(sphere_radius=sphere_radius, sphere_resolution=sphere_resolution,
                                     position=landmark, color=landmark_color)
            landmarks_mesh_list.append(sphere)
        if combine_into_single_mesh:
            return Mesh.combine_meshes(landmarks_mesh_list)
        else:
            return landmarks_mesh_list

    @staticmethod
    def load(path_to_landmarks: Union[str, Path]) -> Landmarks:
        """
        Load landmarks from a text file, assuming individual coordinates are comma-separated, and different
        points are line-separated.
        :return: Landmarks object containing the loaded points as attribute.
        """
        with open(path_to_landmarks, "r") as f:
            landmark_lines = f.readlines()
        def convert_entry(current_entry: str) -> Union[str, float]:
            current_entry = current_entry.rstrip()
            try:
                return float(current_entry)
            except ValueError:
                return current_entry.lower()
        landmark_matrix = [[convert_entry(entry) for entry in landmark_line.split(",")]
                           for landmark_line in landmark_lines]
        if len(landmark_matrix) > 1:
            assert all([len(landmark_matrix[0]) == len(landmark_row) for landmark_row in landmark_matrix[1:]])
        elif len(landmark_matrix) == 0:
            return Landmarks.get_empty_landmarks()
        if all([isinstance(entry, str) for entry in landmark_matrix[0]]):
            data_start_index = 1
            landmark_header = landmark_matrix[0]
        else:
            data_start_index = 0
            landmark_header = ["x", "y", "z", "confidence", "name"][:len(landmark_matrix[0])]
        data_entry = {col_name: [landmark_matrix[i][num_col] for i in range(data_start_index, len(landmark_matrix))]
                      for num_col, col_name in enumerate(landmark_header)}
        # For points and confidence, we assume here they were correctly converted to float.
        # Numpy warns if this is not the case anyway.
        points = np.stack([data_entry[coord] for coord in ["x", "y", "z"] if coord in data_entry], axis=1)
        confidence = data_entry.get("confidence", None)
        names = data_entry.get("name", None)
        # The name could actually be a number, so we convert back to string here
        if names is not None:
            names = [str(name) for name in names]

        return Landmarks(points, names=names, confidence=confidence)

    def export(self, path_to_landmarks: Union[str, Path], overwrite: bool = True,
               only_save_points: bool = False) -> None:
        """
        Export landmarks into a text file, where individual coordinates are comma-separated, and different
        points are line-separated. If available, confidence and names are also saved. A header specifies the order
        of the entries. You can also choose to not have a header and to not save confidence and names via
        only_save_points.
        :param path_to_landmarks: Path to landmarks text file
        :param overwrite: Overwrite existing landmarks file (default True)
        :param only_save_points: Do not save a landmark header, and do not save confidence and/or names.
        :return: None
        """
        if os.path.exists(str(path_to_landmarks)) and not overwrite:
            warnings.warn(f"Path {path_to_landmarks} already exists")
            return
        landmark_header = ["x", "y"]
        if self.dim == 3:
            landmark_header.append("z")
        if not only_save_points:
            if self.confidence is not None:
                landmark_header.append("confidence")
            if self.names is not None:
                landmark_header.append("name")
        with open(path_to_landmarks, "w") as f:
            f.write(",".join(landmark_header) + "\n")
            for num_point, point in enumerate(self.points):
                current_string = ",".join([str(point_c) for point_c in point])
                if not only_save_points:
                    if self.confidence is not None:
                        current_string += f",{self.confidence[num_point]}"
                    if self.names is not None:
                        current_string += f",{self.names[num_point]}"
                if num_point < len(self.points) - 1:
                    current_string += "\n"
                f.write(current_string)

    @staticmethod
    def get_all_landmark_files_in_path(
        path: Union[List[Union[str, Path]], Union[str, Path]], file_suffixes: Union[str, List[str]] = None,
        nested: bool = True, regex_names: Union[str, List[str], None] = "*",
        exclude_regex_names: Union[str, List[str], None] = None,
        remove_suffix_duplicates: bool = False, priority_suffix_list: List[str] = None) -> List[Path]:
        """
        Return all file paths within a directory that are landmarks.
        Cf. get_all_files_in_path() in utils.py documentation. for more details.
        By default, this method will simply filter for "csv" files only,
        but you can also overwrite this behavior to get other file types.
        """
        file_suffixes = [".csv"] if file_suffixes is None else file_suffixes
        return utils.get_all_files_in_path(
            path, file_suffixes=file_suffixes, nested=nested, regex_names=regex_names,
            exclude_regex_names=exclude_regex_names,
            remove_suffix_duplicates=remove_suffix_duplicates,
            priority_suffix_list=priority_suffix_list)

    def procrustes(
            self, target_landmarks: Union[Landmarks, np.ndarray, Union[str, Path]],
            landmark_mask_or_indices: Union[List, np.ndarray, None] = None, reflection: bool = True,
            translation: bool = True, scale: bool = True, in_place: bool = False,) -> Tuple[np.ndarray, Landmarks]:
        """
        Uses procrustes method to get the best alignment from self to other landmarks.
        Implementation is based on the one from trimesh.
        This can be useful if you want to align one mesh with another using both mesh's landmarks.
        :param target_landmarks: Landmarks to align this object to.
                                 Note that it needs to have the same number of points as this Landmark object.
        :param landmark_mask_or_indices: Optionally provide a mask or index array to indicate which landmarks
                                         to use for alignment.
        :param reflection: Set False to rule out the possibility of flipping/reflecting the landmarks.
        :param translation: Set False to rule out any translation (not so common).
        :param scale: Set False to rule out any rescaling (common when you want to align one mesh with another,
                      but preserve its size).
        :param in_place: Set True to transform the points array of this object. Doesn't change the return.
        :return: Tuple:
            1) transformation transform_matrix (4x4 homogeneous numpy array)
            2) Landmarks object with transformed points.
        """
        source_points = self.get_points(copy=True)
        target_points = Landmarks(target_landmarks).get_points(copy=True)
        if landmark_mask_or_indices is not None:
            source_points = source_points[landmark_mask_or_indices]
            target_points = target_points[landmark_mask_or_indices]
        if not len(source_points) >= self.dim:
            raise AssertionError(f"You need at least {self.dim} points for procrustes alignment in "
                                 f"{self.dim} dimensions.")
        if not len(source_points) == len(target_points):
            raise AssertionError("Target landmarks must have the same number of points as the source.")
        if not source_points.shape[1] == target_points.shape[1]:
            raise AssertionError("Source and target must have the same dimension.")


        # Remove translation component
        if translation:
            # source_center and b_center are the average of their individual points.
            source_center = np.mean(source_points, axis=0)
            target_center = np.mean(target_points, axis=0)
        else:
            source_center = np.zeros(source_points.shape[1])
            target_center = np.zeros(target_points.shape[1])

        # Remove scale component
        if scale:
            # source_scale/target_scale are the square roots of the
            # squared difference between each point and the combine center.
            source_scale = np.sqrt(((source_points - source_center) ** 2).sum() / len(source_points))
            target_scale = np.sqrt(((target_points - target_center) ** 2).sum() / len(target_points))
        else:
            source_scale = 1
            target_scale = 1

        # Use SVD to find optimal orthogonal rotation matrix
        # constrained to det(rotation_mat) = 1 if necessary (if reflection is False).

        target = np.dot(((target_points - target_center) / target_scale).T,
                        ((source_points - source_center) / source_scale))

        u, s, vh = np.linalg.svd(target)

        if reflection:
            rotation_mat = np.dot(u, vh)
        else:
            # no reflection allowed, so determinant must be 1.0
            diag_mat = np.diag(np.ones(self.dim))
            diag_mat[-1, -1] = np.linalg.det(np.dot(u, vh))
            rotation_mat = np.dot(np.dot(u, diag_mat), vh)

        # Compute our (d+1)x(d+1) transformation matrix (homogeneous) from
        # translation, scaling, and rotation in the correct order
        translation = target_center - (target_scale / source_scale) * np.dot(rotation_mat, source_center)
        transform_matrix = np.hstack((target_scale / source_scale * rotation_mat, translation.reshape(-1, 1)))
        transform_matrix = np.vstack(
            (transform_matrix, np.array([0.] * (source_points.shape[1]) + [1.]).reshape(1, -1)))

        # Transform points with transformation matrix
        transformed_points = self.transform_points(self.get_points(), transform_matrix)
        if in_place:
            self.set_points(transformed_points, confidence=self.confidence, names=self.names)
        transformed_landmarks = Landmarks(transformed_points)
        return transform_matrix, transformed_landmarks


class CurvilinearFeatures:
    """
    Curvilinear Features, or landmark lines, are another type of label for meshes, cf.
    also the INFACE and INCRAN papers.

    The principle is that one landmark line/curvilinear feature comprises
    a connection of two or more points, which is useful when landmarking lines/curves
    on a mesh, i.e., features where you cannot locate single points, but still label
    their course. One example would be the eyebrows or nasolabial folds in the face.
    During registration, the line-landmarked vertices on the template/model can
    find the closest point on the connection of these landmark lines, so that the curve
    is matched without requiring precise point-to-point correspondences.

    This class is a wrapper around a line_sets attribute, which is a list where each
    entry is a numpy array of several 2D/3D points.
    It offers:
        - some basic setter and getter functions
        - transforming, reordering, filtering, loading, saving the line sets
        - flattening to 2-axis numpy array, and reinflating to line_sets for compatibility of some downstream processing
        - handling (detecting, removing...) of undefined points, also doing the same removal for multiple
          instances of line_sets to ensure label compatibility (for registration)
        - parsing point arrays into line sets
        - returning a mesh representation that comprises cylinders for each line segment
        - computing closest points between line sets (important for registration)

    This class mostly also works with 2D points, but note that the mesh class only works in 3D,
    so any CurvlinearFeatures attribute for a mesh must be 3D.
    """
    def __init__(
            self,
            line_sets_or_points_or_class_or_path: Union[str, Path, np.ndarray, List[np.ndarray], CurvilinearFeatures],
            parse_order: Union[List[Union[List, np.ndarray, int]], None] = None, strip: bool = False) -> None:
        """
        Initialize method for the curvilinear features.
        :param line_sets_or_points_or_class_or_path: CurvilinearFeatures object, line sets, point array, or path to file
                                                     from which to load the curvilinear features, optionally
                                                     rearranged via parse_order.
        :param parse_order: Optionally rearrange the line_sets by providing a list of index arrays, cf. also method
                            parse_points_to_line_sets().
        :param strip: Set True to strip curvilinear features of any undefined points, potentially altering the length.
        :return: None
        """
        if isinstance(line_sets_or_points_or_class_or_path, CurvilinearFeatures):
            line_sets = line_sets_or_points_or_class_or_path.get_line_sets()
        elif isinstance(line_sets_or_points_or_class_or_path, (str, Path)):
            line_sets = CurvilinearFeatures.load(
                line_sets_or_points_or_class_or_path, parse_order=parse_order).get_line_sets()
        else:
            line_sets = line_sets_or_points_or_class_or_path
        if parse_order is not None:
            line_sets = self.parse_points_to_line_sets(line_sets, parse_order=parse_order)

        self.set_line_sets(line_sets, strip=strip)

    def set_line_sets(self, line_sets: List[np.ndarray], strip: bool = False) -> None:
        """
        Set/overwrite the line sets of this object, i.e., the central attribute of this class.
        :param line_sets: List of numpy arrays of 2D/3D points to be assigned to this object.
        :param strip: Set True to remove any undefined entries. This can also split the line sets if there are
                      intermediate undefined points, or remove an entire line set if there are no two neighboring
                      points that are defined.
        :return: None (in-place)
        """
        line_sets = [np.array(line_set) for line_set in line_sets]
        for num_line_set, line_set in enumerate(line_sets):
            if not line_set.ndim == 2:
                raise AssertionError(f"Given line set {line_set} does not have two axes. It should have shape mx(3/2) "
                                     f"(m 2D/3D points per line set).")
            if not line_set.shape[0] >= 2:
                raise AssertionError(f"Given line set {line_set} must contain at least two points, "
                                     f"but it only contains {line_set.shape[0]}.")
            if num_line_set == 0:
                if line_set.shape[1] not in [2, 3]:
                    raise AssertionError(f"The points in line set {line_set} are not 2D or 3D.")
                self.dim = line_set.shape[1]
            else:
                if not line_set.shape[1] == self.dim:
                    raise AssertionError("Provided points don't have the same dimension as the first line set.")
        if strip:
            line_sets = self.strip_landmark_line_sets(line_sets)
        self.line_sets = line_sets


    def get_num_line_sets(self) -> int:
        """
        :return: the number (int) of line sets (just the length of the line_sets attribute).
        """
        return len(self.line_sets)

    def get_num_lines(self) -> int:
        """
        Return the total number of lines from all line sets combined.
        """
        return sum([len(line_set) for line_set in self.line_sets])

    def get_line_sets(self, indices_or_mask: Union[None, np.ndarray] = None, strip: bool = False,
                      copy: bool = False) -> List[np.ndarray]:
        """
        Get the line sets of this object, optionally stripped of undefined entries
        and/or filtered to specific line sets.
        :param indices_or_mask: Optionally provide indices/mask array to filter the returned line sets by,
                                e.g. [1, 2] will only return the second and third line set.
        :param strip: Set True to strip the line sets of undefined entries before they're returned.
        :param copy: Set True to make sure the returned line sets are copied,
                     so you can't change the original attribute.
        :return: List of numpy array of 2D/3D points, representing the line sets.
        """
        line_sets = self.line_sets
        if indices_or_mask is not None:
            indices = IndicesAndMasks.get_indices(indices_or_mask, make_unique=False)
            line_sets = [line_sets[idx] for idx in indices]
        if strip:
            line_sets = self.strip_landmark_line_sets(line_sets)
        if copy:
            line_sets = [line_set.copy() for line_set in line_sets]
        return line_sets

    def get_line_set(self, line_set_index: int) -> np.ndarray:
        """
        Get specific line set from this object's line sets.
        :param line_set_index: Index of the line set to be returned.
        :return: numpy array of 2D/3D points, representing the line set.
        """
        return self.line_sets[line_set_index]

    def filter(self, indices_or_mask: np.ndarray) -> None:
        """
        Filter the line sets of this object via some indices or mask, i.e., remove the remaining points.
        :param indices_or_mask: Index or mask array, specifying which line sets to keep.
        :return: None (in-place)
        """
        self.set_line_sets(self.get_line_sets(indices_or_mask=indices_or_mask))

    def get_filtered(self, indices_or_mask: np.ndarray) -> CurvilinearFeatures:
        """
        Get a new CurvilinearFeatures instance with a subset of the line sets, specified by the argument filter array.
        :param indices_or_mask: Index or mask array, specifying which line sets to keep in the new instance.
        :return: New CurvilinearFeatures object with the filtered line sets as attribute.
        """
        return CurvilinearFeatures(self.get_line_sets(indices_or_mask=indices_or_mask))

    def get_copy(self):
        """
        :return: New CurvilinearFeatures instance with the same line sets (new object) as attribute.
        """
        return CurvilinearFeatures(self.get_line_sets(copy=True))

    def remove_line_sets(self, indices_or_mask: Union[np.ndarray, List]) -> None:
        """
        Remove indexed line sets from current CurvilinearFeatures instance (in-place, no return).
        :param indices_or_mask: Indices or mask array, must fit to line_sets list.
        :return: None (in-place change)
        """
        self.set_line_sets(self.get_line_sets(indices_or_mask=IndicesAndMasks.invert(
            np.asarray(indices_or_mask), self.get_num_line_sets())))

    def remove_line_set(self, line_set_index: int) -> None:
        """
        Remove indexed line set from current CurvilinearFeatures instance  (in-place, no return).
        :param line_set_index: integer index of line set to remove
        :return: None (in-place change)
        """
        self.remove_line_sets([line_set_index])

    def truncate(self, truncate_length: int) -> None:
        """
        Truncate the number of line sets to the provided length, so remove all line sets from the back come after
        the first truncate_length line sets.
        :param truncate_length: integer number of line sets to keep.
        :return: None (in-place change)
        """
        self.remove_line_sets(list(range(truncate_length, self.get_num_line_sets())))

    @staticmethod
    def get_flattened_line_sets(line_sets: List[np.ndarray]) -> Tuple[np.ndarray, List[int]]:
        """
        This is a static method that requires the line_sets to be provided as arguments.
        It returns a flattened version of these line sets, as well as a list that contains
        the cumulative sum of the individual line set lengths. You can apply some processing
        to the flattened line sets (without requiring to loop through them), and when you're
        done, you can reinflate them via get_unflattened_line_sets(), also providing this list
        of cumulative length sums as argument.
        :param line_sets: List of line sets to be flattened. (Note that this also works with indices,
                          so the individual arrays need not be 2D/3D)
        :return: Tuple
            1) flattened line sets numpy array of shape mx(2/3)
               (m is the total number of points in all line sets combined).
            2) List of lengths equal to the number of line sets, containing integers that represent the cumulative sum
               of the lengths of each individual line set, e.g., [3, 7, 12] means the first line set has length 3,
               the second length 4, and the third length 5. (The last entry always equals the length of the flattened
               line sets)
        """
        line_set_lengths = [len(line_set) for line_set in line_sets]
        line_set_lengths_cumul_sum = list(accumulate(line_set_lengths))
        line_sets_flattened = np.concatenate(line_sets, axis=0)
        return line_sets_flattened, line_set_lengths_cumul_sum

    @staticmethod
    def get_unflattened_line_sets(line_sets_flattened: np.ndarray,
                                  line_set_lengths_cumul_sum: List[int]) -> List[np.ndarray]:
        """
        This is a static method that requires the line_sets to be provided as arguments.
        Reinflate/unflatten a flattened version of line sets into the original form.
        :param line_sets_flattened: The flattened numpy array of line sets containing 2D/3D points.
        :param line_set_lengths_cumul_sum: Second return variable of get_flattened_line_sets().
                                           List of lengths equal to the number of line sets, containing
                                           integers that represent the cumulative sum of the lengths of
                                           each individual line set.
        :return: List of numpy arrays of 2D/3D points, representing the unflattened line sets.
        """
        return [line_sets_flattened[start:end] for start, end in zip(
            [0] + line_set_lengths_cumul_sum[:-1], line_set_lengths_cumul_sum)]

    @staticmethod
    def transform_line_sets(line_sets: List[np.ndarray], transform_matrix: np.ndarray) -> List[np.ndarray]:
        """
        This is a static method that requires the line_sets to be provided as arguments.
        Given the line sets, this method efficiently transforms all points within all line sets by the same
        provided transformation matrix, i.e., it computes the matrix-vector product between the transformation
        matrix and every point.
        :param line_sets: List of numpy array of 2D/3D points.
        :param transform_matrix: Numpy transformation matrix. Can be dxd (without translation)
                                 or (d+1)x(d+1) (homogeneous, supporting translation)
        :return: Transformed line sets, so list of numpy arrays of 2D/3D points.
        """
        # For more efficient computation, we first flatten the line sets
        line_sets_flattened, line_set_lengths_cumul_sum = CurvilinearFeatures.get_flattened_line_sets(line_sets)
        # Apply efficient numpy multiplication on all points combined
        line_sets_flattened_transformed = Landmarks.transform_points(line_sets_flattened, transform_matrix)
        # And unflatten the line sets again to their original structure
        return CurvilinearFeatures.get_unflattened_line_sets(
            line_sets_flattened_transformed, line_set_lengths_cumul_sum)

    @staticmethod
    def flip_line_sets(line_sets: List[np.ndarray]) -> List[np.ndarray]:
        """
        This is a static method that requires the line_sets to be provided as arguments.
        Flip the provided line sets along x-axis.
        :param line_sets: List of numpy arrays of 2D/3D points
        :return: List of numpy arrays of flipped points (same shape as input)
        """
        dim = np.asarray(line_sets[0]).shape[-1]
        flip_matrix = utils.get_flip_matrix(flip_axis=0, dim=dim, homogeneous=True)
        return CurvilinearFeatures.transform_line_sets(line_sets, flip_matrix)

    @staticmethod
    def translate_line_sets(line_sets: List[np.ndarray], translate: np.ndarray) -> List[np.ndarray]:
        """
        This is a static method that requires the line_sets to be provided as arguments.
        Translate the provided line sets by adding the same translation vector to each point.
        :param line_sets: List of numpy arrays of 2D/3D points
        :param translate: 2D/3D numpy translation vector
        :return: List of numpy arrays of translated points (same shape as input)
        """
        translate_matrix = utils.get_translation_matrix(translate=translate)
        return CurvilinearFeatures.transform_line_sets(line_sets, translate_matrix)

    def transform(self, transform_matrix: np.ndarray) -> None:
        """
        Transform the line sets of the current object (in-place operation, no return). Cf. also docs
        of transform_line_sets() method.
        :param transform_matrix: 3x3 or 4x4 (homogeneous) numpy transformation matrix.
        :return: None (in-place operation)
        """
        self.set_line_sets(self.transform_line_sets(self.get_line_sets(), transform_matrix=transform_matrix))

    def flip(self) -> None:
        """
        Flip the line sets of the current object along x-axis (in-place operation, no return). Cf. also docs
        of flip_line_sets() method.
        :return: None (in-place operation)
        """
        self.set_line_sets(self.flip_line_sets(self.get_line_sets()))

    def translate(self, translate: np.ndarray):
        """
        Translate the line sets of the current object (in-place operation, no return). Cf. also docs
        of translate_line_sets() method.
        :param translate: 2D/3D numpy translation vector
        :return: None (in-place operation)
        """
        self.set_line_sets(self.translate_line_sets(self.get_line_sets(), translate=translate))

    def reorder(self, new_order_indices: Union[List, np.ndarray]) -> None:
        """
        Reorder line sets of this object given the provided index array (in-place, no return).
        Quite similar to filter() method, but prints warning if a mask is provided.
        :param new_order_indices: Index array to reorder landmarks' points by. A warning is printed if this is a
                                  mask, since masks cannot preserve order.
        :return: None (in-place operation)
        """
        if IndicesAndMasks.is_mask(new_order_indices):
            warnings.warn("You called the reorder method, but provided a mask, which can only be used to "
                          "filter the line sets, but cannot reorder them.")
            new_order_indices = IndicesAndMasks.get_indices(new_order_indices, make_unique=False)
        reordered_line_sets = self.get_line_sets(new_order_indices, strip=False, copy=False)
        self.set_line_sets(reordered_line_sets)

    def reverse(self, reverse_indices_or_mask: Union[List, np.ndarray]) -> None:
        """
        Reverse (invert the order of the points) of specific line sets.
        Could be useful after flipping the points.
        :param reverse_indices_or_mask: Indices or mask array specifying which of the line sets to reverse.
        :return: None (in-place operation)
        """
        reverse_mask = IndicesAndMasks.get_mask(reverse_indices_or_mask, self.get_num_line_sets())
        self.set_line_sets([line_set[::-1] if reverse else line_set for reverse, line_set
                            in zip(reverse_mask, self.line_sets)])

    @staticmethod
    def strip_landmark_line_sets(*line_sets: List[np.ndarray]) -> Union[List[np.ndarray], Tuple[List[np.ndarray], ...]]:
        """
        We strip one or multiple sets of lines. One set has multiple points, representing lines inside of it.
        Each line is stripped the leading and ending None values using the same strip for all sets,
        so for two sets, if a line in the first set is full, but the respective line in the second set is not, e.g.

        [1, 2, 3, 4, 5] vs. [None, 7, None, 8, None],

        then we strip them equally to

        [2, 3, 4] and [7, None, 8].

        Next, we split all lines at None points in any line, so in the example above, we would get no lines,
        since the middle None in the second line would lead both lines to be split at the middle index,
        resulting in two individual points, and since we need at least two points for a line, we get no lines.
        Another example:

        [2, 3, 4, 5, 6, 7, 8, 9] vs. [10, 11, 12, None, 13, 14, None, 15]

        would result in

        [[2, 3, 4], [6, 7]] and [[10, 11, 12], [13, 14]].

        :param line_sets: One or multiple sets of lines. Note that we also support List of index arrays as arguments,
                          the list length and individual array lengths must match, but we don't require 3D entries.
        :return: Same lines, but stripped of leading and ending None values,
                 and stripped of and split at inner None values.
                 Possibly whole lines are stripped if there's no overlap.
                 (if multiple line_sets provided, tuple is returned;
                  if only one, not a tuple but just that one stripped is returned).
        """

        def strip_line_set(*line_set: np.ndarray):
            if len(line_set) == 0:
                raise ImportError("Provide at least one line.")

            # Check that input lines all have the same length
            if len(line_set) > 1:
                assert all([len(current_line) == len(next_line) for current_line, next_line
                            in zip(line_set[:-1], line_set[1:])])

            def line_point_is_none(line_point: Union[List, np.ndarray, None]) -> bool:
                """
                Checks a point in a line for undefined values (nan or inf)
                """
                return line_point is None or np.any([np.logical_or(np.isnan(line_point), np.isinf(line_point))])


            line_length = len(line_set[0])

            def contains_none(line_index: int) -> bool:
                """
                Check all line sets if any of them has a none point at the given line index.
                """
                return any([line_point_is_none(landmark_line_ind[line_index]) for landmark_line_ind in line_set])

            def find_latest_nonan_point(start_index: int) -> int:
                """
                We check from the given start_index for the latest line index that's not none in all lines.
                Note that we return actually the index 1 up, so the index of the first none point
                (or just the end of the line), such that it better works with python indexing.
                """
                for j in range(start_index + 1, line_length):
                    if contains_none(j):
                        return j
                return line_length

            # We search for the individual start and end indices when splitting the line sets at each none point.
            individual_line_indices = []
            start_index = 0
            while start_index < line_length:
                # We skip none points
                if contains_none(start_index):
                    start_index += 1
                    continue
                # Find the end index corresponding to the current start index.
                end_index = find_latest_nonan_point(start_index)
                # We only allow lines if they have two or more points. A single point is not allow for a line.
                if end_index > start_index + 1:
                    individual_line_indices.append((start_index, end_index))
                # If we got to the end with end_index or we got so close
                # that no line of min two points can still be found, we break
                if end_index >= line_length - 2:
                    break
                start_index = end_index + 1

            # Return Nones if no partial lines could be found that don't contain any Nones for all input lines.
            if len(individual_line_indices) == 0:
                return [None] * len(line_set)
            # Return all partial lines that don't contain any None points for all input lines.
            return [np.asarray(landmark_line_ind[start_index:end_index])
                    for start_index, end_index in individual_line_indices for landmark_line_ind in line_set]

        if len(line_sets) == 0:
            raise ImportError("Provide at least one set of landmark lines.")

        # Check that each set of lines has the same length
        if len(line_sets) > 1:
            assert all([len(current_lines) == len(next_lines) for current_lines, next_lines
                        in zip(line_sets[:-1], line_sets[1:])])

        if len(line_sets[0]) == 0:
            return line_sets

        # Get all stripped line sets (some may be None, because the inner strip method couldn't find any stripped line)
        stripped_landmark_lines = [strip_line_set(*line_sets_ind) for line_sets_ind in zip(*line_sets)]
        # We remove these None lines here
        stripped_landmark_lines_nonan = [[line_set for line_set in line_sets_ind if line_set
                                          is not None] for line_sets_ind in zip(*stripped_landmark_lines)]
        if len(line_sets) == 1:
            return stripped_landmark_lines_nonan[0]
        else:
            return tuple(stripped_landmark_lines_nonan)

    def refine_with_cubic_splines(self, refine_factor: int = 10) -> None:
        """
        Refine/Upscale the curvilinear features via cubic spline interpolation.
        Since we typically approximate some natural curves with a couple of points
        that we linearly interpolate (hence the term "curvilinear features"), we might
        in some cases want to improve the approximation of the true curve if we don't
        have enough points. We do this here with a smooth, curvy function, using
        scipy's implementation of cubic splines. We then discretize these splines again
        with smaller steps than the initial number of given points, yielding still linear,
        but more refined curvilinear features.
        :param refine_factor: Choose the factor to increase the resolution of the line sets
                              by. For example, a line set that consisted of 3 points, can be
                              increased to 21 points by choosing the default factor 10.
                              (3 points equals two lines, each line is separated into 10 points).
        :return: None (in-place).
        """
        import scipy
        # The cubic line step intervals are defined via the length of each line set segment,
        # so longer segments get a bigger interval.
        def get_cubic_spline(current_line_set: np.ndarray):
            segment_lengths = [np.linalg.norm(line_beg-line_end) for line_beg, line_end in
                               zip(current_line_set[:-1], current_line_set[1:])]
            total_length = np.sum(segment_lengths)
            return scipy.interpolate.CubicSpline(
                np.asarray([np.sum(segment_lengths[:i]) / total_length for i in range(len(current_line_set))]),
                current_line_set)

        line_set_splines = [get_cubic_spline(line_set) for line_set in self.line_sets]

        def refine_list_with_interpolation(numbers: Union[List, np.ndarray]):
            refined_list = []
            for i in range(len(numbers) - 1):
                # Generate n values between numbers[i] and numbers[i + 1], inclusive
                interpolated = np.linspace(numbers[i], numbers[i + 1], refine_factor+1)
                refined_list.extend(interpolated[:-1])  # Exclude the last value to avoid duplicates

            refined_list.append(numbers[-1])  # Include the last value
            return np.asarray(refined_list)

        line_sets_refined = [line_set_spline(refine_list_with_interpolation(line_set_spline.x))
                             for line_set_spline in line_set_splines]
        self.set_line_sets(line_sets_refined)

    @staticmethod
    def get_closest_points_to_line_set(
            points: Union[List, np.ndarray], line_set: np.ndarray, connected_last_to_first: bool = False) \
            -> Tuple[np.ndarray, Union[np.ndarray, float], Union[np.ndarray, int]]:
        """
        Compute per point the closest point and respective distance to a set of lines.
        :param points: One or multiple 2D/3D points, so shape (2/3) or (n, 2/3).
        :param line_set: Multiple 2D/3D points, shape (m, 2/3). These are converted to m-1 line segments,
                         taking the points in consecutive order, so the first point is connected to the second,
                         the second to the third, etc.
                         The indices returned refer to the index of the segment within these connected line segments.
                         E.g., index 0 would mean that the segment between the first and second point is the closest.
                         You can also choose to connect the last point to the first point again via
                         connected_last_to_first.
        :param connected_last_to_first: Choose this to connect the last point from the line_set also to the first,
                                        thus yielding m line segments in total.
        :return Tuple
            1) closest points on line set of shape (2/3) or (n, 2/3) (same as points)
            2) respective distances of points to line set.
               Single float if points' shape is (2/3), other n-dim numpy array
            3) Indices of closest line segments

        """
        # setup inputs
        points = np.asarray(points)
        if not points.ndim in [1, 2]:
            raise AssertionError(f"Given points array {points} is neither a single point nor multiple points.")
        if points.shape[-1] not in [2, 3]:
            raise AssertionError(f"Given points array {points} does not contain 2D/3D points.")
        # If we only have one point, a single distance will be returned instead of an array of distances.
        if len(points.shape) == 1:
            points = np.expand_dims(points, axis=0)
            single_point = True
        else:
            single_point = False

        if Landmarks(points).landmarks_contain_nan():
            raise AssertionError("The provided points contain undefined values. "
                                 "Strip them before you use this function.")

        line_set = np.asarray(line_set)
        if not line_set.ndim == 2:
            raise AssertionError(f"Given line_set array {line_set} should have two axis, "
                                 f"since it should contain multiple 3D points.")
        if line_set.shape[-1] != points.shape[-1]:
            raise AssertionError(f"Given line_set array {line_set} does not contain {points.shape[-1]}D points.")

        if Landmarks(line_set).landmarks_contain_nan():
            raise AssertionError("The provided line_set contains undefined values. "
                                 "Strip them before you use this function.")

        line_segments = np.asarray([[start_point, end_point] for start_point, end_point
                                    in zip(line_set[:-1], line_set[1:])])
        if connected_last_to_first:
            line_segments = np.concatenate([line_segments, [[line_set[-1], line_set[0]]]], axis=0)

        # num points and num segments
        n, m, point_dim = len(points), len(line_segments), points.shape[-1]

        # connection vectors
        segment_vec = line_segments[:, 1] - line_segments[:, 0]
        begin_vec = np.expand_dims(points, axis=0) - np.expand_dims(line_segments[:, 0], axis=1)
        end_vec = np.expand_dims(points, axis=0) - np.expand_dims(line_segments[:, 1], axis=1)

        # flattened distance array
        distances = np.zeros(m * n)

        # flattened connection vectors
        begin_vec_linear = np.reshape(begin_vec, (m * n, point_dim))
        end_vec_linear = np.reshape(end_vec, (m * n, point_dim))
        segment_vec_repeated = np.repeat(segment_vec, n, axis=0)

        # Check which points lie outside of the beginning of which segment.
        # For those points, the distance is just the distance between the point and the segment start point.
        outside_begin = utils.dot_vector_list(begin_vec_linear, segment_vec_repeated) <= 0
        distances[outside_begin] = np.linalg.norm(begin_vec_linear[outside_begin], axis=1)

        # Check which points lie outside of the end of which segment.
        # For those points, the distance is just the distance between the point and the segment end point.
        not_outside_begin = np.logical_not(outside_begin)
        outside_end_notbeginframe = utils.dot_vector_list(
            end_vec_linear[not_outside_begin], segment_vec_repeated[not_outside_begin]) >= 0
        outside_end = IndicesAndMasks.convert_inner_frame_mask_to_outer_frame(
            outside_end_notbeginframe, not_outside_begin)
        distances[outside_end] = np.linalg.norm(end_vec_linear[outside_end], axis=1)

        # The remaining points have a closest point on the actual line.
        inside = np.logical_and(not_outside_begin, np.logical_not(outside_end))
        cross_prod = np.cross(segment_vec_repeated[inside], begin_vec_linear[inside])
        if cross_prod.ndim == 2:
            cross_prod_norm = np.linalg.norm(cross_prod, axis=1)
        else:
            cross_prod_norm = cross_prod
        distances[inside] = cross_prod_norm / np.linalg.norm(segment_vec_repeated[inside], axis=1)

        distances = np.reshape(distances, (m, n))
        min_axis = 0

        def get_min_vector(vector) -> np.ndarray:
            """
            Get line-reduced vector, so the vector that corresponds to the closest line. This uses the distances_argmin
            variable, which found for each point the closest line. All other vectors, e.g. outside_begin, can be reduced
            along the line dimension (first dimension of length n) by calling this function.
            @param vector: numpy array of shape (m, n, ...), so first two dimensions are fixed,
                           but more are allowed after that. E.g., begin_vec_linear has one additional dimension (3),
                           and line_segments should have two (2, 3).
            @return: numpy array vector of shape (m, ...), so first dimension has been eliminated.
            """
            assert vector.shape[0] == m and vector.shape[1] == n
            current_argmin = distances_argmin
            # account for higher dim vectors by repeating the argmin variable
            for shape_adjust in vector.shape[2:]:
                current_argmin = np.expand_dims(current_argmin, axis=-1).repeat(shape_adjust, axis=-1)
            return np.take_along_axis(vector, np.expand_dims(current_argmin, axis=min_axis), axis=min_axis). \
                squeeze(axis=min_axis)

        # Reduce the distances array along the line dimension, i.e.,
        # find the smallest distance to any line segment for each point.
        distances_argmin = np.argmin(distances, axis=min_axis)
        distances = get_min_vector(distances)

        # We find the closest point by going from the segment beginning point towards
        # the ending point by a distance found via Pythagoras using the connection vector
        # between the point and the segment beginning point and the computed distances
        def reshape_and_filter(vector):
            if len(vector.shape) == 1:
                vector = np.reshape(vector, (m, n))
            else:
                vector = np.reshape(vector, (m, n, point_dim))
            return get_min_vector(vector)

        closest_points = np.zeros((n, point_dim))
        # all previous arrays are reduced along their line dimension
        outside_begin, outside_end = reshape_and_filter(outside_begin), reshape_and_filter(outside_end)
        inside = reshape_and_filter(inside)
        begin_vec_linear = reshape_and_filter(begin_vec_linear)
        segment_vec_repeated = reshape_and_filter(segment_vec_repeated)


        # We need to expand repeat the line segments along the point dimension first, such that we can filter out
        # the closest line segment for each point afterwards.
        closest_line_segments = get_min_vector(np.expand_dims(line_segments, axis=1).repeat(n, axis=1))
        # Fill the closest points array. The two boundary cases are simple.
        closest_points[outside_begin] = closest_line_segments[outside_begin, 0]
        closest_points[outside_end] = closest_line_segments[outside_end, 1]
        # The intermediate line point case uses Pythagoras to find the length of the vector going from the line
        # beginning point to the closest point. (Threshold at 0 to avoid sqrt error due to slight numerical
        # inaccuracies)
        closest_points[inside] = (
                closest_line_segments[inside, 0] + np.expand_dims(np.sqrt(np.maximum(np.sum(np.square(
            begin_vec_linear[inside]), axis=1) - np.square(distances[inside]), 0)), axis=1)
                * segment_vec_repeated[inside] / np.linalg.norm(segment_vec_repeated[inside], axis=1,
                                                                keepdims=True))

        # return results.
        if single_point:
            distances = distances[0]
            closest_points = closest_points[0]
            distances_argmin = distances_argmin[0]

        return closest_points, distances, distances_argmin


    def get_closest_points(
            self, target: Union[CurvilinearFeatures, List[np.ndarray]]) -> Tuple[List[np.ndarray], List[np.ndarray]]:
        """
        For each line set of the current object, find the closest point on and distance to a respective target line set.
        Make sure this object's line sets are stripped of undefined entries, and the target likewise
        (call strip_landmark_line_sets() method on both line sets together to make sure they stay compatible).
        Note that the target line sets do not need to have the same number of points per line set as the source.
        :param target: Target line sets, so list of numpy array of 2D/3D points.
                       The number of line sets must be the same as that of this object,
                       since we compute the closest point for each line set on the respective
                       target line set (same index). However, the number of points in the individual line sets can
                       differ, but make sure all points are defined.
        :return: Tuple
            1) List of numpy arrays of closest 2D/3D points.
            2) List of numpy arrays of distances.

        """
        source_line_sets = self.get_line_sets()
        target_line_sets = target.get_line_sets() if isinstance(target, CurvilinearFeatures) else target
        if not len(source_line_sets) == len(target_line_sets):
            raise AssertionError(f"The provided target contains {len(target_line_sets)} sets of lines, "
                                 f"whereas self contains {len(source_line_sets)} sets of lines. "
                                 f"These numbers must match.")
        closest_points_and_distances = [self.get_closest_points_to_line_set(source_line_set, target_line_set)
                                        for source_line_set, target_line_set in zip(source_line_sets, target_line_sets)]
        return [cd[0] for cd in closest_points_and_distances], [cd[1] for cd in closest_points_and_distances]


    @staticmethod
    def get_line_set_colors_static(
            curv_features_line_sets_or_length: Union[CurvilinearFeatures, List[np.ndarray], int],
            individual_colors: bool = False) -> Union[np.ndarray, List[np.ndarray]]:
        """
        The default colors for the curvilinear features are defined as a gradient that is blue for the
        first feature, then transitions to green for the feature in the middle,
        and ends at red for the last feature. This is the same color scheme used in the Landmarks class;
        the difference is that each feature consists of multiple points/lines that are all assigned the same color.
        :param curv_features_line_sets_or_length: CurvilinearFeatures object, or the line sets,
                                                  or just the number of line sets.
        :param individual_colors: Set True to get a unique color for each line within a line set.
        :return: nx3 numpy array where n is the length of the line sets (one RGB color for each line sets).
        """
        if isinstance(curv_features_line_sets_or_length, (CurvilinearFeatures, list)):
            curv_features = CurvilinearFeatures(curv_features_line_sets_or_length)
            if individual_colors:
                line_sets_flattened, line_set_len_cumul_sum = curv_features.get_flattened_line_sets(
                    curv_features.get_line_sets())
                colors = CurvilinearFeatures.get_line_set_colors_static(
                    len(line_sets_flattened), individual_colors=False)
                return [colors[start_index:end_index] for start_index, end_index
                        in zip([0] + line_set_len_cumul_sum[:-1], line_set_len_cumul_sum)]
            else:
                num_line_sets = curv_features.get_num_line_sets()
        else:
            num_line_sets = curv_features_line_sets_or_length
        return Landmarks.get_landmark_colors_static(num_line_sets)

    def get_line_set_colors(self, individual_colors: bool = False) -> np.ndarray:
        """
        Non-static version of get_line_set_colors_static().
        """
        return self.get_line_set_colors_static(self, individual_colors=individual_colors)

    def get_mesh(self, colorized: bool = True, combine_into_single_mesh: bool = True, cylinder_radius: float = 0.75,
                 custom_colors: Union[np.ndarray, List[np.ndarray], None] = None, cylinder_resolution: int = 5,
                 individual_colors: bool = False) -> Union[Mesh, List[Mesh]]:
        """
        Return a mesh that contains a cylinder between each neighboring pair of points in each line set.
        Size, resolution, and colors of the cylinders are adjustable.
        Note that the line sets are automatically stripped of undefined entries,
        so you might get more or fewer line sets as mesh as you expected.
        :param colorized: Set this to False if you don't want the cylinder meshes to contain colors.
        :param combine_into_single_mesh: Set this to False if you want a separate Mesh object for each line set of
                                         cylinders. List of meshes is returned in that case.
                                         Note that the individual line sets may still consist of multiple
                                         cylinder meshes that are always joined.
        :param custom_colors: Optionally assign any other colors to the line sets than the default gradient.
        :param cylinder_resolution: default 5, increase for smoother cylinders, decrease for better performance.
        :param cylinder_radius: Radius of the cylinder (float), default 0.75 (half of the default sphere radius
                                used for landmarks)
        :param individual_colors: Set True to color each cylinder with a unique color instead of coloring all
                                  cylinders of the same line set with the same color.
        :return Mesh containing all line set cylinders, or list of individual line set cylinders if not
                combine_into_single_mesh.
        """
        if self.dim == 2:
            raise NotImplementedError("Only 3D curvilinear features can currently be converted to a mesh.")
        line_sets_mesh_list = []
        line_sets_stripped = self.get_line_sets(strip=True)
        num_line_sets = len(line_sets_stripped)
        if colorized:
            if custom_colors is None:
                line_set_colors = self.get_line_set_colors(individual_colors=individual_colors)
            else:
                line_set_colors = custom_colors
        else:
            line_set_colors = [None]*num_line_sets
        if not isinstance(line_set_colors[0], (list, np.ndarray)):
            line_set_colors = np.repeat(np.expand_dims(line_set_colors, axis=0), axis=0, repeats=num_line_sets)
        for line_set, line_set_color in zip(line_sets_stripped, line_set_colors):
            line_set_colors_current = np.array(line_set_color)
            if line_set_colors_current.ndim == 1:
                line_set_colors_current = np.expand_dims(line_set_colors_current, axis=0).repeat(len(line_set)-1, axis=0)
            cylinders = Mesh.combine_meshes([Mesh.get_cylinder(
                end_point, radius=cylinder_radius, start_point=start_point, resolution=cylinder_resolution,
                color=line_set_color_current) for start_point, end_point, line_set_color_current
                in zip(line_set[:-1], line_set[1:], line_set_colors_current)])
            line_sets_mesh_list.append(cylinders)
        if combine_into_single_mesh:
            return Mesh.combine_meshes(line_sets_mesh_list)
        else:
            return line_sets_mesh_list

    @staticmethod
    def parse_points_to_line_sets(points_or_line_sets: Union[List[np.ndarray], np.ndarray],
                                  parse_order: List[Union[List, np.ndarray, int]]) -> List[np.ndarray]:
        """
        Rearrange some input array of points to line sets compatible with this class.
        You must provide the indices for this rearrangement.
        :param points_or_line_sets: Numpy array of 2D/3D points. Can also be already some kind of line sets.
                                    E.g., it's valid to provide [[p1, p2, p3], [p4, p5], p6].
                                    Note that if you then provide a parse order like [[0, 2], 1], you
                                    would get [[p1, p2, p3, p6], [p4, p5]] as output.
        :param parse_order: List of index arrays or integers, cf. example above.
        :return: List of numpy array of 2D/3D points, representing the rearranged/parsed line sets.
        """
        parse_order = [[parse_order_ind] if isinstance(parse_order_ind, int) else parse_order_ind
                       for parse_order_ind in parse_order]
        line_sets_init = [np.asarray(line_set).reshape(-1, np.asarray(line_set).shape[-1])
                          for line_set in points_or_line_sets]
        line_sets = [np.asarray([point for idx in parse_order_ind for point in line_sets_init[idx]])
                     for parse_order_ind in parse_order]
        return line_sets

    @staticmethod
    def load(path_to_curvilinear_features: Union[str, Path],
             parse_order: Union[List[Union[List, np.ndarray, int]], None] = None,
             dim: int = 3) -> CurvilinearFeatures:
        """
        Import curvilinear features from a file. Optionally provide a parse_order to rearrange imported
        points into line sets.
        :param path_to_curvilinear_features: Path to file containing curvilinear features or points.
        :param parse_order: List of index arrays or integers, cf. parse_points_to_line_sets().
        :param dim: Optionally set to 2 to load 2D points.
        :return: CurvilinearFeatures object, containing imported line sets as attribute.
        """
        if dim not in [2, 3]:
            raise ValueError("dim must be 2 or 3.")
        line_sets = []
        with open(path_to_curvilinear_features) as f:
            for num_line, line in enumerate(f.readlines()):
                vals = np.fromstring(line.strip(), sep=",")
                if len(vals) > 0:
                    if len(vals) % dim != 0:
                        raise AssertionError(f"Line {num_line} in file {path_to_curvilinear_features} "
                                             f"contains {vals}, which cannot be separated into {dim}D vectors, "
                                             f"since the number of numbers is not divisible by {dim}.")
                    line_sets.append(vals.reshape(-1, dim))
        if parse_order is not None:
            line_sets = CurvilinearFeatures.parse_points_to_line_sets(line_sets, parse_order)
        return CurvilinearFeatures(line_sets)

    def export(self, path_to_curvilinear_features: Union[str, Path], overwrite: bool = True) -> None:
        """
        Save the line sets of this object into a file. Each line will contain mx(2/3) numbers, where m is the length
        of the respective line set (first line is first line set).
        :param path_to_curvilinear_features: File to write line sets to.
        :param overwrite: Set False to avoid overwriting an existing file.
        :return: None
        """
        path_to_curvilinear_features = Path(path_to_curvilinear_features)
        if path_to_curvilinear_features.exists() and not overwrite:
            warnings.warn(f"Path {path_to_curvilinear_features} already exists")
            return
        with open(path_to_curvilinear_features, "w") as f:
            for line_set in self.line_sets:
                f.write(f"{','.join(line_set)}\n")


