"""
Mesh object class to support basic triangle mesh handling without requiring many libraries.

Open3D must be installed for certain import, export, and visualization functions.

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

from __future__ import annotations
from typing import Union, Dict, List

import numpy as np
import json
import warnings

class Mesh:
    """
    Simple mesh object class that basically only contains the vertices and triangles as attributes.
    A mesh can be loaded from a json file that contains the vertices and triangles; usual mesh files, i.e.,
    ply, stl, obj, are also supported, but open3d must be installed. If open3d is installed, the mesh
    can also be visualized.
    """
    def __init__(self, vertices: np.ndarray, triangles: np.ndarray):
        self.vertices = vertices
        self.triangles = triangles

    def get_vertices(self) -> np.ndarray:
        return self.vertices

    def get_triangles(self) -> np.ndarray:
        return self.triangles

    def get_num_vertices(self) -> int:
        return len(self.vertices)

    def get_num_triangles(self) -> int:
        return len(self.triangles)

    def get_copy(self) -> "Mesh":
        """
        Return a copy of the mesh. A new Mesh instance is created with both vertices and triangles arrays
        being copied.
        :return a new Mesh instance with identical vertices and triangles as attributes.
        """
        return Mesh(self.vertices.copy(), self.triangles.copy())

    def get_open3d_mesh(self, compute_normals: bool = False, use_legacy: bool = False):
        """
        Convert self to an open3d triangle mesh. Optionally use legacy type.
        :param compute_normals: Whether to compute the normals of the triangle mesh (useful for visualization)
        :param use_legacy: Whether to use open3d's legacy triangle mesh class, which supports some other methods.
        :return triangle mesh of type open3d.geometry.TriangleMesh or open3d.t.geometry.TriangleMesh
        """
        import open3d
        open3d_mesh = open3d.geometry.TriangleMesh(
            vertices=open3d.utility.Vector3dVector(self.vertices),
            triangles=open3d.utility.Vector3iVector(self.triangles))
        if compute_normals:
            open3d_mesh.compute_vertex_normals()
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
        import open3d
        if isinstance(open3d_mesh, open3d.geometry.TriangleMesh):
            return Mesh(vertices=np.asarray(open3d_mesh.vertices), triangles=np.asarray(open3d_mesh.triangles))
        elif isinstance(open3d_mesh, open3d.t.geometry.TriangleMesh):
            return Mesh(open3d_mesh.vertex.positions.numpy(), open3d_mesh.triangle.indices.numpy())
        else:
            raise TypeError(f"Unknown input for open3d_mesh: {open3d_mesh}")

    def get_trimesh_mesh(self):
        import trimesh
        return trimesh.Trimesh(vertices=self.vertices, faces=self.triangles, process=False)

    @staticmethod
    def get_from_trimesh_mesh(trimesh_mesh):
        return Mesh(vertices=trimesh_mesh.vertices, triangles=trimesh_mesh.faces)

    def remove_vertices(self, indices: np.ndarray) -> Mesh:
        """
        Remove indexed vertices from mesh. This does not happen in-place, i.e.,
        a new Mesh instance is created and returned.
        """
        mesh_open3d = self.get_open3d_mesh()
        mesh_open3d.remove_vertices_by_index(indices)
        return self.get_from_open3d_mesh(mesh_open3d)

    def get_edges_unique(self) -> np.ndarray:
        """
        Return all unique triangle mesh edges (not directional, so [0, 1] and [1, 0] are the same edge).
        :return numpy array of shape [num_edges, 2]
        """

        # Collect all edges from the triangles
        edges = np.vstack([
            self.triangles[:, [0, 1]],
            self.triangles[:, [1, 2]],
            self.triangles[:, [2, 0]]
        ])

        # Sort vertex indices in each edge so that [a, b] and [b, a] are considered the same
        edges = np.sort(edges, axis=1)

        # Use numpy unique to get only unique edges
        unique_edges = np.unique(edges, axis=0)

        return unique_edges

    def get_edges_per_vertex(self, return_dict: bool = True) -> Union[Dict, List[np.ndarray]]:
        """
        Return for each mesh vertex all edges that are attached to that vertex.
        :param return_dict: Whether to return dictionary or list.
        :return dictionary {vertex_ind: [connected_vertex_ind0, connected_vertex_ind1,...]
                or list [[connected_vertex_ind0, connected_vertex_ind1,...], [connected_vertex_ind0, connected_vertex_ind1,...], ..]
        """
        edges = self.get_edges_unique()
        num_vertices = self.get_num_vertices()
        if return_dict:
            edges_per_vertex = {vertex_ind: [] for vertex_ind in range(num_vertices)}
        else:
            edges_per_vertex = [[] for _ in range(num_vertices)]
        for edge in edges:
            edges_per_vertex[edge[0]].append(edge[1])
            edges_per_vertex[edge[1]].append(edge[0])
        return edges_per_vertex

    def get_vertex_neighbors(self) -> List[np.ndarray]:
        """
        Returns each vertex' neighbor as a list, cf. docs of get_edges_per_vertex()
        """
        return self.get_edges_per_vertex(return_dict=False)

    def get_adjacency_matrix(self):
        """
        Build a sparse adjacency matrix for the mesh.
        :return scipy csc_matrix that contains all vertex neighbors
        """
        import scipy.sparse as spsp
        neighbors = self.get_vertex_neighbors()
        row_idx = []
        col_idx = []

        for v, nbrs in enumerate(neighbors):
            row_idx.extend([v] * len(nbrs))
            col_idx.extend(nbrs)

        data = np.ones(len(row_idx), dtype=np.int8)
        num_vertices = self.get_num_vertices()
        adj = spsp.csc_matrix((data, (row_idx, col_idx)), shape=(num_vertices, num_vertices))

        return adj

    def smooth_laplacian(self, num_iterations: int, lambda_fact: float = 0.5, indices: np.ndarray = None) -> "Mesh":
        """
        Apply Laplacian smoothing with a simple forward Euler step v <- v + lambda*(avg(N) - v).
        Not in-place, i.e., a new Mesh instance is created and returned.
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
        new_vertices = self.get_vertices().copy()
        adj = self.get_vertex_neighbors()
        num_vertices = self.get_num_vertices()
        indices = np.arange(num_vertices) if indices is None else np.asarray(indices)

        for _ in range(num_iterations):
            averages = np.asarray([np.mean(new_vertices[adj[idx]], axis=0) for idx in indices])
            new_vertices[indices] = (1 - lambda_fact) * new_vertices[indices] + lambda_fact * averages

        return Mesh(vertices=new_vertices, triangles=self.triangles)

    def show(self):
        try:
            import open3d
        except ImportError:
            print("Install open3d to visualize meshes.")
            return
        open3d_mesh = self.get_open3d_mesh(compute_normals=True)
        open3d.visualization.draw_geometries([open3d_mesh], mesh_show_wireframe=True, mesh_show_back_face=True)

    @staticmethod
    def show_multiple_meshes(*meshes: Mesh):
        """
        Show multiple meshes in an Open3D visualizer. This is a static method, so even if you call it from an instance,
        you still need to provide this instance as argument to include it in the visualizer.
        - First 10 meshes can be toggled with number keys:
            "1" -> mesh 1, "2" -> mesh 2, ..., "0" -> mesh 10
        - Any meshes beyond the tenth are always visible.
        :param meshes: All meshes (type Mesh) to show
        """
        try:
            import open3d
        except ImportError:
            print("Install open3d to visualize meshes.")
            return
        # Convert meshes to open3d for visualization
        open3d_meshes = [mesh.get_open3d_mesh(compute_normals=True) for mesh in meshes]

        # visibility state for first 10 meshes
        visible = [True] * min(len(open3d_meshes), 10)

        def make_toggle_callback(idx):
            def toggle(vis):
                # Save current camera
                view = vis.get_view_control().convert_to_pinhole_camera_parameters()

                visible[idx] = not visible[idx]
                vis.clear_geometries()
                # Add togglable meshes and fixed meshes (above number 10)
                for i, m in enumerate(open3d_meshes):
                    if i >= 10 or visible[i]:
                        vis.add_geometry(m)

                # Restore camera (avoid resetting camera whenever a mesh is shown/hidden)
                vis.get_view_control().convert_from_pinhole_camera_parameters(view)

                return False

            return toggle

        vis = open3d.visualization.VisualizerWithKeyCallback()
        vis.create_window()

        # Add all meshes initially
        for m in open3d_meshes:
            vis.add_geometry(m)

        # Register number keys 1–9, 0 for first 10 meshes
        keymap = [ord(str(i)) for i in range(1, 10)] + [ord("0")]
        for i, key in enumerate(keymap[:len(visible)]):
            vis.register_key_callback(key, make_toggle_callback(i))

        # print info for visualizer
        print("Controls: press 1–9 to toggle meshes 1–9, 0 for the 10th mesh.")
        if len(meshes) > 10:
            print(f"{len(meshes) - 10} meshes beyond the 10th are always shown.")

        # Get render options and adjust two default options (can still be changed with "w" and "b")
        opt = vis.get_render_option()
        opt.mesh_show_wireframe = True
        opt.mesh_show_back_face = True

        # Start visualizer
        vis.run()
        vis.destroy_window()

    def get_volume(self) -> float:
        """
        Compute and return scalar volume of a mesh. The mesh should be watertight.
        We use trimesh for the computation, since it seems to be less strict with the watertightness, though.
        :return volume (float)
        """
        try:
            return self.get_trimesh_mesh().volume
        except ImportError:
            raise ImportError("To get the volume of a mesh, please install trimesh.")


    @staticmethod
    def load(path_to_mesh: str) -> Mesh:
        """
        Load mesh from path. We recommend common formats like ply, stl, or obj, but since loading
        from them requires open3d to be installed, we also support json as an alternative.
        :param path_to_mesh: Path from which to load the mesh (string).
        """
        if path_to_mesh.endswith('.json'):
            with open(path_to_mesh) as file:
                data = json.load(file)
            vertices = np.asarray(data['vertices'])
            triangles = np.asarray(data['triangles'])
        elif any([path_to_mesh.endswith(suffix) for suffix in [".ply", ".stl", ".obj"]]):
            try:
                import open3d
            except ImportError:
                raise ImportError("Install open3d to load a mesh from a mesh file.")
            open3d_mesh = open3d.io.read_triangle_mesh(path_to_mesh, enable_post_processing=False)
            vertices = np.asarray(open3d_mesh.vertices)
            triangles = np.asarray(open3d_mesh.triangles)
        else:
            raise ImportError(f"Unknown file type: {path_to_mesh}")
        return Mesh(vertices, triangles)

    def export(self, path_to_mesh: str):
        """
        Export mesh to a file. We recommend common formats like ply, stl, or obj, but since saving
        to them requires open3d to be installed, we also support json as an alternative.
        :param path_to_mesh: Path to which to save the mesh (string).
        """
        if path_to_mesh.endswith('.json'):
            mesh_dict = {"vertices": self.vertices.tolist(), "triangles": self.triangles.tolist()}
            with open(path_to_mesh, 'w') as file:
                json.dump(mesh_dict, file, indent=4)
        elif any([path_to_mesh.endswith(suffix) for suffix in [".ply", ".stl", ".obj"]]):
            try:
                import open3d
            except ImportError:
                raise ImportError("Install open3d to save the mesh.")
            open3d.io.write_triangle_mesh(path_to_mesh, self.get_open3d_mesh(), write_ascii=True,
                                          write_vertex_normals=False)
        else:
            raise ImportError(f"Unknown file ending for mesh export: {path_to_mesh}")
