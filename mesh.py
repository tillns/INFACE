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
import numpy as np
import json

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

    def get_open3d_mesh(self, compute_normals: bool = False):
        import open3d
        open3d_mesh = open3d.geometry.TriangleMesh(
            vertices=open3d.utility.Vector3dVector(self.vertices),
            triangles=open3d.utility.Vector3iVector(self.triangles))
        if compute_normals:
            open3d_mesh.compute_vertex_normals()
        return open3d_mesh

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
        try:
            import open3d
        except ImportError:
            print("Install open3d to visualize meshes.")
            return
        open3d_meshes = [mesh.get_open3d_mesh(compute_normals=True) for mesh in meshes]
        open3d.visualization.draw_geometries(open3d_meshes, mesh_show_wireframe=True, mesh_show_back_face=True)


    @staticmethod
    def load(path_to_mesh: str) -> Mesh:
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
        if path_to_mesh.endswith('.json'):
            mesh_dict = {"vertices": self.vertices.tolist(), "triangles": self.triangles.tolist()}
            with open(path_to_mesh, 'w') as file:
                json.dump(mesh_dict, file, indent=4)
        elif any([path_to_mesh.endswith(suffix) for suffix in [".ply", ".stl", ".obj"]]):
            try:
                import open3d
            except ImportError:
                raise ImportError("Install open3d to load a mesh from a mesh file.")
            open3d.io.write_triangle_mesh(path_to_mesh, self.get_open3d_mesh(), write_ascii=True,
                                          write_vertex_normals=False)
        else:
            raise ImportError(f"Unknown file ending for mesh export: {path_to_mesh}")
