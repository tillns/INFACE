"""
Basic example how showing meshes works. It explains how meshes can be toggled on and off and how you can
switch between meshes. You can provide your own mesh paths or visualize some default shapes.
"""
import numpy as np
from src.objects.mesh import Mesh

# You can enter paths to your own meshes here to load and show
your_own_mesh_paths = []

if len(your_own_mesh_paths) > 0:
    meshes = [Mesh.load(path) for path in your_own_mesh_paths]
    Mesh.show_multiple_meshes(*meshes)
else:
    # Tetrahedron
    mesh1 = Mesh(
        vertices=np.array([
            [0.0, 1.0, 0.0], [-0.866, -0.5, -0.5], [0.866, -0.5, -0.5], [0.0, -0.5, 1.0]]),
        triangles=np.array([[0, 1, 2], [0, 2, 3], [0, 3, 1], [1, 3, 2]]))

    # Square Pyramid
    mesh2 = Mesh(
        vertices=np.array([
            [0.0, 1.0, 0.0], [-0.5, 0.0, -0.5], [0.5, 0.0, -0.5], [0.5, 0.0, 0.5], [-0.5, 0.0, 0.5]]),
        triangles=np.array([[0, 1, 2], [0, 2, 3], [0, 3, 4], [0, 4, 1],  [1, 3, 2], [1, 4, 3]]))

    # Triangular Prism
    mesh3 = Mesh(
        vertices=np.array([
            [0.0, 1.0, 0.5], [-0.866, -0.5, 0.5], [0.866, -0.5, 0.5],
            [0.0, 1.0, -0.5], [-0.866, -0.5, -0.5], [0.866, -0.5, -0.5]]),
        triangles=np.array([[0, 1, 2], [5, 4, 3], [0, 2, 5], [0, 5, 3], [1, 0, 3], [1, 3, 4], [2, 1, 4], [2, 4, 5]]))

    # Cube
    mesh4 = Mesh(
        vertices=np.array([
            [-0.5, -0.5, -0.5], [0.5, -0.5, -0.5], [0.5, 0.5, -0.5], [-0.5, 0.5, -0.5],
            [-0.5, -0.5, 0.5], [0.5, -0.5, 0.5], [0.5, 0.5, 0.5], [-0.5, 0.5, 0.5]]),
        triangles=np.array([
            [0, 2, 1], [0, 3, 2], [4, 5, 6], [4, 6, 7], [0, 1, 5], [0, 5, 4],
            [2, 3, 7], [2, 7, 6], [0, 4, 7], [0, 7, 3], [1, 2, 6], [1, 6, 5]]))

    # Toggle mesh1 on/off with "1", mesh2 with "2",
    # switch between mesh3 and mesh4 with arrow keys and toggle the currently shown one on/off with "3"
    Mesh.show_multiple_meshes(mesh1, mesh2, switch_meshes=[mesh3, mesh4])