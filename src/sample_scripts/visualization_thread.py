"""
Showcase how the visualization thread works by sampling a mesh and rotating it in a while True loop,
providing the rotation updates to the visualization thread each iteration. Script loops continuously,
so you need to stop it.
"""
from typing import Callable
from src.objects.morphable_model import MorphableModel
from src.objects.visualizer import Visualizer

# Load skull model. Feel free to adjust.
model = MorphableModel.load_correct_morphable_model("INCRAN/pca_skull_model.h5")
# Sample a mesh to rotate.
mesh = model.sample_meshes(1, include_colors_if_available=True)[0]
# Also add some landmarks that are rotated along.
# Note that the indices here may need to be adjusted if you choose another model.
landmark_vertex_indices = model.get_indices(
    "nasion", "menton", "opisthion", "alveolon", "basion", "gnathion", "hormion", "infradentale")
mesh.set_landmarks(mesh.vertices[landmark_vertex_indices])

# Method to be passed to visualization_thread. It takes visualize_geometries as argument.
# This is the update method that the visualization_thread provides rotate_mesh with, so
# that rotate_mesh can call this method with its mesh updates.
def rotate_mesh(visualize_geometries: Callable):
    while True:
        # rotate mesh by a small angle each iteration
        mesh.rotate(angle=0.001, axis=[0, 1, 0])
        # pass mesh and landmarks (as mesh) to visualization_thread.
        visualize_geometries(mesh, mesh.get_landmark_mesh())

# Start visualization_thread, pass rotate_mesh to it.
Visualizer.visualization_thread(rotate_mesh, fps=60)


