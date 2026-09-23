"""
Visualize our provided sample expressions on different identity samples at two age stages.
The visualization_thread is used for interactive updates of the expression, also check out the
visualization_thread.py sample script for usage of the visualization_thread.
"""
import time
from src.objects.AE_morphable_model import AEMorphableModel
from typing import Union, Callable
import numpy as np
from sample_expressions import sample_expression_folder_path
from src.objects.visualizer import Visualizer

# Feel free to adjust these
num_meshes_to_sample: int = 4  # For each mesh, a young and an old version will be shown
age_deviation_factor: float = 2.0  # increase for more extreme young vs old, decrease for more average
num_expression_interpolations: int = 20  # increase for more fine-grained interpolation between the expressions
fps: Union[int, float] = 60  # Frames per second for rendering
#


# Load disentangled model
disentangled_model = AEMorphableModel("INFACE/ae_disent_model.h5")
# Sample expressions to be interpolated
expressions = ["crying", "smiling", "surprised", "whining"]
# Sample mesh identities (age and expression are adjusted in the loop below)
sampled_meshes = disentangled_model.sample_meshes(num_meshes_to_sample)
age_direction = -1  # hard-coded direction in the latent space to make face older
age_mean, age_std = disentangled_model.latent_mean[-1], disentangled_model.latent_std[-1]
# All meshes are stored in here
meshes = [{"young": [], "old": []} for _ in range(num_meshes_to_sample)]
for num_sample, sampled_mesh in enumerate(sampled_meshes):
    for expression, next_expression in zip(expressions, [*expressions[1:], expressions[0]]):
        for num_interpolation in range(num_expression_interpolations):
            # Load two expressions to interpolate between
            current_expression_latent = np.load(sample_expression_folder_path.joinpath(f"{expression}_expression.npy"))
            next_expression_latent = np.load(sample_expression_folder_path.joinpath(f"{next_expression}_expression.npy"))
            a = num_interpolation / num_expression_interpolations
            # expression latent is an interpolation between the two loaded ones
            expression_latent = current_expression_latent * (1 - a) + next_expression_latent * a
            # Set expression for the current mesh sample
            mesh_transferred_expression = disentangled_model.set_expression(sampled_mesh, expression_latent)
            # Adjust age once to young and once to old
            mesh_transferred_expression_young = disentangled_model.adjust_age(
                mesh_transferred_expression, age_mean - age_deviation_factor*age_direction*age_std)
            mesh_transferred_expression_old = disentangled_model.adjust_age(
                mesh_transferred_expression, age_mean + age_deviation_factor*age_direction*age_std)
            # Add meshes to dictionary
            meshes[num_sample]["young"].append(mesh_transferred_expression_young)
            meshes[num_sample]["old"].append(mesh_transferred_expression_old)

# Method loops over interpolations and distributes samples meshes evenly in space before giving the current
# interpolation to the visualization thread.
def show_meshes(visualize_geometries: Callable):
    current_index = 0  # interpolation index
    mesh_width = 140  # x direction
    mesh_height = 130 # y direction
    while True:
        # Get meshes from current interpolation
        meshes_young = [meshes_ind["young"][current_index%len(meshes_ind["young"])] for meshes_ind in meshes]
        meshes_old = [meshes_ind["old"][current_index%len(meshes_ind["old"])] for meshes_ind in meshes]
        # Distribute meshes spatially
        meshes_young = [mesh.translate([mesh_width * (num_mesh - (len(meshes_young)-1)/2), mesh_height/2, 0], in_place=False)
                        for num_mesh, mesh in enumerate(meshes_young)]
        meshes_old = [mesh.translate([mesh_width * (num_mesh - (len(meshes_old)-1)/2), - mesh_height/2, 0], in_place=False)
                      for num_mesh, mesh in enumerate(meshes_old)]
        # Visualize meshes
        visualize_geometries(*meshes_young, *meshes_old)
        current_index += 1
        # Wait a little bit before going to next one
        time.sleep(1/100)

# Start visualization thread and give it the show_meshes function that runs in parallel, updating the meshes
# to be rendered.
Visualizer.visualization_thread(show_meshes, fps=fps)


















