"""
Neutralize the facial expression of a model sample or a custom mesh, or transfer expression from another
mesh to it.
"""
from src.objects.AE_morphable_model import AEMorphableModel
from src.objects.mesh import Mesh
from pathlib import Path
from typing import Union

# Adjust these paths to your liking
mesh_path: Union[str, Path, None] = None  # add path to a registered mesh
expression_mesh_path: Union[str, Path, None] = None  # add path to a registered mesh to transfer the expression from
model_path: Union[str, Path] = "ae_disent_model_reprojected.h5"  # you may specify another model here
# optionally provide a path where to save mesh; it's just visualized otherwise
mesh_export_path: Union[str, Path, None] = None
#

# First load the disentangled model
disentangled_model = AEMorphableModel(model_path)

# Then load the mesh (if provided)
if mesh_path is not None:
    mesh = Mesh.load(mesh_path)
else:
    # Sample a mesh if none provided.
    mesh = disentangled_model.sample_meshes(1)[0]

# If expression_mesh_path is provided, load the expression_mesh and transfer its expression to mesh
if expression_mesh_path is not None:
    expression_mesh = Mesh.load(expression_mesh_path)
    mesh_exp_adjusted = disentangled_model.transfer_expression(expression_mesh, mesh)
# Set the expression to the latent mean, so that the expression is changed to the average
else:
    mesh_exp_adjusted = disentangled_model.set_expression(mesh, disentangled_model.latent_mean)

# Show or export mesh with adjusted expression
if mesh_export_path is None:
    Mesh.show_multiple_meshes(mesh, mesh_exp_adjusted)
else:
    mesh_exp_adjusted.export(mesh_export_path)