"""
Reconstruct a registered mesh with a morphable model, or fill in missing regions.
"""
from src.objects.indices_and_masks import IndicesAndMasks
from src.objects.mesh import Mesh
from src.objects.morphable_model import MorphableModel
from pathlib import Path
from typing import Union

# Adjust these paths to your liking
mesh_path: Union[str, Path, None] = None  # add path to a registered mesh
model_path: Union[str, Path] = "pca_model.h5"  # you may specify a model here
# optionally add path to an index selection that specifies mesh vertex indices to exclude from reconstruction
unknown_vertex_mask_path: Union[str, Path, None] = None
# optionally provide a path where to save mesh; it's just visualized otherwise
recon_mesh_export_path: Union[str, Path, None] = None
#

# First load the morphable model
model = MorphableModel.load_correct_morphable_model(model_path)

# Then load the mesh (if available)
if mesh_path is not None:
    mesh = Mesh.load(mesh_path)
else:
    # Sample a mesh if none provided. Reconstructed mesh should look identical in that case,
    # unless an unknown_vertex_mask is provided.
    mesh = model.sample_meshes(1)[0]

# Optionally load the mask of unknown vertices
unknown_vertex_mask = None if unknown_vertex_mask_path is None else IndicesAndMasks.load(unknown_vertex_mask_path)

# Reconstruct the registered mesh with the model, and fill in the unknown vertices (if provided).
# back_match_known_vertices is set True only if unknown_vertex_mask is provided, since the mesh
# will just look identical to the input otherwise.
reconstructed_mesh = model.reconstruct_mesh(
    mesh, unknown_vertex_mask=unknown_vertex_mask, back_match_known_vertices=unknown_vertex_mask is not None)

# Show or export reconstructed mesh
if recon_mesh_export_path is None:
    reconstructed_mesh.show_multiple_meshes(mesh, reconstructed_mesh)
else:
    reconstructed_mesh.export(recon_mesh_export_path)