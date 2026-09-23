"""
Reconstruct a registered mesh with a morphable model, or fill in missing regions.
"""
from src.objects.indices_and_masks import IndicesAndMasks
from src.objects.linear_regressor import LinearRegressor
from src.objects.mesh import Mesh
from src.objects.morphable_model import MorphableModel
from pathlib import Path
from typing import Union

# Adjust these paths to your liking
mesh_path: Union[str, Path, None] = None  # add path to a registered mesh
first_model_path: Union[str, Path] = "INCRAN/pca_head_model.h5"  # model for the mesh, feel free to adjust
second_model_path: Union[str, Path] = "INCRAN/pca_skull_model.h5" # other model to infer from mesh, feel free to adjust
# linear regressor used for inference; must be compatible with both models
linear_regressor_path: Union[str, Path] = "INCRAN/regressor_pca_head_model_to_pca_skull_model.json"
#

# First load the morphable models
first_model = MorphableModel.load_correct_morphable_model(first_model_path)
second_model = MorphableModel.load_correct_morphable_model(second_model_path)

# Then load the mesh (if available)
if mesh_path is not None:
    mesh = Mesh.load(mesh_path)
else:
    # Sample a mesh if none provided.
    mesh = first_model.sample_meshes(1)[0]

# Finally, load the linear regressor
linear_regressor = LinearRegressor.from_json(linear_regressor_path)

# Infer the mesh by translating the latent of the mesh in the first model's space to a latent
# in the second model's space via the linear regressor.
inferred_mesh = first_model.convert_mesh_with_linear_regressor(linear_regressor, mesh, second_model, include_colors_if_available=True)

# Show the input and inferred mesh.
Mesh.show_multiple_meshes(mesh, inferred_mesh)