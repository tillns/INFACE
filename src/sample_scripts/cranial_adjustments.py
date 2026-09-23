"""
Adjust the cranial attributes of model-sampled meshes to optimal values
and visualize the output along with the input.
Also check out the blender-based measurement_correction_interface for a more interactive experience.
"""
from src.objects.mesh import Mesh
from src.objects.morphable_model import MorphableModel
from src.objects.cranial_attributes import CranialAttributes
from pathlib import Path
from typing import Union

# Feel free to adjust these
model_path: Union[str, Path] = "INCRAN/pca_head_model.h5"  # you may specify a model here
cranial_attributes_path: Union[str, Path] = "INCRAN/pca_head_attribute_correlation.json" # must fit to model
num_meshes_to_sample: int = 5
#

# Load model and cranial attributes
model = MorphableModel.load_correct_morphable_model(model_path)
cranial_attributes = CranialAttributes(model_path, cranial_attributes_path)
# Generate samples; you could also load some samples here via
# mesh_paths = ["/FIRST/MESH.ply", "/SECOND/MESH.ply"]
# sample_meshes = [Mesh.load(path) for path in mesh_paths]
sample_meshes = model.sample_meshes(num_meshes_to_sample)

# Adjust meshes toward optimal cranial settings.
corrected_meshes, *_ = cranial_attributes.get_fully_corrected_meshes(sample_meshes)

# Toggle input mesh on/off with "1" and corrected mesh with "2".
# Switch between different samples via arrow keys
Mesh.show_multiple_meshes(switch_meshes=[[sample_mesh, corrected_mesh] for sample_mesh, corrected_mesh
                                         in zip(sample_meshes, corrected_meshes)])
