"""
Sample meshes with an explicit or implicit model and show or export the results.
"""
from src.objects.implicit_model import ImplicitShapeModel
from src.objects.mesh import Mesh
from src.objects.morphable_model import MorphableModel
from pathlib import Path
from typing import Union

# Adjust these paths to your liking
#
# you may specify a model here, explicit or implicit
model_path: Union[str, Path] = "INCLEFT/implicit_full_sdfcorrection.pt"
# optionally provide a path where to save samples meshes; they're just visualized otherwise
sampled_mesh_folder: Union[str, Path, None] = None
# optionally adjust the number of meshes you want to sample
num_meshes_to_sample: int = 10
# put this closer to 0 for more average samples, and further away from 0 for more extreme samples
deviation_factor: float = 1
#

# Load implicit or explicit model and sample meshes
if Path(model_path).suffix == ".pt":
    # Load implicit model
    model = ImplicitShapeModel.load_static(model_path)
    # implicit model mesh sampling
    sampled_meshes = model.sample_multiple(num_meshes_to_sample, deviation_factor=deviation_factor)
else:
    # Load explicit model
    model = MorphableModel.load_correct_morphable_model(model_path)
    # morphable model mesh sampling
    sampled_meshes = model.sample_meshes(num_meshes_to_sample, deviation_factor=deviation_factor)


# Show or export sampled meshes
if sampled_mesh_folder is None:
    Mesh.show_multiple_meshes(switch_meshes=sampled_meshes)
else:
    for num_mesh, mesh in enumerate(sampled_meshes):
        mesh.export(Path(sampled_mesh_folder).joinpath(f"sampled_mesh_num{num_mesh+1}.ply"))