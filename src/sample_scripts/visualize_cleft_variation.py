"""
Visualize the cleft
"""
from pathlib import Path
from typing import Union
import numpy as np
from src.objects.implicit_model import ImplicitShapeModel
from src.objects.mesh import Mesh

# Feel free to adjust these
#
# You may choose any of the implicit models here
model_path: Union[str, Path] = "INCLEFT/implicit_onlypalates_sdfcorrection.pt"
# Generate more in-between meshes
num_interpolations: int = 10
# How far you want to go (empirically determined number)
cleft_extent: float = 0.1
# Leave empty for left lip/alveolar cleft variation, or choose among
# "w_lip_cleft_left", "w_lip_cleft_right", "w_alveolar_cleft_left", "w_alveolar_cleft_right", "w_palatal_cleft".
# Sometimes, the average state already has the respective cleft unfused, in which case you can set the cleft_extent
# negative to fuse the cleft.
cleft_attribute: Union[str, None] = "w_palatal_cleft"
# Some attributes are correlated, so if you vary one cleft, you might also vary another cleft.
# By default, we remove the correlations with this argument, but feel free to disable it.
fix_other_attributes: bool = True
#

# Load implicit model
model = ImplicitShapeModel.load_static(model_path)
# Choose the cleft to vary
if cleft_attribute is None:
    if "palate" in str(model_path):
        cleft_attribute = "w_alveolar_cleft_left"
    else:
        cleft_attribute = "w_lip_cleft_left"

cleft_meshes = []
for i in np.linspace(0, cleft_extent, num_interpolations):
    # Decode mesh from average to open cleft
    cleft_meshes.append(model.decode(model.adjust_latent_along_attribute(
        cleft_attribute, attribute_extent=i, fix_other_attributes=fix_other_attributes),
        highlight_correspondences=True))

# Show meshes, switch between them with the arrow keys.
Mesh.show_multiple_meshes(switch_meshes=cleft_meshes)