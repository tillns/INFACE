"""
Reconstruct a mesh with an implicit model, optionally filling missing regions.
Note that the mesh should be aligned with the model space.
"""
from src.objects.implicit_model import ImplicitShapeModel
from src.objects.indices_and_masks import IndicesAndMasks
from src.objects.mesh import Mesh, Landmarks
from pathlib import Path
from typing import Union
import numpy as np

from src.objects.morphable_model import MorphableModel
from src.objects.registration_file_communicator import RegistrationFileCommunicator

# Adjust these paths to your liking
#
# Add path to an input mesh to be reconstructed. We sample a mesh otherwise.
# Note that if the mesh has not been aligned and normalized, you should also
# provide the path to a registered mesh (cf. below).
mesh_path: Union[str, Path, None] = None
# Optionally add path to landmarks fitting to the mesh corresponding to the model template landmarks.
landmarks_path: Union[str, Path, None] = None
# Optionally add path to the mesh registered via the explicit INCLEFT model to the input mesh above.
# We recommend using this to align the input mesh to the model space.
registered_mesh_path: Union[str, Path, None] = None
# Feel free to try another model. It should fit to the mesh, though.
model_path: Union[str, Path] = "INCLEFT/implicit_full_sdfcorrection.pt"
# optionally add path to an index selection that specifies mesh vertex indices to exclude from reconstruction
unknown_vertex_mask_path: Union[str, Path, None] = None
# optionally provide a path where to save mesh; it's just visualized otherwise
recon_mesh_export_path: Union[str, Path, None] = None
# Enable this to align the mesh to the model via its landmarks in case you do not provide a registered mesh.
# Since we used the explicit model registrations to normalize and align all scans, we don't recommend this
# option, as it will probably yield suboptimal reconstruction results.
align_with_landmarks: bool = False
#

# First load the morphable model
model = ImplicitShapeModel.load_static(model_path)

# By default, the mesh is not transformed (only if landmarks are provided)
transform = None

# Then load the mesh (if available)
if mesh_path is not None:
    mesh = Mesh.load(mesh_path)
    # If landmarks are provided, we use them to align the mesh with the model template landmarks.
    # If none are provided, the mesh is assumed to already be aligned.
    if landmarks_path is not None:
        mesh_landmarks = Landmarks.load(landmarks_path)
        template_landmarks = model.template_landmarks.get_copy()
        mesh_landmarks, template_landmarks = Landmarks.get_corresponding_landmarks(
            mesh_landmarks, template_landmarks, strip_landmarks=True)
        mesh.set_landmarks(mesh_landmarks)
        # Optionally landmarks to align mesh
        if registered_mesh_path is None and align_with_landmarks:
            mesh, transform = mesh.align_procrustes(
                template_landmarks, translation=True, scale=True, reflection=False, return_transform=True)
    # (recommended) use registered mesh to align input mesh
    if registered_mesh_path is not None:
        # Initial transformation aligns the mesh to the registered mesh
        try:
            transform_init = RegistrationFileCommunicator.get_transform(registered_mesh_path, invert=False)
        except FileNotFoundError:
            # If there's no transformation saved along with the registered mesh, we assume
            # the input mesh was already aligned to the registered mesh.
            # Note that if it's the other way around, you could first align the registered
            # mesh with the average mesh from the explicit model and use that transform here.
            transform_init = np.eye(4)
        # If the palate model is used, we need to normalize the mesh to the palate only instead of the full head mesh.
        # We use the palate indices selection from the explicit morphable model for this.
        if "onlypalates" in Path(model_path).stem:
            explicit_model = MorphableModel.load_correct_morphable_model("INCLEFT/explicit_full_pca.h5")
            indices = explicit_model.get_indices("palate")
        else:
            indices = None
        # Normalization transformation puts recenters and rescales mesh into the unit cube
        transform_norm = model.get_mesh_normalization_transform_from_registered_mesh(
            Mesh.load(registered_mesh_path), indices_to_norm_to=indices)
        # transformation order (first align, then normalize)
        transform = transform_norm @ transform_init
        # Align and normalize mesh (and its landmarks if given).
        mesh.transform(transform, in_place=True)
else:
    # Sample a mesh if none provided. Reconstructed mesh should look identical in that case,
    # unless an unknown_vertex_mask is provided.
    mesh = model.sample()
    # smooth mesh to remove bumpiness from marching cubes reconstruction
    mesh = mesh.smooth_laplacian(10)

# Optionally load the mask of unknown vertices
unknown_vertex_mask = None if unknown_vertex_mask_path is None else IndicesAndMasks.load(unknown_vertex_mask_path)

# Reconstruct the raw mesh with the model, and fill in the unknown vertices (if provided).
reconstructed_mesh = model.reconstruct_mesh(
    mesh, unknown_vertex_mask=unknown_vertex_mask, highlight_correspondences=mesh.has_landmarks())

# Show or export reconstructed mesh
if recon_mesh_export_path is None:
    reconstructed_mesh.show_multiple_meshes(mesh, reconstructed_mesh)
else:
    if transform is not None:
        reconstructed_mesh.transform(transform, in_place=True, invert=True)
    reconstructed_mesh.export(recon_mesh_export_path)