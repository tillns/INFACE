"""
This is a blender script that can only be run from blender. It should be called by calling
blender from the command line and passing the path to this file as additional argument:
blender --python PATH/TO/src/blender/landmarking.py -- --mesh_paths /PATH/TO/SOME/MESH_FILE.ply/obj/stl
Use --help as additional argument to get info about all arguments.
Alternatively, you can also use the python interface where you call the function set_landmarks()

The purpose of this script is to landmark arbitrary meshes manually via a custom blender interface
using panels and alternative keyboard shortcuts. The user can additionally segment geometrical and textural
artifacts in the mesh. Landmarks are saved in a .csv file containing the 3D positions of each point.
The segmentations are saved as .txt file containing the selected vertex indices, and additionally
as .csv file containing the 3D position of each selected vertex. This redundancy is a safety measure,
since there are mesh libraries that may alter mesh topologies, which would make the .txt indices useless.

This script does not need any additional libraries to be installed in Blender.


Author: Till Schnabel (contact till.schnabel@inf.ethz.ch)

MIT License

Copyright (c) 2026 ETH Zurich, Till Schnabel

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""

# normal imports
import json
import re
import warnings
from argparse import ArgumentParser
from dataclasses import field, dataclass
from enum import Enum
from typing import Union, Tuple, List
import sys, os
from pathlib import Path

# blender imports
import bpy
import numpy as np
from bpy.types import Operator
from bpy.props import IntProperty, FloatProperty, BoolProperty, StringProperty
from bpy_extras import view3d_utils
from bpy_extras.io_utils import ExportHelper
import bmesh

# We need to add the src module to the system path to add functions from other files,
# since this file is executed from blender.
if __name__ == "__main__":
    file_path = Path(os.path.realpath(__file__))
    src_dir = file_path.parents[1]
    assert src_dir.name == "src" and "inface" in src_dir.parent.name.lower()
    sys.path.append(str(src_dir.parent))

# imports from own code
from src.blender import blender_utils
from src.objects.mesh import Mesh, Landmarks
from src.objects.indices_and_masks import IndicesAndMasks
from src import utils

# For the blender session, we disable the default function of the keys "S", "R", and "G". We don't need scaling or any
# other types of object manipulations. In fact, we must avoid them, since they would change the landmark positions.
# Additionally, the user can use S to toggle off/on view syncing, so this ensures there's no overlap. Since the user
# can select all vertices when pressing "L", we disable this default behavior as well to avoid that while selecting
# certain vertices and switching the sync toggle, suddenly all vertices are selected.
# I've also added the T and I keys, the former is now used for triggering exclude_texture mode,
# the latter is just unnecessary.
# Generally note that default behavior should also be suppressed by not returning {"PASS_THROUGH"} in the respective
# running modal operator. This here is more just a safety measure.
wm = bpy.context.window_manager
kc = wm.keyconfigs.default

# List of keys and their associated operations to disable
keys_to_disable_per_km = {
    '3D View': {
        # we don't want to accidentally change the scan or landmarks with the blender shortcuts
        'G': 'transform.translate',
        'R': 'transform.rotate',
        'S': 'transform.resize',
        'W': 'wm.tool_set_by_id',  # this can be used similarly as G to translate
        # Also some other adjustment methods
        'D': 'object.duplicate_move',
        'I': 'anim.keyframe_insert_menu',
        'T': 'wm.call_panel',
        'H': 'object.hide_view_set',
        'NUMPAD_PLUS': 'view3d.zoom',
        'NUMPAD_MINUS': 'view3d.zoom'
    },
    "Mesh": {
        'L': 'mesh.select_linked_pick',
        'I': 'mesh.inset'
    }
}

# Disable the keys
for km_type, keys_to_disable in keys_to_disable_per_km.items():
    for key, operation in keys_to_disable.items():
        km = kc.keymaps[km_type]
        for kmi in km.keymap_items:
            if kmi.idname == operation and kmi.type == key:
                if key == 'T' and kmi.properties.name not in ['VIEW3D_PT_tools_object', 'VIEW3D_PT_tools_mesh']:
                    continue
                km.keymap_items.remove(kmi)

class EditMode(Enum):
    """
    Simple data class to handle the different region selections for the different meshes via a consistent naming scheme.
    """
    NONE = None
    EXCLUDE_GEOMETRY = "exclude_geometry"
    EXCLUDE_TEXTURE = "exclude_texture"

    def get_value(self, num_mesh):
        value = self.value
        if value is None:
            return None
        return f"mesh{num_mesh}_{value}"

@dataclass
class Config:
    """
    This data class is instantiated globally and handles the communication between all the methods and operators
    in this script. It contains all the information about the current state of the landmarking tool.
    """
    # Variable to hold the mouse sphere object to be reused
    MOUSE_SPHERE = None
    # This contains the Landmarks object
    landmarks: Landmarks = Landmarks.get_empty_landmarks()
    # The radius is adjusted once relative to the first mesh's size.
    # It's a good general measure for size in this script.
    landmark_radius: float = 1.0
    # The landmark scale, unlike the landmark radius, can be adjusted by the user.
    landmark_scale = 1.0
    # Dynamic variable that specifies how many landmarks are currently set by the user.
    landmark_counter: int = 0
    # Static variable that may define the maximum number of landmarks to be set by the user.
    # If this is set, the landmarks are also colored from blue to green to red.
    num_landmarks: int = None
    # This variable contains the colors of the landmark objects, and it's only defined if num_landmrks is defined.
    landmark_colors: np.ndarray = None
    # An undefined landmark is colored in white, since this is well visible against Blender's default dark background
    landmark_undefined_color = [1, 1, 1]
    # Path to export the landmarks to. This can be pre-defined via CLI or manually chosen via a File window.
    export_landmarks_path: str = None
    # This variable comprises the paths of the export region selection for each mesh.
    # It only contains the paths for the geometry selection. The texture selection paths are automatically
    # computed based on the geometry selection paths.
    export_region_paths: dict = field(default_factory=dict)
    # The names of the meshes to be shown and landmarked.
    input_mesh_names: list[str] = None
    # List of python meshes (we keep this list in addition to the blender objects for some convenient processing steps)
    meshes: list[Mesh] = None
    # We also keep track of each mesh's center (mean of all vertex positions),
    # which is used in the mesh slicing.
    mesh_centers: list[np.ndarray] = None
    # Dictionary that specifies for each view, which mesh is currently shown in that view.
    mesh_per_view: dict = field(default_factory=dict)
    # List of events that are passed through to Blender even when we're in a blocking mode.
    # This allows the user to still e.g. rotate mesh while they're setting the landmarks.
    pass_through_events: list = field(default_factory=lambda: [
        "MIDDLEMOUSE", "TRACKPADPAN", "WHEELINMOUSE", "WHEELOUTMOUSE", "WHEELUPMOUSE", "WHEELDOWNMOUSE",
        "TRACKPADZOOM"])
    # Specifies for each mesh the EditMode it's currently in
    # (including None when the user is not selecting any region for that mesh at the moment).
    edit_modes: list[EditMode] = None
    # Flag that's initially False, so when the user tries to export landmarks despite having set fewer than
    # num_landmarks, they're asked again if they really want to export.
    # If they choose "Yes", this flag is permanently set to True.
    allow_incomplete_lm: bool = False
    # We define a couple of shortcut keys on the keyboard, which can be used to switch around the different modes
    # more efficiently.
    mark_keys: dict = field(default_factory=lambda: {"G": EditMode.EXCLUDE_GEOMETRY, "T": EditMode.EXCLUDE_TEXTURE})
    set_landmarks_key = "S"
    edit_landmarks_key = "E"
    sync_key = "L"
    # This attribute controls the syncing between the potentially multiple views shown for the user.
    # Switching is done by pressing "L" (see above).
    sync_active: bool = False
    # This is the collection meshes are assigned to by default in Blender.
    # We use different collections for proper hiding and showing of different meshes.
    default_collection = None
    # This is a flag that can be set True via CLI. If True, landmark updates are continuously saved,
    # instead of waiting for the user to manually press on "Export".
    save_continuously: bool = False
    # Flag that specifies whether the landmarks are currently visible (True) or not (False)
    landmarks_visible: bool = True
    # Flag that avoids infinite recursion problems while updating the index of a landmark.
    currently_updating_lm_indices: bool = False
    # Flag that can be set True via CLI, in which case Blender closes immediately after the landmarks are exported.
    quit_blender_after_landmark_export: bool = False

    # Next we have some methods to handle the pressing of the relevant keys to trigger the correct events

    @staticmethod
    def any_key_pressed(event):
        return event.value == "PRESS"

    @staticmethod
    def specific_key_pressed(event, key_or_list_of_keys: Union[List[str], str]):
        """
        We if the current event translates to a specific key having been pressed.
        We first check if any key as been pressed, then we check if the key name matches with one or multiple
        potential matches, then we also check if the ascii translation of the currently pressed key(s) matches with
        any of those, which is useful if the combination of multiple keys translates to a specific sign, e.g.,
        shift and a number giving a special sign.
        """
        if config.any_key_pressed(event):
            list_of_keys = key_or_list_of_keys if isinstance(key_or_list_of_keys, list) else [key_or_list_of_keys]
            return event.type in [specific_key.upper() for specific_key in list_of_keys] or getattr(event, "ascii", "") in list_of_keys
        else:
            return False

    def trigger_key_pressed(self, event):
        return (
                self.sync_key_pressed(event) or
                self.mark_key_pressed(event) or
                self.set_landmarks_key_pressed(event) or
                self.edit_landmarks_key_pressed(event) or
                self.plus_key_pressed(event) or
                self.minus_key_pressed(event)
        )

    def sync_key_pressed(self, event):
        # The user can toggle off/on view syncing by pressing "L" or "S"
        return config.specific_key_pressed(event, self.sync_key)

    def pass_event(self, event):
        return event.type in self.pass_through_events or self.sync_key_pressed(event)

    def mark_key_pressed(self, event):
        # The user can toggle off/on view are marking modes via buttons defined in mark_keys
        return self.specific_key_pressed(event, list(self.mark_keys.keys()))

    def invoke_mark_event(self, event):
        assert self.mark_key_pressed(event)
        operator = getattr(bpy.ops.landmarking, self.mark_keys[event.type].value)
        operator('INVOKE_DEFAULT')

    # Blender context override for a specific area and region
    @staticmethod
    def get_override(area_type, region_type):
        for area in bpy.context.screen.areas:
            if area.type == area_type:
                for region in area.regions:
                    if region.type == region_type:
                        override = {'area': area, 'region': region}
                        return override
        # No override context found, return None
        return None

    def set_landmarks_key_pressed(self, event):
        return self.specific_key_pressed(event, self.set_landmarks_key)

    def invoke_set_landmarks(self, event):
        assert self.set_landmarks_key_pressed(event)
        lm_override = self.get_override('VIEW_3D', 'WINDOW')
        blender_utils.blender_call_with_override(bpy.ops.landmarking.infinite_landmarks, lm_override, 'INVOKE_DEFAULT')

    def edit_landmarks_key_pressed(self, event):
        return self.specific_key_pressed(event, self.edit_landmarks_key)

    def invoke_edit_landmarks(self, event):
        assert self.edit_landmarks_key_pressed(event)
        lm_override = self.get_override('VIEW_3D', 'WINDOW')
        blender_utils.blender_call_with_override(bpy.ops.landmarking.edit_landmarks, lm_override, 'INVOKE_DEFAULT')

    def plus_key_pressed(self, event):
        return self.specific_key_pressed(event, ["NUMPAD_PLUS", "EQUAL+SHIFT", "+"])

    def minus_key_pressed(self, event):
        return self.specific_key_pressed(event, ["NUMPAD_MINUS", "-"])

    def set_landmark_colors(self, num_landmarks: int = None) -> None:
        if num_landmarks is not None:
            self.num_landmarks = num_landmarks
            self.landmark_colors = Landmarks.get_landmark_colors_static(self.num_landmarks)

    def get_landmark_color(self, num_landmark: int = None) -> np.ndarray:
        num_landmark = self.landmark_counter if num_landmark is None else num_landmark
        if self.landmark_colors is not None and num_landmark < len(self.landmark_colors):
            return self.landmark_colors[num_landmark]
        else:
            return np.asarray([0., 0., 1.])

    def requires_more_landmarks(self) -> bool:
        """
        :return: True/False if the number of landmarks is defined. False otherwise.
        """
        if self.num_landmarks is not None:
            return self.landmark_counter < self.num_landmarks
        else:
            return False

    def import_landmarks(self, landmarks_path: Union[str, Path]) -> None:
        """
        Import landmarks, add them to self.landmarks attribute and adjust landmark counter.
        Note that if these are more landmarks than allowed by self.num_landmarks, they are truncated.
        :param landmarks_path: path to landmarks file.
        :return: None
        """
        landmarks = Landmarks.load(landmarks_path)
        if landmarks.confidence is None:
            landmarks.confidence = np.ones(landmarks.get_num_points(), dtype=np.float32)
        self.landmarks.add_points(point_array=landmarks.points, confidence=landmarks.confidence)
        num_current_landmarks = self.landmarks.get_num_points()
        if self.num_landmarks is not None and num_current_landmarks > config.num_landmarks:
            warnings.warn(f"Too many landmarks to import. Given are {num_current_landmarks}, "
                          f"but the max is {config.num_landmarks}. Truncating number.")
            self.landmarks.truncate(config.num_landmarks)
        self.landmark_counter = self.landmarks.get_num_points()

    def add_landmark(self, location, confidence: float, index: int = None) -> None:
        """
        Add a landmark to self.landmarks and increase self.landmark_counter by 1.
        By default, the landmark is appended at the end, but it may also be added at another index.
        This does not handle the coloring or indexing of the Blender objects.
        :param location: 3D location at which to add the new landmark.
        :param confidence: Confidence of the new landmark (undefined landmarks get confidence 0)
        :param index: Index of the landmark to add. If not specified, landmark is appended at the end.
        :return: None
        """
        if self.num_landmarks is not None:
            assert self.landmark_counter < self.num_landmarks

        self.landmarks.add_point(location, index=index, confidence=confidence)
        self.landmark_counter += 1

    def change_landmark_index(self, old_index, new_index) -> None:
        """
        Change the index of a specific landmark. All other landmarks are re-indexed accordingly.
        This does not handle the coloring or indexing of the Blender objects.
        :param old_index: Current landmark index.
        :param new_index: The new index to assign to the landmark.
        :return: None
        """
        current_location = self.landmarks.get_point(old_index)
        current_confidence = self.landmarks.confidence[old_index]
        self.delete_landmark(old_index)
        self.add_landmark(current_location, confidence=current_confidence, index=new_index)

    def is_landmark_object_defined(self, lm_obj: bpy.types.Object) -> bool:
        """
        Check if the landmark Blender object is defined or not. This method simply uses the coloring
        of the landmark object by comparing it to self.landmark_undefined_color.
        :param lm_obj: Landmark Blender object.
        """
        mat = blender_utils.get_obj_materials(lm_obj)[0]
        return not np.allclose(mat.diffuse_color[:3], self.landmark_undefined_color)

    def change_landmark_object_index(self, landmark_object_or_shown_index: Union[bpy.types.Object, int], shown_index_to_assign: int) -> None:
        """
        We change the shown index of the given landmark object. This is unlike the method change_landmark_index in that
        it actually concerns the Blender object. The color of the object is also adjusted in case it's a defined
        landmark.
        :param landmark_object_or_shown_index: Landmark object or the index of it that's displayed in Blender.
        :param shown_index_to_assign: New index to be displayed for the landmark object.
        :return: None
        """
        if isinstance(landmark_object_or_shown_index, int):
            current_lm_obj = self.get_landmark_object(self.get_landmark_index_from_shown(landmark_object_or_shown_index))
        else:
            current_lm_obj = landmark_object_or_shown_index
        actual_index_to_assign = self.get_landmark_index_from_shown(shown_index_to_assign)

        # We make the check here just to avoid making more calls to the update function
        if current_lm_obj.landmark_shown_index != shown_index_to_assign:
            previous_lm_index_updating_state = self.currently_updating_lm_indices
            try:
                self.currently_updating_lm_indices = True
                current_lm_obj.landmark_shown_index = shown_index_to_assign
            finally:
                self.currently_updating_lm_indices = previous_lm_index_updating_state
        current_lm_obj.name = self.get_landmark_name(actual_index_to_assign)

        if self.is_landmark_object_defined(current_lm_obj):
            assign_landmark_material(current_lm_obj, color=self.get_landmark_color(actual_index_to_assign))

    def delete_landmarks(self) -> None:
        """
        Delete all landmarks. self.landmarks is emptied, and self.landmark_counter is set to 0.
        This does not concern the Blender objects.
        :return: None
        """
        self.landmarks = Landmarks.get_empty_landmarks()
        self.landmark_counter = 0

    def delete_landmark(self, index) -> None:
        """
        Delete a specific landmark from self.landmarks and reduce self.landmark_counter by 1.
        This does not concern the Blender objects.
        :param index: index of the landmark to be deleted
        :return None
        """
        self.landmarks.remove_point(index)
        self.landmark_counter -= 1

    def adjust_landmark(self, index, location, confidence: float = 1) -> None:
        """
        Adjust the position of a specific landmark within the self.landmarks attribute.
        This does not concern the Blender objects.
        """
        self.delete_landmark(index)
        self.add_landmark(location, index=index, confidence=confidence)

    def adjust_landmark_coord(self, index: int, coord: int, loc: Union[int, float]) -> None:
        """
        Adjust individual coordinate component of specific landmark.
        :param index: index of the landmark.
        :param coord: coordinate component of the landmark (0 for x-, 1 for y-, 2 for z coordinate).
        :param loc: New value to set the coordinate component to.
        :return None
        """
        full_loc = self.landmarks.get_point(index)
        coord = min(max(coord, 0), 2)
        full_loc[coord] = loc
        self.adjust_landmark(index, full_loc)

    # Hard-coded landmark coordinate adjustment methods for x, y, and z
    def adjust_landmark_x(self, index: int, loc: Union[int, float]):
        self.adjust_landmark_coord(index, 0, loc)

    def adjust_landmark_y(self, index: int, loc: Union[int, float]):
        self.adjust_landmark_coord(index, 1, loc)

    def adjust_landmark_z(self, index: int, loc: Union[int, float]):
        self.adjust_landmark_coord(index, 2, loc)


    # Whenever we write "shown_index", we refer to the index that's displayed for the landmark Blender object.
    # Otherwise, we write only "index" and mean the original 0-based index.
    @staticmethod
    def get_landmark_shown_index(index: int) -> int:
        """
        This method basically transforms the 0-based under-the-hood landmark indices
        to 1-based indices that are displayed. That means that the internal self.landmarks attribute uses
        the internal 0-based indices, while the landmark Blender objects start at 1.
        If this behavior should be changed, you only need to adjust this method and get_landmark_index_from_shown()
        :param index: 0-based landmark index to be converted to the displayed one.
        :return: displayed landmark index (int).
        """
        return index + 1

    @staticmethod
    def get_landmark_index_from_shown(shown_index: int) -> int:
        """
        Reverse of get_landmark_shown_index()
        :param shown_index: Index that's displayed for the landmark Blender object.
        :return: Original 0-based landmark index (int).
        """
        return shown_index - 1

    def get_landmark_name(self, index: int = None) -> str:
        """
        Get landmark name from its index (basically just its shown index converted to string).
        If no index is provided, the name of a landmark to be appended at the end is returned.
        :param index: Landmark index.
        :return: Landmark name (str)
        """
        index = self.landmark_counter if index is None else index
        return str(self.get_landmark_shown_index(index))

    def change_landmark_scale(self, scale: Union[float, int]) -> None:
        """
        Change the scale of all landmark Blender objects. This does not change anything about the internal landmarks.
        Note that the radius of the landmark Blender objects is not changed, but the spheres are scaled down/up.
        :param scale: Uniform scale to set for all landmark spheres.
        :return: None
        """
        self.landmark_scale = scale
        for num_lm in range(self.landmarks.get_num_points()):
            lm_blender_obj = blender_utils.get_obj_by_name(self.get_landmark_name(num_lm))
            lm_blender_obj.scale = (scale, scale, scale)
        if self.MOUSE_SPHERE is not None and hasattr(self.MOUSE_SPHERE, "scale"):
            self.MOUSE_SPHERE.scale = (scale, scale, scale)

    @staticmethod
    def get_landmark_index(landmark_obj) -> int:
        """
        More or less the reverse of get_landmark_name(). You provide the landmark Blender object, and this
        methods returns its internal 0-based index.
        :param landmark_obj: Landmark Blender object.
        :return: Landmark index (int) in [0, self.landmark_counter - 1]
        """
        return Config.get_landmark_index_from_shown(int(landmark_obj.name.strip()))

    def get_landmark_object(self, landmark_index):
        """
        Reverse of get_landmark_index(). You give the landmark index, this method returns the respective Blender object.
        :param landmark_index: Landmark index (int) in [0, self.landmark_counter - 1]
        :return: Blender object.
        """
        if landmark_index >= self.landmarks.get_num_points():
            raise IndexError(f"Index {landmark_index} exceeds number of landmarks.")
        return blender_utils.get_obj_by_name(self.get_landmark_name(landmark_index))

    def get_all_landmark_objects(self) -> List:
        """
        Return list of all landmark Blender objects.
        """
        return [self.get_landmark_object(landmark_index) for landmark_index in range(self.landmarks.get_num_points())]

    @staticmethod
    def is_landmark_object(obj) -> bool:
        """
        Check if the given Blender object is a landmark (used to detect whether the user has currently selected a
        landmark).
        :param obj: Blender object
        :return: True if the given Blender object is a landmark, False otherwise.
        """
        obj_name = blender_utils.get_obj_name(obj.name)
        try:
            int(obj_name)
        except ValueError:
            return False
        return obj_name not in config.input_mesh_names

    def select_input_mesh(self, num_mesh: int) -> None:
        """
        Select specific input mesh.
        :param num_mesh: Number of mesh to select.
        :return: None
        """
        blender_utils.select_object_by_name(self.input_mesh_names[num_mesh], make_active=True)

    def load_meshes(self, *mesh_paths: Union[str, Path]) -> None:
        """
        Load meshes from paths to be displayed in the interface. The meshes are loaded with blender
        and also converted to our Mesh class to enable some more processing.
        :param mesh_paths: Paths of the meshes to be loaded.
        :return: None
        """

        # First initialize the mesh-related attributes

        # Default Blender collection that any loaded mesh is first assigned to
        self.default_collection = bpy.data.collections.get("Collection")
        self.input_mesh_names: List[str] = []
        self.meshes: List[Mesh] = []
        self.mesh_centers: List[np.ndarray] = []
        self.edit_modes: List[EditMode] = []

        assert len(mesh_paths) > 0, "Need to provide at least one mesh path for landmarking"

        for num_mesh, mesh_path in enumerate(mesh_paths):
            # Load mesh with Blender
            mesh_path = Path(mesh_path)
            mesh_obj = blender_utils.load_mesh(mesh_path)

            # Save its name
            mesh_name = blender_utils.get_obj_name(mesh_obj)
            self.input_mesh_names.append(mesh_name)

            # We convert the blender object to a python mesh, so that we don't rely
            # on any other python mesh library (e.g. open3d) for the import.
            python_mesh = blender_utils.convert_blender_to_python_mesh(mesh_obj)
            self.meshes.append(python_mesh)

            # Mesh center is defined as the average of all vertices (uniformly weighted)
            mesh_vertices = blender_utils.get_vertices(mesh_obj, copy=True)
            self.mesh_centers.append(np.mean(mesh_vertices, axis=0))

            # define edit mode for mesh; we start with None, since the mesh starts in object mode
            self.edit_modes.append(EditMode.NONE)

            # Create new collection for every mesh, so that we can hide/show
            # the individual collections in specific views.
            # Otherwise, we could only show/hide meshes in all views together.
            mesh_collection = bpy.data.collections.new(self.input_mesh_names[-1])
            bpy.context.scene.collection.children.link(mesh_collection)
            mesh_collection.objects.link(mesh_obj)
            # Unlink mesh from default collection
            if self.input_mesh_names[-1] in self.default_collection.objects:
                self.default_collection.objects.unlink(mesh_obj)

            # We make some specific mesh-size-based adjustments only for the first mesh,
            # so the first mesh serves as size reference
            if num_mesh == 0:
                # We define the mesh size as the length of the mesh's longest axis
                mesh_size = max(np.max(mesh_vertices, axis=0) - np.min(mesh_vertices, axis=0))
                for area in bpy.context.screen.areas:
                    if area.type == 'VIEW_3D':
                        for space in area.spaces:
                            if space.type == 'VIEW_3D':
                                # Set view location at mesh center
                                space.region_3d.view_location = self.mesh_centers[-1]
                                # And distance 1.5x the mesh size
                                space.region_3d.view_distance = mesh_size * 1.5
                                break

                # Here we set the landmark radius to be 1% of the mesh size.
                # This attribute stays fixed and is used as size reference in multiple places.
                # Adjustments to the landmark sphere sizes are done via self.landmark_scale.
                self.landmark_radius = mesh_size * 0.01

    def toggle_collection_visibility_for_num_mesh(
            self, current_num_mesh: int, current_override, toggle: bool = True, extend: bool = False) -> None:
        """
        Toggle collection visibility for a specific mesh, essentially to show/hide the mesh in a specific view.
        This method essentially calls bpy.ops.object.hide_collection() for the mesh's collection, with the given
        context and arguments.
        :param current_num_mesh: Mesh index.
        :param current_override: Blender contex override.
        :param toggle: Toggle boolean (arg of bpy.ops.object.hide_collection)
        :param extend: (arg of bpy.ops.object.hide_collection)
        :return: None
        """
        blender_utils.toggle_collection_visibility(
            self.input_mesh_names[current_num_mesh], current_override, toggle=toggle, extend=extend)

    def set_mesh_per_view(self, view, num_mesh: int) -> None:
        """
        Define for a specific view which mesh to show in that view.
        :param view: Blender view
        :param num_mesh: Mesh index.
        :return: None
        """
        try:
            override_context = blender_utils.get_3d_view_context_overrides(specific_views=view)
        except ValueError:
            raise AssertionError("No region was found to change the collection visibility in.")
        bpy.context.window_manager.windows.update()
        view.spaces.active.use_local_collections = True
        # Disable default collection
        blender_utils.toggle_collection_visibility(
            config.default_collection.name, override_context, toggle=False, extend=False)
        # enable specific mesh collection
        self.toggle_collection_visibility_for_num_mesh(num_mesh, override_context, toggle=True, extend=False)
        # Write the mesh index into the mesh_per_view attribute to keep track which mesh is shown in which view.
        self.mesh_per_view[view] = num_mesh

    def show_all_meshes_in_all_views(self) -> None:
        """
        Make all mesh collections visible in all views. This is meant for internal use only, since normally, only
        one mesh should be shown in each view. This method also does not adjust self.mesh_per_view, so use
        reset_meshes_per_view() after you're done with this method.
        :return: None
        """
        view_areas = blender_utils.get_all_areas(area_types="VIEW_3D")

        if len(view_areas) < 1:
            raise AssertionError("Unable to find 3D view areas.")

        for num_view, view_area in enumerate(view_areas):
            try:
                override_context = blender_utils.get_3d_view_context_overrides(specific_views=view_area)
            except ValueError:
                raise AssertionError("No region was found to change the collection visibility in.")
            bpy.context.window_manager.windows.update()
            view_area.spaces.active.use_local_collections = True
            blender_utils.toggle_collection_visibility(
                config.default_collection.name, override_context, toggle=False, extend=False)
            for num_mesh in range(len(self.meshes)):
                self.toggle_collection_visibility_for_num_mesh(num_mesh, override_context, toggle=True, extend=False)

    def reset_meshes_per_view(self) -> None:
        """
        This method counteracts the effect of show_all_meshes_in_all_views. It returns the views to the state that only
        one mesh is shown per view, as specified in self.mesh_per_view.
        :return: None
        """
        for view, num_mesh in self.mesh_per_view.items():
            self.set_mesh_per_view(view, num_mesh)

    def get_current_indices(self, num_mesh: int) -> np.ndarray:
        """
        Return the currently selected mesh indices. Crucially, this also includes those selected indices
        of vertices that are currently sliced away, using the info from the vertex groups.
        :param num_mesh: Mesh index.
        :return numpy int array of selected mesh vertex indices.
        """
        mesh_obj = self.get_mesh_obj(num_mesh)
        # The current vertex selection only contains True values for vertices that are shown,
        # so all vertices currently sliced away have False, even if they were selected before.
        current_vertex_selection = blender_utils.get_current_vertex_selection(mesh_obj, return_as_indices=False)
        if self.is_in_any_edit_mode(num_mesh):
            # Hence, we retrieve what was previously selected in the vertex group
            vertex_group_selection = blender_utils.get_vertex_selection_from_groups(
                mesh_obj, self.edit_modes[num_mesh].get_value(num_mesh), return_as_indices=False)
            hidden_vertex_mask = blender_utils.get_hidden_vertices(mesh_obj, return_as_indices=False)
            # And we add the True values from the vertex group to the current vertex selection.
            current_vertex_selection[hidden_vertex_mask] = vertex_group_selection[hidden_vertex_mask]
        return IndicesAndMasks.get_indices(current_vertex_selection)

    def get_region_path(self, region_type: EditMode, num_mesh: int) -> Union[str, None]:
        """
        Get the file path to save the specific region selection for the specific mesh.
        :param region_type: EditMode object specifying for which type of region selection to get the path.
        :param num_mesh: Mesh index.
        :return: Absolute path as string if available, otherwise None.
        """
        if num_mesh not in self.export_region_paths:
            return None
        export_region_path = Path(self.export_region_paths[num_mesh])
        if region_type == EditMode.EXCLUDE_TEXTURE:
            return str(export_region_path.with_stem(f"{export_region_path.stem}_texture"))
        elif region_type == EditMode.EXCLUDE_GEOMETRY:
            return str(export_region_path)
        else:
            raise TypeError(f"Invalid region type: {region_type}.")

    def set_region_path(self, num_mesh: int, export_region_path: Union[Path, str]) -> None:
        """
        Set the region path for the specific mesh (as Path with .txt ending)
        :param num_mesh: Mesh index.
        :param export_region_path: Absolute file path (str or Path).
        :return: None
        """
        self.export_region_paths[num_mesh] = Path(export_region_path).with_suffix(".txt")

    def set_indices_region(self, region_type: EditMode, num_mesh: int, indices=None) -> None:
        """
        Set the indices for a specific region type, i.e., select the vertices on the object,
        assign the selection to the respective vertex group, then set the object to object mode.
        We assume that this function is called while the object is in edit mode, and when the user
        is done selecting, which is why after we do the assignment, the object is put to object mode again.
        :param region_type: EditMode object specifying the type of region selection
                            (for assigning the selection to the correct vertex group).
        :param num_mesh: Mesh index.
        :param indices: Provide the indices to assign to the vertex group. If not provided,
                        the saved indices will be used. If that doesn't exist, nothing happens.
        :return: None
        """
        export_region_path = self.get_region_path(region_type, num_mesh=num_mesh)
        if export_region_path is None and indices is None:
            return
        blender_utils.select_vertices(
            export_region_path if indices is None else indices, self.input_mesh_names[num_mesh],
            vertex_group_name=region_type.get_value(num_mesh))
        blender_utils.set_to_object_mode(self.input_mesh_names[num_mesh])
        self.edit_modes[num_mesh] = EditMode.NONE

    def set_input_mesh_to_edit_mode(self, num_mesh: int) -> None:
        """
        Set specific mesh to edit mode and select it.
        :param num_mesh: Mesh index.
        :return: None
        """
        if blender_utils.is_in_edit_mode(self.input_mesh_names[num_mesh]):
            warnings.warn("You're already in edit mode.")
            return
        blender_utils.set_to_edit_mode(self.input_mesh_names[num_mesh])
        blender_utils.select_obj(blender_utils.get_obj_by_name(self.input_mesh_names[num_mesh]))

    def set_input_mesh_to_obj_mode(self, num_mesh: int) -> None:
        """
        Set specific mesh to object mode.
        :param num_mesh: Mesh index.
        :return: None
        """
        blender_utils.set_to_object_mode(self.input_mesh_names[num_mesh])

    def is_input_mesh_in_object_mode(self, num_mesh: int) -> bool:
        """
        :param num_mesh: Mesh index.
        :return: True if mesh is in object mode. False otherwise.
        """
        return blender_utils.is_in_object_mode(self.input_mesh_names[num_mesh])

    def is_in_any_edit_mode(self, num_mesh: int):
        """
        :param num_mesh: Mesh index.
        :return: True if mesh is in an edit mode used for a region selection. False otherwise.
        """
        return self.edit_modes[num_mesh] != EditMode.NONE

    def is_in_region_edit_mode(self, region_type: EditMode, num_mesh: int):
        """
        :param region_type: EditMode object specifying the type of region selection to check if the mesh is in.
        :param num_mesh: Mesh index.
        :return: True if mesh is in the specific edit mode. False otherwise.
        """
        return self.is_in_any_edit_mode(num_mesh) and self.edit_modes[num_mesh] == region_type

    def is_in_other_edit_mode(self, region_type: EditMode, num_mesh: int):
        """
        :param region_type: EditMode object specifying the type of region selection to check if the mesh is not in.
        :param num_mesh: Mesh index.
        :return: True if mesh is in an edit mode, but not for the given region type. False otherwise.
        """
        return self.is_in_any_edit_mode(num_mesh) and not self.is_in_region_edit_mode(region_type, num_mesh=num_mesh)

    def set_to_region_edit_mode(self, region_type: EditMode, num_mesh: int) -> None:
        """
        Set a specific mesh into a specific edit mode. In Blender, this simply puts the object into edit mode,
        but it also checks if the mesh is already in edit mode, and it uses the self.edit_modes attribute to
        ensure we move properly between the different region selections.
        raises TypeError if the region_type is not a proper edit mode.
        :param region_type: EditMode object specifying the type of region selection to set the mesh to.
        :param num_mesh: Mesh index.
        :return: None
        """
        # Check input
        if not isinstance(region_type, EditMode):
            raise TypeError(f"Invalid region type {region_type}.")
        if region_type == EditMode.NONE:
            raise TypeError("Don't select None as the region type when switching to edit mode.")

        if self.is_in_other_edit_mode(region_type, num_mesh=num_mesh):
            warnings.warn(f"You're already in {self.edit_modes[num_mesh]} edit mode.")
            return
        self.set_input_mesh_to_edit_mode(num_mesh=num_mesh)

        # for texture selection, we select from geometry and texture,
        # assuming that any type of geometric artifact is also a texture artifact.
        selection_groups = [EditMode.EXCLUDE_GEOMETRY.get_value(num_mesh),
                            EditMode.EXCLUDE_TEXTURE.get_value(num_mesh)] \
            if region_type == EditMode.EXCLUDE_TEXTURE else region_type.get_value(num_mesh)

        if not blender_utils.has_vertex_group(self.input_mesh_names[num_mesh], selection_groups):
            raise AssertionError("This shouldn't happen. The vertex groups should all be set at the "
                                 "beginning of this script.")

        # select the indices by grabbing the current vertex group selection(s)
        blender_utils.select_vertices_from_groups(
            self.input_mesh_names[num_mesh], selection_groups, deselect_current_selection=True)
        self.edit_modes[num_mesh] = region_type

    def get_num_mesh_from_view(self, view) -> int:
        """
        Return the index of the mesh object shown in the given view.
        :param view: Blender view.
        :return: Mesh index
        """
        if view not in self.mesh_per_view:
            raise KeyError("Provided view is not saved in mesh_per_view dict. Dunno why.")
        return self.mesh_per_view[view]

    def get_mesh_obj(self, num_mesh: int):
        """
        Get the Blender object for the provided mesh index.
        :param num_mesh: Mesh index
        :return: Blender object
        """
        return blender_utils.get_obj_by_name(self.input_mesh_names[num_mesh])

    def switch_landmarks_visible(self):
        """
        Hide/show all landmark blender objects. This doesn't alter anything else about the objects, so color, position,
        and index stay the same, and the internal self.landmarks attribute is left untouched.
        The self.landmarks_visible attribute is switched in its state to keep track if the landmarks are currently
        visible or not.
        """
        for num_lm in range(self.landmark_counter):
            blender_utils.set_object_render_invisible(self.get_landmark_name(num_lm), state=self.landmarks_visible)
        self.landmarks_visible = not self.landmarks_visible

#
#
# Next we define some global methods.
#
#

def assign_landmark_material(landmark_obj, color: Union[list, np.ndarray] = None,
                             material_name: str = "landmark_material") -> None:
    """
    Assign material with diffuse color to landmark object.
    :param landmark_obj: Landmark Blender object.
    :param color: Material color. If not provided, blue color is used.
    :param material_name: Material name (str), default is "landmark_material".
    :return: None
    """
    mat = bpy.data.materials.new(name=material_name)
    diffuse_color = [0.0, 0.0, 1.0, 1.0]
    if color is not None:
        diffuse_color[:len(color)] = color
    mat.diffuse_color = diffuse_color
    landmark_obj.active_material = mat


def get_landmark_sphere(location: Union[list[float], tuple[float, float, float], None] = None,
                        color: Union[list, np.ndarray] = None,
                        landmark_name: str = None, material_name: str = "landmark_material"):
    """
    Create a new landmark sphere object or return existing one.
    :param location: Where to position the landmark sphere. If not provided, sphere is assigned the undefined color.
    :param color: The color to assign to the landmark sphere (overwritten in case no location is provided).
    :param landmark_name: If this is provided, and there is an existing landmark sphere object, no new object is
                          created, but the existing one returned (potentially still rescaled, though).
    :param material_name: Material name (str), default is "landmark_material".
    :return: Landmark Blender object (sphere).
    """
    landmark_name = config.get_landmark_name() if landmark_name is None else landmark_name
    # Use existing object if it has been created already
    if blender_utils.exists_obj(landmark_name):
        landmark_sphere = blender_utils.get_obj_by_name(landmark_name)
    else:
        # Create new sphere otherwise
        bpy.ops.mesh.primitive_uv_sphere_add(radius=config.landmark_radius)
        landmark_sphere = bpy.context.active_object
        landmark_sphere.name = landmark_name
        if location is not None and not np.isnan(location[0]):
            landmark_sphere.location = location
        else:
            color = config.landmark_undefined_color

        # color landmark sphere
        assign_landmark_material(landmark_sphere, color=color, material_name=material_name)

    landmark_sphere.scale = (config.landmark_scale, config.landmark_scale, config.landmark_scale)

    # Return newly created landmark sphere
    return landmark_sphere


def get_mouse_landmark(mouse_landmark_name: str = "MouseSphere", material_name: str = "mouse_sphere"):
    """
    Get the mouse landmark sphere (Basically just another landmark, but with undefined color, and a different
    object and material name).
    :param mouse_landmark_name: Landmark name (str), default is "MouseSphere".
    :param material_name: Material name (str), default is "mouse_sphere".
    :return: Landmark Blender object (sphere).
    """
    mouse_sphere = get_landmark_sphere(landmark_name=mouse_landmark_name, material_name=material_name)
    return mouse_sphere


def place_landmark(location: Union[list[float], tuple[float, float, float]],
                   index: int, color: Union[list, np.ndarray]) -> None:
    """
    Place a landmark sphere with the given color at the given location. It's assigned the displayed index
    as name and landmark_shown_index attribute; the name is shown in the Blender interface.
    :param location: 3d location (float), must be provided.
    :param color: Color to assign to the landmark sphere; must be provided, since the landmark may not be the last.
    :param index: 0-based index of the landmark.
    :return: None
    """
    assert location is not None and color is not None
    landmark_name = config.get_landmark_name(index)
    landmark_sphere = get_landmark_sphere(location=location, color=color, landmark_name=landmark_name,
                                          material_name="landmark_material")
    landmark_sphere.landmark_shown_index = config.get_landmark_shown_index(index)
    # Show names of defined landmarks. Makes them easier to distinguish.
    bpy.context.object.show_name = True


def import_landmarks(lm_path) -> None:
    """
    Import landmarks from path. Load them into config, and add blender meshes.
    Normally, we keep the positions of undefined landmarks and only adjust their confidence to 0.
    However, for those landmarks where the position is actually not defined, we distribute the blender spheres
    around the first defined landmark (which takes a bit of additional time).
    :param lm_path: Path to the landmarks.
    :return: None
    """
    config.import_landmarks(lm_path)
    if config.landmarks.get_num_points() > 0:
        mesh = config.meshes[0]
        mesh_vertices = mesh.get_vertices()
        # For landmark positioning, we only need to redistribute those landmarks without a defined position.
        # Those with 0 confidence still have a position encoded, which we can use for placement.
        nan_rows = config.landmarks.get_nan_rows(count_zero_confidence_as_nan=False)

        # We only need to define the distribution if there are any landmarks with undefined positions
        # (save time otherwise).
        if len(nan_rows) > 0:
            # We first check if there are any defined landmarks (with larger than zero confidence).
            non_nan_rows = config.landmarks.get_non_nan_rows(count_zero_confidence_as_nan=True)
            # If that's not the case, we also consider landmarks with defined position but 0 confidence.
            if len(non_nan_rows) == 0:
                non_nan_rows = config.landmarks.get_non_nan_rows(count_zero_confidence_as_nan=False)
            # We need to distribute the nan landmarks somehow. If at least one landmark is defined, we
            # distribute the undefined landmarks around the first define one.

            if len(non_nan_rows) > 0:
                mesh_vertex_dists = np.linalg.norm(
                    mesh_vertices - np.expand_dims(config.landmarks.get_point(non_nan_rows[0]), axis=0), axis=1)
                start_vertex_index = np.argmin(mesh_vertex_dists)
            # Otherwise, we need to distribute them around the first mesh vertex. This should be an exceptional case,
            # because why would all the landmarks you import be undefined.
            else:
                start_vertex_index = 0
                mesh_vertex_dists = np.linalg.norm(
                    mesh_vertices - mesh_vertices[start_vertex_index:start_vertex_index + 1], axis=1)

            # We distribute the undefined landmarks along the closest vertices around the defined starting vertex.
            # We try to find the closest vertices along the mesh triangulation instead of just computing
            # Euclidean distances, so that we avoid the case where if there's an inner surface close to the outer
            # surface where the starting vertex is defined, we would put a lot of undefined landmarks on the
            # inner surface that's hard to reach in the interface.

            # As a speed optimization, we only compute the graph for the Dijkstra computation
            # around the vertices that are relatively close to the starting vertex in terms of
            # Euclidean distance.

            # We set the cutoff conservatively, such that we should normally include way more vertices than necessary.
            cutoff_num_vertex = min(max(5000, 50*(len(nan_rows))+1), len(mesh_vertices) - 1)
            vertex_dist_cutoff = np.partition(mesh_vertex_dists, cutoff_num_vertex)[cutoff_num_vertex]

            # Edge-distance per vertex, computed using Dijkstra
            lengths = mesh.dijkstra_lengths(
                start_vertex_index, indices_or_mask=mesh_vertex_dists <= vertex_dist_cutoff, return_as_dict=True)

            # exclude starting vertex from list -- we don't want to put an undefined
            # landmark on the same vertex as a defined one.
            lengths.pop(start_vertex_index)

            # It could be that lengths doesn't have enough entries if most of the vertices are
            # disconnected from the chosen vertex. In that case, we just get the closest vertices
            # based on Euclidean distance, which is undesirable, but this shouldn't happen usually.
            if len(lengths) < len(nan_rows):
                sorted_lengths = np.nonzero(mesh_vertex_dists < vertex_dist_cutoff)[0]
                sorted_lengths = sorted_lengths[sorted_lengths != start_vertex_index]
                sorted_lengths = sorted_lengths[np.argsort(mesh_vertex_dists[sorted_lengths])]
            # In the usual case, we should have more than enough lengths entries, so we get the
            # closest vertex indices based on the triangulation distance.
            else:
                sorted_lengths = sorted(lengths.keys(), key=lambda vertex_idx: lengths[vertex_idx])

        nan_lm_counter = 0
        for num_lm, lm in enumerate(config.landmarks.points):
            if Landmarks.is_point_undefined(lm):
                # We get a better spacing with this kind of vertex choosing.
                # If we just take one closest vertex after the next, the undefined landmarks are very close together.
                lm = mesh_vertices[sorted_lengths.pop((50*(nan_lm_counter + 1)) % len(sorted_lengths))]
                color = config.landmark_undefined_color
                nan_lm_counter += 1
            else:
                if config.landmarks.confidence[num_lm] == 0:
                    color = config.landmark_undefined_color
                else:
                    color = config.get_landmark_color(num_lm)
            place_landmark(lm, color=color, index=num_lm)


def get_global_mouse_coords(context, event) -> Tuple[float, float]:
    """
    Get the global mouse coordinates based on the current context by adding the local region coordinates
    to the region boundary coordinates.
    :param context: Blender context
    :param event: Blender event
    :return: x and y coordinates (tuple of two floats)
    """
    if context.region is None:
        raise NotImplementedError("Please provide a valid context")
    else:
        region_x, region_y = context.region.x, context.region.y
    return event.mouse_region_x + region_x, event.mouse_region_y + region_y


def find_3d_view_under_mouse(mouse_x, mouse_y):
    """
    Find the 3D view the mouse is currently located in. This is important for deciding view-based actions, such
    as view syncing, or which mesh to show in which view.
    :param mouse_x: x coordinate of the mouse
    :param mouse_y: y coordinate of the mouse
    :return: 3D view area or None if no valid view could be found
    """
    for area in bpy.context.window.screen.areas:
        if area.type == 'VIEW_3D':
            x, y, width, height = area.x, area.y, area.width, area.height
            if (x <= mouse_x <= x + width) and (y <= mouse_y <= y + height):
                return area
    return None


def get_current_num_mesh(event) -> Union[int, None]:
    """
    Get the mesh index from the Blender event. Basically, the event first determines the view the event was triggered
    in, and we only have one mesh per view, so from the view, we return the mesh index.
    :param event: Blender event
    :return: mesh index or None if no view could be found for the given event.
    """
    view = find_active_area(event)
    if view is None:
        return None
    return config.get_num_mesh_from_view(view)


def mouse_location_on_mesh(context, event, mesh) -> Tuple[bool, Union[List[float], None]]:
    """
    Get the mouse raycast position on the mesh for the given context and event.
    :param context: Blender context
    :param event: Blender event
    :param mesh: Blender mesh (not the object, but the actual mesh)
    :return Tuple:
        1) Boolean whether hit was found
        2) List of 3 floats for 3D location if a hit was found, otherwise None
    """
    global_mouse_x, global_mouse_y = get_global_mouse_coords(context, event)
    area_3d_view = find_3d_view_under_mouse(global_mouse_x, global_mouse_y)
    if area_3d_view is not None:
        local_mouse_x = global_mouse_x - area_3d_view.x
        local_mouse_y = global_mouse_y - area_3d_view.y
        mouse_coord = (local_mouse_x, local_mouse_y)

        # Find the 'WINDOW' region and its associated 'RegionView3D' data
        region = None
        rv3d = None
        for reg in area_3d_view.regions:
            if reg.type == 'WINDOW':
                region = reg
                break

        for space in area_3d_view.spaces:
            if space.type == 'VIEW_3D':
                rv3d = space.region_3d
                break

        if region and rv3d:
            mouse_origin = view3d_utils.region_2d_to_origin_3d(region, rv3d, mouse_coord)
            vector = view3d_utils.region_2d_to_vector_3d(region, rv3d, mouse_coord)
            hit_result, hit_loc, _, _ = mesh.ray_cast(mouse_origin, vector)
            return hit_result, list(hit_loc)
        else:
            warnings.warn("Could not find region or region data for the current view. This is unexpected, so check.")
            return False, None
    else:
        return False, None


def get_closest_landmark(location: Union[List[float], np.ndarray]):
    """
    Get the closest landmark blender object given a 3D position. This location should probably be computed based
    on the 2D mouse location projected onto the respective mesh.
    :param location: 3D float location
    :return: Blender landmark object or None if no landmark is close by.
    """
    landmark_objects = config.get_all_landmark_objects()
    landmark_locations = np.asarray([np.asarray(lm.location) for lm in landmark_objects])
    distances = np.linalg.norm(np.expand_dims(location, axis=0) - landmark_locations, axis=1)
    closest_lm = np.argmin(distances)
    # We check that the user has clicked into the close proximity of the landmark by comparing the distance
    # between the landmark and the clicked position to the general landmark radius
    # (which is scaled along with the mesh size)
    if distances[closest_lm] <= config.landmark_radius*2:
        return blender_utils.get_obj_by_name(config.get_landmark_name(closest_lm))
    return None


def find_active_area(event):
    """
    Find active area corresponding to the given Blender event.
    :param event: Blender event
    :return: Blender area
    """
    if event is None:
        return None
    for area in bpy.context.screen.areas:
        if area.type == blender_utils.AreaType.VIEW_3D:
            if (area.x < event.mouse_x < area.x + area.width) and (
                    area.y < event.mouse_y < area.y + area.height):
                return area
    return None


def find_active_area_view_spaces(event=None, active_area=None) -> List:
    """
    Return all 3D view spaces from the areas corresponding to the given Blender event or the provided active area.
    :param event: Blender event
    :param active_area: Active blender area, overrides event parameter if provided.
    :return: 3D view spaces as list
    """
    active_area = find_active_area(event) if active_area is None else active_area
    if active_area is not None:
        return [space for space in active_area.spaces if space.type == "VIEW_3D"]
    return []

def is_single_lm_selected() -> bool:
    """
    Check if we currently have only one landmark Blender object selected (no other object).
    This is useful for some additional options in the panel to adjust a specific landmark.
    :return: True if we have exactly one Blender object selected and if that object is a landmark sphere,
             False otherwise.
    """
    objs = blender_utils.get_selected_objects(obj_type="MESH")
    if len(objs) != 1:
        return False
    obj = objs[0]
    return config.is_landmark_object(obj)

def is_single_defined_lm_selected():
    """
    :return: True if we have exactly one Blender object selected and if that object is a landmark sphere
             and if that landmark is defined, False otherwise.
    """
    if not is_single_lm_selected():
        return False
    lm_obj = blender_utils.get_selected_objects(obj_type="MESH")[0]
    return config.landmarks.is_point_defined(config.get_landmark_index(lm_obj))

def update_lm_position(coord: str) -> None:
    """
    Update specific coordinate component of a landmark. Used as a generic update function for updating landmark
    positions via the sliders. The new location is retrieved from the lm_pos_properties object attribute.
    :param coord: coordinate component to update ("x", "y", or "z")
    :return: None
    """
    # check if single landmark is currently selected
    if not is_single_lm_selected():
        return

    # Check coordinate argument for validity
    coord = coord.lower()
    if "x" in coord:
        coord = "x"
        coord_ind = 0
    elif "y" in coord:
        coord = "y"
        coord_ind = 1
    elif "z" in coord:
        coord = "z"
        coord_ind = 2
    else:
        raise TypeError(f"Unknown coordinate: {coord}. Choose between 'x', 'y' and 'z'")

    # Get landmark object
    obj = blender_utils.get_selected_objects(obj_type="MESH")[0]
    # Get new_loc from lm pos properties
    lm_props = obj.lm_pos_properties
    new_loc = getattr(lm_props, f"lm_{coord}")
    current_loc = getattr(obj.location, coord)
    # Only update position if there was an actual change
    if (current_loc == 0 and new_loc != 0) or (current_loc != 0 and not (0.9999 < new_loc / current_loc < 1.0001)):
        setattr(obj.location, coord, new_loc)
        lm_index = config.get_landmark_index(obj)
        config.adjust_landmark_coord(lm_index, coord_ind, new_loc)
        assign_landmark_material(obj, color=config.get_landmark_color(lm_index))
        blender_utils.select_obj(obj)

# Specific component update methods.
def update_lm_pos_x(self, context):
    update_lm_position("x")


def update_lm_pos_y(self, context):
    update_lm_position("y")


def update_lm_pos_z(self, context):
    update_lm_position("z")

# The coordinate component update methods are added as callbacks for the slider updates.
class LMPosProperties(bpy.types.PropertyGroup):
    lm_x: bpy.props.FloatProperty(name="X", description="Change landmark position in x", update=update_lm_pos_x)
    lm_y: bpy.props.FloatProperty(name="Y", description="Change landmark position in y", update=update_lm_pos_y)
    lm_z: bpy.props.FloatProperty(name="Z", description="Change landmark position in z", update=update_lm_pos_z)


def update_slider_values(self, context):
    """
    Callback function used to update the landmark position properties if the object location is changed.
    """
    if not is_single_lm_selected():
        return
    obj = blender_utils.get_selected_objects(obj_type="MESH")[0]
    lm_props = obj.lm_pos_properties
    lm_props.lm_x = obj.location.x
    lm_props.lm_y = obj.location.y
    lm_props.lm_z = obj.location.z

def call_op(id_name: str, op_context: str = 'INVOKE_DEFAULT', **kwargs):
    """
    Call an operator from a string like 'landmarking.export_landmarks'.
    :param id_name: id of the operator
    :param op_context: Basically, what method of the operator to call.
    :param kwargs: Additional keyword arguments to pass to the operator's function.
    :return: operator's output.
    """
    if "." not in id_name:
        raise ValueError(f"Invalid operator {id_name=}")
    cat, op = id_name.split(".", 1)
    op_group = getattr(bpy.ops, cat)
    op_func = getattr(op_group, op)
    return op_func(op_context, **kwargs)

def landmark_index_update(self, context):
    """
    Update function called whenever the landmark_shown_index property is updated for a specific landmark.
    This function handles changing the index of the selected landmark, which involves updating the internal
    landmark array in the config, but also updating the displayed landmark names of this and other landmarks
    affected by the change, as well as changing their landmark_shown_index property.
    Additionally, we clamp the landmark index to be no lower than the very first landmark and no higher
    than the last currently defined landmark.
    """
    # This is to avoid recursion, because when we update the landmark_shown_index further below,
    # this method is automatically called again.
    if config.currently_updating_lm_indices:
        return

    # the shown index is clipped into the allowed range
    min_allowed = config.get_landmark_shown_index(0)
    max_allowed = config.get_landmark_shown_index(config.landmark_counter-1)
    new_shown_index = np.clip(int(self.landmark_shown_index), min_allowed, max_allowed)

    # Write back clipped value, and update others (with recursion guard)
    config.currently_updating_lm_indices = True
    try:
        # Adjust landmark_shown_index in case it was clipped to another value
        if self.landmark_shown_index != new_shown_index:
            self.landmark_shown_index = new_shown_index

        # Adjust the shown index, and also the ones of other affected landmarks.
        current_shown_index = config.get_landmark_shown_index(config.get_landmark_index(self))
        if current_shown_index != new_shown_index:
            # Assign temporary name to avoid that two blender objects are assigned the same name,
            # which would lead to Blender adjusting one of the names.
            self.name = "lm_tmp"
            # If the index was decreased, we increase the indices by 1 of all landmarks that come after the new
            # but before the old index. We go in descending order to avoid any name collisions.
            if current_shown_index > new_shown_index:
                for shown_index_to_shift in range(current_shown_index - 1, new_shown_index - 1,  - 1):
                    config.change_landmark_object_index(shown_index_to_shift, shown_index_to_shift + 1)
            # If the index was increased, we decrease the indices by 1 of all landmarks that come after the old
            # but before the new index. We go in ascending order to avoid any name collisions.
            else:
                for shown_index_to_shift in range(current_shown_index + 1, new_shown_index + 1):
                    config.change_landmark_object_index(shown_index_to_shift, shown_index_to_shift - 1)

            # We finally change the current landmark after having changed all other affected landmarks.
            config.change_landmark_object_index(self, new_shown_index)
            config.change_landmark_index(config.get_landmark_index_from_shown(current_shown_index),
                                         config.get_landmark_index_from_shown(new_shown_index))

    # Set currently_updating_lm_indices back to False at the end, even if something in code above fails.
    finally:
        config.currently_updating_lm_indices = False

# A simple item to hold the landmark objects for searching them
class LMSearchItem(bpy.types.PropertyGroup):
    # 'name' is what the search field displays
    # 'description' is what shows up when you hover an item in the list
    description: bpy.props.StringProperty()


# Show all available landmarks in the search list
def update_lm_search_list(scene):
    scene.my_lm_search_items.clear()

    # Get objects and sort them naturally
    def natural_key(string_):
        return [int(s) if s.isdigit() else s.lower() for s in re.split('([0-9]+)', string_)]
    sorted_objs = sorted(config.default_collection.objects, key=lambda o: natural_key(o.name))

    # Here we add all landmarks (sorted by number) to the dropdown list
    for obj in sorted_objs:
        if config.is_landmark_object(obj):
            item = scene.my_lm_search_items.add()
            item.name = obj.name
            item.description = f"Select Landmark {obj.name}"

#
#
# Next come the panels that have buttons, sliders, etc. and thus make use of the methods above, and operators
# defined further below
#
#

class LANDMARKING_PT_mesh_panel(bpy.types.Panel):
    """
    This is the panel that contains the mesh view options. If there are multiple meshes given, this panel includes
    a row that lets the user choose for each view which mesh to show in that view. Additionally, there are three rows
    that let the user slice the mesh in x-, y-, and z-direction from one and from the other side.
    """
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = "Landmarking"
    bl_label = "Mesh View Options"

    def get_status_icon(self, status):
        return "PAUSE" if 'inactive' in status.lower() else "MOUSE_RMB"

    def draw(self, context):
        layout = self.layout
        layout.use_property_decorate = False

        col = layout.column()
        col.scale_y = 2

        # Buttons to switch between the different meshes (performed per view)
        if len(config.input_mesh_names) > 1:
            row = col.row(align=True)
            row.label(icon="MESH_MONKEY")
            for num_mesh, mesh_name in enumerate(config.input_mesh_names):
                op = row.operator(f"landmarking.show_meshes", text=mesh_name)
                op.index = num_mesh
                op.tooltip = f"Show mesh {mesh_name} in the current window."

        # Buttons to slice the mesh from different sides and along the different axes (done to all meshes in all views)
        row = col.row(align=True)
        row.label(icon="AXIS_SIDE")
        row.prop(context.scene, 'x_plus', slider=True)
        row.prop(context.scene, 'x_minus', slider=True)

        row = col.row(align=True)
        row.label(icon="AXIS_FRONT")
        row.prop(context.scene, 'y_plus', slider=True)
        row.prop(context.scene, 'y_minus', slider=True)

        row = col.row(align=True)
        row.label(icon="AXIS_TOP")
        row.prop(context.scene, 'z_plus', slider=True)
        row.prop(context.scene, 'z_minus', slider=True)

class LANDMARKING_PT_segment_panel(bpy.types.Panel):
    """
    This is the segmentation panel, where we have geometry and texture selection options.
    """
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = "Landmarking"
    bl_label = "Segment Exclude Regions"

    def get_status_icon(self, status):
        return "PAUSE" if 'inactive' in status.lower() else "MOUSE_RMB"

    def draw(self, context):
        layout = self.layout
        layout.use_property_decorate = False

        col = layout.column()
        col.scale_y = 2

        # Buttons to segment regions to exclude. We offer two separate segmentations:
        # One for geometry (can also be triggered by pressing "G" key)...
        row = col.row(align=True)
        row.label(icon="EDITMODE_HLT")
        row.operator('landmarking.exclude_geometry', text="Exclude Geometry", )

        # ...one for texture (can also be triggered by pressing "T" key).
        row = col.row(align=True)
        row.label(icon="TEXTURE_DATA")
        row.operator('landmarking.exclude_texture', text="Exclude Texture", )


class LANDMARKING_PT_landmark_panel(bpy.types.Panel):
    """
    This is the most important panel. It includes all the options for the actual landmarking of the mesh(es).
    It includes options to set one landmark after another, edit the landmarks that were already set, hide/show
    the current landmarks, change the landmark size, delete all landmarks, and export the set landmarks.
    Additionally, in case a specific landmark is selected, further options are offered to change the landmark's
    position freely in x-, y-, z-direction, change its index, or delete it.
    """
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = "Landmarking"
    bl_label = "Landmarking"

    def get_status_icon(self, status):
        return "PAUSE" if 'inactive' in status.lower() else "MOUSE_RMB"

    def draw(self, context):
        layout = self.layout
        layout.use_property_decorate = False

        col = layout.column()
        col.scale_y = 2

        # Buttons to adjust the position of a selected landmark along x, y, and z, and for reattaching to mesh surface
        if is_single_lm_selected():
            box = col.box()
            row = box.row(align=True)
            row.label(text="Selected Landmark")

            lm_obj = blender_utils.get_selected_objects(obj_type="MESH")[0]
            lm_props = lm_obj.lm_pos_properties

            row = box.row(align=True)
            row.label(icon='EMPTY_ARROWS')
            for coord_dir in ["x", "y", "z"]:
                op = row.operator("wm.adjust_slider", text=f"{coord_dir.upper()}: {getattr(lm_props, f'lm_{coord_dir}'):.2f}")
                op.update_func_kind = f'lm_{coord_dir}'
                op.tooltip = (f"Adjust landmark position freely in {coord_dir}-direction by pressing this button "
                              f"and then moving the mouse up and down."
                              f"\nPress left or right mouse button or Esc to exit.")

            row.operator("landmarking.reattach_landmark", text="R", icon="LOOP_BACK")

            # --- draw the slider ---
            row = box.row(align=True)
            # The slider's internal min/max are static (1..1_000_000),
            # but logically we'll only ever keep it in [1..dyn_max]
            row.label(icon="FORCE_CHARGE")
            row.prop(lm_obj, "landmark_shown_index", text="Index")



            # Second row (additional button, same box)
            row = box.row(align=True)
            row.label(icon="X")
            row.operator("landmarking.delete_landmark", text="Delete Landmark")

        # Button to set landmarks, one after another
        row = col.row(align=True)
        row.label(icon="RESTRICT_SELECT_OFF")
        row.operator('landmarking.infinite_landmarks', text="Set Landmarks")

        # Button to edit the landmarks that have already been set
        row = col.row(align=True)
        row.label(icon='GREASEPENCIL')
        row.operator('landmarking.edit_landmarks', text="Edit Landmarks")

        # Search button to find a specific landmark.
        # This is especially useful if there are a lot of landmarks, or just a lot very close together.
        row = col.row(align=True)
        row.label(icon='OBJECT_DATA')
        row.prop_search(context.scene, "selected_lm_name",  context.scene, "my_lm_search_items", text="", icon='NONE')


        # Button to hide/show landmarks.
        # While landmarks are hidden, new landmarks can still be set; they will be shown at first, but can
        # still be hidden later on.
        row = col.row(align=True)
        row.label(icon='HIDE_ON' if config.landmarks_visible else "HIDE_OFF",)
        row.operator('landmarking.hide_landmarks',
                     text="Hide Landmarks" if config.landmarks_visible else "Show Landmarks")

        # Change the scaling of the landmarks. Their initial radius is defined relative to the mesh size,
        # but the user can still make them smaller or larger.
        row = col.row(align=True)
        row.label(icon="FIXED_SIZE")
        row.prop(context.scene, 'lm_scale', slider=True)

        # Button to delete all landmarks.
        row = col.row(align=True)
        row.label(icon="TRASH")
        row.operator('landmarking.delete_landmarks', text="Delete All Landmarks")

        # Button to export landmarks. If they are incomplete, the user is asked if they really want to export.
        # Blender is closed after this button is pressed.
        row = col.row(align=True)
        row.label(icon="FILE")
        row.operator('landmarking.export_landmarks', text="Export Landmarks")

#
#
# Next we have the operators, these are used for event triggering, as well as buttons
# and sliders used in the panels above.
#
#


class VIEW3D_OT_sync_viewports(bpy.types.Operator):
    """
    This is an operator that is continuously run. Its modal method is called whenever there's an event.
    We have the timer event, which periodically calls this operator to sync windows while view syncing is active.
    But we also have any other event triggering this method, in which case we check if the event is any
    type of our own trigger events (meaning some key was pressed that we want to use), in which case we
    trigger this specific event, such as setting view syncing on/off, or placing landmarks.
    In any other case, we let the event pass through to allow any other Blender event to still function normally.
    """
    bl_idname = "view3d.sync_viewports"
    bl_label = "Synchronize Viewports"
    bl_description = "Synchronize the rotation of all 3D viewports based on the active one"

    _timer = None
    def modal(self, context, event):
        if event.type == 'TIMER' and config.sync_active:
            # Find the 3D view under the mouse
            active_area = find_active_area(event)
            active_area_spaces = find_active_area_view_spaces(active_area=active_area)

            # If we found a 3D view under the mouse, get its properties
            for space in active_area_spaces:
                active_rot = space.region_3d.view_rotation.copy()
                active_distance = space.region_3d.view_distance
                active_location = space.region_3d.view_location.copy()

                # Apply these properties to all other 3D views
                for area in bpy.context.screen.areas:
                    if area.type == 'VIEW_3D' and area != active_area:
                        for other_space in area.spaces:
                            if other_space.type == 'VIEW_3D':
                                other_space.region_3d.view_rotation = active_rot
                                other_space.region_3d.view_distance = active_distance
                                other_space.region_3d.view_location = active_location

        elif config.trigger_key_pressed(event):
            # Check for the "L" key press
            if config.sync_key_pressed(event):
                config.sync_active = not config.sync_active  # Toggle the synchronization state
                self.report({'INFO'}, f"Synchronization {'ON' if config.sync_active else 'OFF'}")
                # We avoid the default behavior of the sync keys from being triggered with this return statement
                # If we returned pass through, then the behavior would still be triggered.
            elif config.mark_key_pressed(event):
                config.invoke_mark_event(event)
            elif config.set_landmarks_key_pressed(event):
                config.invoke_set_landmarks(event)
            elif config.edit_landmarks_key_pressed(event):
                config.invoke_edit_landmarks(event)
            elif config.plus_key_pressed(event):
                context.scene.lm_scale *= 2
            elif config.minus_key_pressed(event):
                context.scene.lm_scale /= 2
            # Avoid default behavior from being triggered with this return here.
            return {"RUNNING_MODAL"}

        return {'PASS_THROUGH'}

    def execute(self, context):
        # Here we add the timer, which periodically calls the modal function of this operator
        wm = context.window_manager
        self._timer = wm.event_timer_add(0.1, window=context.window)
        wm.modal_handler_add(self)
        return {'RUNNING_MODAL'}

    def cancel(self, context):
        wm = context.window_manager
        wm.event_timer_remove(self._timer)


class MY_OT_adjust_slider(bpy.types.Operator):
    """
    This is the general slider used for updating either x-, y-, or z-position of a specific landmark.
    It's not a normal Blender slider, since it doesn't have a minimum or maximum, and it can be triggered
    via a simple press and then updates the respective position based on the mouse movement in y-direction
    with a sensitivity based on the mesh size.
    """
    bl_idname = "wm.adjust_slider"
    bl_label = "Adjust Slider"

    update_func_kind: bpy.props.StringProperty()
    initial_value: bpy.props.FloatProperty()

    # Tooltip is the variable that is returned by the description function.
    # It can be specified for individual instances of this slider operator.
    tooltip: bpy.props.StringProperty()

    @classmethod
    def description(cls, context, operator) -> str:
        return operator.tooltip

    def get_lm_prop(self):
        if "x" in self.update_func_kind:
            prop_name = "lm_x"
        elif "y" in self.update_func_kind:
            prop_name = "lm_y"
        elif "z" in self.update_func_kind:
            prop_name = "lm_z"
        else:
            raise TypeError(f"Unknown update func kind: {self.update_func_kind}")
        return prop_name



    def modal(self, context, event):
        if event.type == 'MOUSEMOVE':
            # The landmark radius serves as a sensitivity measure, since it's scaled with the mesh size,
            # and we additionally multiply by 0.01 just to have a reasonable scale of the sensitivity.
            new_value = self.initial_value + (event.mouse_y - self.mouse_y_initial) * config.landmark_radius * 0.01
            setattr(context.object.lm_pos_properties, self.get_lm_prop(), new_value)
        elif event.type in ['LEFTMOUSE', 'ESC', 'RIGHTMOUSE']:
            return {'FINISHED'}

        return {'RUNNING_MODAL'}

    def invoke(self, context, event):
        self.mouse_y_initial = event.mouse_y
        self.initial_value = getattr(context.object.lm_pos_properties, self.get_lm_prop())
        context.window_manager.modal_handler_add(self)
        return {'RUNNING_MODAL'}


class OBJECT_OT_show_meshes(bpy.types.Operator):
    """
    This is the operator to show a specific mesh in a specific view
    (it simply finds the current view from the trigger event and then calls the respective config function).
    """
    bl_idname = "landmarking.show_meshes"
    bl_label = "Show Meshes"

    # mesh index (there's a separate operator instance in the panel for each mesh)
    index: bpy.props.IntProperty()

    # Tooltip to be displayed for each mesh to be shown
    tooltip: bpy.props.StringProperty()

    @classmethod
    def description(cls, context, operator) -> str:
        return operator.tooltip

    def invoke(self, context, event):
        view = find_active_area(event)
        if view is None:
            return {"Cancelled"}
        config.set_mesh_per_view(view=view, num_mesh=self.index)
        # Call your function with self.index here
        return {'FINISHED'}



class LANDMARKING_OT_reattach_landmark(Operator):
    """
    This operator is triggered via a button and concerns a specific landmark to be reattached to the mesh in the
    current view. We use the closest_point_on_mesh() function from Blender, which automatically excludes hidden
    regions that were sliced away by the user. Since this is dynamic, we don't build a hash tree around the mesh,
    but since we only do this reattachment very sparsely, the speed should be fine.
    """
    bl_idname = "landmarking.reattach_landmark"
    bl_label = "R"

    @classmethod
    def description(cls, context, operator) -> str:
        return "Reattach landmark to the scan (closest vertex)"

    def invoke(self, context, event):
        if not is_single_lm_selected():
            return {'FINISHED'}
        obj = blender_utils.get_selected_objects(obj_type="MESH")[0]
        obj_position = np.asarray(obj.location)
        num_mesh = get_current_num_mesh(event)
        if num_mesh is None:
            raise AssertionError("Reattach landmark was pressed, but no mesh could be found in the view. Check...")
        _, closest_mesh_pos, _, _ = config.get_mesh_obj(num_mesh).closest_point_on_mesh(obj_position)
        obj.location = list(closest_mesh_pos)
        lm_index = config.get_landmark_index(obj)
        config.adjust_landmark(lm_index, closest_mesh_pos)
        assign_landmark_material(obj, color=config.get_landmark_color(lm_index))
        return {'FINISHED'}

class LANDMARKING_OT_delete_landmark(Operator):
    """
    This operator is used as a button to delete a single landmark. It's only used in the panel while a single
    landmark is selected, so we assume here that it concerns the first selected Blender object.
    Besides deleting the Blender object, we adjust all shown indices (names and attributes) of the landmarks
    that come after the deleted landmark. Additionally, we make the same adjustments to the internal config landmarks
    attribute.
    """
    bl_idname = "landmarking.delete_landmark"
    bl_label = "Delete"

    @classmethod
    def description(cls, context, operator) -> str:
        return ("Delete the current landmark.\n"
                "You cannot revert this, but to fix accidental deletion you can set a new landmark, "
                "then select it, and change its index to this one.")

    def invoke(self, context, event):
        if not is_single_lm_selected():
            return {'FINISHED'}
        lm_obj_to_delete = blender_utils.get_selected_objects(obj_type="MESH")[0]
        lm_index = config.get_landmark_index(lm_obj_to_delete)
        blender_utils.remove_objects(lm_obj_to_delete)
        for old_index in range(lm_index+1, config.landmark_counter):
            shown_index = config.get_landmark_shown_index(old_index)
            config.change_landmark_object_index(landmark_object_or_shown_index=shown_index, shown_index_to_assign=shown_index-1)
        config.delete_landmark(lm_index)
        return {'FINISHED'}

class LANDMARKING_OT_delete_landmarks(Operator):
    """
    This operator is used as a button to delete all landmarks currently set.
    Since this is a dangerous operation, we ask the user if they're sure before
    executing.
    """
    bl_idname = "landmarking.delete_landmarks"
    bl_label = "Delete_all"

    @classmethod
    def description(cls, context, operator) -> str:
        return "Delete all landmarks. This cannot be reverted"

    def execute(self, context):
        # This runs only after user confirmed (or if called directly via EXEC).
        for lm_index in range(config.landmark_counter):
            blender_utils.remove_objects(config.get_landmark_object(lm_index))
        config.delete_landmarks()
        return {'FINISHED'}

    def invoke(self, context, event):
        # Ask first, and if confirmed, call THIS operator again, but its execute method.
        bpy.ops.wm.generic_yesno(
            'INVOKE_DEFAULT',
            message="Delete all landmarks? This cannot be undone.",
            yes_operator=type(self).bl_idname,
            yes_context='EXEC_DEFAULT',  # call execute() directly after confirm
        )
        return {'FINISHED'}


class LANDMARKING_OT_edit_mode(Operator, ExportHelper):
    """
    This is a superclass operator used for exporting the different regions the user selected.
    Mainly, we have two inheriting classes further below for exporting the geometry and texture
    selections. This superclass operator has an invoke_super method, which is called once one of
    the subclass' invoke method is called by pressing the respective button in the panel (or "G" or "T" key).
    The invoke_super method checks which mesh is concerned based on the Blender event, and if the mesh
    is currently in object mode (EditMode.None), it's switched to edit mode, the respective index selection
    is assigned to the mesh, and the config.edit_modes variable is adjusted accordingly. If the mesh
    is already in edit mode, we switch back to object mode and save the user selection, unless the
    EditMode of the mesh does not match with the subclass that triggered the event, then we warn the user.
    E.g., if the mesh is currently in EditMode.exclude_texture, but the user presses the Exclude Geometry
    button, the warning is raised that the user should first finish the texture selection.
    If no path has been selected yet for exporting the region selection, a file window is opened.
    The actual file export is handled in the execute method.
    """
    bl_options = {'PRESET', 'UNDO'}

    filename_ext = '.txt'

    filter_glob: StringProperty(
        default='*.txt',
        options={"HIDDEN"}
    )
    marked_indices = None
    marked_points = None
    num_mesh = None
    region_type = None

    def invoke_super(self, context, event, region_type: EditMode):
        self.num_mesh = get_current_num_mesh(event)
        self.region_type = region_type
        if self.num_mesh is None:
            self.report({'WARNING'}, f"Couldn't find corresponding mesh...")
            return {'CANCELLED'}
        if config.is_input_mesh_in_object_mode(num_mesh=self.num_mesh):
            config.set_to_region_edit_mode(self.region_type, num_mesh=self.num_mesh)
            return {'FINISHED'}
        if config.is_in_other_edit_mode(self.region_type, num_mesh=self.num_mesh):
            self.report({'WARNING'}, f"You're currently in {config.edit_modes[self.num_mesh]} mode. Finish that first.")
            return {'CANCELLED'}
        self.marked_indices = config.get_current_indices(num_mesh=self.num_mesh)
        if IndicesAndMasks.get_num_indexed(self.marked_indices) == 0:
            self.report({'WARNING'}, f"Please select a region first.")
            return {'CANCELLED'}
        self.marked_points = config.meshes[self.num_mesh].get_vertices(copy=False)[self.marked_indices]
        config.set_indices_region(self.region_type, num_mesh=self.num_mesh, indices=self.marked_indices)
        config.set_input_mesh_to_obj_mode(num_mesh=self.num_mesh)
        if self.num_mesh in config.export_region_paths and config.export_region_paths[self.num_mesh] is not None:
            self.filepath = config.get_region_path(self.region_type, num_mesh=self.num_mesh)
            return self.execute(context)
        else:
            print(f"Please choose file path for mesh {self.num_mesh} with region {self.region_type}")
        context.window_manager.fileselect_add(self)
        return {'RUNNING_MODAL'}

    def execute(self, context):
        export_region_path = Path(self.filepath).with_suffix(self.filename_ext)
        if self.num_mesh not in config.export_region_paths or config.export_region_paths[self.num_mesh] is None:
            if self.region_type == EditMode.EXCLUDE_GEOMETRY:
                geometry_region_path = export_region_path
            elif self.region_type == EditMode.EXCLUDE_TEXTURE:
                if export_region_path.stem.endswith("_texture"):
                    geometry_region_path = export_region_path.with_stem(export_region_path.stem.replace("_texture", ""))
                else:
                    geometry_region_path = None
            else:
                raise AssertionError(f"region type {self.region_type} is neither geometry nor texture.")
            if geometry_region_path is None:
                warnings.warn("Please use proper naming convention for texture exclusion and geom inclusion files.")
            else:
                config.set_region_path(num_mesh=self.num_mesh, export_region_path=geometry_region_path)
        # Export selected vertices as index array
        IndicesAndMasks.export(self.marked_indices, export_region_path)
        # Also export selected vertices as points as a safety measure
        Landmarks(self.marked_points).export(export_region_path.with_name(f"{export_region_path.stem}points.csv"))
        return {"FINISHED"}


class LANDMARKING_OT_exclude_geometry(LANDMARKING_OT_edit_mode):
    """
    This operator inherits from LANDMARKING_OT_edit_mode. It handles the region selection for the geometry
    to exclude.
    """

    bl_idname = f"landmarking.{EditMode.EXCLUDE_GEOMETRY.value}"
    bl_label = "Export Selection"

    @classmethod
    def description(cls, context, operator) -> str:
        mark_key = [key for key, value in config.mark_keys.items() if value == EditMode.EXCLUDE_GEOMETRY][0]
        return ("Mark and export selected geometry region to exclude\nYou can press the \"C\" "
                "key on your keyboard while you're in this mode to select with Blender's circle tool\n"
                "Press this button again to exit the mode.\nYou can also activate/deactivate "
                f"this mode by pressing the \"{mark_key}\" key on your keyboard.")

    def invoke(self, context, event):
            return self.invoke_super(context, event, EditMode.EXCLUDE_GEOMETRY)



class LANDMARKING_OT_exclude_texture(LANDMARKING_OT_edit_mode):
    """
    This operator inherits from LANDMARKING_OT_edit_mode. It handles the region selection for the texture
    to exclude.
    """

    bl_idname = f"landmarking.{EditMode.EXCLUDE_TEXTURE.value}"
    bl_label = "Export Selection"

    @classmethod
    def description(cls, context, operator) -> str:
        mark_key = [key for key, value in config.mark_keys.items() if value == EditMode.EXCLUDE_TEXTURE][0]
        return ("Mark and export selected texture region to exclude\nYou can press the \"C\" "
                "key on your keyboard while you're in this mode to select with Blender's circle tool\n"
                "Press this button again to exit the mode.\nYou can also activate/deactivate "
                f"this mode by pressing the \"{mark_key}\" key on your keyboard.")

    def invoke(self, context, event):
        return self.invoke_super(context, event, EditMode.EXCLUDE_TEXTURE)


class LANDMARKING_OT_infinite_landmarks(Operator):
    """
    This is the operator that is included as a button in the landmarking panel. It's called "infinite",
    because you set one landmark after another. The invoke method handles what happens after the button
    has been pressed; we enter the landmarking mode which blocks most Blender events except the ones
    we pass through. The modal method is continuously run after invoke that until the user decides to exit.
    We track the mouse movement and project the sphere onto the mesh in the mouse's respective view.
    If the user left-clicks, they confirm the landmark position. If they press "N", it means they position
    the Blender object at the current location, but make the underlying landmark position undefined.
    """
    bl_idname = "landmarking.infinite_landmarks"
    bl_label = "Infinite Landmarks"

    @classmethod
    def description(cls, context, operator) -> str:
        return ("Place landmarks on selected mesh, one after another.\nPress \"N\" while positioning a landmark to "
                f"make it undefined.\nRight click on mouse or press Esc to exit tool.\nYou can also activate/"
                f"deactivate this mode by pressing the \"{config.set_landmarks_key}\" key on your keyboard.")

    # Properties
    hit_result: BoolProperty()
    location: (FloatProperty(), FloatProperty(), FloatProperty())

    def modal(self, context, event):
        if config.num_landmarks is not None and config.landmark_counter >= config.num_landmarks:
            blender_utils.hide_object(config.MOUSE_SPHERE)
            return {'FINISHED'}
        num_mesh = get_current_num_mesh(event)

        # Handling of mouse movement
        if event.type == 'MOUSEMOVE':
            # perform ray cast from mouse onto Mesh
            if num_mesh is not None:
                self.hit_result, hit_loc = mouse_location_on_mesh(context, event, config.get_mesh_obj(num_mesh))
                self.location = hit_loc
            else:
                self.hit_result, hit_loc = False, None
            # Mouse sphere tracking on Mesh
            if self.hit_result:
                config.MOUSE_SPHERE.location = hit_loc
                blender_utils.show_object(config.MOUSE_SPHERE)
            else:
                blender_utils.hide_object(config.MOUSE_SPHERE)

        # Location is approved bc mouse left click or by pressing N or Enter.
        # N means that the location is nan, whereas the left mouse click and enter pressing are equivalent.
        elif (event.type == 'LEFTMOUSE' or event.type == "N" or event.type == "RET") \
                and event.value == 'PRESS' and self.hit_result:
            if event.type == "N":
                landmark_color = config.landmark_undefined_color
                landmark_confidence = 0
            else:
                landmark_color = config.get_landmark_color()
                landmark_confidence = 1

            config.add_landmark(self.location, confidence=landmark_confidence)
            place_landmark(location=self.location,
                           color=landmark_color, index=config.landmark_counter-1)

            if config.save_continuously and config.export_landmarks_path is not None:
                LANDMARKING_OT_export_landmarks.do_export(config.export_landmarks_path)
            config.select_input_mesh(num_mesh)

        # allow to rotate camera during landmark setting
        elif config.pass_event(event):
            return {'PASS_THROUGH'}

        # exit condition
        elif event.type in {'RIGHTMOUSE', 'ESC'} or (event.type == 'LEFTMOUSE' and event.value == 'PRESS'
                                                     and not self.hit_result):
            blender_utils.hide_object(config.MOUSE_SPHERE)
            context.scene.landmarking_tool_status = "Landmarking tool inactive."
            return {'FINISHED'}

        return {'RUNNING_MODAL'}

    def invoke(self, context, event):
        num_mesh = get_current_num_mesh(event)
        if config.is_in_any_edit_mode(num_mesh):
            self.report({'WARNING'}, f"Please finish the {config.edit_modes[num_mesh].get_value(num_mesh)} "
                                     f"marking first.")
            return {'CANCELLED'}
        if context.space_data.type == 'VIEW_3D':
            if config.num_landmarks is not None and config.landmark_counter >= config.num_landmarks:
                self.report({'WARNING'}, f"All {config.num_landmarks} already defined. No more landmarks required.")
                return {'CANCELLED'}

            # initialize properties
            self.hit_result = False
            # Here we create the mouse sphere object.
            # It's later repositioned.
            config.MOUSE_SPHERE = get_mouse_landmark()

            context.window_manager.modal_handler_add(self)
            context.scene.landmarking_tool_status = "Tool active. Right click to exit."
            return {'RUNNING_MODAL'}
        else:
            self.report({'WARNING'}, "No active object, could not finish")
            return {'CANCELLED'}


class LANDMARKING_OT_edit_landmarks(Operator):
    """
    This operator is also activated via a button in the panel. It's similar to the infinite_landmarks operator
    in that it blocks most other Blender operations while it's active, and it has an invoke and modal method
    with similar functionalities. The invoke method is called once the button is pressed (the first time),
    and it enters the blocking mode. The modal method then runs. It tracks the mouse movement, but only
    moves a landmark, when it's been pressed on first and only as long as the left mouse keeps being pressed,
    so it's a drag and drop action.
    """

    bl_idname = "landmarking.edit_landmarks"
    bl_label = "Edit Landmarks"

    @classmethod
    def description(cls, context, operator) -> str:
        return ("Edit landmarks via drag and drop.\nPress \"N\" while dragging to make the landmark undefined.\n"
                f"Right click on mouse or press Esc to exit.\nYou can also activate/deactivate this mode by "
                f"pressing the \"{config.edit_landmarks_key}\" on your keyboard.")

    # properties
    hit_result: BoolProperty()
    location: (FloatProperty(), FloatProperty(), FloatProperty())
    current_lm, current_lm_index = None, None
    currently_moving = BoolProperty()

    def modal(self, context, event):
        num_mesh = get_current_num_mesh(event)
        if event.type == 'MOUSEMOVE' or (event.type == "N" and event.value == "PRESS"):
            if num_mesh is not None:
                # Perform ray cast from mouse onto mesh
                self.hit_result, hit_loc = mouse_location_on_mesh(context, event, config.get_mesh_obj(num_mesh))
                self.location = hit_loc
            else:
                self.hit_result = False

            # Only move landmark if it's been pressed on first and only as long as mouse button is still pressed.
            if self.currently_moving and self.current_lm.select_get():
                self.current_lm.location = self.location
                if event.type == "N":
                    assign_landmark_material(self.current_lm, color=config.landmark_undefined_color)
                    config.adjust_landmark(self.current_lm_index, location=self.location, confidence=0)
                    self.currently_moving = False
                    bpy.ops.object.select_all(action='DESELECT')
                else:
                    config.adjust_landmark(self.current_lm_index, location=self.location)

        # If we press the left mouse button (before releasing), check if a landmark is close by and only
        # then do we enter the dragging mode.
        elif event.type == 'LEFTMOUSE' and event.value == 'PRESS' and self.hit_result:
            bpy.ops.object.select_all(action='DESELECT')
            closest_landmark = get_closest_landmark(self.location)
            if closest_landmark is not None:
                self.current_lm = closest_landmark
                self.current_lm_index = config.get_landmark_index(self.current_lm)
                self.current_lm.select_set(True)
                self.currently_moving = True
                assign_landmark_material(self.current_lm, color=config.get_landmark_color(self.current_lm_index))

        # Once mouse button is released, exit the dragging mode.
        elif event.type == 'LEFTMOUSE' and event.value == 'RELEASE' and self.hit_result:
            self.currently_moving = False
            if config.save_continuously and config.export_landmarks_path is not None:
                LANDMARKING_OT_export_landmarks.do_export(config.export_landmarks_path)
            blender_utils.deselect_all_objects()
            config.select_input_mesh(num_mesh)

        # allow to rotate camera during landmark setting
        elif config.pass_event(event):
            return {'PASS_THROUGH'}

        # exit condition
        elif event.type in {'RIGHTMOUSE', 'ESC'} or (
                event.type == 'LEFTMOUSE' and event.value == 'PRESS' and not self.hit_result):
            context.scene.landmarking_tool_status = "Landmarking tool inactive."
            return {'FINISHED'}

        return {'RUNNING_MODAL'}

    def invoke(self, context, event):
        num_mesh = get_current_num_mesh(event)
        if config.is_in_any_edit_mode(num_mesh):
            self.report({'WARNING'}, f"Please finish the {config.edit_modes[num_mesh].get_value(num_mesh)} "
                                     f"marking first.")
            return {'CANCELLED'}

        if context.space_data.type == 'VIEW_3D':
            # initialize properties
            self.hit_result = False
            self.currently_moving = False

            # Make mouse sphere invisible
            if config.MOUSE_SPHERE is not None:
                blender_utils.hide_object(config.MOUSE_SPHERE)

            context.window_manager.modal_handler_add(self)
            context.scene.landmarking_tool_status = "Tool active. Right click to exit."
            return {'RUNNING_MODAL'}
        else:
            self.report({'WARNING'}, "No active object, could not finish")
            return {'CANCELLED'}


class LANDMARKING_OT_hide_landmarks(Operator):
    """
    Simple operator that's used as a button in the panel to hide or show again the currently set landmarks.
    It simply calls the config function switch_landmarks_visible(), nothing else. In the panel, we switch
    the button text and label based on whether the landmarks are currently shown or hidden.
    """

    bl_idname = "landmarking.hide_landmarks"
    bl_label = "Toggle Landmarks' Visibility"

    @classmethod
    def description(cls, context, operator) -> str:
        return ("Hide/show landmarks.\nNote that you can still set new landmarks while in hide mode, in which "
                "case the new ones are still shown. Press Show and then Hide again to also hide the new ones")

    def invoke(self, context, event):
        config.switch_landmarks_visible()
        return {'FINISHED'}


class LANDMARKING_OT_export_landmarks(Operator, ExportHelper):
    """
    Landmark export operator, handled as a button. If self.num_landmarks was set by the user, but
    they haven't defined all landmarks yet, a popup window asks the user whether they really want to
    export. We also save the current index selection in case the user presses the button while
    in edit mode. Blender is quit after the files have been exported in case the user set
    quit_blender_after_landmark_export True.
    """

    bl_idname = "landmarking.export_landmarks"
    bl_label = "Export Landmarks"
    bl_options = {'PRESET', 'UNDO'}

    filename_ext = '.csv'

    filter_glob: StringProperty(
        default='*.csv',
        options={"HIDDEN"}
    )

    @classmethod
    def description(cls, context, operator) -> str:
        return ("Export all landmarks as a csv-file (3D locations written on each line, "
                "line number corresponds to landmark index)")

    @staticmethod
    def do_export(export_path=None):
        export_path = config.export_landmarks_path if export_path is None else export_path
        if export_path is not None:
            config.landmarks.export(export_path)
        for num_mesh in range(len(config.meshes)):
            if config.is_in_any_edit_mode(num_mesh=num_mesh):
                getattr(bpy.ops.landmarking, f"{config.edit_modes[num_mesh].get_value(num_mesh)}")("INVOKE_DEFAULT")
        if config.quit_blender_after_landmark_export:
            blender_utils.quit_blender()
        return {'FINISHED'}

    def invoke(self, context, event):
        if not config.allow_incomplete_lm:
            if config.num_landmarks is not None and config.landmark_counter < config.num_landmarks:
                self.report({'WARNING'}, f"{config.num_landmarks} landmarks required for export. "
                                         f"{config.landmark_counter} landmarks currently specified.")
                bpy.ops.wm.generic_yesno(
                    'INVOKE_DEFAULT',
                    message=f"Only {config.landmark_counter} from {config.num_landmarks} landmarks currently specified. "
                            f"Still export?",
                    yes_operator=type(self).bl_idname,
                    # We call invoke again, but with config.allow_incomplete_lm set to True
                    yes_context='INVOKE_DEFAULT',
                    config_attr="allow_incomplete_lm",
                    config_value=True
                )
                return {'CANCELLED'}
        if config.export_landmarks_path is not None:
            return self.do_export()
        else:
            context.window_manager.fileselect_add(self)
            return {'RUNNING_MODAL'}

    def execute(self, context):
        config.export_landmarks_path = str(Path(self.filepath).with_suffix(".csv"))
        return self.do_export()


class WM_OT_generic_yesno(bpy.types.Operator):
    """
    This operator is used as a general confirmation panel that can be called by any other operator.
    We pass arguments to allow for calling functions (of operators) in case "Yes" has been pressed,
    and additionally to change config attributes if "Yes" has been pressed.
    """
    bl_idname = "wm.generic_yesno"
    bl_label = "Confirmation"
    bl_options = {'INTERNAL'}

    # what the popup shows
    message: bpy.props.StringProperty(
        name="Message",
        default="Are you sure?"
    )

    # what happens when pressing Yes
    yes_operator: bpy.props.StringProperty(
        name="Yes operator idname",
        default=""
    )

    # JSON dict of kwargs to pass into the operator call
    yes_kwargs: bpy.props.StringProperty(
        name="Yes kwargs (JSON)",
        default="{}"
    )

    # optional: set config.<attr> = <value> before calling yes_operator
    config_attr: bpy.props.StringProperty(
        name="Config attribute",
        default=""
    )
    config_value: bpy.props.BoolProperty(
        name="Config value",
        default=True
    )

    # how to run the operator (INVOKE_DEFAULT vs EXEC_DEFAULT)
    yes_context: bpy.props.EnumProperty(
        name="Yes call context",
        items=[
            ('INVOKE_DEFAULT', "Invoke", ""),
            ('EXEC_DEFAULT', "Execute", ""),
        ],
        default='INVOKE_DEFAULT'
    )

    # internal: which button was clicked
    confirm: bpy.props.BoolProperty(options={'HIDDEN'}, default=False)
    pressed: bpy.props.BoolProperty(options={'HIDDEN'}, default=False)

    def invoke(self, context, event):
        return context.window_manager.invoke_popup(self, width=420)

    def draw(self, context):
        layout = self.layout
        layout.label(text=self.message, icon='QUESTION')

        row = layout.row()
        row.scale_y = 1.5

        # IMPORTANT: ensure button presses go to execute(), not invoke()
        row.operator_context = 'EXEC_DEFAULT'

        # YES button: call THIS operator again with confirm=True
        op_yes = row.operator(self.bl_idname, text="Yes", icon='CHECKMARK')
        op_yes.message = self.message
        op_yes.yes_operator = self.yes_operator
        op_yes.yes_kwargs = self.yes_kwargs
        op_yes.config_attr = self.config_attr
        op_yes.config_value = self.config_value
        op_yes.yes_context = self.yes_context
        op_yes.confirm = True
        op_yes.pressed = True

        # NO button: call THIS operator again with confirm=False
        op_no = row.operator(self.bl_idname, text="No", icon='CANCEL')
        op_no.message = self.message
        op_no.confirm = False
        op_no.pressed = True

    def execute(self, context):
        # First call is just to show popup; second call is from button press.
        if not self.pressed:
            return {'FINISHED'}

        # Close the popup now (before running follow-up ops is fine too)
        wm = context.window_manager
        try:
            wm.popup_menu_end()
        except (AttributeError, RuntimeError):
            try:
                wm.popmenu_end()
            except (AttributeError, RuntimeError):
                pass

        if self.confirm:
            # optional config tweak
            if self.config_attr:
                setattr(config, self.config_attr, self.config_value)

            # call target operator
            if self.yes_operator:
                kwargs = json.loads(self.yes_kwargs or "{}")
                if not isinstance(kwargs, dict):
                    raise ValueError("yes_kwargs must be a JSON object/dict")
                call_op(self.yes_operator, self.yes_context, **kwargs)

        return {'FINISHED'}


# =================================================
# CLASSES TO REGISTER / UNREGISTER
# -------------------------------------------------



classes = [LANDMARKING_PT_mesh_panel, LANDMARKING_PT_segment_panel, LANDMARKING_PT_landmark_panel,
           LANDMARKING_OT_hide_landmarks, OBJECT_OT_show_meshes,  #VIEW3D_PT_dynamic_slider,
           LANDMARKING_OT_infinite_landmarks, LANDMARKING_OT_export_landmarks, LANDMARKING_OT_edit_landmarks,
           LANDMARKING_OT_exclude_geometry, LANDMARKING_OT_exclude_texture,
           VIEW3D_OT_sync_viewports,
           LMPosProperties, MY_OT_adjust_slider, LANDMARKING_OT_reattach_landmark, LANDMARKING_OT_delete_landmark,
           LANDMARKING_OT_delete_landmarks, WM_OT_generic_yesno, LMSearchItem]


def register():
    """
    We register all relevant classes, but also define the relevant callbacks.
    """

    def hide_layers(self, context, axis: str, plus_or_minus: str):
        """
        Callback function that's called whenever any of the axis slicing sliders is used.
        It defines which vertices to show and which to hide for all meshes by checking which side
        on the slicing planes they're currently on. Although this function is called for a specific plane,
        the actual vertex showing/hiding is done for all planes. We just update the selection of vertices
        to hide based on the current plane, but the masking uses all combined selections.
        We also handle special cases of slicing while the objects are in edit mode.
        They're supposed to keep the vertex selections for regions to exclude, which Blender
        normally wouldn't keep, by keeping track of the selections via the vertex groups.
        """
        plus_or_minus = "plus" if "p" in plus_or_minus.lower() or "+" in plus_or_minus.lower() else "minus"
        cutoff = getattr(self, f"{axis}_{plus_or_minus}")
        plane_normal = np.zeros(3)
        plane_normal[["x", "y", "z"].index(axis.lower())] = 1
        if plus_or_minus == "minus":
            plane_normal *= -1
        # (the plane anchor is defined dynamically based on mesh in the loop further below)
        hide_layer_plane = utils.Plane([0, 0, 0], plane_normal)

        # We temporally show all meshes in all views, such that the index selection works on all meshes in all views
        if len(config.meshes) > 1:
            config.show_all_meshes_in_all_views()
        selected_objects = blender_utils.get_selected_objects()
        active_obj = blender_utils.get_active_obj()
        for num_mesh, (mesh, mesh_center, mesh_name) in enumerate(zip(config.meshes, config.mesh_centers, config.input_mesh_names)):
            vertex_group_name = f"mesh{num_mesh}_hide_{axis.lower()}_{plus_or_minus}_layer"
            modifier_name = f"mesh{num_mesh}_mask_layer_{axis.lower()}_{plus_or_minus}"
            blender_utils.select_object_by_name(mesh_name)
            blender_obj = blender_utils.get_obj_by_name(mesh_name)
            initial_mode = blender_utils.get_mode(blender_obj)
            initial_vertex_selection = config.get_current_indices(num_mesh)
            if config.is_in_any_edit_mode(num_mesh):
                blender_utils.select_vertices(
                    initial_vertex_selection, config.input_mesh_names[num_mesh],
                    vertex_group_name=config.edit_modes[num_mesh].get_value(num_mesh))

            # First make adjustments in object mode by using a mask modifier
            mesh_vertices = mesh.get_vertices(copy=False)
            hide_layer_plane.set_point(mesh_center)
            vertex_plane_distances = hide_layer_plane.compute_distance_points_to_plane(mesh_vertices, compute_absolute=False)
            min_dist, max_dist = np.min(vertex_plane_distances), np.max(vertex_plane_distances)
            vertex_plane_distances_normed = (vertex_plane_distances - min_dist) / (max_dist - min_dist)
            vertices_to_show = vertex_plane_distances_normed < cutoff + 1e-5
            blender_utils.select_vertices(vertices_to_show, blender_obj, vertex_group_name=vertex_group_name)
            if modifier_name not in blender_obj.modifiers:
                mask_modifier = blender_obj.modifiers.new(name=modifier_name, type='MASK')
                mask_modifier.vertex_group = vertex_group_name


            # then do hiding also in edit mode
            all_vertex_group_names = [f"mesh{num_mesh}_hide_{axis_local}_{sign_local}_layer" for axis_local in
                                      ["x", "y", "z"] for sign_local in ["plus", "minus"]]
            all_vertices_to_show_mask = blender_utils.get_vertex_selection_from_groups(
                blender_obj, vertex_group_names=all_vertex_group_names, intersect=True, return_as_indices=False)
            blender_utils.set_to_edit_mode(mesh_name)
            blender_mesh_data = blender_utils.get_obj_data(blender_obj)
            bm = bmesh.from_edit_mesh(blender_mesh_data)

            for v in bm.verts:
                v.hide_set(not all_vertices_to_show_mask[v.index])
            blender_utils.select_vertices(initial_vertex_selection, mesh_name, vertex_group_name=None)

            blender_utils.set_mesh_to_mode(initial_mode, mesh_name)



        for selected_object in selected_objects:
            blender_utils.select_obj(selected_object, make_active=selected_object == active_obj)
        # Reset the previous state of meshes shown
        if len(config.meshes) > 1:
            config.reset_meshes_per_view()

    # Actual individual callbacks for slicing the meshes
    def make_hide_layers_callback(axis, sign):
        def _cb(self, context):
            hide_layers(self, context, axis=axis, plus_or_minus=sign)
        return _cb
    # We have six callbacks, so for each axis from front and from back.
    for hide_axis in ["x", "y", "z"]:
        for hide_sign in ["+", "-"]:
            cb = make_hide_layers_callback(hide_axis, hide_sign)
            setattr(
                bpy.types.Scene,
                f"{hide_axis}_{'plus' if hide_sign == '+' else 'minus'}",
                bpy.props.FloatProperty(
                    name=f"{hide_axis.upper()}{hide_sign} Slice",
                    description=f"Hide mesh layers/slice mesh along {hide_axis.upper()}-axis "
                                f"in {'positive' if hide_sign == '+' else 'negative'} direction (drag slider)",
                    min=0, max=1, update=cb, default=1))

    # Callback for changing the landmark scale.
    def change_lm_scale_callback(self, context):
        config.change_landmark_scale(self.lm_scale)
    bpy.types.Scene.lm_scale = bpy.props.FloatProperty(
        name="Landmark Size", description="Change size of landmark spheres",
        min=0, soft_max=5, update=change_lm_scale_callback, default=config.landmark_scale)

    # Register all classes
    for c in classes:
        bpy.utils.register_class(c)

    # lm_pos_properties, used to update the landmark positions via sliders
    bpy.types.Object.lm_pos_properties = bpy.props.PointerProperty(type=LMPosProperties)

    # The shown index of a landmark. We use this int property to allow the user to change the index,
    # thus triggering the calling of an update function that adjusts this and all affected landmarks.
    bpy.types.Object.landmark_shown_index = bpy.props.IntProperty(
        name="Landmark Index",
        description="Change the index of the currently selected landmark. The indices of the other landmarks are "
                    "shifted accordingly.",
        default=1, min=1, max=1000000,  # Assumed to be far, far larger than any maximum a user would ever choose
        update=landmark_index_update,
    )

    bpy.types.Scene.my_lm_search_items = bpy.props.CollectionProperty(type=LMSearchItem)

    # If another landmark was selected via the search property, select it via this method.
    def on_searched_lm_name_change(self, context):
        obj = bpy.data.objects.get(self.selected_lm_name)
        if obj is not None and config.is_landmark_object(obj):
            blender_utils.select_obj(obj, deselect_remaining=True)

    # Stores the current landmark selection, and calls the method above for any change
    bpy.types.Scene.selected_lm_name = bpy.props.StringProperty(
        name="Select Landmark",
        # This solves Issue #3: Pre-defined text when hovering the field
        description="Type the landmark index or select it from the dropdown list.",
        update=on_searched_lm_name_change)

    # Necessary for landmark position sliders as well.
    bpy.app.handlers.depsgraph_update_post.append(update_slider_values)
    # Updates the landmark dropdown list for any scene change
    # (so also if we add or remove landmarks, the list is updated automatically)
    bpy.app.handlers.depsgraph_update_post.append(update_lm_search_list)


def unregister():
    del bpy.types.Object.landmark_shown_index
    for c in reversed(classes):
        bpy.utils.unregister_class(c)
    bpy.app.handlers.depsgraph_update_post.remove(update_slider_values)
    bpy.app.handlers.depsgraph_update_post.remove(update_lm_search_list)


def main():
    register()
    argv = sys.argv
    if "--" not in argv:
        argv = []
    else:
        argv = argv[argv.index("--") + 1:]
    parser = ArgumentParser("You can run blender from CLI with this script: "
                            "blender [blend_file] --python " + __file__ + " -- [options]")
    parser.add_argument('--mesh_paths', type=str, required=True, nargs="+",
                        help="Path to input meshes to be landmarked.")
    parser.add_argument('--input_landmarks_path', type=str, default=None,
                        help="Optional path to existing landmarks.")
    parser.add_argument('--output_landmarks_path', type=str, default=None,
                        help="Path where landmarks should be exported. Will ask for path if not specified.")
    parser.add_argument('--output_region_paths', type=str, default=[], nargs="+",
                        help="Path where optionally marked region should be exported. "
                             "Will ask for path if not specified.")
    parser.add_argument('--save_continuously', default=False, action="store_true",
                        help="Save landmarks every time you make any change to the landamrks.")
    parser.add_argument('--num_landmarks', type=int, default=None)
    parser.add_argument('--num_window_splits', type=int, default=1,
                        help="How many times to split the 3D View window, so you'll end up with 2^num_window_splits "
                             "windows. Don't go too high. ;)")
    parser.add_argument("--quit_blender_after_landmark_export", default=False, action="store_true",
                        help="Choose this to quit the blender window immediately after exporting the landmarks.")
    args = parser.parse_args(argv)
    blender_utils.remove_all_objects()

    # Set some contex variables.
    bpy.types.Scene.landmarking_tool_status = bpy.props.StringProperty(
        name="Tool Active. Right Click to Exit", default="Tool inactive")

    config.set_landmark_colors(args.num_landmarks)
    config.save_continuously = args.save_continuously
    config.quit_blender_after_landmark_export = args.quit_blender_after_landmark_export

    config.load_meshes(*args.mesh_paths)

    if args.input_landmarks_path is not None:
        import_landmarks(args.input_landmarks_path)

    config.export_landmarks_path = args.output_landmarks_path
    for num_mesh, output_region_path in enumerate(args.output_region_paths):
        if num_mesh >= len(config.meshes):
            warnings.warn("More region output paths provided than meshes. Additional paths are ignored. "
                          "Check your input arguments.")
            break
        config.set_region_path(num_mesh=num_mesh, export_region_path=output_region_path)

    for num_mesh in range(len(config.meshes)):
        for region_type in [EditMode.EXCLUDE_GEOMETRY, EditMode.EXCLUDE_TEXTURE]:
            region_path = config.get_region_path(region_type, num_mesh=num_mesh)

            if region_path is not None and Path(region_path).exists() and IndicesAndMasks.get_num_indexed(IndicesAndMasks.load(region_path)) > 0:
                initial_select = IndicesAndMasks.get_indices(region_path)
            else:
                initial_select = config.meshes[num_mesh].get_small_and_large_triangles(
                    triangle_thresh_perc=0.999, return_vertices=True)
            config.set_indices_region(
                region_type, num_mesh=num_mesh, indices=initial_select)

    # Open landmarking panel
    blender_utils.adjust_view(
        cursor=False, relationship_lines=False, object_origins=False, cameras=False, lights=False,
        axes=False, floor=False)
    for num_split in range(max(args.num_window_splits, 0)):
        # We switch between vertical and horizontal splitting
        direction = ["VERTICAL", "HORIZONTAL"][num_split % 2]
        # We make the first vertical split with a facto 0.6, such that it gets evened out when we later remove the
        # other side panels on the right.
        factor = 0.6 if num_split == 0 else 0.5
        blender_utils.split_window(blender_utils.AreaType.VIEW_3D, factor=factor, direction=direction)

    # We always activate the viewport syncing in the beginning.
    # Even if we only have a single window, we need to do this, because otherwise, the event listener that
    # registers key presses like "S" doesn't react.
    config.sync_active = True
    bpy.ops.view3d.sync_viewports()

    view_areas = blender_utils.get_all_areas(area_types="VIEW_3D")

    if len(view_areas) < 1:
        raise AssertionError("Unable to find 3D view areas.")

    for num_view, view_area in enumerate(view_areas):
        config.set_mesh_per_view(view_area, min(num_view, len(config.meshes)-1))

        # We change all views from material to solid mode, unless the mesh to be shown in the current view has
        # colors and is not already shown in another view
        if num_view >= len(config.meshes) or not (blender_utils.has_vertex_colors(config.get_mesh_obj(num_view)) or
                                                  blender_utils.has_textures(config.get_mesh_obj(num_view))):
            spaces = [space for space in view_area.spaces if space.type == 'VIEW_3D']
            for space in spaces:
                space.shading.type = "SOLID"

    blender_utils.toggle_panel("UI", focus_tab="Landmarking")
    blender_utils.toggle_panel(["toolbar", "tool_header"])
    blender_utils.set_solid_light(blender_utils.SolidLightType.MATCAP)

    blender_utils.close_areas(areas_to_keep_open=[blender_utils.AreaType.VIEW_3D])

    config.select_input_mesh(0)




if __name__ == "__main__":
    # config is a global variable, so that it can be accessed from all operators and functions.
    config = Config()
    main()
