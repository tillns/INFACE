"""
This is a blender script that can only be run from blender. It should be called by calling
blender from the command line and passing the path to this file as additional argument:
blender --python Path/TO/src/blender/measurement_correction_GUI.py -- --head_shape_factor_function /PATH/TO/pca_head_attribute_correlation.json --head_model /PATH/TO/pca_head_model.h5
Use --help as additional argument to get info about all arguments.
Alternatively, you can also use the python interface where you call the function.
Note that you need to install h5py in Blender for this interface to work, cf. README.md.


The purpose of this script is to visualize the variation of the cranial attributes that were correlated with the space
of the head morphable model. Additionally, it can be used to compute a healthy ("optimal") cranial shape for any
input shape. Any mesh loaded into this model must first be registered, cf. src/processing/registration.py.
The user can choose which attributes/measurements to include into the shape adjustments and which to leave out.
Additionally, the user can enable a color coding which highlights the deviation of the adjusted to the input shape.

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
from argparse import ArgumentParser
import sys
import os
from copy import deepcopy
from dataclasses import dataclass
from typing import Callable, Dict, Union, List

import bpy
import numpy as np
from pathlib import Path

from bpy.types import Operator
from bpy.props import IntProperty, FloatProperty, BoolProperty, StringProperty
from bpy_extras import view3d_utils
from bpy_extras.io_utils import ImportHelper, ExportHelper
import bmesh

# We need to add the src module to the system path to add functions from other files,
# since this file is executed from blender.
if __name__ == "__main__":
    file_path = Path(os.path.realpath(__file__))
    src_dir = file_path.parents[1]
    assert src_dir.name == "src" and "inface" in src_dir.parent.name.lower()
    sys.path.append(str(src_dir.parent))

from src.blender import blender_utils
from src import utils
from src.objects.cranial_attributes import CranialAttributes


# config Class contains all attributes relevant to perform the cranial shape adjustments
@dataclass
class Config:
    # These contain fixed optimal values for some measurements, but not necessarily all
    measurement_optima_fixed: dict = None
    # These contain all the optimal values
    measurement_optima: dict = None
    # Boolean dictionary indicating for each measurement whether it is currently enabled or disabled
    measurements_enabled: Dict[str, bool] = None
    # This array contains all the relevant measurement kinds as well as descriptions to explain the purpose of each
    measurement_kinds_and_descriptions: dict = None
    # The values that represent the estimation of the measurements for the input mesh
    measurements_relevant_for_correction: List[str] = None
    measurements_estimated: Dict = None
    # What values the measurements should be adjusted to
    measurements_current_values: Dict = None
    # Functions to retrieve the current and original mesh (as blender or python meshes)
    get_current_mesh: Callable = None
    get_original_mesh: Callable = None
    # CranialAttributes class, which handles most of the actual computations for the cranial shape correction
    cranial_attributes: CranialAttributes = None
    # If the user imports a mesh, this attribute is set to that path, such that
    # the user starts again at that path when trying to import another mesh
    last_mesh_import_file = None
    # Boolean that encodes whether the mesh should receive a color coding to highlight its deviation
    # from the original input mesh
    color_coding_enabled: bool = False

    # Parameters of the registered input mesh
    registered_mesh_vertices: np.ndarray = None
    registered_mesh_vertex_normals: np.ndarray = None
    registered_mesh_vertex_colors: np.ndarray = None

    # This function must be called before the GUI is run.
    # It basically instantiates all relevant attributes for computing the shape correction.
    def load_cranial_attributes(self, model_path: Union[str, Path], shape_factor_function_path: Union[str, Path]):
        self.cranial_attributes = CranialAttributes(
            model_path=model_path, shape_factor_function_path=shape_factor_function_path)
        self.measurement_optima_fixed = self.cranial_attributes.measurement_optima
        all_measurement_kinds = self.cranial_attributes.get_measurement_kinds()
        measurements_to_show = list(all_measurement_kinds.keys())
        # We only show the signed versions of measurements, e.g., CVAI itself cannot be linearly correlated, since
        # it cannot distinguish between left and right asymmetry, so there's no use including the unsigned version
        # into the GUI.
        measurements_to_show = [measurement_kind for measurement_kind in measurements_to_show
                                if f"{measurement_kind}_signed" not in measurements_to_show]
        if not self.cranial_attributes.has_age():
            measurements_to_show.remove("age")

        self.measurements_relevant_for_correction = self.cranial_attributes.get_measurements_relevant_for_correction(
            add_measurements_to_fix=True)

        self.measurement_kinds_and_descriptions = {
            measurement_kind: measurement_desc for measurement_kind, measurement_desc in
            all_measurement_kinds.items() if measurement_kind in measurements_to_show}
        self.measurements_enabled = {
            measurement_kind: True for measurement_kind in self.measurement_kinds_and_descriptions}

    def get_slider_name(self, measurement_kind: str) -> str:
        assert measurement_kind in list(self.measurement_kinds_and_descriptions.keys())
        return f"{measurement_kind}_slider"

    def get_checkbox_name(self, measurement_kind: str) -> str:
        assert measurement_kind in list(self.measurement_kinds_and_descriptions.keys())
        return f"{measurement_kind}_checkbox"

    #
    def update_color_coding(self):
        # These are the adjusted and original blender mesh objects
        adjusted_mesh_obj = self.get_current_mesh(return_as_python=False)
        original_registered_mesh_obj = self.get_original_mesh(return_as_python=False)

        # If color coding is enabled, we measure the deviation of the current vertices to the original
        # vertices, and we assign a color coding to the meshes that goes from -5mm to 5mm.
        if config.color_coding_enabled:
            new_vertices = blender_utils.get_vertices(adjusted_mesh_obj)
            vertex_diff = new_vertices - self.registered_mesh_vertices
            vertex_diff_sign = np.sign(utils.dot_vector_list(vertex_diff, self.registered_mesh_vertex_normals))
            vertex_error = np.linalg.norm(vertex_diff, axis=1) * vertex_diff_sign
            blender_utils.set_vertex_colors(utils.get_error_heatmap_colors(vertex_error, min_value=-5, max_value=5), adjusted_mesh_obj)
            blender_utils.set_vertex_colors(utils.get_error_heatmap_colors(-vertex_error, min_value=-5, max_value=5), original_registered_mesh_obj)

        # Uniform [0.6, 0.6, 0.6] colors are assigned to the mesh
        else:
            uniform_colors = np.ones_like(self.registered_mesh_vertices)*0.6
            blender_utils.set_vertex_colors(uniform_colors, adjusted_mesh_obj)
            # The original input mesh receives its original colors if it has any
            if self.registered_mesh_vertex_colors is not None:
                blender_utils.set_vertex_colors(self.registered_mesh_vertex_colors, original_registered_mesh_obj)
            else:
                blender_utils.set_vertex_colors(uniform_colors, original_registered_mesh_obj)

# This is the panel that contains all buttons and sliders.
class VISUALIZER_PT_PANEL(bpy.types.Panel):
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = "Visualizer"
    bl_label = "Visualizer"

    def get_status_icon(self, status):
        return "PAUSE" if 'inactive' in status.lower() else "MOUSE_RMB"

    def draw(self, context):
        layout = self.layout
        layout.use_property_decorate = False
        col = layout.column()
        col.scale_y = 2

        row = col.row(align=True)
        row.operator('object.import_mesh', text="Import Mesh", icon="FILE")

        row = col.row(align=True)
        row.operator('object.export_mesh', text="Export Mesh", icon="FILE")

        row = col.row(align=True)
        row.operator("object.reset_sliders", text=f"Reset Sliders")

        row = col.row(align=True)
        row.operator("object.change_all_sliders", text=f"Tick/Untick All")

        row = col.row(align=True)
        row.operator("object.set_sliders_to_optimum", text=f"Set to Optimum")

        row = col.row(align=True)
        row.operator("object.error_color_coding", text=f"Enable/Disable Error Colors")

        for measurement_kind in config.measurement_kinds_and_descriptions.keys():
            row = col.row(align=True)
            checkbox_width = 0.1
            # Create a split layout with two columns
            split = row.split(factor=checkbox_width)
            left_col = split.column()
            right_col = split.column()

            checkbox_name = config.get_checkbox_name(measurement_kind)
            # Draw the checkbox in the left column
            left_col.prop(context.object, checkbox_name, text="")

            # Enable/disable the slider based on the checkbox in the right column
            right_col.enabled = getattr(context.object, checkbox_name, False)
            right_col.prop(context.object, config.get_slider_name(measurement_kind), slider=True)



# Method to enable/disable specific slider, i.e., set its checkbox to True/False
def change_specific_slider(context, measurement_kind: str, set_to: bool):
    obj = context.object
    checkbox_name = config.get_checkbox_name(measurement_kind)
    if hasattr(obj, checkbox_name):
        setattr(obj, checkbox_name, set_to)

# Method to enable/disable all sliders, i.e., set all their checkboxes to True/False.
def change_all_sliders(context, set_to: bool):
    for measurement_kind in config.measurement_kinds_and_descriptions.keys():
        change_specific_slider(context, measurement_kind=measurement_kind, set_to=set_to)

# Operator to reset all sliders to their initial values.
# This automatically also enables all sliders.
class OBJECT_OT_reset_sliders(bpy.types.Operator):
    bl_idname = "object.reset_sliders"
    bl_label = "Reset Sliders"
    bl_description = "Reset all slider values to initial values"

    def execute(self, context):
        change_all_sliders(context, True)
        obj = context.object
        # Loop over all sliders and change their values to the initially estimated ones.
        for measurement_kind in config.measurement_kinds_and_descriptions.keys():
            slider_name = config.get_slider_name(measurement_kind)
            if hasattr(obj, slider_name):
                setattr(obj, slider_name, config.measurements_estimated[measurement_kind])
        return {'FINISHED'}

# Operator that sets all sliders to the optimal value, thus giving the user
# a quick way to create a respective cranial shape with "perfect" attributes.
# This automatically also enables all sliders.
class OBJECT_OT_set_sliders_to_optimum(bpy.types.Operator):
    bl_idname = "object.set_sliders_to_optimum"
    bl_label = "Set sliders to optimum"
    bl_description = "Change all sliders to the optimal value"

    def execute(self, context):
        change_all_sliders(context, False)
        obj = context.object
        # Loop over each slider and change its value to the optimum
        for measurement_kind in config.measurements_relevant_for_correction:
            change_specific_slider(context, measurement_kind=measurement_kind, set_to=True)
            slider_name = config.get_slider_name(measurement_kind)
            if hasattr(obj, slider_name):
                setattr(obj, slider_name, config.measurement_optima[measurement_kind])
        return {'FINISHED'}

# Operator to enable or disable all sliders.
class OBJECT_OT_change_all_sliders(bpy.types.Operator):
    bl_idname = "object.change_all_sliders"
    bl_label = "Tick/untick Sliders"
    bl_description = "If all sliders are ticked, they're all unticked. Else, all sliders are ticked."

    def execute(self, context):
        obj = context.object
        set_to = False
        # Loop over all sliders and check their current state
        for measurement_kind in config.measurement_kinds_and_descriptions.keys():
            checkbox_name = config.get_checkbox_name(measurement_kind)
            if hasattr(obj, checkbox_name):
                is_checkbox_ticked = getattr(obj, checkbox_name)
                if not is_checkbox_ticked:
                    # If at least one checkbox is unticked, we choose to enable all
                    set_to = True
                    break
        # We only disable all if all are currently enabled
        change_all_sliders(context, set_to=set_to)
        return {'FINISHED'}


# Operator that is simply a button to show or hide the color coding.
class OBJECT_OT_error_color_coding(bpy.types.Operator):
    bl_idname = "object.error_color_coding"
    bl_label = "Tick/Enable/Disable Error Colors"
    bl_description = ("Enable or disable the color coding of the error between the adjusted and the original mesh. "
                      "-5mm is blue, 0mm green, 5mm red.")

    def execute(self, context):
        config.color_coding_enabled = not config.color_coding_enabled
        config.update_color_coding()
        return {'FINISHED'}

# This Operator allows the user to export the mesh in its currently adjusted state,
# which can be useful for later downstream analysis.
class OBJECT_OT_export_mesh(Operator, ExportHelper):
    """Export current mesh"""

    bl_idname = "object.export_mesh"
    bl_label = "Export Mesh"
    bl_options = {'PRESET', 'UNDO'}

    filename_ext = '.ply'

    filter_glob: StringProperty(
        default='*.ply',
        options={"HIDDEN"}
    )

    def invoke(self, context, event):
        context.window_manager.fileselect_add(self)
        return {'RUNNING_MODAL'}

    def execute(self, context):
        current_mesh_obj = config.get_current_mesh(return_as_python=False)
        blender_utils.export_mesh_blender(current_mesh_obj, self.filepath)
        return {"FINISHED"}

# Operator to import a new registered mesh. As such, the user doesn't need
# to restart the GUI each time they want to look at a new mesh.
class OBJECT_OT_import_mesh(bpy.types.Operator, ImportHelper):
    """
    Load a registered mesh from file.
    """
    bl_idname = "object.import_mesh"
    bl_label = "Import Mesh"
    init_dir, init_file = None, None

    check_extension = False
    filter_glob: StringProperty(
        default='*.ply',
        options={"HIDDEN"}
    )
    filepath = ""

    # This is executed when pressing the import button within the import window.
    def execute(self, context):
        start_gui(self.filepath)
        config.last_mesh_import_file = self.filepath
        return {'FINISHED'}

    # This is executed when pressing import button in the custom panel.
    def invoke(self, context, event):
        if self.filepath == "" and config.last_mesh_import_file is not None and str(config.last_mesh_import_file).endswith(".ply"):
            self.filepath = config.last_mesh_import_file
        context.window_manager.fileselect_add(self)
        return {'RUNNING_MODAL'}

classes = [VISUALIZER_PT_PANEL, OBJECT_OT_reset_sliders, OBJECT_OT_export_mesh, OBJECT_OT_change_all_sliders,
           OBJECT_OT_set_sliders_to_optimum, OBJECT_OT_import_mesh, OBJECT_OT_error_color_coding]

def start_gui(registered_mesh_path: Union[str, Path, None] = None) -> None:

    # We first remove all objects that are already in the blender scene (mostly the cube)
    blender_utils.remove_all_objects()

    blender_mesh_load_kwargs = dict(name="registered mesh", shade_smooth=True)
    # We either load the registered mesh from the provided path
    if registered_mesh_path is not None:
        registered_mesh_original_obj = blender_utils.load_mesh(
            registered_mesh_path, **blender_mesh_load_kwargs)
    # or we use the average model mesh
    else:
        registered_mesh_original_obj = blender_utils.load_mesh(
            config.cranial_attributes.model.get_average_mesh(), **blender_mesh_load_kwargs)

    # We need the python mesh just for compatibility with a few methods.
    registered_python_mesh = blender_utils.convert_blender_to_python_mesh(registered_mesh_original_obj)

    # Initialize attributes from registered mesh
    config.registered_mesh_vertices = blender_utils.get_vertices(registered_mesh_original_obj, copy=True)
    config.registered_mesh_vertex_normals = registered_python_mesh.get_vertex_normals()
    if blender_utils.has_vertex_colors(registered_mesh_original_obj):
        config.registered_mesh_vertex_colors = blender_utils.get_blender_vertex_colors(registered_mesh_original_obj)


    # We then create a copy of the registered mesh -- one we leave untouched
    # and one is used for the shape adjustments.
    registered_mesh_obj = blender_utils.copy_object(registered_mesh_original_obj, new_object_name="adjusted mesh")
    # We hide the untouched one. The user can choose to manually show it using Blender's interface.
    blender_utils.hide_object(registered_mesh_original_obj)
    blender_utils.select_obj(registered_mesh_obj)

    # Projected registered mesh into PCA space (size is 1 x latent_size)
    registered_vertices_projected = config.cranial_attributes.model.get_projected_vertices(
        registered_python_mesh, normalize_projection=True)

    # Estimate all measurements for the mesh
    # We show all measurements in their original space to avoid confusion with the user.
    measurements_est = config.cranial_attributes.get_estimated_measurements(
        registered_vertices_projected, list(config.measurement_kinds_and_descriptions.keys()),
        apply_back_mapping=True)
    config.measurements_estimated = {
        measurement_kind: measurement_est[0] for measurement_kind, measurement_est in
        zip(config.measurement_kinds_and_descriptions.keys(), measurements_est)}

    # Set the optimal values to either the fixed optima from the CranialAttributes class,
    # or to the dataset average if that's not available
    config.measurement_optima = {
        measurement_kind: config.measurement_optima_fixed[measurement_kind]
        if measurement_kind in config.measurement_optima_fixed else
        (config.measurements_estimated[measurement_kind] if measurement_kind in ["age", "v_tot"]
         else config.cranial_attributes.get_original_mean(measurement_kind))
        for measurement_kind in config.measurement_kinds_and_descriptions.keys()}

    # This variable encodes the current state of the slider values, so every update of the measurements
    # goes through here.
    config.measurements_current_values = deepcopy(config.measurements_estimated)

    # This is the update function that receives a new value for a specific measurement
    # and adjusts the mesh accordingly.
    def update_measurement(measurement_kind: str, measurement_value: float):
        # Update the current measurement value
        config.measurements_current_values[measurement_kind] = measurement_value

        # Check which measurements are currently enabled
        current_measurement_kinds = [
            measurement_kind for measurement_kind in config.measurement_kinds_and_descriptions.keys()
            if config.measurements_enabled[measurement_kind]]

        # Get the measurement estimates and correction values for the enabled measurements
        def filter_measurement_array(measurements_dict):
            return np.asarray([measurements_dict[measurement_kind] for measurement_kind in current_measurement_kinds])

        measurements_estimated_current = filter_measurement_array(config.measurements_estimated)
        correction_values_current = filter_measurement_array(config.measurements_current_values)

        # Compute the updated vertices based on the enabled measurements
        vertices_shifted_measurement = config.cranial_attributes.correct_estimated_measurements(
            measurements_estimated=np.expand_dims(measurements_estimated_current, axis=1),
            measurement_kinds=current_measurement_kinds,
            measurement_targets=correction_values_current,
            registered_vertices_projected=registered_vertices_projected,
            are_measurements_in_original_space=True
        )[0]

        # Update the vertices of the blender mesh
        blender_utils.set_vertices(vertices_shifted_measurement, registered_mesh_obj)

        # Also update the color coding, so the difference between the adjusted and the input mesh
        config.update_color_coding()

    # toggle between enabled and disabled measurement
    def enable_disable_measurement(measurement_kind: str, set_to: bool):
        config.measurements_enabled[measurement_kind] = set_to

    # We loop over all the measurements in the GUI and add a checkbox and a slider for each
    for measurement_kind, measurement_description in config.measurement_kinds_and_descriptions.items():
        measurement_init = config.measurements_estimated[measurement_kind]
        slider_name = config.get_slider_name(measurement_kind)

        # Define an update function for each measurement to be passed on to the respective slider
        def make_update_func(measurement_kind_current, slider_name_current):
            def update_func(self, context):
                val = getattr(self, slider_name_current)
                update_measurement(measurement_kind_current, val)

            return update_func

        # Here we define the slider
        update_function_current = make_update_func(measurement_kind, slider_name)
        if measurement_kind in ["age", "v_tot"]:
            center = config.cranial_attributes.get_original_mean(measurement_kind)
        else:
            center = config.measurement_optima[measurement_kind]
        stdev = config.cranial_attributes.get_original_stdev(measurement_kind)
        setattr(bpy.types.Object, slider_name, bpy.props.FloatProperty(
            name=measurement_kind, description=measurement_description,
            # We allow the user to go four times the standard deviation in plus and minus direction
            min=center-4*stdev, max=center+4*stdev,
            update=update_function_current, default=measurement_init))

        # Next we define the checkbox
        checkbox_name = config.get_checkbox_name(measurement_kind)
        # The update function simply toggles the enabled status of the respective measurement
        def make_checkbox_update_func(measurement_kind_current, checkbox_name_current):
            def update_func(self, context):
                val = getattr(self, checkbox_name_current)
                enable_disable_measurement(measurement_kind_current, val)
            return update_func

        checkbox_update_func = make_checkbox_update_func(measurement_kind, checkbox_name)
        setattr(bpy.types.Object, checkbox_name, bpy.props.BoolProperty(
            name=f"Enable/disable {measurement_kind}",
            default=config.measurements_enabled[measurement_kind],
            update=checkbox_update_func
        ))

    # Next we define getter functions for the currently adjusted mesh...
    def get_current_mesh(return_as_python: bool = True):
        if return_as_python:
            return blender_utils.convert_blender_to_python_mesh(registered_mesh_obj)
        else:
            return registered_mesh_obj

    config.get_current_mesh = get_current_mesh

    # and for the original mesh
    def get_original_mesh(return_as_python: bool = True):
        if return_as_python:
            return blender_utils.convert_blender_to_python_mesh(registered_mesh_original_obj)
        else:
            return registered_mesh_original_obj

    config.get_original_mesh = get_original_mesh

    # We start by updating all measurements to the current values, which initializes the mesh state.
    for measurement_kind in config.measurement_kinds_and_descriptions.keys():
        update_measurement(measurement_kind, config.measurements_current_values[measurement_kind])
        break


def main():
    argv = sys.argv
    if "--" not in argv:
        argv = []
    else:
        argv = argv[argv.index("--") + 1:]
    parser = ArgumentParser("You can run blender from CLI with this script: "
                            "blender [blend_file] --python " + __file__ + " -- [options]")
    parser.add_argument('--head_shape_factor_function', type=str, required=True,
                        help="Path to the correlated cranial attributes, i.e., the shape factor function (.json).")
    parser.add_argument('--head_model', type=str, required=True,
                        help="Path to head morphable model (.h5).")
    parser.add_argument("--registered_mesh_path", type=str, default=None,
                        help="Optionally provide a path to a registered mesh for which you want to "
                             "vary/adjust/correct the cranial attributes. "
                             "If not provided, the average model mesh will be used.")
    args = parser.parse_args(argv)

    # Pass the head model and the shape factor function to the configuration,
    # so that it can instantiate the CranialAttributes class
    config.load_cranial_attributes(args.head_model, args.head_shape_factor_function)

    # Start the interface
    start_gui(registered_mesh_path=args.registered_mesh_path)

    # Just some Blender interface adjustments
    for c in classes:
        bpy.utils.register_class(c)

    # Open as much of the relevant GUI panel as possible given Blender's limited options in that regard
    blender_utils.toggle_panel("UI", focus_tab="Visualizer")
    blender_utils.toggle_panel(["toolbar", "tool_header"])
    # remove floor and other unnecessary stuff
    blender_utils.adjust_view(
        cursor=False, relationship_lines=False, object_origins=False, cameras=False, lights=False,
        axes=False, floor=False)
    blender_utils.set_to_material_mode()


def unregister():
    for c in classes:
        bpy.utils.unregister_class(c)


if __name__ == "__main__":
    # config object is defined with file-scope, so that all classes can access it.
    config = Config()
    main()
