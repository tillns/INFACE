"""
This is a blender script that can only be run from blender. It should be called by calling
blender from the command line and passing the path to this file as additional argument:
blender --python Path/TO/src/blender/morphable_model_visualizer.py -- --PATH/TO/3DMM.h5.
Use --help as additional argument to get info about all arguments.
Alternatively, you can also use the python interface where you call the function visualize_morphable_model().
Note that you need to install h5py in Blender for this interface to work, cf. README.md.

The purpose of this script is to visualize a morphable model with blender, which works a bit better
than the alternative Open3D-based interface, since it offers good transparency options.
The user gets a slider to adjust the transparency of each disconnected mesh component.
Otherwise, the interface works relatively similarly to the Open3D-based interface.
The user is expected to provide a path to the morphable model they want to visualize.
Optionally, the user can specify specific components to show the sliders for.
By default, sliders for all components are available.
The user can also optionally provide one or more linear regressors and respective other morphable models
to translate the weights of the initial morphable model to the other morphable models
and show the decoded meshes along with the initial mesh.

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
from dataclasses import field, dataclass
from typing import Callable, Union, List

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
from src.objects.morphable_model import MorphableModel
from src.objects.mesh import Mesh
from src.objects.linear_regressor import LinearRegressor


# configuration data class to communicate between main method and the different panel and operator classes
@dataclass
class Config:
    # The specific components to show sliders for
    components: Union[List[int], None] = None
    # minimum slider value
    min_val: float = -3
    # maximum slider value
    max_val: float = 3
    # Names of the sliders
    sliders: list = field(default_factory=list)
    # The mesh vertex components -- transparency can be changed via sliders
    vertex_components = None
    # Callable function to retrieve the current Mesh given the current model weights
    get_current_mesh: Callable = None

# This is the main panel that contains all operators, so mainly all the sliders
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

        # Top button to export the mesh in its current form, so from the current model weights
        row = col.row(align=True)
        row.operator('object.export_mesh', text="Export Mesh", icon="FILE")

        # For each vertex component, we add a slider to adjust its transparency
        if config.vertex_components is not None:
            box = layout.box()
            box.label(text=f"Mesh Component Transparency", icon='MATERIAL')
            for num_comp in range(len(config.vertex_components)):
                box.prop(context.object, f"transp{num_comp}", slider=True)

        box = layout.box()
        box.label(text=f"Shape Components", icon='MESH_DATA')
        # Reset all shape components to 0 (average)
        box.operator("object.reset_sliders", text=f"Reset Sliders")
        # And for each model component, we add a slider to adjust the component's weight
        for comp in config.components:
            box.prop(context.object, str(comp), slider=True)

# Operator to reset all shape component sliders
class OBJECT_OT_reset_sliders(bpy.types.Operator):
    bl_idname = "object.reset_sliders"
    bl_label = "Reset Sliders"
    bl_description = "Reset all slider values to 0"

    def execute(self, context):
        obj = context.object
        # Reset the slider values to 0 if they exist on the object
        for name in config.sliders:
            if hasattr(obj, name) and getattr(obj, name) != 0:
                setattr(obj, name, 0)
        return {'FINISHED'}

# Operator to export the mesh in its current form. The user is asked where to save the mesh.
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

    # Here, a blender window is opened, s.t. the user can specify where to save the current mesh.
    def invoke(self, context, event):
        context.window_manager.fileselect_add(self)
        return {'RUNNING_MODAL'}

    # Actually do the saving after file path has been specified be user.
    def execute(self, context):
        current_mesh: Mesh = config.get_current_mesh()
        blender_utils.export_mesh_blender(current_mesh, self.filepath)
        return {"FINISHED"}

# attribute to register and deregister Blender classes
classes = [VISUALIZER_PT_PANEL, OBJECT_OT_reset_sliders, OBJECT_OT_export_mesh]

def main():
    argv = sys.argv
    if "--" not in argv:
        argv = []
    else:
        argv = argv[argv.index("--") + 1:]
    parser = ArgumentParser("You can run blender from CLI with this script: "
                            "blender [blend_file] --python " + __file__ + " -- [options]")
    parser.add_argument('--path_to_morphable_model', type=str, required=True,
                        help="Path to morphable model h5 file.")
    parser.add_argument('--min_val', type=float, default=-3,
                        help="Min val for sliders (deviation from the mean, relative to standard deviation)")
    parser.add_argument('--max_val', type=float, default=3,
                        help="Max val for sliders (deviation from the mean, relative to standard deviation)")
    parser.add_argument("--components", type=int, default=None, nargs='+',
                        help="Which shape components of the morphable model to show sliders for. "
                             "By default, all components are shown.")
    parser.add_argument("--linear_regressors", type=str, default=None, nargs='+',
                        help="Optionally provide one or more paths to linear regressors to translate the"
                             "morphable model latent to that of other morphable models. "
                             "This requires you to also provide translated morphable models.")
    parser.add_argument("--translated_morphable_models", type=str, default=None, nargs='+',
                        help="Other morphable model paths that, in combination with the linear regressors, "
                             "also give you the meshes from the models' correlated spaces.")

    args = parser.parse_args(argv)

    # This is the primary morphable model
    morphable_model = MorphableModel.load_correct_morphable_model(args.path_to_morphable_model)

    # Here we have the linear regressors that translate the weights based on the slider selection of the user,
    # which represent the setting of the primary morphable model's latent space, to other morphable models.
    linear_regressors = []
    if args.linear_regressors is not None:
        linear_regressors = [LinearRegressor.from_json(lin_path) for lin_path in args.linear_regressors]
    # These other morphable models are defined here. The latents translated by the linear regressors
    # are decoded to get the translated meshes.
    translated_morphable_models = []
    if args.translated_morphable_models is not None:
        translated_morphable_models = [MorphableModel.load_correct_morphable_model(mm_path)
                                       for mm_path in args.translated_morphable_models]
    # The user doesn't need to provide any linear regressors or translated morphable models, but the count must match.
    assert len(linear_regressors) == len(translated_morphable_models)

    # Number of shape components
    num_components = morphable_model.get_number_of_components()

    # The weights are initialized to 0, so the average in normalized model space.
    current_weights = np.zeros(num_components)

    def get_current_vertices() -> Union[np.ndarray, List[np.ndarray]]:
        """
        If we only have the primary morphable model, this method simply returns the vertices as numpy array
        decoded from the current weights via that model.
        If we also have translated models, this method returns a list with the first entry being the vertices
        of the primary model and the remaining entries the vertices decoded from the translated weights via
        the translated models.
        """
        unnormed_latent = morphable_model.unnormalize_weights(current_weights)
        mm_vertices = morphable_model.decode(unnormed_latent)
        if len(translated_morphable_models) == 0:
            return mm_vertices
        else:
            translated_vertices = [translated_model.decode(linear_regressor.translate_latent_code(unnormed_latent))
                                   for linear_regressor, translated_model in
                                   zip(linear_regressors, translated_morphable_models)]
            return [mm_vertices, *translated_vertices]

    def get_current_mesh() -> Mesh:
        """
        This method returns only one mesh. Even if there are multiple morphable models, this method
        combines all their meshes into one. The order of the vertices is preserved, so the first
        vertices are those from the primary morphable model, etc.
        """
        current_vertices = get_current_vertices()
        if isinstance(current_vertices, list):
            return Mesh.combine_meshes([
                morphable_model.get_mesh_from_vertices(current_vertices[0], include_colors_if_available=True),
                *[translated_model.get_mesh_from_vertices(vertices_ind, include_colors_if_available=True)
                  for vertices_ind, translated_model in zip(current_vertices[1:], translated_morphable_models)]])
        else:
            return morphable_model.get_mesh_from_vertices(current_vertices, include_colors_if_available=True)

    config.get_current_mesh = get_current_mesh

    # Given the combined mesh from all models, we compute its disconnected components
    # and provide the individual components as separate meshes to blender, such that
    # we can assign separate materials with varying transparency to the components.
    init_mesh = get_current_mesh()
    _, _, vertex_components = init_mesh.get_disconnected_components()
    # These are the materials we assign to the individual components
    materials = [blender_utils.get_colored_material(blender_utils.MaterialSettings(
        color="vertex" if init_mesh.has_colors() else (0.5, 0.5, 0.5), alpha=1, kind="principled"),
        force_new_material=True) for _ in range(len(vertex_components))]
    config.vertex_components = vertex_components
    blender_utils.remove_all_objects()
    # Here we do the separation by disconnected components
    meshes = [init_mesh.remove_vertices(vertex_component, invert=True) for vertex_component in vertex_components]
    mesh_obj_names = []
    # Here we loop over the individual meshes, and create a separate mesh blender object with
    # its individual material.
    for num_mesh, mesh in enumerate(meshes):
        mesh_obj_names.append(f"3DMM_mesh_comp{num_mesh}")
        obj = blender_utils.load_mesh(mesh, name=mesh_obj_names[-1], shade_smooth=True)
        blender_utils.assign_material_to_object(materials[num_mesh], obj, remove_current_materials=True)

    def update_transparency(num_material: int, transparency_value: float) -> None:
        """
        This method changes the transparency of the specific vertex components.
        For transparency 1 we use Blender's OPAQUE blend mode, whereas for <1 values,
        we switch to BLEND mode.
        :param num_material: index of the vertex component and respective material.
        :param transparency_value: The new value for the transparency.
        """
        current_object = blender_utils.get_obj_by_name(mesh_obj_names[num_material])
        current_object_active_material = current_object.active_material
        current_object_active_material.diffuse_color = (0.8, 0.8, 0.8, transparency_value)
        # node is hard-coded, but we make sure we created principled BSDF materials in the init_materials method
        blender_utils.change_node_value(materials[num_material].node_tree.nodes["Principled BSDF"],
                                        "alpha", transparency_value)
        if transparency_value < 1:
            current_object_active_material.blend_method = 'BLEND'
            current_object_active_material.show_transparent_back = False
        else:
            current_object_active_material.blend_method = 'OPAQUE'

    def update_vertices():
        """
        Update the vertices of the mesh, a.k.a. of all vertex component blender meshes.
        """
        current_vertices = get_current_vertices()
        if isinstance(current_vertices, list):
            # The vertex components were computed on the combined mesh,
            # which has preserved the order of the vertices, so we can
            # simply concatenate the individual vertex arrays here
            # and index them via the vertex_components to update the individual
            # blender meshes
            current_vertices = np.concatenate(current_vertices, axis=0)
        for num_mesh, vertex_component in enumerate(vertex_components):
            blender_utils.set_vertices(
                current_vertices[vertex_component], blender_utils.get_obj_by_name(mesh_obj_names[num_mesh]))

    def update_mesh_component(n, val):
        """
        Update the value of the current weight component, and then also update the vertices of the mesh(es).
        """
        current_weights[n] = val
        update_vertices()

    # Here we loop over the vertex components and define a separate transparency update function for each.
    for num_component, vertex_component in enumerate(vertex_components):
        slider_name = f"transp{num_component}"

        def make_transparency_update_func(mesh_comp_current, slider_name_current):
            def update_func(self, context):
                val = getattr(self, slider_name_current)
                update_transparency(mesh_comp_current, val)
            return update_func

        update_function_current = make_transparency_update_func(num_component, slider_name)
        setattr(bpy.types.Object, slider_name, bpy.props.FloatProperty(
            name=str(num_component), description=f"Change transparency of mesh component {num_component}",
            min=0, max=1, update=update_function_current, default=1))

    # Parse the components argument. If the user hasn't specified it,
    # we simply show sliders for all components. If the user has
    # specified a single nonzero number, we assume it's a range, otw we assume
    # the user wanted to show sliders for exactly these components.
    components = args.components
    if components is None:
        components = list(range(num_components))
    else:
        assert isinstance(components, list)
        if len(components) == 1:
            if components[0] == -1:
                components = list(range(num_components))
            elif components[0] != 0:
                components = list(range(min(components[0], num_components)))
    config.components = components

    # Loop over weights components to add sliders for
    for comp in components:
        if comp >= len(current_weights):
            continue
        current_val = current_weights[comp]
        slider_name = str(comp)

        def make_update_func(comp_current, slider_name_current):
            def update_func(self, context):
                val = getattr(self, slider_name_current)
                update_mesh_component(comp_current, val)
            return update_func

        update_function_current = make_update_func(comp, slider_name)
        setattr(bpy.types.Object, slider_name, bpy.props.FloatProperty(
            name=str(comp), description=f"Change shape component {comp}",
            min=min(args.min_val, current_val), max=max(args.max_val, current_val),
            update=update_function_current, default=current_val))
        config.sliders.append(slider_name)

    for c in classes:
        bpy.utils.register_class(c)

    # Open as much of the relevant GUI panel as possible given Blender's limited options in that regard
    blender_utils.toggle_panel("UI", focus_tab="Visualizer")
    blender_utils.toggle_panel(["toolbar", "tool_header"])
    # remove floor and other unnecessary stuff
    blender_utils.adjust_view(
        cursor=False, relationship_lines=False, object_origins=False, cameras=False, lights=False,
        axes=False, floor=False)

def unregister():
    for c in classes:
        bpy.utils.unregister_class(c)


if __name__ == "__main__":
    config = Config()
    main()
