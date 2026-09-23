"""
This is a file that contains utility Blender functions that are used in one or more of the
custom Blender interfaces. Due to time constraints, the functions in this file are not thoroughly
documented. Please reach out to the author in case you have any questions.

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

# Standard imports
import numpy as np
import warnings
import re
from dataclasses import dataclass
from pathlib import Path
from typing import ClassVar, Union, List, Tuple

# Imports from own code
from src import utils
from src.objects.mesh import Mesh
from src.objects.indices_and_masks import IndicesAndMasks

# Blender imports
import bpy, bmesh
import mathutils
from mathutils.bvhtree import BVHTree


@dataclass
class BlenderObjType:
    MESH: ClassVar[str] = "MESH"
    LIGHT: ClassVar[str] = "LIGHT"
    SURFACE: ClassVar[str] = "SURFACE"
    CAMERA: ClassVar[str] = "CAMERA"

    @staticmethod
    def is_blender_obj_type(obj_type):
        return obj_type in BlenderObjType.__dict__.values()

    @staticmethod
    def get_possible_obj_types():
        return BlenderObjType.__dict__.values()

@dataclass()
class AreaType:
    """
    Contains the strings for the different available area types in blender (element of list bpy.context.screen.areas
    have attribute called type which can be compared to the strings in this class).
    """
    VIEW_3D: ClassVar[str] = "VIEW_3D"
    CONSOLE: ClassVar[str] = "CONSOLE"
    INFO: ClassVar[str] = "INFO"
    IMAGE_EDITOR: ClassVar[str] = "IMAGE_EDITOR"

@dataclass
class SolidLightType:
    MATCAP: ClassVar[str] = "MATCAP"
    STUDIO: ClassVar[str] = "STUDIO"
    FLAT: ClassVar[str] = "FLAT"

@dataclass
class MaterialSettings:
    """
    @param color: numbers for uniform color, "vertex" for vertex color
    @param kind: "diffuse", "BSDF", "palate"
    @param alpha: value between 0 and 1
    """
    color: Union[Tuple[float, float, float], List[float], str] = (0.5, 0.5, 0.5)
    kind: str = "diffuse"
    alpha: float = 1.0
    show_backface: bool = False
    uv: Union[str, bool] = None
    texture_scale: float = None
    allow_brightnesscontrast_change: bool = False
    shader_adjust_settings: dict[str, float] = None

    def get_mat_name(self):
        return f"{self.color}_{self.kind}_{self.alpha}"

    @staticmethod
    def get_color_static(custom_color: Union[float, int, tuple, list] = None, num_components: int = 4):
        color = ([custom_color] if not isinstance(custom_color, (list, tuple)) else custom_color)
        if len(color) == 1:
            color = [color[0], color[0], color[0]]
        if len(color) < num_components:
            color = color + [1] * (num_components - len(color))
        else:
            color = color[:num_components]
        return tuple(color)

    def get_color(self, num_components: int = 4, custom_color: Union[float, int, tuple, list] = None):
        color = [1, 1, 1] if isinstance(self.color, str) else list(self.color)
        return self.get_color_static(color if custom_color is None else self.color, num_components=num_components)


def get_obj_name(obj):
    if hasattr(obj, "name"):
        return obj.name
    elif isinstance(obj, str):
        return obj
    else:
        raise TypeError(f"Input {obj} of unknown type.")

def get_obj_by_name(name):
    return bpy.context.scene.objects[name]

def get_obj_by_name_or_obj(name_or_obj):
    if isinstance(name_or_obj, str):
        return get_obj_by_name(name_or_obj)
    else:
        return name_or_obj

def exists_obj(obj_name):
    try:
        obj_name_actual = get_obj_name(obj_name)
    except (TypeError, ReferenceError):
        return False
    return obj_name_actual in bpy.context.scene.objects

def get_obj_data(obj_mesh_or_name):
    if isinstance(obj_mesh_or_name, str):
        return get_obj_by_name(obj_mesh_or_name).data
    elif hasattr(obj_mesh_or_name, "data"):
        return obj_mesh_or_name.data
    else:
        return obj_mesh_or_name

def get_obj_names(objs: list):
    return [get_obj_name(obj) for obj in objs]

def get_active_obj():
    try:
        return bpy.context.active_object
    except AttributeError:
        return bpy.context.view_layer.objects.active

def get_selected_objects(regex_filter=None, obj_type=None):
    return get_all_objects(regex_filter, obj_type, only_selected=True)


def get_selected_object_names(regex_filter=None, obj_type=None):
    selected_objects = get_selected_objects(regex_filter=regex_filter, obj_type=obj_type)
    return [get_obj_name(selected_object) for selected_object in selected_objects]

def deselect_all_objects():
    for obj in get_all_objects(only_selected=True):
        obj.select_set(False)


def select_obj(obj, make_active: bool = True, deselect_remaining: bool = False):
    if deselect_remaining:
        deselect_all_objects()
    if make_active:
        bpy.context.view_layer.objects.active = obj
    obj.select_set(True)


def select_object_by_name(*name, make_active=True):
    deselect_all_objects()
    for name_ind in name:
        ob = get_obj_by_name(name_ind)
        # Don't remember what this try-except is for...
        try:
            bpy.context.view_layer.objects.mouse_active = ob
        except AttributeError:
            pass
        select_obj(ob, make_active=make_active)

def set_all_objects_render_invisible(state, regex_filter=None, obj_type=None, only_selected=False):
    all_objects = get_all_objects(regex_filter=regex_filter, obj_type=obj_type, only_selected=only_selected)
    for obj in all_objects:
        set_object_render_invisible(get_obj_name(obj), state)

def set_all_meshes_render_invisible(state, regex_filter=None, only_selected=False):
    set_all_objects_render_invisible(state, regex_filter=regex_filter, obj_type=BlenderObjType.MESH,
                                     only_selected=only_selected)


def set_all_lights_render_invisible(state, regex_filter=None, only_selected=False):
    set_all_objects_render_invisible(state, regex_filter=regex_filter, obj_type=BlenderObjType.LIGHT,
                                     only_selected=only_selected)

def set_all_cameras_render_invisible(state, regex_filter=None, only_selected=False):
    set_all_objects_render_invisible(state, regex_filter=regex_filter, obj_type=BlenderObjType.CAMERA,
                                     only_selected=only_selected)

def set_object_render_invisible(name_or_obj: str, state: bool) -> None:
    """
    Make an object invisible or visible.
    :param name_or_obj: Name of the object.
    :param state: True makes it invisible, False makes it visible.
    :return: None
    """
    obj = get_obj_by_name_or_obj(name_or_obj)

    obj.hide_render = state
    obj.hide_set(state)

def switch_object_render_invisible(name_or_obj: str) -> None:
    set_object_render_invisible(name_or_obj, not is_object_hidden(name_or_obj))

def is_object_hidden(obj_or_name):
    obj = get_obj_by_name_or_obj(obj_or_name)
    return obj.hide_render

def hide_object(obj_or_name):
    set_object_render_invisible(obj_or_name, True)

def hide_all_objects(regex_filter=None, obj_type=None, only_selected=False):
    set_all_objects_render_invisible(True, regex_filter=regex_filter, obj_type=obj_type, only_selected=only_selected)

def show_object(obj_or_name):
    set_object_render_invisible(obj_or_name, False)

def copy_object(obj_or_name, new_object_name: str, collection_to_assign_to = None):
    obj = get_obj_by_name_or_obj(obj_or_name)
    obj_copy = obj.copy()
    obj_copy.data = obj.data.copy()  # copy mesh datablock too
    change_object_name(new_name=new_object_name, obj_or_current_name=obj_copy)
    if collection_to_assign_to is not None:
        collection_to_assign_to.objects.link(obj_copy)
    else:
        bpy.context.collection.objects.link(obj_copy)
    return obj_copy

def set_overlays(show=False):
    spaces = get_all_context_spaces()
    for space in spaces:
        space.overlay.show_overlays = show

def blender_call_with_override(blender_method, override, *blender_method_args, **blender_method_kwargs):
    try:
        blender_method(override, *blender_method_args, **blender_method_kwargs)
    except ValueError:
        with bpy.context.temp_override(**override):
            blender_method(*blender_method_args, **blender_method_kwargs)

def get_all_objects(regex_filter: Union[str, bpy.types.Object, List, None] = None, obj_type=None, only_selected=False,
                    exclude_regex_filter: Union[str, List[str], None] = None) -> List[bpy.types.Object]:

    objs = bpy.context.scene.objects
    if only_selected:
        objs = [o for o in objs if o.select_get()]
    if obj_type is not None:
        if not BlenderObjType.is_blender_obj_type(obj_type):
            warnings.warn(f"Given obj_type {obj_type} is not part of the allowed obj types: "
                          f"{BlenderObjType.get_possible_obj_types()}")
        objs = [ob for ob in objs if ob.type == obj_type]
    if isinstance(regex_filter, str):
        pattern = re.compile(regex_filter)
        objs = [obj for obj in objs if pattern.fullmatch(get_obj_name(obj))]
    elif isinstance(regex_filter, list):
        objs_tmp = []
        for regex_filter_ind in regex_filter:
            objs_tmp.extend(get_all_objects(regex_filter_ind, obj_type=obj_type, only_selected=only_selected))
        objs = list(set(objs_tmp))
    elif isinstance(regex_filter, bpy.types.Object):
        objs = [regex_filter]

    if exclude_regex_filter is not None:
        if isinstance(exclude_regex_filter, str):
            exclude_regex_filter = [exclude_regex_filter]
        if not isinstance(exclude_regex_filter, list):
            raise ValueError("Provide string or list of strings for exclude regex filter")
        exclude_patterns = [re.compile(exclude_regex_filter_ind) for exclude_regex_filter_ind in exclude_regex_filter]
        objs = [obj for obj in objs if not any(exclude_pattern.fullmatch(get_obj_name(obj))
                                               for exclude_pattern in exclude_patterns)]

    return objs

def get_all_meshes(regex_filter: Union[str, bpy.types.Object, List, None] = None,
                   exclude_regex_filter: Union[str, List[str], None] = None) -> List[bpy.types.Object]:
    """
    Returns a list of all mesh objects in the scene. You can filter the list via the regex filter argument.
    If regex_filter is a blender object, this method will return a list with this single obj.
    """
    return get_all_objects(regex_filter, obj_type=BlenderObjType.MESH, exclude_regex_filter=exclude_regex_filter)

def get_all_lights(regex_filter=None):
    """
    Returns a list of all light objects in the scene. You can filter the list via the regex filter argument.
    If regex_filter is a blender object, this method will return a list with this single obj.
    """
    return get_all_objects(regex_filter, obj_type=BlenderObjType.LIGHT)

def remove_objects(regex_filter=None, exclude_regex_filter=None):
    """
    Remove all mesh objects in the scene, optionally only those that match with the provided regex.
    """
    objs = get_all_meshes(regex_filter, exclude_regex_filter=exclude_regex_filter)
    if len(objs) > 0:
        for obj in objs:
            bpy.data.objects.remove(obj, do_unlink=True)
        # old code
        """
        # objects can only be deleted in object mode.
        override = {"selected_objects": objs}
        blender_call_with_override(bpy.ops.object.mode_set, override, mode='OBJECT')
        blender_call_with_override(bpy.ops.object.delete, override)
        """

def remove_objects_also_nonmeshes(regex_filter=None, exclude_regex_filter=None):
    """
    Remove all objects in the scene, optionally only those that match with the provided regex.
    """
    objs = get_all_objects(regex_filter, exclude_regex_filter=exclude_regex_filter)
    if len(objs) > 0:
        # objects can only be deleted in object mode.
        override = {"selected_objects": objs}
        blender_call_with_override(bpy.ops.object.mode_set, override, mode='OBJECT')
        blender_call_with_override(bpy.ops.object.delete, override)

def remove_all_objects():
    remove_objects()

def remove_lights(regex_filter=None):
    """
    Remove all light objects in the scene, optionally only those that match with the provided regex.
    """
    objs = get_all_lights(regex_filter)
    if len(objs) > 0:
        # objects can only be deleted in object mode.
        bpy.ops.object.mode_set({"selected_objects": objs}, mode='OBJECT')
        bpy.ops.object.delete({"selected_objects": objs})

def add_light(light_type: str = "SUN", radius: float = 1, location: Tuple[float] = None, rotation: Tuple[float] = None,
              strength: float = None) -> None:
    """
    Add light source, cf. Blender's docs of bpy.ops.object.light_add() for most arguments.
    :param light_type: "SUN", "POINT", "SPOT", or "AREA"
    """
    location = (0, 0, 0) if location is None else location
    rotation = (0, 0, 0) if rotation is None else rotation
    bpy.ops.object.light_add(type=light_type.upper(), radius=radius, align='WORLD', location=location, rotation=rotation,
                             scale=(1, 1, 1))
    light_obj = get_selected_objects()[0]
    if strength is not None:
        light_obj.data.energy = strength

def get_all_areas(area_types: Union[str, List[str]] = None, filter_areas: str = None, context=None):
    context = bpy.context if context is None else context
    area_types = [area_types] if isinstance(area_types, str) else area_types
    areas = [area for area in context.window.screen.areas if area.type in area_types]
    # Filter areas if necessary
    if filter_areas is not None:
        filter_areas = filter_areas.lower()
        areas_x = [area.x for area in areas]
        areas_y = [area.y for area in areas]
        if "left" in filter_areas:
            areas = [area for area in areas if area.x == min(areas_x)]
        elif "right" in filter_areas:
            areas = [area for area in areas if area.x == max(areas_x)]
        elif "top" in filter_areas or "up" in filter_areas:
            areas = [area for area in areas if area.x == max(areas_y)]
        elif "bott" in filter_areas or "down" in filter_areas:
            areas = [area for area in areas if area.x == min(areas_y)]

    return areas

def get_all_context_spaces(filter_spaces: str = None, context=None):
    if context is not None and context.space_data is not None and context.space_data.type == AreaType.VIEW_3D:
        return [context.space_data]
    view3d_areas = get_all_areas(AreaType.VIEW_3D, filter_areas=filter_spaces, context=context)

    # Convert areas to spaces
    spaces = [space for area in view3d_areas for space in area.spaces if space.type == 'VIEW_3D']

    return spaces

def toggle_panel(panel_type: Union[str, List[str]] = "ui", focus_tab: str = None, filter_spaces=None,
                 toggle_mode: str = "toggle", context=None) -> None:
    """
    This method toggles a panel to open or close.
    It is currently not possible to open a specific tab of the panel, though.
    :param panel_type: UI, toolbar, etc.
    :param focus_tab: Specify to focus on a specific panel tab (currently not supported, cf. comments below)
    :param num_open: If there's multiple 3D views open, how many times to toggle the panel,
                     e.g., set to 1 to only open panel in one 3D view.
    :param toggle_mode: Set this to "open" or "close" if you want to specifically open or close a panel, not toggle it.
    :return: None
    """

    panel_type = [panel_type] if isinstance(panel_type, str) else panel_type
    panel_type = [panel_type_ind.lower() for panel_type_ind in panel_type]

    #view_3d_areas = [area for area in bpy.context.screen.areas if area.type == AreaType.VIEW_3D]
    #view_3d_areas = get_all_areas(AreaType.VIEW_3D, context=context)
    context_spaces = get_all_context_spaces(filter_spaces=filter_spaces, context=context)
    for space in context_spaces:
        for panel_type_ind in panel_type:
            # todo: extend list if needed using docs https://docs.blender.org/api/current/bpy.types.SpaceView3D.html
            if "ui" in panel_type_ind:
                toggle_type = "show_region_ui"
            elif "gizmo" in panel_type_ind:
                toggle_type = "show_gizmo_tool"
            elif "tool" in panel_type_ind:
                if "head" in panel_type_ind:
                    toggle_type = "show_region_tool_header"
                else:
                    toggle_type = "show_region_toolbar"
            elif "hud" in panel_type_ind:
                toggle_type = "show_region_hud"
            else:
                raise TypeError(f"Unknown panel type: {panel_type_ind}.")

            if toggle_mode.lower() == "toggle":
                new_mode = not getattr(space, toggle_type)
            else:
                new_mode = "open" in toggle_type.lower()
            setattr(space, toggle_type, new_mode)

            if focus_tab is not None:
                # todo: Once blender supports opening a specific panel, add the code here.
                #  cf. here
                #  https://blender.stackexchange.com/questions/285829/how-can-i-set-the-active-category-in-the-n-panel.
                #  The proposed solution from Blender 4.2 doesn't seem to work like that, since it always says that
                #  the panel category is read-only. Adding a delay doesn't help either.
                #  Possible queues to google to check if this has changed:
                #  1. "Blender Python API select sidebar tab"
                #  2. "Blender Python API focus on custom panel"
                #  3. "Blender Python API set active N-panel"
                #  4. "Blender bpy set active sidebar tab"
                #  5. "Blender bpy open specific panel"
                #  6. "Blender Python API programmatically select UI tab"
                #  7. "Blender Python API change active panel"
                #  8. "Blender bpy set active UI category"
                pass

def set_vertices(new_vertices, obj=None):
    if obj is None:
        obj = bpy.context.object
    obj_data = get_obj_data(obj)
    for num_vert, vert in enumerate(obj_data.vertices):
        vert.co = new_vertices[num_vert]
    obj_data.update()


def get_node_value_index(node, value_name: str, is_output_value: bool = False):
    node_value_names = []
    node_values = node.outputs if is_output_value else node.inputs
    for i, node_value in enumerate(node_values):
        node_value_name = node_value.name
        if node_value_name.lower() == value_name.lower():
            return i
        node_value_names.append(node_value_name)
    raise ValueError(f"Node {value_name} not found in {node.name}. Available nodes: {', '.join(node_value_names)}")


def change_node_value(node, value_name: str, value, is_output_value: bool = False):
    value_index = get_node_value_index(node, value_name, is_output_value=is_output_value)
    if is_output_value:
        node.outputs[value_index].default_value = value
    else:
        node.inputs[value_index].default_value = value

def get_blender_default_vertex_color_name():
    return "Col"

def get_colored_material(material_settings: MaterialSettings = None, force_new_material=False):
    material_settings = material_settings if material_settings is not None else MaterialSettings()
    material_name = material_settings.get_mat_name()
    if material_name not in bpy.data.materials or force_new_material:
        mat = bpy.data.materials.new(name=material_name)
        mat.use_nodes = True
        links = mat.node_tree.links
        nodes = mat.node_tree.nodes
        nodes.clear()

        color_setting = material_settings.color
        use_vertex_colors = False
        if isinstance(color_setting, str):
            color_setting = color_setting.lower()
            if "vertex" in color_setting:
                # vertex color node
                use_vertex_colors = True
                node_vertex = nodes.new(type='ShaderNodeVertexColor')
                node_colors = node_vertex
            elif "texture" in color_setting:
                if "check" in color_setting:
                    node_colors = nodes.new(type="ShaderNodeTexChecker")
                    if material_settings.texture_scale is not None:
                        change_node_value(node_colors, "Scale", material_settings.texture_scale)
                        change_node_value(node_colors, "Color1", (0.4, 0.4, 0.4, 1))
                        change_node_value(node_colors, "Color2", (0.6, 0.6, 0.6, 1))
                elif "grad" in color_setting:
                    node_colors = nodes.new(type="ShaderNodeTexGradient")
                elif "wave" in color_setting:
                    node_colors = nodes.new(type="ShaderNodeTexWave")
                    node_colors.wave_type = "RINGS"
                    node_colors.rings_direction = "Spherical"
                    if material_settings.texture_scale is not None:
                        change_node_value(node_colors, "Scale", material_settings.texture_scale)
                else:
                    node_colors = nodes.new(type="ShaderNodeTexChecker")
                if isinstance(material_settings.uv, str) or \
                        (isinstance(material_settings.uv, bool) and material_settings.uv):
                    node_uv = nodes.new(type="ShaderNodeUVMap")
                    node_uv.uv_map = material_settings.uv if isinstance(material_settings.uv, str) else "UVMap"
                    links.new(node_uv.outputs[0], node_colors.inputs[0])
            else:
                node_colors = nodes.new(type="ShaderNodeRGB")
        else:
            # RGB color node
            node_colors = nodes.new(type="ShaderNodeRGB")
            change_node_value(node_colors, "Color", material_settings.get_color(num_components=4),
                              is_output_value=True)
        node_colors.location = 0, 0

        # Material output node
        node_output = nodes.new(type='ShaderNodeOutputMaterial')
        node_output.location = 700, 0

        # more palate-like material
        if "diffuse" in material_settings.kind and material_settings.alpha == 1:
            node_main = nodes.new(type='ShaderNodeBsdfDiffuse')
            links.new(node_main.outputs[0], node_output.inputs[0])
        elif "emiss" in material_settings.kind.lower() or "shaderless" in material_settings.kind.lower():
            node_main = nodes.new(type='ShaderNodeEmission')
            print("Creating emissive shader")
            if material_settings.alpha < 1:
                print("Alpha is smaller than 1. Creating a mixed shader.")
                node_mix = nodes.new(type='ShaderNodeMixShader')
                node_transparent = nodes.new(type="ShaderNodeBsdfTransparent")
                node_transparent.inputs[0].default_value[3] = 0
                links.new(node_main.outputs[0], node_mix.inputs[1])
                links.new(node_transparent.outputs[0], node_mix.inputs[2])
                node_mix.inputs[0].default_value = material_settings.alpha
                links.new(node_mix.outputs[0], node_output.inputs[0])
            else:
                links.new(node_main.outputs[0], node_output.inputs[0])
        else:
            node_main = nodes.new(type='ShaderNodeBsdfPrincipled')
            node_main.location = 200, 0
            if material_settings.alpha < 1:
                mat.blend_method = 'BLEND'
                mat.show_transparent_back = material_settings.show_backface
                # Alpha should be on place 21, but this can change with blender versions
                change_node_value(node_main, "alpha", material_settings.alpha)
                mat.diffuse_color = (*mat.diffuse_color[:3], material_settings.alpha)

            links.new(node_main.outputs[0], node_output.inputs[0])

        if material_settings.shader_adjust_settings is not None:
            for setting_name, setting_value in material_settings.shader_adjust_settings.items():
                change_node_value(node_main, setting_name, setting_value)
        node_main.location = 200, 0
        links.new(node_colors.outputs[0], node_main.inputs[0])

        return mat
    else:
        return bpy.data.materials[material_name]

def assign_material_to_object(material, obj_mesh_or_name, remove_current_materials: bool = False):
    blender_mesh = get_obj_data(obj_mesh_or_name)
    if remove_current_materials:
        blender_mesh.materials.clear()
    blender_mesh.materials.append(material)
    blender_mesh.update()

def set_vertex_colors(vertex_colors, blender_mesh, force_add_color=False, assign_material=True, color_name=None):
    if not hasattr(blender_mesh, "vertex_colors") and hasattr(blender_mesh, "data"):
        blender_mesh = blender_mesh.data
    vertex_color_name = get_blender_default_vertex_color_name() if color_name is None else color_name
    if not vertex_color_name in blender_mesh.vertex_colors or force_add_color:
        color_layer = blender_mesh.vertex_colors.new(name=vertex_color_name)
    else:
        color_layer = blender_mesh.vertex_colors[vertex_color_name]
    # colors in blender are defined per triangle per vertex (so one vertex can have multiple colors, smh)
    i = 0
    for poly in blender_mesh.polygons:
        for idx in poly.vertices:
            # todo: Consider gamma correction, so take vertex colors to the power of 0.4545
            color_layer.data[i].color = (*vertex_colors[idx], 1.0)
            i += 1

    if assign_material:
        material_settings = MaterialSettings("vertex_colors")
        assign_material_to_object(get_colored_material(material_settings), blender_mesh, remove_current_materials=True)

def get_mode(obj_or_name):
    if isinstance(obj_or_name, str):
        obj = get_obj_by_name(obj_or_name)
    else:
        obj = obj_or_name
    if hasattr(obj, "mode"):
        return obj.mode
    else:
        raise TypeError("Provided argument unknown: {}. It should be either a blender object or its name".
                        format(obj_or_name))

def check_mode(mode):
    mode = mode.upper()
    allowed_modes = ['OBJECT', 'EDIT', 'SCULPT', 'VERTEX_PAINT', 'WEIGHT_PAINT', 'TEXTURE_PAINT',
                     'PARTICLE_EDIT', 'POSE']
    if mode not in allowed_modes:
        raise TypeError("Provided mode {} not one of the available modes {}".format(mode, deselect_all_objects()))

def set_mesh_to_mode(mode: str = "OBJECT", regex_filter=None):
    """
    Set all mesh objects -- or only those that match the provided regex_filter -- to specific mode.
    """
    mode = mode.upper()
    check_mode(mode)
    objs = get_all_meshes(regex_filter)
    all_objs = get_all_meshes()
    if mode == "OBJECT":
        remaining_objs = [obj for obj in all_objs if obj not in objs]
        remaining_objs_initial_state = [get_mode(obj) for obj in remaining_objs]
        remaining_objs_edit_mode_selection = [get_current_vertex_selection(obj) if initial_state == "EDIT" else None for
                                              obj, initial_state in zip(remaining_objs, remaining_objs_initial_state)]
    for obj in objs:
        if get_mode(obj) == mode:
            continue
        select_object_by_name(get_obj_name(obj), make_active=True)
        is_hidden = is_object_hidden(obj)
        show_object(obj)
        bpy.ops.object.mode_set(mode=mode)
        set_object_render_invisible(obj, is_hidden)

    # We can't only set one object to Object mode, so if we tried that before,
    # we set the remaining objects back to their initial state.
    if mode == "OBJECT":
        for obj, initial_state, edit_mode_selection in zip(remaining_objs, remaining_objs_initial_state, remaining_objs_edit_mode_selection):
            if initial_state != "OBJECT":
                set_mesh_to_mode(initial_state, obj)
                if edit_mode_selection is not None:
                    select_vertices(edit_mode_selection, obj)



def is_in_mode(mode, obj_or_name):
    mode = mode.upper()
    check_mode(mode)
    return get_obj_by_name_or_obj(obj_or_name).mode == mode


def set_to_edit_mode(regex_filter=None):
    """
    Set all mesh objects to edit mode.
    """
    set_mesh_to_mode("EDIT", regex_filter)


def is_in_edit_mode(obj_or_name):
    """
    Check if object is in edit mode.
    """
    return is_in_mode("EDIT", obj_or_name)


def set_to_object_mode(regex_filter=None):
    """
    Set all mesh objects to object mode.
    """
    set_mesh_to_mode("OBJECT", regex_filter)


def is_in_object_mode(obj_or_name):
    return is_in_mode("OBJECT", obj_or_name)


def convert_python_to_blender_mesh(python_mesh: Mesh, name=None):
    name = "default" if name is None else name
    # Create mesh as object data
    blender_mesh = bpy.data.meshes.new(name=name)

    vertices = python_mesh.get_vertices(copy=True)
    blender_mesh.vertices.add(len(vertices))
    blender_mesh.vertices.foreach_set("co", vertices.flatten())

    faces = python_mesh.get_triangles(copy=True)
    flattened_faces = faces.flatten()

    blender_mesh.loops.add(len(flattened_faces))
    blender_mesh.loops.foreach_set("vertex_index", flattened_faces)

    num_loops = len(faces)
    loop_start = np.asarray([i * 3 for i in range(num_loops)], dtype=np.int32)
    loop_total = np.asarray([3 for _ in range(num_loops)], dtype=np.int32)

    blender_mesh.polygons.add(len(faces))
    blender_mesh.polygons.foreach_set("loop_start", loop_start)
    blender_mesh.polygons.foreach_set("loop_total", loop_total)

    if python_mesh.has_colors():
        vertex_colors = python_mesh.get_colors(copy=True)
        set_vertex_colors(vertex_colors, blender_mesh)

    # We're done setting up the mesh values, update mesh object and
    # let Blender do some checks on it
    blender_mesh.update()
    blender_mesh.validate()

    return blender_mesh

def convert_blender_to_python_mesh(obj=None, include_transform=False) -> Mesh:
    """
    Takes the current/provided blender mesh object and converts it to python Mesh,
    optionally include the blender transform into the mesh.
    :param obj: Blender mesh object. If not provided, active object will be used.
    :param include_transform: Include transform into the mesh. Default is False.
    :return Mesh type instance
    """
    obj = bpy.context.object if obj is None else obj
    obj_current_mode = get_mode(obj)
    set_to_edit_mode(obj)
    me = obj.data
    blender_mesh = bmesh.from_edit_mesh(me)
    mesh_dict = {"vertices": [list(vertex.co[:3]) for vertex in blender_mesh.verts],
                 "triangles": [[vert.index for vert in face.verts] for face in blender_mesh.faces]}

    set_mesh_to_mode(obj_current_mode, obj)
    if get_blender_default_vertex_color_name() in me.vertex_colors:
        mesh_dict["colors"] = get_blender_vertex_colors(obj).tolist()
    mesh = Mesh(**mesh_dict)

    if include_transform:
        transf = obj.matrix_world
        mesh.transform(transf, in_place=True)
    return mesh

def export_mesh_blender(mesh: Union[Mesh, bpy.types.Object], file_path: Union[Path, str]) -> None:
    """
    Export a Mesh or blender object as a mesh.
    """
    file_path = Path(file_path)
    selected_objects = get_selected_objects()

    # If we have a Mesh as input. We first convert it to a blender object and delete the object after we're done.
    if isinstance(mesh, Mesh):
        blender_mesh = convert_python_to_blender_mesh(mesh)
        blender_obj = bpy.data.objects.new("temp_export_obj", blender_mesh)
        bpy.context.collection.objects.link(blender_obj)
        delete_object_after = True
    else:
        blender_obj = mesh
        delete_object_after = False

    select_obj(blender_obj, deselect_remaining=True)

    try:
        # Call the bpy operators for exporting.
        filepath_str = str(file_path)
        if file_path.suffix == ".stl":
            if bpy.app.version[0] >= 4:
                bpy.ops.wm.stl_export(filepath=filepath_str, export_selected_objects=True)
            else:
                bpy.ops.export_mesh.stl(filepath=filepath_str, use_selection=True)
        elif file_path.suffix == ".ply":
            if bpy.app.version[0] >= 4:
                bpy.ops.wm.ply_export(filepath=filepath_str, export_selected_objects=True)
            else:
                bpy.ops.export_mesh.ply(filepath=filepath_str, use_selection=True)
        else:
            raise NotImplementedError(f"{file_path.suffix} currently not supported.")

    finally:
        if delete_object_after:
            remove_objects(blender_obj)
        if len(selected_objects) > 0:
            for selected_obj in selected_objects:
                select_obj(selected_obj, deselect_remaining=False)


def import_mesh_blender(mesh_or_path, name=None):
    """
    Import mesh into blender from ply, stl, or obj file.
    Blender has this weird default with obj's swapping z and y axis, which we adjust.
    """
    if utils.is_existing_path(mesh_or_path):
        mesh_or_path = str(mesh_or_path)
        if mesh_or_path.endswith(".ply"):
            if bpy.app.version[0] >= 4:
                bpy.ops.wm.ply_import(filepath=mesh_or_path)
            else:
                bpy.ops.import_mesh.ply(filepath=mesh_or_path)
        elif mesh_or_path.endswith(".stl"):
            if bpy.app.version[0] >= 4:
                bpy.ops.wm.stl_import(filepath=mesh_or_path)
            else:
                bpy.ops.import_mesh.stl(filepath=mesh_or_path)
        elif mesh_or_path.endswith(".obj"):
            if bpy.app.version[0] >= 4:
               bpy.ops.wm.obj_import(filepath=mesh_or_path, forward_axis="Y", up_axis="Z")
            else:
                print("Trying to import obj. May still not work properly. Check axes forward and up. "
                      "Upgrade to newer Blender could also break this.")
                try:
                    bpy.ops.import_scene.obj(filepath=mesh_or_path, axis_forward="Y", axis_up="Z")
                except AttributeError:
                    bpy.ops.wm.obj_import(filepath=mesh_or_path, forward_axis="Y", up_axis="Z")
            # Apparently, importing an obj like that does not make the imported object active.
            select_object_by_name(Path(mesh_or_path).stem)
        else:
            raise TypeError("Currently, only .ply, .stl, and .obj are supported types for blender imports. "
                            "{} not supported.".format(mesh_or_path))

    elif isinstance(mesh_or_path, Mesh):
        name = "mesh" if not isinstance(name, str) else name
        obj_data = convert_python_to_blender_mesh(mesh_or_path, name)
        # Create Object whose Object Data is our new mesh
        obj = bpy.data.objects.new(name, obj_data)

        # Add *Object* to the scene collection, not the mesh
        bpy.context.collection.objects.link(obj)
        # Make object active
        bpy.context.view_layer.objects.active = obj
    else:
        raise ImportError("Unknown input for blender mesh import {}".format(mesh_or_path))

    obj = get_active_obj()
    if isinstance(name, str):
        change_object_name(new_name=name, obj_or_current_name=obj)

    return obj

def set_to_material_mode(filter_spaces: str = None) -> None:
    """
    Set all views to material mode.
    """
    spaces = get_all_context_spaces(filter_spaces=filter_spaces)
    for space in spaces:
        space.shading.type = 'MATERIAL'

def set_to_solid_mode(filter_spaces: str = None) -> None:
    """
    Set all views to solid mode.
    """
    spaces = get_all_context_spaces(filter_spaces=filter_spaces)
    for space in spaces:
        space.shading.type = "SOLID"

def set_solid_light(solid_light: str = SolidLightType.MATCAP, filter_spaces: str = None) -> None:
    """
    :param solid_light: Choose any from SolidLightType
    """
    spaces = get_all_context_spaces(filter_spaces=filter_spaces)
    for space in spaces:
        space.shading.light = solid_light

def has_textures(obj):
    if not hasattr(obj, "data"):
        raise TypeError("Given obj is not a blender object that contains data.")
    if hasattr(obj.data, "materials") and len(obj.data.materials) > 0:
        for mat_key, mat_value in obj.data.materials.items():
            if mat_value.use_nodes and "Image Texture" in mat_value.node_tree.nodes.keys():
                return True
    return False

def get_nodes(material):
    use_nodes = material.use_nodes
    material.use_nodes = True
    nodes = material.node_tree.nodes
    material.use_nodes = use_nodes
    return nodes


def get_links(material):
    use_nodes = material.use_nodes
    material.use_nodes = True
    links = material.node_tree.links
    material.use_nodes = use_nodes
    return links


def get_node(material, node_name: str, check_contains: bool = False):
    node_name = node_name.lower()
    nodes = get_nodes(material)
    for node in nodes:
        current_node_name = node.name.lower()
        if (check_contains and node_name in current_node_name) or (not check_contains and current_node_name == node_name):
            return node
    return None


def get_image_texture_node(material):
    return get_node(material, node_name="Image Texture", check_contains=False)


def get_output_node(material):
    return get_node(material, node_name="output", check_contains=True)

def get_materials():
    return bpy.data.materials

def get_obj_materials(obj_or_name) -> List[bpy.types.Material]:
    obj_data = get_obj_data(obj_or_name)
    materials = obj_data.materials
    return list(materials)


def get_materials_with_textures():
    all_materials = get_materials()
    tex_materials = []
    for mat in all_materials:
        if get_image_texture_node(mat) is not None:
            tex_materials.append(mat)
    return tex_materials

def get_connected_nodes(material, node):
    links = get_links(material)
    connected_nodes = []

    for link in links:
        if link.from_node == node:
            connected_nodes.append(link.to_node)
    return connected_nodes

def get_texture_connected_node(material):
    tex_node = get_image_texture_node(material)
    if tex_node is None:
        raise AssertionError("Couldn't locate texture node.")

    connected_nodes = get_connected_nodes(material, tex_node)
    if len(connected_nodes) == 0:
        return None
    elif len(connected_nodes) > 1:
        warnings.warn("Multiple nodes connected with texture node. May lead to unexpected behavior.")
    return connected_nodes[0]


def is_node_kind(node, kind: str, check_contains: bool = False):
    kind = kind.lower()
    node_name = node.name.lower()
    return (check_contains and kind in node_name) or (not check_contains and node_name == kind)


def is_mixer_node(node):
    return is_node_kind(node, "mix", check_contains=True)


def is_bsdf_node(node):
    return is_node_kind(node, "bsdf", check_contains=True)


def is_brightcontrast_node(node):
    brightcontrast_name = "Brightness/Contrast" if bpy.app.version[0] >= 4 else "Bright/Contrast"
    return is_node_kind(node, brightcontrast_name, check_contains=False)


def get_texture_material_brightness_mixer(material):
    connected_node = get_texture_connected_node(material)
    if not is_mixer_node(connected_node):
        warnings.warn("Texture node is not connected to mixer.")
        return None
    return connected_node


def get_texture_material_brightnesscontrast_node(material):
    connected_node = get_texture_connected_node(material)
    if not is_brightcontrast_node(connected_node):
        warnings.warn("Texture node is not connected to mixer.")
        return None
    return connected_node

def get_materials_with_texture_connected_to_brightnesscontrast_node():
    tex_materials = get_materials_with_textures()
    tex_materials_with_bc_node = []
    for mat in tex_materials:
        connected_node = get_texture_connected_node(mat)
        if is_brightcontrast_node(connected_node):
            tex_materials_with_bc_node.append(mat)
    return tex_materials_with_bc_node

def adjust_brightcontrast_node(brightcontrast_node, brightness: Union[int, float] = None,
                               contrast: Union[int, float] = None):
    assert is_brightcontrast_node(brightcontrast_node)
    if brightness is not None:
        change_node_value(brightcontrast_node, "Bright", brightness)
    if contrast is not None:
        change_node_value(brightcontrast_node, "Contrast", contrast)

def adjust_texture_brightness_and_contrast(brightness: Union[float, int] = None, contrast: Union[int, float] = None):
    tex_materials = get_materials_with_texture_connected_to_brightnesscontrast_node()
    for mat in tex_materials:
        brightnesscontrast_node = get_texture_connected_node(mat)
        assert is_brightcontrast_node(brightnesscontrast_node)
        adjust_brightcontrast_node(brightnesscontrast_node, brightness=brightness, contrast=contrast)
        if brightness is not None:
            solid_brightness = max(min(1 + brightness, 1), 0)
            mat.diffuse_color = (solid_brightness, solid_brightness, solid_brightness, 1)

def edit_texture_material(material, material_settings: MaterialSettings):
    nodes = get_nodes(material)
    links = get_links(material)
    tex_node = get_image_texture_node(material)
    if tex_node is None:
        raise AssertionError("Couldn't locate texture node.")
    out_node = get_output_node(material)

    connected_texture_node = get_texture_connected_node(material)
    if is_brightcontrast_node(connected_texture_node):
        node_color_brightcontrast = connected_texture_node
        shader_node = get_connected_nodes(material, node_color_brightcontrast)
        if len(shader_node) != 1:
            raise AssertionError(f"{len(shader_node)} nodes connected to brightness/contrast node. "
                                 f"It should be exactly 1!")
        shader_node = shader_node[0]
    else:
        shader_node = connected_texture_node
        node_color_brightcontrast = None
    if not is_bsdf_node(shader_node):
        raise AssertionError("Could not locate shader node. It's neither connected directly to texture node, "
                             f"nor indirectly over brightness/contrast node. The one we found is {shader_node}.")

    if material_settings.kind == "diffuse":
        nodes.remove(shader_node)
        #shader_node.delete()
        shader_node = nodes.new(type='ShaderNodeBsdfDiffuse')
        links.new(tex_node.outputs[0], shader_node.inputs[0])
        links.new(shader_node.outputs[0], out_node.inputs[0])
        if node_color_brightcontrast is not None:
            links.new(node_color_brightcontrast.outputs[0], shader_node.inputs[0])
    elif "emiss" in material_settings.kind.lower() or "shaderless" in material_settings.kind.lower():
        nodes.remove(shader_node)
        #shader_node.delete()
        shader_node = nodes.new(type='ShaderNodeEmission')
        links.new(tex_node.outputs[0], shader_node.inputs[0])
        print("Creating emissive shader")
        if material_settings.alpha < 1:
            print("Alpha is smaller than 1. Creating a mixed shader.")
            node_mix = nodes.new(type='ShaderNodeMixShader')
            node_transparent = nodes.new(type="ShaderNodeBsdfTransparent")
            node_transparent.inputs[0].default_value[3] = 0
            links.new(shader_node.outputs[0], node_mix.inputs[1])
            links.new(node_transparent.outputs[0], node_mix.inputs[2])
            node_mix.inputs[0].default_value = material_settings.alpha
            links.new(node_mix.outputs[0], out_node.inputs[0])
        else:
            links.new(shader_node.outputs[0], out_node.inputs[0])

        if node_color_brightcontrast is not None:
            links.new(node_color_brightcontrast.outputs[0], shader_node.inputs[0])

    if material_settings.allow_brightnesscontrast_change and node_color_brightcontrast is None:
        node_color_brightcontrast = nodes.new(type='ShaderNodeBrightContrast')
        links.new(tex_node.outputs[0], node_color_brightcontrast.inputs[0])
        links.new(node_color_brightcontrast.outputs[0], shader_node.inputs[0])

def edit_texture_materials(material_settings: MaterialSettings):
    tex_materials = get_materials_with_textures()
    for mat in tex_materials:
        edit_texture_material(mat, material_settings=material_settings)

def has_vertex_colors(active_object):
    return get_blender_default_vertex_color_name() in active_object.data.vertex_colors

def quit_blender():
    bpy.ops.wm.quit_blender()

def get_num_vertices(obj_or_data_or_name):
    obj_data = get_obj_data(obj_or_data_or_name)
    return len(obj_data.vertices)

def get_blender_vertex_colors(obj, color_name=None):
    obj_data = get_obj_data(obj)
    vertex_colors_default_name = get_blender_default_vertex_color_name() if color_name is None else color_name
    if vertex_colors_default_name in obj_data.vertex_colors:
        color_layer = obj_data.vertex_colors[vertex_colors_default_name]
        vertex_colors = np.zeros((get_num_vertices(obj), 3))
        # colors in blender are defined per triangle per vertex (so one vertex can have multiple colors, smh)
        i, counts_per_idx = 0, np.zeros(len(vertex_colors), dtype=int)
        for poly in obj_data.polygons:
            for idx in poly.vertices:
                vertex_colors[idx] += np.asarray(color_layer.data[i].color)[:3]
                i += 1
                counts_per_idx[idx] += 1
        vertex_colors /= np.expand_dims(counts_per_idx, axis=1)
        return vertex_colors
    return None

def get_vertices(obj=None, copy: bool = False) -> np.ndarray:
    obj_data = get_obj_data(obj)
    vert_list = [vert.co for vert in obj_data.vertices]
    if copy:
        return np.array(vert_list)
    else:
        return np.asarray(vert_list)

def load_mesh(mesh_or_path, name=None, material_settings: MaterialSettings = None, shade_smooth=False,
              vertex_colors=None, force_new_material=False, vertex_uvs=None):
    """
    Load a mesh into blender and add material based on vertex colors if available.
    :param mesh_or_path: Python mesh or path to it.
    :param name: Optional object name.
    :param palate_material: True:  assigns a shiny material that resembles the palatal skin.
                            False: assigns simple diffuse material.
    :return: blender object (should be the active object after method call)
    """
    obj = import_mesh_blender(mesh_or_path, name=name)

    if shade_smooth:
        for poly in obj.data.polygons:
            poly.use_smooth = True

    if has_textures(obj):
        set_to_material_mode()
        if material_settings is not None:
            edit_texture_materials(material_settings)

    else:
        if vertex_colors is not None:
            set_vertex_colors(vertex_colors, obj.data)
        else:
            if has_vertex_colors(obj):
                set_vertex_colors(get_blender_vertex_colors(obj), obj.data)

        # check if newly loaded mesh has vertex colors and show them if available
        # and not explicitly discouraged by material settings.
        obj_has_vertex_colors = has_vertex_colors(obj)
        if obj_has_vertex_colors and material_settings is None:
            material_settings = MaterialSettings("vertex_colors")

        # If material settings are chosen, assign material to mesh
        if material_settings is not None:
            if "vertex" in material_settings.color and not obj_has_vertex_colors:
                material_settings.color = (0.5, 0.5, 0.5)

            # assign material possibly with vertex colors
            obj.data.materials.clear()
            obj.data.materials.append(get_colored_material(material_settings, force_new_material=force_new_material))
            # set all blender 3D views to material mode, s.t. material with vertex colors is actually displayed
            set_to_material_mode()

    return obj

def toggle_collection_visibility(collection_name, current_override, toggle: bool, extend: bool = False):
    collection_map = {}
    for index, layer_collection in enumerate(bpy.context.view_layer.layer_collection.children):
        collection_map[layer_collection.collection.name] = index
    collection_index = collection_map[collection_name]

    # There's an index shift by 1 inside blender when calling the hide_collection method.
    # Maybe the general Scene Collection has index 0.
    collection_index_shifted = collection_index + 1
    # the extend argument isn't available in older blender versions, e.g. 3.0, so we catch that error
    try:
        blender_call_with_override(bpy.ops.object.hide_collection, current_override,
                                   collection_index=collection_index_shifted, toggle=toggle, extend=extend)
    except TypeError:
        if extend:
            warnings.warn("I think you need to update the blender version to use the extend arg.")
        blender_call_with_override(bpy.ops.object.hide_collection, current_override,
                                   collection_index=collection_index_shifted, toggle=toggle)

def get_3d_view_context_overrides(filter_areas=None, context=None, specific_views=None):
    if specific_views is None:
        view_areas = get_all_areas(AreaType.VIEW_3D, filter_areas=filter_areas, context=context)
    else:
        view_areas = specific_views
    if not isinstance(view_areas, list):
        view_areas = [view_areas]
        return_single_override = True
    else:
        return_single_override = False
    overrides = []
    for area in view_areas:
        for region in area.regions:
            if region.type == 'WINDOW':
                override = bpy.context.copy()
                override['area'] = area
                override['region'] = region
                overrides.append(override)
    if return_single_override:
        if len(overrides) == 0:
            raise ValueError("No window was found in the provided view.")
        overrides = overrides[0]
    return overrides

def get_hidden_vertices(obj=None, return_as_indices: bool = False):
    obj = get_obj_by_name_or_obj(obj) if obj is not None else get_active_obj()
    initial_mode = get_mode(obj)
    set_to_edit_mode(obj)
    me = obj.data
    bm = bmesh.from_edit_mesh(me)
    vertices = bm.verts
    num_vertices = len(vertices)
    hide_indices = [vert.index for vert in vertices if vert.hide]
    set_mesh_to_mode(initial_mode, obj)
    return IndicesAndMasks.get_mask_or_indices(hide_indices, return_as_indices=return_as_indices,
                                               num_indices_or_mesh=num_vertices)

def get_current_vertex_selection(obj=None, return_as_indices: bool = True):
    """
    @return: Current index selection as a numpy array.
    """
    obj = get_obj_by_name_or_obj(obj) if obj is not None else get_active_obj()
    initial_mode = get_mode(obj)
    if initial_mode != "EDIT":
        set_to_edit_mode(obj)
    me = obj.data
    bm = bmesh.from_edit_mesh(me)
    vertices = bm.verts
    num_vertices = len(vertices)
    indices = np.asarray([vert.index for vert in vertices if vert.select])
    if initial_mode != "EDIT":
        set_mesh_to_mode(initial_mode, obj)
    return IndicesAndMasks.get_mask_or_indices(
        indices, return_as_indices=return_as_indices,  num_indices_or_mesh=num_vertices)


def has_vertex_group(obj_or_name, vertex_group_name: Union[str, list]):
    vertex_group_name = [vertex_group_name] if not isinstance(vertex_group_name, list) else vertex_group_name
    obj = get_obj_by_name_or_obj(obj_or_name)
    return any([vertex_group_name_ind in obj.vertex_groups for vertex_group_name_ind in vertex_group_name])

def get_vertex_selection_from_groups(obj_or_name, vertex_group_names: Union[str, list[str]], intersect: bool = False, return_as_indices: bool = True):
    obj = get_obj_by_name_or_obj(obj_or_name)
    vertex_group_names = [vertex_group_names] if not isinstance(vertex_group_names, list) else vertex_group_names
    vertex_indices = []
    for vertex_group_name_ind in vertex_group_names:
        if has_vertex_group(obj, vertex_group_name_ind):
            vertex_group = obj.vertex_groups.get(vertex_group_name_ind)
            vertex_group_index = vertex_group.index
            current_indices = [num_v for num_v, v in enumerate(obj.data.vertices)
                               if vertex_group_index in [vg.group for vg in v.groups]]
            vertex_indices.append(current_indices)
    if len(vertex_indices) == 0:
        return []
    if intersect:
        vertex_indices = IndicesAndMasks.intersect(*vertex_indices)
    else:
        vertex_indices = IndicesAndMasks.join(*vertex_indices)
    return IndicesAndMasks.get_mask_or_indices(vertex_indices, return_as_indices=return_as_indices,
                                               num_indices_or_mesh=len(obj.data.vertices))

def select_vertices_from_groups(obj_or_name, vertex_group_names: Union[str, list], intersect: bool = False,
                                deselect_current_selection: bool = True):
    obj = get_obj_by_name_or_obj(obj_or_name)
    vertex_indices = get_vertex_selection_from_groups(obj, vertex_group_names=vertex_group_names, intersect=intersect)
    select_vertices(vertex_indices, obj, deselect_current_selection=deselect_current_selection,
                    vertex_group_name=None)

def select_vertices(vertex_indices_mask_or_path, obj_or_name, deselect_current_selection=True, vertex_group_name=None,
                    create_new_group=False, add_to_selection: bool = False):
    """
    Select the provided vertex indices for the provided object.
    """
    obj = get_obj_by_name_or_obj(obj_or_name)
    initial_mode = get_mode(obj)
    set_to_edit_mode(obj)
    me = obj.data
    bm = bmesh.from_edit_mesh(me)

    vertices = [e for e in bm.verts]
    num_vertices = len(vertices)

    selection_mask = IndicesAndMasks.get_mask(vertex_indices_mask_or_path, num_indices_or_mesh=num_vertices)

    if add_to_selection:
        selection_mask = IndicesAndMasks.join(
            selection_mask, get_current_vertex_selection(obj, return_as_indices=False), return_as_indices=False)

    for num_vert, vert in enumerate(vertices):
        if deselect_current_selection:
            vert.select_set(selection_mask[vert.index])
        else:
            if selection_mask[vert.index]:
                vert.select_set(True)

    # Don't think this is necessary, but just to be safe
    bmesh.update_edit_mesh(me)
    bm = bmesh.from_edit_mesh(me)
    bm.verts.ensure_lookup_table()

    # (Create and) assign selection to vertex group
    if vertex_group_name is not None:
        if has_vertex_group(obj, vertex_group_name) and not create_new_group:
            group = obj.vertex_groups.get(vertex_group_name)
            obj.vertex_groups.remove(group)
        vg = obj.vertex_groups.new(name=vertex_group_name)

        # assign index selection to vertex group (need to set object to OBJECT mode for this)
        set_mesh_to_mode('OBJECT', obj)
        vg.add(IndicesAndMasks.get_indices(selection_mask).astype(int).tolist(), 1.0, 'REPLACE')

    set_mesh_to_mode(initial_mode, obj)

def change_object_name(new_name, obj_or_current_name=None):
    if obj_or_current_name is not None:
        if isinstance(obj_or_current_name, str):
            deselect_all_objects()
            obj = get_obj_by_name(obj_or_current_name)
        else:
            obj = obj_or_current_name
    else:
        selected_objects = get_all_objects(only_selected=True)
        if len(selected_objects) > 1:
            warnings.warn("More than one object currently selected, which makes renaming not possible.")
            return
        else:
            obj = selected_objects[0]
    obj.name = new_name

def adjust_view(cursor: bool = None, relationship_lines: bool = None, object_origins: bool = None, axes: bool = None,
                floor: bool = None, text: bool = None, cameras: bool = None, lights: bool = None) -> None:
    """
    Adjust elements/overlays to show in 3D View. Set all to False for a minimal version
    :return: None
    """
    if isinstance(lights, bool):
        set_all_lights_render_invisible(not lights)
    if isinstance(cameras, bool):
        set_all_cameras_render_invisible(not cameras)

    spaces = get_all_context_spaces()
    for space in spaces:
        def set_space_overlay(overlay_name: str, overlay_state: bool):
            setattr(space.overlay, f"show_{overlay_name}", overlay_state)

        for local_name, local_value in locals().items():
            if local_name in ["cameras", "lights"]:
                continue
            if isinstance(local_value, bool):
                if local_name == "axes":
                    for axis in ["x", "y", "z"]:
                        set_space_overlay(f"axis_{axis}", local_value)
                else:
                    set_space_overlay(local_name, local_value)

def split_window(area_types: Union[str, List[str]] = "VIEW_3D", direction: str = "VERTICAL", factor: float = 0.5) \
        -> None:
    """
    Split window of specified types.
    :param area_types: str or list of strings specifying all window types to be split. Possible string choices include
                       'VIEW_3D', 'CONSOLE', 'INFO', 'IMAGE_EDITOR' etc. Cf AreaType entries.
    :param direction: 'HORIZONTAL' or 'VERTICAL' (default)
    :param factor: split factor. Default 0.5 means even splitting along specified direction
    todo: There may be some bug in the Blender API, resulting in the split window to have width and height = (0,0),
          which causes further issues when trying to split again. See how this works in newer Blender versions.
    """
    area_types = [area_types] if isinstance(area_types, str) else area_types

    # Store a reference to the current screen
    current_screen = bpy.context.window.screen

    original_areas_ids = [area.as_pointer() for area in current_screen.areas if area.type in area_types]

    for original_area_id in original_areas_ids:
        # Find the area by its unique identifier
        area_to_split = next((area for area in current_screen.areas if area.as_pointer() == original_area_id), None)

        if area_to_split:
            override = bpy.context.copy()
            override['area'] = area_to_split
            blender_call_with_override(bpy.ops.screen.area_split, override, direction=direction, factor=factor)

def close_areas(*areas_to_close, areas_to_keep_open: Union[str, List[str]] = AreaType.VIEW_3D):
    areas_to_keep_open = [areas_to_keep_open] if isinstance(areas_to_keep_open, str) else areas_to_keep_open

    # Store a reference to the current window
    current_window = bpy.context.window

    # Store a reference to the current screen
    current_screen = bpy.context.window.screen

    while True:
        # Search for the next area to close
        area_to_close = next((area for area in current_screen.areas if
                              (len(areas_to_close) > 0 and area.type in areas_to_close) or
                              area.type not in areas_to_keep_open), None)

        # If we found an area to close, close it
        if area_to_close:
            override = {'window': current_window, 'screen': current_screen, 'area': area_to_close}
            blender_call_with_override(bpy.ops.screen.area_close, override)
        else:
            # If there are no more areas to close, break out of the loop
            break

def turn_off_floor():
    for area in bpy.context.screen.areas:
        if area.type == 'VIEW_3D':
            for space in area.spaces:
                if space.type == 'VIEW_3D':
                    space.overlay.show_floor = False

def turn_off_direction_lines():
    for area in bpy.context.screen.areas:
        if area.type == 'VIEW_3D':
            for space in area.spaces:
                if space.type == 'VIEW_3D':
                    space.overlay.show_axis_x = False
                    space.overlay.show_axis_y = False
                    space.overlay.show_axis_z = False

def turn_off_floor_and_direction_lines():
    turn_off_floor()
    turn_off_direction_lines()