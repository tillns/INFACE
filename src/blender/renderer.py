"""

This is a blender script that can only be run from blender. It should be called by calling
blender from the command line and passing the path to this file as additional argument:
blender --python PATH/TO/src/blender/renderer.py -- --mesh_paths /PATH/TO/SOME/MESH_FILE.ply/obj/stl
Use --help as additional argument to get info about all arguments.
Alternatively, you can also use the python interface where you call the function render_image()

The purpose of this script is to render one or multiple meshes from python, offering
some limited options for the camera, lighting, and materials, including optional transparency.


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

# standard imports
from argparse import ArgumentParser
import sys
import os
from pathlib import Path

# blender imports
import bpy
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

# blender utils import
from src.blender import blender_utils


def main():
    argv = sys.argv
    if "--" not in argv:
        argv = []
    else:
        argv = argv[argv.index("--") + 1:]
    parser = ArgumentParser("You can run blender from CLI with this script: "
                            "blender [blend_file] --python " + __file__ + " -- [options]")
    parser.add_argument('--mesh_paths', type=str, required=True, nargs="+",
                        help="Path to meshes to be rendered.")
    parser.add_argument('--mesh_alphas', type=float, default=None, nargs="+",
                        help="Alpha transparency for each mesh. 1 by default.")
    parser.add_argument('--mesh_cast_shadows', type=str, default=None, nargs="+",
                        help="Whether to cast shadows or not for each mesh. "
                             "Give a string like F/false/none or something to disable shadows for a specific mesh.")
    parser.add_argument('--camera_location', required=True, type=float, nargs=3,
                        help="Three coordinates for camera position.")
    parser.add_argument('--camera_rotation_euler_rad', required=True, type=float, nargs=3,
                        help="Three angles (radian), specifying the Euler rotation of the camera around x, y, and z.")
    parser.add_argument('--light_location', default=None, type=float, nargs=3,
                        help="Three coordinates for light position. "
                             "By default, the light is positioned the same as the camera.")
    parser.add_argument('--light_rotation_euler_rad', default=None, type=float, nargs=3,
                        help="Three angles (radian), specifying the Euler rotation of the light around x, y, and z. "
                             "By default, the light is rotated the same as the camera.")
    parser.add_argument('--camera_focal_length', default=None, type=float,
                        help="Camera focal length in mm.")
    parser.add_argument('--camera_sensor_width', default=None, type=float,
                        help="Camera sensor width in mm.")
    parser.add_argument('--resolution', default=[1000, 1000], type=int, nargs=2,
                        help="Two integers, specifying the resolution of the final rendered image. "
                             "Default is 1000x1000.")
    parser.add_argument('--default_uniform_color', default=[0.7, 0.7, 0.7], type=float, nargs=3,
                        help="If the mesh doesn't have any colors, a uniform vertex color is assigned, "
                             "which you can optionally specify here via three numbers (floats between 0 and 1).")
    parser.add_argument('--light_strength', default=None, type=float,
                        help="Light strength (Blender setting).")
    parser.add_argument('--light_type', default=None, type=str,
                        help="Light type. Can be SUN, POINT, SPOT, or AREA.")
    parser.add_argument('--output_image_path', type=str, required=True,
                        help="Path where to save the rendered image.")
    parser.add_argument('--render_engine', type=str, default=None,
                        help="Choose Cycles or Eevee")
    parser.add_argument('--switch_sensor_fit', default=False, action="store_true",
                        help="For non-square resolutions, you can choose this to switch Blender's camera's sensor fit.")
    parser.add_argument('--material_settings', type=str, default=None, nargs="+",
                        help="Optionally choose some material settings for each mesh. Currently, we have diffuse "
                             "shading by default, but support a specular setting when you choose the string "
                             "'specular' here.")
    args = parser.parse_args(argv)

    blender_utils.remove_objects()
    mesh_objs = []
    is_transparent = False
    if len(args.mesh_paths) == 0:
        raise AssertionError("Provide at least one mesh path.")

    # Import the meshes and assign them proper materials
    for num_mesh, mesh_path in enumerate(args.mesh_paths):
        if args.mesh_alphas is None:
            alpha = 1
        else:
            if len(args.mesh_alphas) <= num_mesh:
                alpha = 1
            else:
                alpha = args.mesh_alphas[num_mesh]
        if alpha < 1:
            is_transparent = True

        additional_mat_setting = {}
        if args.material_settings is not None and len(args.material_settings) > num_mesh:
            current_mat_setting = args.material_settings[num_mesh].lower()
            if "shiny" in current_mat_setting or "spec" in current_mat_setting:
                additional_mat_setting = dict(kind="PBSDF", shader_adjust_settings={"Specular": 0.622, "Roughness": 0.283})
            elif "shaderless" in current_mat_setting:
                additional_mat_setting = dict(kind="shaderless")
        mesh_obj = blender_utils.load_mesh(mesh_path, shade_smooth=True)
        material_default_settings = dict(alpha=alpha)
        material_settings_dict = dict(**material_default_settings, **additional_mat_setting)
        # We currently assume that textured meshes shall not be lit. Change if necessary.
        if blender_utils.has_textures(mesh_obj):
            material_settings_dict.setdefault("kind", "shaderless")
        elif blender_utils.has_vertex_colors(mesh_obj):
            material_settings_dict.setdefault("color", "vertex_colors")
        else:
            material_settings_dict.setdefault("color", args.default_uniform_color)
        material_settings = blender_utils.MaterialSettings(**material_settings_dict)
        blender_utils.assign_material_to_object(blender_utils.get_colored_material(material_settings), mesh_obj, remove_current_materials=True)

        mesh_objs.append(mesh_obj)
        if args.mesh_cast_shadows is not None:
            if len(args.mesh_cast_shadows) > num_mesh:
                if "f" or "n" in args.mesh_cast_shadows[num_mesh].lower():
                    mesh_obj.visible_shadow = False

    scene = bpy.context.scene

    # Set up camera
    camera_obj = scene.camera
    cam_data = camera_obj.data
    if args.camera_focal_length is not None:
        cam_data.lens = args.camera_focal_length
    if args.camera_sensor_width is not None:
        cam_data.sensor_width = args.camera_sensor_width
    cam_data.clip_start = 0.01
    cam_data.clip_end = 10000
    if args.switch_sensor_fit:
        if args.resolution[0] > args.resolution[1]:
            cam_data.sensor_fit = 'VERTICAL'
        else:
            cam_data.sensor_fit = 'HORIZONTAL'
    # Set camera location and rotation
    camera_obj.location = args.camera_location
    camera_obj.rotation_euler = args.camera_rotation_euler_rad

    # Set up light
    blender_utils.remove_lights()
    if args.light_type is None:
        light_type = "SUN"
    else:
        light_type = args.light_type

    if args.light_strength is None:
        light_strength = 10 if is_transparent else 1
        if light_type.lower() != "sun":
            light_strength *= 1000000
    else:
        light_strength = args.light_strength

    light_loc = args.camera_location if args.light_location is None else args.light_location
    light_rot = args.camera_rotation_euler_rad if args.light_rotation_euler_rad is None else args.light_rotation_euler_rad
    # We move light to same position and with same rotation as camera, such that an object is evenly lit
    blender_utils.add_light(light_type, location=tuple(light_loc), rotation=tuple(light_rot), strength=light_strength)

    # Set render resolution
    scene.render.resolution_x = args.resolution[0]
    scene.render.resolution_y = args.resolution[1]
    scene.render.resolution_percentage = 100

    # Enable transparent background
    scene.render.film_transparent = True

    # Render settings
    scene.render.image_settings.file_format = 'PNG'  # Set output format to PNG to support transparency
    scene.render.image_settings.color_mode = 'RGBA'  # Include alpha channel

    # Set output path
    scene.render.filepath = args.output_image_path

    # Choose render engine. By default, we use EEVEE in case we don't use transparency, and only if we do
    # use it, we choose CYCLES
    if args.render_engine is None:
        if is_transparent:
            bpy.context.scene.render.engine = 'CYCLES'
        else:
            bpy.context.scene.render.engine = 'BLENDER_EEVEE'
    else:
        if "cycl" in args.render_engine.lower():
            bpy.context.scene.render.engine = 'CYCLES'
        else:
            bpy.context.scene.render.engine = 'BLENDER_EEVEE'

    # Render the scene
    bpy.ops.render.render(write_still=True)

    # Makes things easier when testing cam and lighting
    blender_utils.turn_off_floor_and_direction_lines()


if __name__ == "__main__":
    main()
