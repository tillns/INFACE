"""
Interface to start blender scripts from python.

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
import pkgutil
import subprocess
from typing import Union, List, Callable
from pathlib import Path
import platform
from src import utils

def get_platform_formatted_path(my_path: Union[str, Path, int, float]) -> str:
    """
    Numbers are only converted to string, whereas paths are formatted for the respective system.
    Windows is handled separately to Linux and MaxOS.
    """
    if isinstance(my_path, (float, str)):
        return str(my_path)
    my_path = str(my_path)
    if Path(my_path).exists():
        if platform.system() == "Windows":
            if not my_path.startswith("\""):
                my_path = f"\"{my_path}"
            if not my_path.endswith("\""):
                my_path = f"{my_path}\""
        else:
            if my_path.startswith("\""):
                my_path = my_path[1:]
            if my_path.endswith("\""):
                my_path = my_path[:-1]
    return my_path

def general_blender_interface(
        tool_path: Union[str, Path, Callable, None] = None, blend_path: Union[str, Path, None] = None,
        background: bool = False, **cli_args: Union[int, float, str, Path, None, List]) -> None:
    """

    @param tool_path: Possibilities: 1) Relative path to tool module within project (e.g. src.utils.blender_arap)
                                     2) Absolute path to python script.
                                     3) Callable function that returns a path to a blender file and the absolute path
                                        to python script (or None for one of them or both).
                                     4) None
    @param blend_path: Path to blender file. A good python script makes this unnecessary. :)
    @param input_arguments: Possibilities: 1) {"argument_name": argument_value or list of argument_values, where one
                                                                argument_value can be a number, bool, or a path.}
                                              You can have 0 to as many entries as necessary.
                                           2) None. Input arguments are then skipped.
    @param background: Set to True to run blender script in the background, so to prevent opening a blender window.
    @return: None
    """
    from config import blender_executable
    blend_file_path, python_script_path = blend_path, None

    # tool_path can be callable method, direct path, or module string
    if callable(tool_path):
        blend_file_path, python_script_path = tool_path()
    elif utils.is_existing_path(tool_path):
        python_script_path = tool_path
    elif isinstance(tool_path, str):
        # only way I could find to get the file path of another module without having to import it
        # (it cannot be imported from outside blender, because it contains blender modules, such as bpy)
        python_script_path = pkgutil.get_loader(tool_path).path

    # basic blender command simply calls blender executable
    blender_command = get_platform_formatted_path(blender_executable)

    if background:
        blender_command += " -b"

    # blender file may be used if available
    if blend_file_path is not None:
        blender_command += f" {get_platform_formatted_path(blend_file_path)}"

    # python script used if available
    if python_script_path is not None:
        blender_command += f" --python {get_platform_formatted_path(python_script_path)}"

        # Add arguments if available
        if len(cli_args) > 0:
            blender_command += " --"
            # input arguments are properly parsed for blender CLI arguments
            for argument_key, argument_values in cli_args.items():
                # None and False values are assumed to be meant to be skipped.
                if argument_values is None or (isinstance(argument_values, list) and len(argument_values) == 0):
                    continue
                if isinstance(argument_values, bool) and not argument_values:
                    continue
                blender_command += " --{}".format(argument_key)
                # None is for action_store=True arguments that still need to be specified but without any entries.
                if isinstance(argument_values, bool):
                    continue
                argument_paths = [argument_values] if not isinstance(argument_values, list) else argument_values
                for argument_path in argument_paths:
                    blender_command += f" {get_platform_formatted_path(argument_path)}"
    if platform.system() == "Windows":
        subprocess.call(blender_command, shell=True)
    else:
        blender_command = blender_command.split(" ")
        subprocess.call(blender_command)

def set_landmarks(
        *mesh_paths: Union[str, Path],
        input_landmarks_path: Union[str, Path, None] = None, output_landmarks_path: Union[str, Path, None] = None,
        num_landmarks: Union[int, None] = None,
        output_region_paths: Union[str, Path, List[Union[str, Path]]] = None,
        save_continuously: bool = False,
        num_window_splits: int = 0, quit_blender_after_landmark_export: bool = False) -> None:
    """
    Set landmarks on one or multiple meshes via Blender GUI.
    :param mesh_paths: One or more mesh paths to be landmarked in Blender. In each view, you can
                       switch between all the provided meshes. The meshes should probably correlate in some way,
                            since you can still only define a single set of landmarks.
    :param input_landmarks_path: Path to landmarks file to be load at start.
    :param output_landmarks_path: Path where to save the landmarks after defining them in Blender.
                                  Blender asks about path if it's not provided.
    :param output_region_paths: One path or a list of paths specifying the segmentation to be saved for each mesh.
                                If not provided, the Blender tool will ask you where to save it.
    :param num_landmarks: Number of landmarks used as sanity check if set landmarks are correct,
                          also defines color coding.
    :param save_continuously: Save each landmark as soon as you set it rather than
                              only saving them after pressing Export.
    :param num_window_splits: Number of times to split the Blender views. 0 means you don't split, so you have one view;
                              1 means you split once, giving you two views, etc. May not work very well above 1.
    :param quit_blender_after_landmark_export: Set True to quit Blender window immediately after exporting the
                                               landmarks.
    :return: None
    """
    assert len(mesh_paths) > 0, "You must provide at least one mesh path"

    return general_blender_interface(
        "src.blender.landmarking",
        mesh_paths=[*mesh_paths],
        input_landmarks_path=input_landmarks_path, output_landmarks_path=output_landmarks_path,
        num_landmarks=num_landmarks, output_region_paths=output_region_paths,
        save_continuously=save_continuously,
        num_window_splits=num_window_splits, quit_blender_after_landmark_export=quit_blender_after_landmark_export)

def visualize_morphable_model(
        path_to_morphable_model: Union[str, Path], min_val: float = -3, max_val: float = 3,
        components: Union[List[int], int, None] = None,
        linear_regressors: Union[Path, str, List[Union[Path, str]]] = None,
        translated_morphable_models: Union[Path, str, List[Union[Path, str]]] = None,) -> None:
    """
    Visualize the variations encoded in the components of a morphable model in a Blender GUI.
    :param path_to_morphable_model: Path to morphable model to be visualized.
    :param min_val: Extent of the sliders into negative (left) direction (relative to standard deviation).
    :param max_val: Extent of the sliders into positive (right) direction (relative to standard deviation).
    :param components: List of component indices or number of components to be display sliders for.
    :param linear_regressors: Optionally provide one or more paths to linear regressors, which are used
                              to translate the code of the morphable model into codes of other morphable models.
                              If you provide this, you also need to provide paths to the respective morphable models
                              you translate the code to.
    :param translated_morphable_models: Provide this along with the linear regressors. In the GUI, all meshes will
                                        be displayed from all the given morphable models.
    :return: None
    """

    general_blender_interface(
        "src.blender.morphable_model_visualizer",
        path_to_morphable_model=path_to_morphable_model, min_val=min_val, max_val=max_val,
        components=components, linear_regressors=linear_regressors,
        translated_morphable_models=translated_morphable_models
    )

def measurement_correction_interface(
        head_shape_factor_function: Union[str, Path], head_model: Union[str, Path],
        registered_mesh_path: Union[str, Path, None] = None) -> None:
    """
    Visualize the Blender GUI in which you can change attributes correlated with the morphable model space.
    :param head_shape_factor_function: Path to the json that includes the linear correlation weights for each attribute.
    :param head_model: Path to the linear head model the attributes were correlated with.
    :param registered_mesh_path: Optionally provide a path to some registered mesh. If not provided, the average
                                 mesh will be displayed in the GUI, but you can also import meshes from the GUI.
    :return: None
    """

    general_blender_interface(
        "src.blender.measurement_correction_GUI", head_shape_factor_function=head_shape_factor_function,
        head_model=head_model, registered_mesh_path=registered_mesh_path)


def render_image(*mesh_paths, camera_location: List, camera_rotation_euler_rad: List, output_image_path: Union[str, Path],
                 camera_focal_length: float = None, camera_sensor_width: float = None,
                 light_location: List = None, light_rotation_euler_rad: List = None,
                 resolution: List[int] = None, mesh_alphas: List[float] = None, mesh_cast_shadows: List[Union[str, bool]] = None,
                 default_uniform_color: List[float] = None, light_strength: float = None,
                 light_type: str = None, material_settings: List[str] = None,
                 render_engine: Union[str, None] = None, test_cam: bool = False, switch_sensor_fit: bool = False
                 ) -> None:
    """
    Provide one or more meshes to render in Blender, along with the camera location.
    To choose a good camera setting, you can use some predefined angles via mesh_render.get_camera_loc_and_rot(),
    or you can easily choose a custom setting like so:
        - align the camera with the viewport via control + option/alt(gr) + Numpad 0 (Preferences -> Input -> Emulate Numpad)
        - attach the camera to the viewport via side panel (N), under View, Camera to View
    To print camera pos and loc from blender, use the following code snippet:
import bpy
cam = bpy.context.scene.camera
loc = cam.location
rot = cam.rotation_euler
print("Camera Location: ({:.3f}, {:.3f}, {:.3f})".format(*loc))
print("Camera Rotation (Radians): ({:.3f}, {:.3f}, {:.3f})".format(*rot))

    For further documentation of the arguments, see the --help message of src.blender.renderer.py.
    """

    return general_blender_interface(
        "src.blender.renderer", background=not test_cam,
        mesh_paths=list(mesh_paths), output_image_path=output_image_path, mesh_alphas=mesh_alphas,
        mesh_cast_shadows=mesh_cast_shadows, camera_location=camera_location, resolution=resolution,
        camera_rotation_euler_rad=camera_rotation_euler_rad, default_uniform_color=default_uniform_color,
        camera_focal_length=camera_focal_length, camera_sensor_width=camera_sensor_width, light_strength=light_strength,
        light_type=light_type, material_settings=material_settings, render_engine=render_engine,
        light_location=light_location, light_rotation_euler_rad=light_rotation_euler_rad, switch_sensor_fit=switch_sensor_fit,
    )