"""
This file offers a Visualizer class, using Open3D to visualize one or several meshes.
Additionally, a visualization_thread() method can be used to continuously run a visualization window
while updating the rendered geometry from a method the thread runs in parallel. This is useful, for instance,
in the registration file, where a model/template can be fit to a target mesh, and the intermediate updates
can be visualized with this method.

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
import threading
import time
from copy import deepcopy
from pathlib import Path
from typing import Union, Callable, List
import open3d
import numpy as np
import trimesh

from src.objects.mesh import Mesh, Landmarks, CurvilinearFeatures


class Visualizer:
    def __init__(self, geometries=None, width: int = 1920, height: int = 1080,
                 switch_geometries: Union[List, List[List], None] = None,
                 color_mode="texture", **render_kwargs):
        self.vis = open3d.visualization.VisualizerWithKeyCallback()
        self.vis.create_window(width=width, height=height, left=0, top=0)
        self.switch_index, self.color_mode = 0, color_mode
        self.add_geometries(geometries, switch_geometries=switch_geometries, first_call=True)


        # Set the render options.
        self.render = self.vis.get_render_option()
        self.render.mesh_show_wireframe = render_kwargs.get("mesh_show_wireframe", True)
        self.render.mesh_show_back_face = render_kwargs.get("mesh_show_back_face", True)
        self.render.light_on = render_kwargs.get("light_on", True)
        self.render.mesh_shade_option = open3d.visualization.MeshShadeOption(int(render_kwargs.get(
            "mesh_shade_option", render_kwargs.get("smooth_shading", True))))

        def switch_background_color(vis, action, mods):
            if action == 1:
                self.render.background_color = np.clip(np.asarray([1, 1, 1]) - self.render.background_color, 0, 1)
                self.show_geometries()
            return False

        def hide_geom(vis, num_geom):
            if num_geom < len(self.hide_geometries):
                self.hide_geometries[num_geom] = not self.hide_geometries[num_geom]
                self.show_geometries()

        def add_to_switch_index(vis, action, mods, num_to_add):
            if action == 1:
                if isinstance(self.switch_geometries, (list, tuple)) and len(self.switch_geometries) > 0:
                    self.switch_index = (self.switch_index + num_to_add) % len(self.switch_geometries)
                    self.show_geometries()
                    return True
            return False

        def switch_color_mode(vis, action, mods):
            if action == 1:
                if self.color_mode == "texture":
                    self.set_color_mode("vertex")
                elif self.color_mode == "vertex":
                    self.set_color_mode("gray")
                elif self.color_mode == "gray":
                    self.set_color_mode("texture")
                return True
            return False

        from functools import partial
        hide_geoms = {i: partial(hide_geom, num_geom=i) for i in range(10)}

        # Switch background color with K, flip mesh with F
        self.vis.register_key_action_callback(ord("K"), switch_background_color)
        self.vis.register_key_action_callback(ord("C"), switch_color_mode)
        # Switch geometries by pressing arrow key back for back switching and arrow key forward for forth switching
        # I don't know if the numbers 262 and 263 correspond to these numbers on every OS,
        # but it worked on Mac and Ubuntu.
        if len(self.switch_geometries) > 0:
            self.vis.register_key_action_callback(263, partial(add_to_switch_index, num_to_add=-1))
            self.vis.register_key_action_callback(262, partial(add_to_switch_index, num_to_add=1))
        for num_hide, hide_geom_func in hide_geoms.items():
            self.vis.register_key_callback(ord(str(num_hide + 1) if num_hide < 9 else str(0)), hide_geom_func)

    @staticmethod
    def get_colors(geometry: open3d.geometry.Geometry) -> np.ndarray:
        if isinstance(geometry, open3d.geometry.TriangleMesh):
            colors = geometry.vertex_colors
        else:
            colors = geometry.colors
        return np.array(colors)

    def set_color_mode(self, color_mode):
        self.color_mode = color_mode

        # Assign colors per vertex to open3D geometry, also supporting point clouds.
        def assign_colors(current_geometry: open3d.geometry.Geometry, current_colors: np.ndarray):
            current_colors = current_colors[:, :3]
            if isinstance(geometry, open3d.geometry.TriangleMesh):
                current_geometry.vertex_colors = open3d.utility.Vector3dVector(current_colors)
            else:
                current_geometry.colors = open3d.utility.Vector3dVector(current_colors)

        # Distinguish our three different color rendering modes
        if self.color_mode == "vertex":
            for geometry, geometry_vertex_colors in zip(self.all_geometries, self.geometry_vertex_colors):
                if hasattr(geometry, "textures"):
                    geometry.textures = []
                assign_colors(geometry, geometry_vertex_colors)

        elif self.color_mode == "gray":
            for geometry in self.all_geometries:
                if hasattr(geometry, "textures"):
                    geometry.textures = []
                if hasattr(geometry, "vertex_colors"):
                    geometry.vertex_colors = open3d.utility.Vector3dVector([])
        elif self.color_mode == "texture":
            for geometry, geometry_textures, geometry_vertex_colors in zip(self.all_geometries, self.geometry_textures, self.geometry_vertex_colors):
                if hasattr(geometry, "textures") and len(geometry_textures) > 0:
                    geometry.textures = deepcopy(geometry_textures)
                else:
                    assign_colors(geometry, deepcopy(geometry_vertex_colors))
        self.show_geometries()

    def get_open3d_geometry(
            self, geometry: Union[Mesh, open3d.geometry.Geometry, trimesh.Trimesh, Landmarks, CurvilinearFeatures, np.ndarray, Path, str, None],
            copy: bool = True) -> open3d.geometry.Geometry:
        if geometry is not None:
            if isinstance(geometry, (str, Path)):
                geometry_path = Path(geometry)
                if geometry_path.suffix == ".csv":
                    geometry = Landmarks.load(geometry_path)
                else:
                    geometry = Mesh.load(geometry)
            if isinstance(geometry, trimesh.Trimesh):
                geometry = Mesh.get_from_trimesh_mesh(geometry)
            elif isinstance(geometry, np.ndarray):
                geometry = open3d.geometry.PointCloud(points=open3d.utility.Vector3dVector(geometry))
            if isinstance(geometry, (Landmarks, CurvilinearFeatures)):
                geometry = geometry.get_mesh()
            if not isinstance(geometry, open3d.geometry.Geometry):
                geometry = geometry.get_open3d_mesh(compute_normals=True, copy=False, use_legacy=False)
        if copy:
            return deepcopy(geometry)
        else:
            return geometry

    def get_open3d_geometries(self, geometries):
        open3d_geometries = []
        for num_geom, geometry in enumerate(geometries if isinstance(geometries, (list, tuple)) else [geometries]):
            if isinstance(geometry, (list, tuple)):
                open3d_geometries.append(self.get_open3d_geometries(geometry))
            else:
                open3d_geometry = self.get_open3d_geometry(geometry, copy=True)
                if open3d_geometry is not None:
                    open3d_geometries.append(open3d_geometry)
        return open3d_geometries

    def add_geometries(self, geometries=None, hide_geometries=None, switch_geometries=None, first_call=False):
        # add the different geometries to the renderer.
        self.geometries = [] if geometries is None else self.get_open3d_geometries(geometries)

        # This is just a generous upper bound, because dynamically computing the number of geometries to
        # hide/show when also incorporating switch_geometries is just too difficult
        num_hide_geometries = 100

        # Possibly hide geometries
        if not isinstance(hide_geometries, (list, tuple)):
            hide_all_geometries = hide_geometries if isinstance(hide_geometries, bool) else False
            self.hide_geometries = [hide_all_geometries for _ in range(num_hide_geometries)]
        else:
            if len(hide_geometries) >= num_hide_geometries:
                self.hide_geometries = hide_geometries[:num_hide_geometries]
            else:
                self.hide_geometries = hide_geometries + [False for _ in
                                                          range(num_hide_geometries - len(hide_geometries))]

        # Switch geometries is a separate list of geometries between which the user can switch via two keys, i.e.,
        # only one antry (possibly containing multiple geometries) is shown at a time.
        self.switch_geometries = [] if switch_geometries is None else self.get_open3d_geometries(switch_geometries)

        # We keep track of all textures and vertex colors. They can be switched by the user via the key 'c',
        # cf. method set_color_mode(). self.all_geometries is linearized, so no sublist of switch geometries is used,
        # because we merely require a list of pointers to the meshes used in order to change all of their textures.
        self.all_geometries = [*self.geometries]
        for switch_geometry in self.switch_geometries:
            if isinstance(switch_geometry, (list, tuple)):
                self.all_geometries.extend(list(switch_geometry))
            else:
                self.all_geometries.append(switch_geometry)
        self.geometry_textures = [deepcopy(geometry.textures) if hasattr(geometry, "textures")
                                  else self.get_colors(geometry) for geometry in self.all_geometries]
        self.geometry_vertex_colors = [self.get_colors(geometry) for geometry in self.all_geometries]


        # Render the geometries in the currently set color mode.
        self.set_color_mode(self.color_mode)
        self.show_geometries(first_call=first_call)

    def show_geometries(self, first_call=False):
        self.vis.clear_geometries()

        # Add geometries from self.geometries and currently chosen self.switch_geometries entry.
        # Geometries may be hidden according to their index.

        def add_geometry(current_geometry):
            try:
                self.vis.add_geometry(current_geometry, reset_bounding_box=first_call)
            except TypeError:
                print("{} not supported.".format(current_geometry))

        # First, add all normal geometries
        for num_geom, geometry in enumerate(self.geometries):
            if not self.hide_geometries[num_geom]:
                add_geometry(geometry)

        # Next, add all switch geometries
        if isinstance(self.switch_geometries, (list, tuple)) and len(self.switch_geometries) > self.switch_index:
            switch_geometry = self.switch_geometries[self.switch_index]
            # Support for multiple geometries per switch geometry entry
            if isinstance(switch_geometry, (list, tuple)):
                for num_geom, switch_geometry_ind in enumerate(switch_geometry, start=len(self.geometries)):
                    if not self.hide_geometries[num_geom]:
                        add_geometry(switch_geometry_ind)
            # Also support single geometry element per switch geometry entry
            else:
                if not self.hide_geometries[len(self.geometries)]:
                    add_geometry(switch_geometry)


    def update_frame(self):
        self.vis.poll_events()
        self.vis.update_renderer()

    def get_render_args(self):
        render_args_current = {key: getattr(self.render, key) for key in self.render_args.keys()}
        smooth_shading_current = self.render.mesh_shade_option == open3d.visualization.MeshShadeOption(1)
        return render_args_current, smooth_shading_current

    def run(self, camera_params=None, interactive_window: bool = True):
        # Set the initial view parameters. The open3d default parameters are used
        # if use_base_view is set to False and no other view parameters are provided.
        ctrl = self.vis.get_view_control()
        if camera_params is not None:
            # camera setting can lead to a warning if the width and height from camera_params.intrinsic
            # don't match with the window width and height. This should only happen if a fullscreen
            # is set differently to the setting saved in camera_params.
            # If this turns out to be a problem, think of a solution.
            ctrl.convert_from_pinhole_camera_parameters(camera_params)

        with threading.Lock():
            if interactive_window:
                # print info for visualizer
                print("Controls: press 1–9 to toggle meshes 1–9, 0 for the 10th mesh.")
                if len(self.geometries) > 10:
                    print(f"{len(self.geometries) - 10} meshes beyond the 10th are always shown.")

                if self.switch_geometries is not None and len(self.switch_geometries) > 0:
                    print("Switch between the provided meshes using the arrow keys.")

                self.vis.run()
                self.vis.destroy_window()

            # render last frame of window (or just the headless setting)
            self.update_frame()

    @staticmethod
    def visualization_thread(method: Callable, method_args: dict = None, fps: Union[int, float] = 24,
                             close_automatically: bool = False, **render_kwargs):
        """
        This method runs an open3d window for visualization in a while loop on the main thread,
        while running the provided method on a parallel thread.
        Make sure the provided method doesn't use visualizations like that, too,
        because openGL must run on the main thread.
        The method is provided with an argument called 'visualize_geometries', which is a callable method
        that takes geometries (meshes) as input. This updates the geometry list on the main thread that visualizes
        all geometries in that list. You can use that to continuously visualize geometry updates as the method is running.
        In case close_automatically is set to True, the method must also accept a parameter called 'event'.
        The method should then terminate in case event.is_set().

        @param method: Callable method. Runs on parallel thread.
        @param method_args: Optional arguments for the method (dictionary).
                            The argument 'visualize_geometries' is added automatically.
        @param fps: How often the visualization window is updated per second.
        @param close_automatically: Whether to close visualization window and thus terminate function automatically
                                    when provided method is finished or to wait for the user to manually close the window.

        @return What provided method returns.
        """
        method_args = {} if not isinstance(method_args, dict) else method_args
        geometry_list = []
        lock = threading.Lock()
        vis = Visualizer(**render_kwargs)

        def visualize_geometries(*my_geometries, **_):
            with lock:
                geometry_list.clear()
                geometry_list.extend(list(my_geometries))

        method_args["visualize_geometries"] = visualize_geometries

        # additional method required that's called via a thread and sets a non_local variable,
        # s.t. method return can be kept
        method_return = None

        event = threading.Event()

        def thread_with_return(method, args):
            nonlocal method_return
            method_return = method(**args)

        method_thread = threading.Thread(target=thread_with_return, args=[method, method_args])
        method_thread.start()

        geometries = []

        ctrl, camera_params, first_geom_set, print_finish_message = vis.vis.get_view_control(), None, False, True
        was_closed = False
        while True:
            time.sleep(1 / fps)
            change = False
            with lock:
                if not len(geometry_list) == 0:
                    change = True
                    if not len(geometries) == 0:
                        vis.vis.clear_geometries()
                    geometries = list(geometry_list)
                    geometry_list.clear()
            if change:
                vis.add_geometries(geometries, first_call=not first_geom_set,
                                   hide_geometries=vis.hide_geometries
                                   if len(geometries) == len(vis.geometries) else None)

                first_geom_set = True
                if camera_params is not None:
                    ctrl.convert_from_pinhole_camera_parameters(camera_params)

            vis.run(interactive_window=False, camera_params=camera_params)
            if not vis.vis.poll_events():
                was_closed = True
            if not method_thread.is_alive():
                if close_automatically:
                    vis.vis.close()
                # Check if the window was closed and terminates loop if it was.
                if was_closed:
                    break
                if print_finish_message:
                    print("Method is finished. Close window to finish program execution.")
                    print_finish_message = False
            if first_geom_set:
                camera_params = deepcopy(ctrl.convert_to_pinhole_camera_parameters())
        return method_return