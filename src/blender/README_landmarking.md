<p align="center">
<img src="images/landmarking_tool_preview.png">
</p>

We describe here the landmarking tool that I (Till Schnabel) developed throughout my PhD and Postdoc at ETH Zurich.
It is meant to make the effort of defining landmarks and segmenting artifacts on a large number of scans 
as easy and efficient as possible. If you have any questions, feel free to contact me via 
[email](till@familie-schnabel.ch).

# Getting Started
## Downloading Blender
Assuming you've already downloaded this repository, you now need to download [Blender](https://www.blender.org). 
Even though we tried to make this code very generic, we cannot guarantee that it will be 
compatible with all future versions of Blender, which is why we recommend 
[Blender 3.1](https://download.blender.org/release/Blender3.1). 
Feel free to try other versions; Blender versions in 4 and 5 also seemed to be working in our short tests.
No additional libraries should be required to run this tool, i.e., even though other code in this repository
may require libraries like torch, Open3D or trimesh, if you only want to use this tool, you don't need to set up
any virtual environment.
We also recommend changing the following options in Blender (Edit -> Preferences):
1. Interface -> Splash Screen (uncheck)
2. Navigation -> Trackball (choose instead of Turntable) + Orbit Around Selection (check)
3. Input -> Emulate 3 Button Mouse (check)

## Finding the Blender Executable
This step might not be necessary. Feel free to first go to the next step and only come back to this one
if you get an ImportError about setting the executable in a config file.
If you want to start Blender from this repo, e.g., if you want to use
[`landmarking_batch.py`](../../src/processing/landmarking_batch.py) script to go through a whole dataset,
you might need to specify the location of the Blender executable in case it's not at a common location. 
For that, check where you downloaded Blender to
your system, look inside the folder and locate the executable file. Copy the path to that executable.
Now go to [`config.py`](../../config.py) and replace the None value of `blender_executable` on line 11 with your path.
E.g., on a Mac, the line could look like this:
`blender_executable: Union[str, None] = "/Applications/Blender.app/Contents/MacOS/Blender"`.


## Starting the tool
You have three ways to start the tool:
1) You can open Blender directly and pass it the landmarking tool via:
```
/PATH/TO/BLENDER/EXECUTABLE --python /PATH/TO/INFACE/src/blender/landmarking.py -- --mesh_paths /PATH/TO/MESH
```
and use `--help` for more details about the arguments.
2) You can also call the tool from python code written within this project:
```
from src.blender.python_interface import set_landmarks
set_landmarks("/ENTER/YOUR/MESH/PATH.obj")
```
The method also has documentation on its arguments.
3) If you want to landmark a whole dataset, one mesh after another, you can use the
[`landmarking_batch.py`](../../src/processing/landmarking_batch.py) script:
```
python -m src.processing.landmarking_batch --dataset_folders /PATH/TO/ONE/OR/MORE/DATASET/FOLDERS
```
The script offers a lot of data filter options, based on names and dates when you landmarked files, 
so that you don't have to landmark every scan in the folder. 
Also here, you can add `--help` for a description of the script's arguments.

## Click the Landmarking Tab
After you started the tool, you should see a Blender window that shows one of the meshes you chose to landmark.
Potentially, you also see two subwindows within the Blender window showing the mesh, or more if you chose 
number above 1 for `num_window_splits`.
In order to see the button interface of the landmarking tool, you still need to click the **Landmarking** tab,
which should be located at the bottom of the panel at top right of each subwindow, cf. image below:
<p align="center">
<img src="images/landmarking_tool_click_tab.png" width="60%" style="display:inline-block;">
</p>
Now you should be ready to start:

# Usage
First, you can hover over any button with your mouse, and it should show you a description of what that button does,
cf. below:
<p align="center">
<img src="images/landmarking_tool_hover_info.png" width="80%" style="display:inline-block;">
</p>

Try to avoid most of Blender's functionality and only interact with the tool's button interface.
Otherwise, you might run into inconsistencies. E.g., Blender allows you to delete the mesh you want to landmark,
and then the tool will obviously not work anymore. 
The most important caveat of this tool: You cannot revert actions with Ctrl Z.

We will shortly describe the important buttons and features:
- **Set Landmarks**: Press this button, or press "S" on your keyboard to activate this mode. Once activate, you can 
go over the mesh with your mouse to set a landmark. Click to set the landmark. Now you can immediately continue
to set the next landmark. If you want to skip a landmark (a.k.a. make it undefined), press "N" on your keyboard while
in setting mode of that landmark. Skipped landmarks are highlighted in gray. If you cannot reach a point on the mesh,
you can also slice the mesh along the x, y, and z axis in both directions, using the respective buttons
under the **Mesh View Options** tab. Exit the mode via right mouse click, or by pressing the escape key.
- **Edit Landmarks**: Press this button, or press "E" on your keyboard to activate this mode. Once activated, you can
drag and drop any landmark that you already placed. While dragging a landmark, you can press "N" to skip it. You can
also unskip landmarks by dragging and dropping a skipped landmark.
- **Exclude Geometry**: Press this button, or press "G" on your keyboard to activate this mode. It will switch the mesh
to Blender's Edit Mode, where you can select vertices. This mode is used to segment geometric artifacts, such as the 
pacifier in the example shown below:
<p align="center">
<img src="images/landmarking_tool_edit_mode.png" width="80%" style="display:inline-block;">
</p> 

- Blender offers a range of options hwo to select vertices. We recommend the circle tool
that you can activate by pressing "C" on your keyboard. You can adjust the circle radius 
via mouse scrolling and then dragging across vertices you want to select. You can also
press+hold the mouse wheel (or press Alt if you've emulated it as third mouse button) while dragging
to de-select vertices again. Press Escape to exit the circle tool, and press **Exclude Geometry**
or "G" again to exit the segmentation mode. This will immediately save a file with your index selection.
If you haven't specified a path, blender will ask you where to save it (only first time).
The button **Exclude Texture** works the same way. It simply saves a separate file so that you
can distinguish between geometric and texture artifacts.

If you press on a landmark, additional buttons will appear:
<p align="center">
<img src="images/landmarking_tool_specific_landmark.png" width="100%" style="display:inline-block;">
</p> 

You can use the first row to adjust the landmark position freely
in space, and vice versa, to reattach it back to the mesh.
You may also change the landmark order with the **Index** button
or even delete the landmark with **Delete Landmark** (cannot be undone).

- If you cannot find a specific landmark, there's a black button
below **Edit Landmarks** you can use to find your landmark.
- You can also hide the landmarks with **Hide Landmarks** in case they're in the way when 
you're segmenting artifacts.
- Alternatively, you can make the landmarks smaller or larger via **Landmark Size**.
- If you want to delete all landmarks, use **Delete All Landmarks**, but bear in mind that you cannot undo this.
- Press **Export Landmarks** to export your selection. Blender will ask you where to save them,
unless you specified this via argument beforehand.
If you chose to quit Blender after exporting the landmarks, the window will close.


# Fastest routine Usage
- Use the [`landmarking_batch.py`](../../src/processing/landmarking_batch.py) script to landmark a full dataset.
- Define num_landmarks as argument in the script to enable color-coding on the landmarks. This will make it easier to spot
errors when comparing to a reference.
- Have a separate window open with such a reference mesh that shows where to position which landmark. 
You can use the arguments `--reference_mesh_path` and `--reference_landmarks_path` with the 
[`landmarking_batch.py`](../../src/processing/landmarking_batch.py) script to automatically open this 
window.
- For each case, first mark the artifacts by pressing "G" and using the circle tool (so that you don't forget).
- Then set the landmarks by using "S", one after another, press "N" to skip some, compare to reference 
if you can't remember order. 
- Press "E" to adjust existing landmarks, also optionally skipping or unskipping existing ones, 
also optionally clicking on individual landmarks to move 
them away from the mesh or change the order.
- Press **Export Landmarks**, and the window will close, and a new one will open with a new case, 
while the reference window will stay open.


# Advanced Features
## Several Meshes in one Tool Instance
You can only define one set of landmarks per landmarking tool instance. 
However, you can still show more than one mesh in one instance by simply providing
more than one mesh path to the tool. If you use the 
[`landmarking_batch.py`](../../src/processing/landmarking_batch.py) script, you 
can also achieve this via the `--additional_mesh_filters` argument (this requires the additional
meshes to be in the same folder as the first mesh, though). If you have loaded the tool with more than
one mesh, you can choose the mesh you want to show in each subwindow with the buttons in the very first 
row of the tool.
A use case would be if you have paired scans of the face and the skull, so when landmarking the face,
you can also see the skull which might provide more information about certain key features, e.g., 
where the bony part of the nose ends or where the pogonion exactly lies. We demonstrate such a case
with our templates below:
<p align="center">
<img src="images/landmarking_tool_two_meshes.png" width="100%" style="display:inline-block;">
</p> 


## Slice Mesh Also for Segmentation
We already mentioned that you can slice your mesh if you want to place landmarks in regions that might otherwise 
be hidden. We implemented the slicing feature in a way that you can also segment artifacts while having the mesh 
sliced. This also works if you have more than one mesh. Cf. below:
<p align="center">
<img src="images/landmarking_tool_slice_edit.png" width="100%" style="display:inline-block;">
</p> 









