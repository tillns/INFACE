"""
This is a script that can be used to batch-landmark a lot of meshes, one after another.
You provide one or more folders that contain meshes you want to landmark, and this script
loops over the meshes and calls the Blender landmarking interface for each mesh.
The script offers several file filtering options, so that you can choose subsets of the meshes
conveniently. Check all arguments via --help argument.

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
# library imports
from datetime import datetime
from tqdm import tqdm
from argparse import ArgumentParser
import threading

# own imports
from src.objects.mesh import Mesh
from src.blender.python_interface import set_landmarks

def main():
    parser = ArgumentParser()
    parser.add_argument('--dataset_folders', type=str, nargs="+", required=True,
                        help="Provide one or more folder paths to load the meshes to be landmarked from.")
    parser.add_argument('--mesh_filter', type=str, default=None, nargs="+",
                        help="Optionally provide one or multiple regexes to filter what meshes you want to "
                             "landmark in the provided folder(s).")
    parser.add_argument('--exclude_mesh_filter', type=str, default=[], nargs="+",
                        help="Negative of --mesh_filter. All mesh files that fulfill these regex(es) are not loaded.")
    parser.add_argument('--additional_mesh_filters', type=str, default=None, nargs="+",
                        help="Use this to load more than one mesh into the landmarking tool. "
                             "For each mesh file, the script will check the same folder for mesh files "
                             "that match these additional filters.")
    parser.add_argument('--num_landmarks', type=int, default=None,
                        help="Optionally provide number of landmarks you want to set for each mesh.")
    parser.add_argument('--start_from_mesh', type=str, default=None,
                        help="Provide the name of the mesh file from which you want to start the landmarking. "
                             "If this is provided, all mesh files that come before this mesh are skipped "
                             "(even if you chose to overwrite).")
    parser.add_argument('--reverse_order', default=False, action="store_true",
                        help="Reverse the file sorting order of the mesh files you want to landmark.")
    parser.add_argument('--overwrite', default=False, action="store_true",
                        help="Also landmark meshes you already landmarked.")
    parser.add_argument('--save_continuously', default=False, action="store_true",
                        help="Save landmark updates in the Blender interface continuously. "
                             "Note that if you choose this, existing landmark files will be overwritten immediately "
                             "when you make any changes in the interface.")
    parser.add_argument('--creator', type=str, default="DefaultCreator",
                        help="The landmark files are named after the mesh file, but with an appended name of a creator,"
                             "so that it's clear who did the landmarking, thus allowing multiple people to landmark "
                             "the same mesh, thus allowing to measure interobserver variability."
                             "By default, the name 'DefaultCreator' is used. You can also choose "
                             "creator1, creator2 to have the same person landmark the same mesh multiple times "
                             "to measure intraobserver variability.")
    parser.add_argument('--init_creators', type=str, nargs="+",
                        default=[],
                        help="Optionally start from landmarks of another creator. You can provide multiple options, "
                             "in which case the first existing creator is chosen. By default, this is empty, so you "
                             "will only start from existing landmarks there exists a landmark file for the creator "
                             "you defined, and you chose to overwrite.")
    parser.add_argument('--date_modify', default=None, type=str,
                        help="Landmark all meshes that haven't been landmarked since the date you can choose here in "
                             "YYYY-MM-DD format.")
    parser.add_argument('--added_since', default=None, type=str,
                        help="Landmark only those meshes that have been added since the date you provide here in "
                             "YYYY-MM-DD format.")
    parser.add_argument('--added_before', default=None, type=str,
                        help="Landmark only those meshes that have been added before the date you provide here in "
                             "YYYY-MM-DD format.")
    parser.add_argument('--num_window_splits', type=int, default=1,
                        help="How many times to split the 3D View window in Blender. "
                             "Default is 1, so you get two windows.")
    parser.add_argument('--reference_mesh_path', default=None, type=str,
                        help="Optionally provide path a reference mesh to show in a separate window along with its "
                             "landmarks (need to specify reference_landmarks as well).")
    parser.add_argument('--reference_landmarks_path', default=None, type=str,
                        help="Optionally provide path reference landmarks to show in a separate window along with their"
                             " mesh (need to specify reference_mesh as well).")
    args = parser.parse_args()

    if args.reference_mesh_path is not None and args.reference_landmarks_path is not None:
        reference_thread = threading.Thread(
            target=set_landmarks,
            args=(args.reference_mesh_path, ),
            kwargs=dict(input_landmarks_path=args.reference_landmarks_path,
                        num_landmarks=args.num_landmarks, num_window_splits=0)
        )
        reference_thread.start()

    # Load all mesh paths in the given folder(s).
    # These mesh paths are sorted.
    mesh_paths = Mesh.get_all_mesh_files_in_path(
        args.dataset_folders, nested=True, regex_names=args.mesh_filter,  exclude_regex_names=args.exclude_mesh_filter,
        added_since=args.added_since, added_before=args.added_before
    )

    # Reverse order of the mesh paths if chosen by user.
    if args.reverse_order:
        mesh_paths = mesh_paths[::-1]

    # Flag to keep track if the start mesh has already been found in case start_from_mesh is defined
    has_started = args.start_from_mesh is None

    # Loop over the mesh paths and check for each mesh path if we should start the Blender landmarking interface
    # with it
    for mesh_path in tqdm(mesh_paths):
        # Skip all meshes that come before the specified mesh (pattern)
        if args.start_from_mesh is not None:
            if args.start_from_mesh in mesh_path.stem:
                has_started = True
            if not has_started:
                continue

        # Output landmarks are named after creator, such that multiple creators can define landmark positions,
        # or even the same one multiple times via "creator1", "creator2", etc.
        output_landmarks_path = mesh_path.with_name(f"{mesh_path.stem}_{args.creator}.csv")

        # Landmarks defined by someone else can serve as a starting point
        # if the creator hasn't defined any landmarks yet.
        for init_creator in args.init_creators:
            initial_landmarks_path = mesh_path.with_name(f"{mesh_path.stem}_{init_creator}.csv")
            if initial_landmarks_path.exists():
                break
        else:
            initial_landmarks_path = None

        # output region path defines the file naming for the vertex indices that should be excluded
        output_region_path = mesh_path.with_name(f"{mesh_path.stem}_ignore.txt")

        # We check when the mesh labels have been last modified, and if the files are older
        # than the user-provided date, we set overwrite to True.
        overwrite = args.overwrite
        if not overwrite and args.date_modify is not None:
            cutoff_date = datetime.strptime(args.date_modify, '%Y-%m-%d')
            last_updated = [datetime.fromtimestamp(file_path.stat().st_mtime) for file_path in
                            [output_region_path, output_landmarks_path] if file_path.exists()]
            overwrite = (min(last_updated) < cutoff_date) if len(last_updated) > 0 else True

        # Do the landmarking if landmark file doesn't exist yet
        # or if existing landmarks are to be overwritten
        if overwrite or not output_landmarks_path.exists():

            # input landmarks are the creator's previous landmarks if they exist,
            # and if not, the first of the init_creators' landmarks if any exist.
            if output_landmarks_path.exists():
                input_landmarks_path = output_landmarks_path
            elif initial_landmarks_path is not None and initial_landmarks_path.exists():
                input_landmarks_path = initial_landmarks_path
            else:
                input_landmarks_path = None

            # From the current mesh's folder, we check if there are more meshes
            # that match the additional_mesh_filters if given by the user.
            # If there are any, we add those paths to the interface, so that
            # we can have multiple meshes in the interface.
            if args.additional_mesh_filters is not None:
                additional_mesh_paths = Mesh.get_all_mesh_files_in_path(
                    mesh_path.parent, regex_names=args.additional_mesh_filters)
            else:
                additional_mesh_paths = []

            # Define output regions paths for all meshes to be shown in the interface.
            output_region_paths = [output_region_path, *[additional_mesh_path.with_name(f"{additional_mesh_path.stem}_ignore.txt")
                                                         for additional_mesh_path in additional_mesh_paths]]
            # Call the blender landmarking script.
            set_landmarks(
                mesh_path, *additional_mesh_paths,
                input_landmarks_path=input_landmarks_path,
                output_landmarks_path=output_landmarks_path,
                num_landmarks=args.num_landmarks,
                output_region_paths=output_region_paths,
                num_window_splits=args.num_window_splits,
                save_continuously=args.save_continuously,
                # When the user clicks "Export Landmarks", Blender is closed, and we loop to the next mesh
                quit_blender_after_landmark_export=True
            )


if __name__ == '__main__':
    main()
