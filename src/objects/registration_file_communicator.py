"""
Defines the RegistrationFileCommunicator class, which defines the paths, imports, and exports
used for the registration (mainly used by src/processing/registration.py).

Author: Till Schnabel (contact till.schnabel@inf.ethz.ch);

MIT License

Copyright (c) 2026 ETH Zurich, Till Schnabel; potentially also other copyright, cf. further below.

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

from typing import Union, List, Tuple
from pathlib import Path
from src.objects.mesh import Mesh, Landmarks
from src.objects.indices_and_masks import IndicesAndMasks
import re
import numpy as np
import warnings

class RegistrationFileCommunicator:
    """
    This is a helper class that serves for saving and loading the correct files required for the registration.
    This includes target scans and their respective landmarks, as well as target ignore index selections,
    """

    def __new__(cls, *args, **kwargs):
        raise TypeError(f"{cls.__name__} cannot be instantiated")

    @staticmethod
    def import_target_scan_for_registration(scan_path: Union[str, Path]) -> Mesh:
        """
        We found it's better to post-process obj scans and remove their duplicate vertices after loading
        them from a file via open3d.
        """
        is_obj = Path(scan_path).suffix == ".obj"
        return Mesh.load(scan_path, enable_post_processing=is_obj, remove_duplicate_vertices=is_obj)

    @staticmethod
    def get_lm_path_from_creators(scan_path: Union[str, Path], lm_creators: Union[str, List[str], None] = None) -> Union[Path, None]:
        """
        Get the path to the file that contains the landmarks for the provided scan that was created by one of the
        specified creators.
        :param lm_creators: str specifying one creator or list of str specifying potentially multiple creators.
                            The order of the creators defines which landmarks to look for first.
                            I.e., if a landmark file is found for the first lm_creator, the other ones won't
                            even be checked for. If you set this argument to None, the algorithm will try to
                            look any suitable landmark file and pick the first one from the sorted list.
        :param scan_path: Path to scan file for which you want to find the landmark file.
        :return: Path to landmark file, or None of none could be found.
        """
        # We created some copies of the original meshes in which we removed some unnecessary regions
        # The two resulting meshes were renamed, so that the original was added the term "_raw" after its name,
        # whereas the adjusted one was added the term "_cleaned". However, only a single landmark file was used
        # for both, which didn't need these terms added, which is why we remove these terms from the name of the scan
        # when looking for respective landmarks.
        scan_path_stem_stripped = re.sub('_raw', '', re.sub('_cleaned', '', scan_path.stem))

        lm_path = None

        if lm_creators is None:
            # We look for all suitable landmarks for the scan and simply pick the first ones.
            # We exclude files with "ignore" in their name, since our landmarking tool also saves
            # regions to ignore as .csv files, because only saving the indices proved unreliable
            # given the occasional vertex resorting of open3d.
            lm_paths = Landmarks.get_all_landmark_files_in_path(
                scan_path.parent, regex_names=f"{scan_path_stem_stripped}*", exclude_regex_names="*ignore*")
            if len(lm_paths) > 0:
                lm_path = lm_paths[0]
        else:
            # Retrieve landmarks, given creator options.
            lm_creators = [lm_creators] if isinstance(lm_creators, str) else lm_creators
            # Retrieve the first existing landmarks corresponding to the chosen lm creator order.
            for current_lm_creator in lm_creators:
                lm_paths = Landmarks.get_all_landmark_files_in_path(
                    scan_path.parent, regex_names=f"{scan_path_stem_stripped}_{current_lm_creator}")
                if len(lm_paths) == 0:
                    continue
                elif len(lm_paths) != 1:
                    raise AssertionError(f"Found multiple fitting landmark paths: {lm_paths}. Check.")
                else:
                    lm_path = lm_paths[0]
                    break
        return lm_path

    @staticmethod
    def get_scan_with_landmarks(
            scan_path: Union[str, Path], lm_creators: Union[str, List[str], None] = None,
            min_num_defined_landmarks: int = 4,
            landmark_indices: Union[List[int], np.ndarray, None] = None,
            curvilinear_feature_indices: Union[List[Union[List[int], np.ndarray]], None] = None) -> Union[Mesh, None]:
        """
        This method returns the loaded mesh, including the correct landmarks as attribute.
        The landmarks are matched with the lm_creators argument, cf. :func:`get_lm_path_from_creators()`
        for more details. If no matching landmarks could be found, this method returns None.
        :param scan_path: path to the scan to be loaded
        :param lm_creators: str or list of string specifying landmark creators to consider for loading the landmarks.
                            cf. :func:`get_lm_path_from_creators()` for more details.
        :param min_num_defined_landmarks: int that specifies how many landmarks need to be actually defined to be
                                          allowed here. This method will not try to match any other of the lm_creators
                                          should this check fail.
        :param landmark_indices: Optionally provide an index array to filter/reorder the landmark points by.
        :param curvilinear_feature_indices: Optionally provide an additional list of index arrays to rearrange a
                                            subset of the landmark points into curvilinear features.
        :return: loaded Mesh with landmarks as attribute, or None if no fitting landmarks could be found.
        """
        lm_path = RegistrationFileCommunicator.get_lm_path_from_creators(scan_path=scan_path, lm_creators=lm_creators)
        if lm_path is None:
            return None

        scan_landmarks = Landmarks.load(lm_path)
        if scan_landmarks.get_num_defined_points() < min_num_defined_landmarks:
            warnings.warn(f"Landmarks {lm_path} contain fewer than {min_num_defined_landmarks} defined points.")
            return None

        # Import scan
        scan = RegistrationFileCommunicator.import_target_scan_for_registration(scan_path)
        scan.set_landmarks(scan_landmarks, parse_order=landmark_indices)
        if curvilinear_feature_indices is not None:
            scan.set_curvilinear_features(scan_landmarks.get_points(), curvilinear_feature_indices)
        return scan

    @staticmethod
    def get_scan_ignore_paths(scan_path: Path) -> Tuple[Path, Path]:
        """
        Return the file paths to the vertex ignore regions for the specified scan,
        including the vertex indices and actual vertex points arrays.
        :return:
            1) Path to vertex ignore vertex indices
            2) Path to vertex ignore vertex positions.
        """
        ignore_vertices_path = scan_path.with_name(f"{scan_path.stem}_ignore.txt")
        ignore_vertices_points_path = scan_path.with_name(f"{scan_path.stem}_ignorepoints.csv")
        return ignore_vertices_path, ignore_vertices_points_path

    @staticmethod
    def get_scan_ignore_indices(scan_path: Path, scan=None, flip: bool = False, dist_thresh=1e-6) -> Union[np.ndarray, None]:
        """
        Unlike get_scan_ignore_paths(), this method returns the actual vertex and color vertex ignore indices.
        It basically loads the actual point positions for both selections, computes the respective indices
        on the current mesh via closest point search, and then compares these converted indices to the
        originally saved indices as a sanity check.
        """
        if scan is None:
            scan = RegistrationFileCommunicator.import_target_scan_for_registration(scan_path)
        scan_vertices = scan.get_vertices()
        ignore_vertices_path, ignore_vertices_points_path = RegistrationFileCommunicator.get_scan_ignore_paths(scan_path)

        ignore_indices = None
        if ignore_vertices_points_path.exists():
            ignore_target_points = Landmarks.load(ignore_vertices_points_path).get_points()
            if flip:
                ignore_target_points = Landmarks.flip_points(ignore_target_points)

            _, closest_point_indices = scan.get_closest_vertices(ignore_target_points)
            closest_point_dists = np.linalg.norm(scan_vertices[closest_point_indices] - ignore_target_points, axis=1)
            ignore_indices = closest_point_indices[closest_point_dists <= dist_thresh]
            if len(ignore_indices) < len(closest_point_indices):
                warnings.warn(f"There's some issue with the {ignore_vertices_points_path} points index selection. "
                              f"It should be {len(closest_point_indices)}, but we only have {len(ignore_indices)} "
                              f"matches. Check if this is correct, maybe adjust dist_thresh.")

        elif ignore_vertices_path.exists():
            warnings.warn("Direct ignore indices are not recommend for registration anymore, "
                          "because Open3D is too inconsistent with its vertex reordering. "
                          f"Please check the scan {scan_path} and save the actual point positions.")

        return ignore_indices

    @staticmethod
    def get_transform(registered_mesh_path: Union[str, Path], invert: bool = False, return_path: bool = False) -> Union[np.ndarray, Path]:
        """
        Get the transformation of a registered mesh, i.e., the transformation that needs to be applied
        to the original input mesh to align it with the registered mesh.
        :param registered_mesh_path: Path to the registered mesh. The transform file is assumed
                                     to be in the same folder and named "transform.txt". Such a file
                                     is automatically saved at the end of the registration process.
        :param invert: Set to True to invert the transformation matrix before returning it.
        :param return_path: Set to True to return the path to the transform file instead of the numpy matrix array.
        :return: 4x4 homogeneous transformation matrix numpy array, or path to file if return_path is True.
        """
        transform_path = Path(registered_mesh_path).with_name("transform.txt")
        if return_path:
            if invert:
                warnings.warn("You chose to invert the transform, but also chose to return the path instead of "
                              "the actual np array, so the path is returned and the invert argument is ignored.")
            return transform_path
        if not transform_path.exists():
            raise FileNotFoundError(f"{transform_path} does not exist. It's required to align the target mesh.")
        with open(transform_path, "r") as f:
            transform_str = f.readline().strip().split(",")
            transform = np.asarray([float(transform_entry.strip()) for transform_entry in transform_str]).reshape((4, 4))

        if invert:
            transform = np.linalg.inv(transform)

        return transform

    @staticmethod
    def write_transform(transform_matrix: np.ndarray, transform_file_path: Union[Path, str]) -> None:
        """
        Write a 4x4 homogeneous transformation matrix to a text file, entries separated by commas.
        Numpy's ravel() method used for linearization of the matrix.
        :param transform_matrix: 4x4 homogeneous transformation matrix numpy array.
        :param transform_file_path: Path to the text file to write the matrix to.
        :return None
        """
        with open(transform_file_path, "w") as f:
            f.write(",".join([str(transf_entry) for transf_entry in transform_matrix.ravel()]))

    @staticmethod
    def get_excluded_regions_on_registered_mesh(
            registered_mesh_path: Union[str, Path], return_paths: bool = False) -> Union[Tuple[Path, Path], Tuple[np.ndarray, np.ndarray]]:
        """
        This method differs from :func:get_scan_ignore_indices() in that it returns the indices that were ignored
        on the registered mesh, so those that couldn't find a matching correspondence in the final iteration
        of the registration process.
        @param registered_mesh_path: Path to the registered mesh of which to return the excluded regions.
        @param return_paths: Set to True to return the two paths to the vertex and color
                             exclude indices instead of the actual arrays.
        @return: Tuple:
            1) Exclude vertex indices (or path to them)
            2) exclude color indices (or path to them)
        """
        registered_mesh_path = Path(registered_mesh_path)
        exclude_vertex_path = registered_mesh_path.with_name(f"{registered_mesh_path.stem.replace('_flipped', '')}_exclude.txt")
        exclude_texture_path = exclude_vertex_path.with_stem(f"{exclude_vertex_path.stem}_texture")
        if return_paths:
            return exclude_vertex_path, exclude_texture_path
        exclude_vertices = IndicesAndMasks.load(exclude_vertex_path) if exclude_vertex_path.exists() else np.asarray([])
        exclude_texture = IndicesAndMasks.join(exclude_vertices, IndicesAndMasks.load(exclude_texture_path) if exclude_texture_path.exists() else np.asarray([]))
        return exclude_vertices, exclude_texture