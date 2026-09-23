"""
This file offers the methods to compute all cranial measurements used for cranial shape correction in the paper.
Additionally, methods are included to load the linear regressors for each attribute and to correct
the cranial shape based on one or more of these attributes. Lastly, new regressors can be trained as well.
A list of all measurements and their explanations can be found under get_all_measurement_kinds().
Since the release of the INCRAN, we additionally included measurements for the clefts,
and another experimental setting for nonlinear mappings. You can enabled those with the
--use_nonlinear_mappings arguments when training a new function. Cf. also the class documentation for more details.

If you call this file, you can provide a directory with registered meshes, and it will automatically
compute the measurements for the meshes and save them in json files, and it will also save a corresponding
healthy cranium with "optimal" measurements in the same folder. You can also choose to train new measurement
regressors on this data. Choose --help when calling this file to get a full description of available arguments.
Also note again that the meshes must already in the space of the model you're providing as arguments.
If you need to create registrations, cf. src/processing/registration.py.

compute_measurements_on_registered_mesh() requires installation of trimesh, shapely, and mapbox_earcut.
compute_shape_factor_function_ls() requires sklearn.

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
import pickle
import warnings
from argparse import ArgumentParser
from pathlib import Path


from src.objects.registration_file_communicator import RegistrationFileCommunicator

from models import get_model_path
from src import utils
from src.objects.indices_and_masks import IndicesAndMasks
from src.objects.mesh import Mesh
from typing import Dict, List, Union, Tuple
from src.objects.morphable_model import MorphableModel
from src.utils import Plane, Line

import json
import numpy as np


class CranialAttributes:
    """
    This is the class that can compute various cranial attributes/measurements for registered infant head meshes.
    Additionally, it can correlate these measurements with the space of morphable model of the infant head.
    It receives the respective model to compute the correlation via (regularized) linear regression,
    and the correlations are saved within a class attribute called shape_factor_function, which is a dictionary
    of linear regressors, so for each kind of measurement, there's a key, and the value is a tuple
    that contains four entries:
        1) the weight vector that translates from the model's latent space to the centered measurement.
        2) the mean of the measurement, so to get the actual measurement estimate, you need to multiply the model
           projection by the weight vector and then add the mean.
        3) the standard deviation of the measurement. This is not necessary for estimation, but gives an
           intuition about the range of the value.
       4) An explanation what the attribute represents.
       maybe 5-8) Potential mapping attributes.

    Note that there are attributes that probably do not have an exact linear relationship to the model space.
    Take for example the cranial volume: Assuming that vertex positions linearly correlate with the model
    space, then cranial volume, which grows cubically with the vertex positions, probably does not correlate
    linearly with the model space. Hence, we added an experimental setting to this class, where certain
    attributes can be transformed before being correlated with the model space. For volumes, we chose
    to take the cubic root (power 1/3) and for circumference the square root (power 1/2).
    Another, more complex attribute, is the age. Babies grow more in the first six months than after that
    until they're two years old, so this is not linear, and probably neither polynomial. We chose to approximate
    this with a piecewise linear function, because it readily integrates into the remaining pipeline. The function
    that increases monotonically within the trained range and cuts off outside, so babies cannot go below 0 months
    or above our training range. You can enable all of these nonlinear mappings with --use_nonlinear_mappings.
    """
    def __init__(self, model_path: Union[str, Path], shape_factor_function_path: Union[str, Path, None] = None):
        """
        :param model_path: Path to head morphable model (.h5)
        :param shape_factor_function_path: If available, already provide the path to the trained shape factor function
                                        (.json). Note that you can also retrain this.
        """
        self.model: MorphableModel = MorphableModel.load_correct_morphable_model(model_path)

        # These are the measurements we actually used to compute optimal healthy cranial shapes
        self.measurements_relevant_for_correction = [
            "FI", "CI", "CVAI_signed", "TRx", "TRy", "TRz", "TRtop",
            "VR_back_left_right", "VR_front_left_right", "VR_front_back",
            "right_lip_cleft_percentage", "left_lip_cleft_percentage", "cleft_palate_percentage",
            "right_alveolar_cleft_size", "left_alveolar_cleft_size"
        ]

        # Some measurements deliberately adjust their values
        # depending on the asymmetry, e.g., the default version of CVAI stays always positive
        # no matter which side is longer, so it doesn't make sense to correlate those with the
        # model space. Note that there are also attributes that probably do not linearly correlate
        # with the model space, cf. class documentation.
        self.exclude_measurements_from_training = [
            "CVA", "CVAI", "*_side-adjusted"
        ]

        self.nonlinear_mappings = {
            # Volumes change cubically with linear changes in the vertex positions
            "v_front_left": "root3", "v_front_right": "root3", "v_back_left": "root3", "v_back_right": "root3",
            "v_tot": "root3",
            # The head circumference changes quadratically with linear changes in the vertex positions
            "circumference": "root2",
            # For the age, we use a monotonic piecewise linear function to approximate the more complex relationship
            # between age and growth.
            "age": "piecewise_linear",
            # We use the same piecewise linear approximation for the clefts as for age. Note we haven't thoroughly
            # experimented with this, though.
            "right_lip_cleft_percentage": "piecewise_linear",
            "left_lip_cleft_percentage": "piecewise_linear",
            "cleft_palate_percentage": "piecewise_linear",
            "right_alveolar_cleft_size": "piecewise_linear",
            "left_alveolar_cleft_size": "piecewise_linear",
        }

        # These are optimal values that may be different from the average values in the dataset
        # For the other measurements, we take the average value as the "optimal" one
        self.measurement_optima = {
            "CI": 80, "CVAI_signed": 0, "TR": 1, "TRx": 1, "TRy": 1, "TRz": 1, "TR_top": 1,
            "VR_left_right": 1, "VR_back_left_right": 1, "VR_front_left_right": 1,
            "VR_left_front_back": 1, "VR_right_front_back": 1,
            "TSR": 1, "EOR": 1, "EIR": 1, "EWR": 1,
            "right_lip_cleft_percentage": 0, "left_lip_cleft_percentage": 0, "cleft_palate_percentage": 0,
            "right_alveolar_cleft_size": 0, "left_alveolar_cleft_size": 0

        }

        # Load the attribute correlator if path was given.
        if shape_factor_function_path is not None:
            self.load_shape_factor_function(shape_factor_function_path)

    def get_measurements_relevant_for_correction(self, add_measurements_to_fix: bool = False) -> List[str]:
        measurements_relevant_for_correction = self.measurements_relevant_for_correction.copy()
        if add_measurements_to_fix:
            measurements_relevant_for_correction.append("v_tot")
            if self.has_age():
                measurements_relevant_for_correction.append("age")
        return measurements_relevant_for_correction


    def load_shape_factor_function(self, shape_factor_function_path: Union[str, Path]) -> None:
        with open(get_model_path(shape_factor_function_path), "r") as f:
            self.shape_factor_function: Dict = json.load(f)

        # Load the weights, mean, and standard deviation from the shape factor function
        self.measurements_w = {
            measurement_kind: measurement_func[0] for measurement_kind, measurement_func
            in self.shape_factor_function.items()}
        self.measurements_mean = {
            measurement_kind: measurement_func[1] for measurement_kind, measurement_func
            in self.shape_factor_function.items()}
        self.measurements_stdev = {
            measurement_kind: measurement_func[2] for measurement_kind, measurement_func
            in self.shape_factor_function.items()}
        self.measurement_mappings = {
            measurement_kind: measurement_func[4:] for measurement_kind, measurement_func
            in self.shape_factor_function.items() if len(measurement_func) > 4
        }

    @staticmethod
    def get_measurement_plane(registered_mesh: Mesh, zero_plane: Plane, num_segments: int = 10,
                              num_iterations: int = 5) -> Plane:
        """
        This is an iterative root finding algorithm. It starts by defining a range of distances, starting from the zero
        plane and going until the vertex furthest away from the zero plane.
        This range is divided into num_segments segments.
        For each segment a plane is created, and the circumference at that plane is computed.
        The plane with the maximum circumference defines the new range of distances.
        Around -1, +1 plane around this max circumference plane,
        we divide the range again into num_segments segments and find the plane with the maximum circumference.
        We do this for num_iterations iterations, so that we converge on a plane with maximum circumference.
        :param registered_mesh: registered mesh with flattened ears and no eyeballs.
        :param zero_plane: zero plane (defines starting point and direction).
        :param num_segments: number of segments to divide the distance into.
        :param num_iterations: number of iterations until convergence is defined.
        :return measurement plane of type Plane.
        """
        registered_vertices = registered_mesh.get_vertices()
        plane_normal = zero_plane.get_normal()
        # signed vertex distances to the zero plane
        vertex_distances = zero_plane.compute_distance_points_to_plane(registered_vertices, compute_absolute=False)
        # maximum positive distance
        max_distance = np.max(vertex_distances)
        # We start by dividing the range from the zero plane to the highest point into num_segments segments.
        start_dist, end_dist = 0, max_distance
        measurement_plane = None
        max_circumference = 0
        for it in range(max(num_iterations, 1)):
            plane_distances = np.linspace(start_dist, end_dist, num_segments)
            # Create a plane for each segment.
            planes = [Plane(point_on_plane=zero_plane.get_point() + dist * plane_normal, plane_normal=plane_normal)
                      for dist in plane_distances]
            # Compute the mesh circumference for each plane.
            plane_circumferences = []
            for plane in planes:
                circ_intersection_line_segments = plane.intersect_mesh(registered_mesh, return_faces=False)
                plane_circumferences.append(np.sum(
                    np.linalg.norm(circ_intersection_line_segments[:, 1] - circ_intersection_line_segments[:, 0],
                                   axis=-1)))
            # Find the plane with maximum circumference.
            circ_sort = np.argsort(plane_circumferences)
            max_circ_index = circ_sort[0]
            # To make sure we don't miss the correct plane, we go to both sides of the plane with max circumference.
            # So we divide the initial range by a factor of num_segments/2.
            # Handle edge cases where max circ plane is either the first...
            if max_circ_index == 0:
                start_index, end_index = 0, 2
            # ...or the last segment
            elif max_circ_index == num_segments - 1:
                start_index, end_index = num_segments - 3, num_segments - 1
            # normal cases
            else:
                start_index, end_index = max_circ_index - 1, max_circ_index + 1
            # Define start and end dist range for next iteration
            start_dist, end_dist = plane_distances[start_index], plane_distances[end_index]
            if plane_circumferences[max_circ_index] > max_circumference:
                max_circumference = plane_circumferences[max_circ_index]
                measurement_plane = planes[max_circ_index]
        # After num_iterations iterations, we define our measurement plane to be the plane with maximum
        # circumference found over all iterations. Usually, this should be the last iteration, but maybe there's
        # some numerical reason that one from a previous iteration was slightly bigger.
        assert measurement_plane is not None
        return measurement_plane

    def compute_measurements_on_registered_mesh(
            self, registered_mesh: Mesh, recompute_planes: bool = False,
            use_measurement_plane_for_cranial_volume: bool = False,
            excluded_regions: Union[List, np.ndarray, None] = None) -> Dict[str, float]:
        """
        This method requires the library trimesh, including shapely and mapbox_earcut.
        Given a registered cranial mesh (soft tissue), compute all medically relevant measurements on the mesh
        that were used for the paper, using the known correspondences and preselected indices.
        The hereby computed measurements for each mesh from our dataset were used to train the linear regressors
        that can then be used to estimate and adjust the meshes to corresponding meshes with a healthy cranium
        using their 3DMM projections, cf. the methods further below.
        :param registered_mesh: registered cranial soft-tissue mesh of type Mesh.
        :param recompute_planes: Set True to recompute planes based on nasion and tragus vertices.
                                 By default, we assume the registration already incorporated this.
        :param use_measurement_plane_for_cranial_volume: We initially used the measurement plane to measure cranial
                                                         volume (everything above zero plane). We are now experimenting
                                                         with using the zero plane instead, but you can still adjust
                                                         this via this argument.
        :param excluded_regions: Optionally provide the indices of vertices that were not registered.
                                 Measurements will be skipped if relevant vertices were not registered.
        :return: dictionary containing as keys the names of the measurements and as values the computed scalar floats.
        """

        if excluded_regions is None:
            excluded_mask = IndicesAndMasks.get_mask([], registered_mesh)
        else:
            excluded_mask = IndicesAndMasks.get_mask(excluded_regions, registered_mesh)

        measurements: Dict[str, float] = {}

        def get_indices(*index_names: str, single_index: bool = False) -> Union[np.ndarray, int]:
            for index_name in index_names:
                if not self.model.has_index(index_name):
                    raise AssertionError(f"{index_name} not available in the model. Did you choose the correct model?")
            indices = self.model.get_indices(*index_names, join_indices=True, make_unique=True)
            if single_index:
                return indices[0]
            else:
                return indices

        # Create a copy of the registered mesh, leave that one unchanged.
        registered_mesh_ears_stand_out = registered_mesh.get_copy()
        registered_mesh_ears_stand_out_vertices = registered_mesh_ears_stand_out.get_vertices()

        # We begin by loading specific vertex indices defined for the topology of the provided model
        # The scalp vertices decide whether we compute the cranial measurements or not:
        scalp_noneck_indices = get_indices("scalp_noneck")
        if not np.all(excluded_mask[scalp_noneck_indices]):
            zero_plane_indices = get_indices("zero_plane_line")
            measurement_plane_indices = get_indices("measurement_plane_line")
            frontal_plane_indices = get_indices("frontal_plane_line")
            nasion_plane_indices = get_indices("nasion_plane_line")
            frontotemporal_indices = get_indices("frontotemporal")
            sa_indices = get_indices("supraorbital")
            tragus_indices = get_indices("ear_tragus_right", "ear_tragus_left")
            nasion_index = get_indices("nasion", single_index=True)
            # This one is also called "vertex", but we don't use that name here to avoid confusion with the mesh vertex
            head_top_index = get_indices("head_top", single_index=True)
            subnasale_index = get_indices("subnasale", single_index=True)
            outer_eye_corner_indices = get_indices("right_eye_outer", "left_eye_outer")
            inner_eye_corner_indices = get_indices("right_eye_inner", "left_eye_inner")
            ear_indices = get_indices("ears")
            tragion_indices = get_indices("tragion")
            orbital_indices = get_indices("orbital")
            # The glabella, opisthocranion, and pronasale don't have fixed correspondences, so their position
            # is computed dynamically by considering a range of possible points.
            glabella_possible_indices = get_indices("glabella_possible_indices")
            opisthocranion_possible_indices = get_indices("opisthocranion_possible_indices")
            pronasale_possible_indices = get_indices("pronasale_possible_indices")
            # The vertical line is an index selection that goes through the center of the face and cranium.
            # We compute a sorting of the selection, so that it goes from one vertex to the next connected one.
            vertical_line_sorted_index_sets = self.model.get_average_mesh().get_edge_sorted_selection(
                get_indices("vertical_line"))
            eyebrow_indices = {side: get_indices(f"{side}_eyebrow_all", single_index=False)
                               for side in ["left", "right"]}

            # Compute tragus to nasion differences
            tragus_left_nasion_vector = (registered_mesh_ears_stand_out_vertices[tragus_indices[0]] -
                                         registered_mesh_ears_stand_out_vertices[nasion_index])
            tragus_right_nasion_vector = (registered_mesh_ears_stand_out_vertices[tragus_indices[1]] -
                                         registered_mesh_ears_stand_out_vertices[nasion_index])
            tragus_left_nasion_coord_distances = np.abs(tragus_left_nasion_vector)
            tragus_right_nasion_coord_distances = np.abs(tragus_right_nasion_vector)


            # Measure distances between important points to assess symmetry of the face and head
            def get_distance_between_points(first_idx, second_idx, return_vector: bool = False):
                vector = registered_mesh_ears_stand_out_vertices[first_idx] - registered_mesh_ears_stand_out_vertices[second_idx]
                if return_vector:
                    return vector
                else:
                    return np.linalg.norm(vector)

            # These distances take important central points in the face/on the cranium
            # and compare the distances to other important points on the left and right;
            # this helps to assess facial and cranial symmetry.
            tragus_left_nasion_distance = get_distance_between_points(
                tragus_indices[0], nasion_index)
            tragus_right_nasion_distance = get_distance_between_points(
                tragus_indices[1], nasion_index)
            tragus_left_top_distance = get_distance_between_points(
                tragus_indices[0], head_top_index)
            tragus_right_top_distance = get_distance_between_points(
                tragus_indices[1], head_top_index)
            tragus_left_subnasale_distance = get_distance_between_points(
                tragus_indices[0], subnasale_index)
            tragus_right_subnasale_distance = get_distance_between_points(
                tragus_indices[1], subnasale_index)
            outer_left_eye_nasion_distance = get_distance_between_points(
                outer_eye_corner_indices[0], nasion_index)
            outer_right_eye_nasion_distance = get_distance_between_points(
                outer_eye_corner_indices[1], nasion_index)
            inner_left_eye_nasion_distance = get_distance_between_points(
                inner_eye_corner_indices[0], nasion_index)
            inner_right_eye_nasion_distance = get_distance_between_points(
                inner_eye_corner_indices[1], nasion_index)
            left_eye_width = get_distance_between_points(
                outer_eye_corner_indices[0], inner_eye_corner_indices[0])
            right_eye_width = get_distance_between_points(
                outer_eye_corner_indices[1], inner_eye_corner_indices[1])


            # The ears are in the way for the measurements, so we just flatten them away
            registered_mesh_flattened_ears = registered_mesh_ears_stand_out.smooth_laplacian(
                num_iterations=1000, indices=ear_indices)
            vertices = registered_mesh_flattened_ears.get_vertices()
            # smooth out eye sockets and remove eye vertices
            registered_mesh_flattened_ears_no_eyes = registered_mesh_flattened_ears.smooth_laplacian(
                indices=IndicesAndMasks.join(get_indices("eye_socket_left"), get_indices("eye_socket_right")),
                num_iterations=10)
            registered_mesh_flattened_ears_no_eyes = registered_mesh_flattened_ears_no_eyes.remove_vertices(
                IndicesAndMasks.join(get_indices("eye_left"), get_indices("eye_right")))

            # Compute planes used to compute measurements
            if recompute_planes:
                # Here we recompute the planes based on the tragus and nasion points
                tragus_points = vertices[tragus_indices]
                nasion_point = vertices[nasion_index]
                # The zero plane goes through these three points
                zero_plane = Plane.fit_plane_to_points([nasion_point, *tragus_points[::-1]])
                # The nasion plane is orthogonal to the zero plane (normal along tragus connection vector)
                # and goes through the nasion point.
                nasion_plane = Plane(point_on_plane=nasion_point, plane_normal=tragus_points[1] - tragus_points[0])
                tragus_center = np.mean(tragus_points, axis=0)
                # The frontal plane is orthogonal to both, the nasion and zero plane, and goes
                # through the tragus center (middle between two tragus points)
                frontal_plane = Plane(tragus_center, np.cross(zero_plane.get_normal(), nasion_plane.get_normal()))
                # The measurement plane is parallel to the zero plane and is positioned
                # at the maximum cranial circumference given its direction.
                measurement_plane = self.get_measurement_plane(registered_mesh, zero_plane)
            else:
                # Since these planes were already pre-defined via vertex indices on the template,
                # a proper registration adheres to that selection, and thus they don't necessarily
                # need to be recomputed.
                zero_plane = Plane.fit_plane_to_points(vertices[zero_plane_indices])
                measurement_plane = Plane.fit_plane_to_points(vertices[measurement_plane_indices])
                nasion_plane = Plane.fit_plane_to_points(vertices[nasion_plane_indices])
                frontal_plane = Plane.fit_plane_to_points(vertices[frontal_plane_indices])
            # The Frankfurter plane is meant to orient the head more straightly.
            # It goes through the tragion points (points at the top of the ear canal)
            # and the orbital points (points at the bottom of the orbital rings -- the bones that go around our eyes)
            frankfurter_plane = Plane.fit_plane_to_points(vertices[[*tragion_indices, *orbital_indices]])
            # We additionally define a plane orthogonal to the Frankfurter plane
            # that connects the two tragion points. This is useful to find the pronasale
            # as the point on the nose farthest away from that plane.
            frankfurter_orth_normal = np.cross(
                frankfurter_plane.get_normal(),
                utils.normalize_vector(vertices[tragion_indices[1]] - vertices[tragion_indices[0]]))
            frankfurter_orth_plane = Plane(vertices[tragion_indices[0]], frankfurter_orth_normal)

            #
            # Find head length
            head_length = -1
            glabella_index, opisthocranion_index = None, None
            # We determine the actual glabella and opisthocranion indices by comparing each possible pairs
            # and taking those with maximal distance to each other.
            for idx in glabella_possible_indices:
                for cf_idx in opisthocranion_possible_indices:
                    current_length = np.linalg.norm(vertices[idx] - vertices[cf_idx])
                    if current_length > head_length:
                        head_length = current_length
                        glabella_index, opisthocranion_index = (idx, cf_idx)
            glabella, opisthocranion = vertices[[glabella_index, opisthocranion_index]]
            # We compute the perimeter between glabella and opisthocranion by adding up the edge lengths for all
            # vertices between them using the sorted vertical line selection
            glabella_opisthocranion_in_between_vertices = None
            for vertical_line_sorted_index_set in vertical_line_sorted_index_sets:
                if glabella_index in vertical_line_sorted_index_set:
                    assert opisthocranion_index in vertical_line_sorted_index_set
                    glabella_set_index = list(vertical_line_sorted_index_set).index(glabella_index)
                    opisthocranion_set_index = list(vertical_line_sorted_index_set).index(opisthocranion_index)
                    glabella_opisthocranion_in_between_vertices = vertical_line_sorted_index_set[
                        min(glabella_set_index, opisthocranion_set_index):max(glabella_set_index, opisthocranion_set_index)+1]
                    break
            assert glabella_opisthocranion_in_between_vertices is not None
            glabella_opisthocranion_perimeter = sum([
                np.linalg.norm(vertices[vertex_idx0]-vertices[vertex_idx1])
                for vertex_idx0, vertex_idx1 in zip(
                    glabella_opisthocranion_in_between_vertices[:-1], glabella_opisthocranion_in_between_vertices[1:])])

            # For metopic index
            metopic_point_indices = []
            for eyebrow_side, vertex_indices in eyebrow_indices.items():
                eyebrow_points = registered_mesh_ears_stand_out_vertices[vertex_indices]
                eyebrow_points_projected = measurement_plane.project_points(eyebrow_points)
                segments = np.diff(eyebrow_points_projected, axis=0)
                lengths = np.linalg.norm(segments, axis=1)

                # Normalize segment directions safely
                directions = segments / lengths[:, None]

                # Angle at point i is between segment i-1 and segment i
                prev_dirs, next_dirs = directions[:-1], directions[1:]

                angles = np.arccos(np.clip(np.einsum("ij,ij->i", prev_dirs, next_dirs), -1, 1))

                # Optional curvature-like weighting:
                # angle divided by average adjacent segment length.
                # This favors sharp bends that occur over short distances.
                avg_lengths = 0.5 * (lengths[:-1] + lengths[1:])
                scores_inner = angles / avg_lengths

                all_scores = np.zeros(len(eyebrow_points_projected))
                all_scores[1:-1] = scores_inner

                max_turn_index = int(np.argmax(all_scores))
                metopic_point_indices.append(vertex_indices[max_turn_index])
            metopic_points = measurement_plane.align_points_along_plane(
                registered_mesh_ears_stand_out_vertices[metopic_point_indices])
            metopic_length = np.linalg.norm(metopic_points[1] - metopic_points[0])

            # For VNO
            # We define a new point at the top of the head (vertex),
            # as the one point farthest away from the Frankfurter plane.
            scalp_vertex_distances = frankfurter_plane.compute_distance_points_to_plane(
                vertices[scalp_noneck_indices], compute_absolute=True)
            head_top_new_index = scalp_noneck_indices[np.argmax(scalp_vertex_distances)]
            # The VNO uses the distances between nasion and top (vertex) and between nasion and opisthocranion
            nasion_top_vector = get_distance_between_points(
                nasion_index, head_top_new_index, return_vector=True
            )
            nasion_opisthocranion_vector = get_distance_between_points(
                nasion_index, opisthocranion_index, return_vector=True
            )

            # For NFA
            nasion_glabella_vector = get_distance_between_points(
                nasion_index, glabella_index, return_vector=True
            )
            # We find the pronasale as the point on the nose farthest away from the orthogonal Frankfurter plane.
            pronasale_distances = frankfurter_orth_plane.compute_distance_points_to_plane(
                vertices[pronasale_possible_indices])
            pronasale_index = pronasale_possible_indices[np.argmax(pronasale_distances)]
            nasion_pronasale_vector = get_distance_between_points(
                nasion_index, pronasale_index, return_vector=True
            )

            # Define circumference plane as the plane that includes the glabella and the opisthocranion
            # while being perpendicular to the nasion plane. CVAI, CVA, and CI are all computed by points
            # within this plane
            circumference_plane = Plane.get_plane_from_two_points_and_one_vector(
                glabella, opisthocranion, nasion_plane.get_normal())

            # Intersect circumference plane with registered mesh to get the line set on which all other points
            # for CVAI, CVA, and CI are found
            # I noticed the eyes are sometimes also intersected. so we remove them
            circ_intersection_line_segments = np.asarray(circumference_plane.intersect_mesh(
                registered_mesh_flattened_ears_no_eyes))

            circumference = float(np.sum(
                np.linalg.norm(circ_intersection_line_segments[:, 1] - circ_intersection_line_segments[:, 0], axis=-1)))

            def find_largest_circumference_width_given_direction(current_direction: np.ndarray) -> Tuple[float, np.ndarray, np.ndarray]:
                """
                This method takes each point on the circumference plane intersection, and finds the point on the opposite
                side, where opposite is defined via current_direction. You can imagine this as a sliding line that goes
                along the closed circle of line segments and finds the longst intersection.
                The longest distance as well as the respective points are returned.
                """
                max_width = -1
                first_point, second_point = None, None
                for num_segment, circ_line_segment in enumerate(circ_intersection_line_segments[:-1]):
                    for point in circ_line_segment:
                        intersection_points, _ = Line(point, current_direction).intersect_line_and_line_segments(
                            circ_intersection_line_segments[num_segment + 1:])

                        if len(intersection_points) == 0:
                            continue
                        furthest_intersection_point = intersection_points[np.argmax([
                            np.linalg.norm(intersection_point - point) for intersection_point in intersection_points])]
                        point_vector = point - furthest_intersection_point
                        current_width = np.linalg.norm(point_vector)
                        if current_width > max_width:
                            max_width = current_width
                            if np.dot(point_vector, current_direction) > 0:
                                first_point, second_point = point, furthest_intersection_point
                            else:
                                second_point, first_point = point, furthest_intersection_point
                return max_width, first_point, second_point

            # Find eurion points (most outer points on circumference plane used for computing head width)
            head_width_direction = nasion_plane.get_normal()
            head_width, eurion_left, eurion_right = find_largest_circumference_width_given_direction(
                head_width_direction)

            # Cranial index is defined as the ratio between head width and head length
            ci = head_width * 100 / head_length

            # Find diagonal points (similar to eurion points, only the direction is different) for CVA(I) measurements
            rot_axis = circumference_plane.get_normal()
            direction_init = utils.normalize_vector(glabella - opisthocranion)
            rot_mat_plus = utils.get_rotation_matrix_from_axis_and_angle(
                axis=rot_axis, angle=np.pi/6, homogeneous=False)
            rot_mat_minus = utils.get_rotation_matrix_from_axis_and_angle(
                axis=rot_axis, angle=-np.pi/6, homogeneous=False)
            diagonal_plus_width_direction = rot_mat_plus @ direction_init
            diagonal_minus_width_direction = rot_mat_minus @ direction_init
            diagonal_width_plus, diagonal_plus_first_point, diagonal_plus_second_point = find_largest_circumference_width_given_direction(
                diagonal_plus_width_direction)
            diagonal_width_minus, diagonal_minus_first_point, diagonal_minus_second_point = find_largest_circumference_width_given_direction(
                diagonal_minus_width_direction)

            # Compute CVA, CVAI and their signed versions
            cva_signed = float(diagonal_width_plus - diagonal_width_minus)
            cvai_signed = float(cva_signed * 100 / min(diagonal_width_plus, diagonal_width_minus))

            cva = abs(cva_signed)
            cvai = abs(cvai_signed)

            frontotemporal_dist = np.linalg.norm(vertices[frontotemporal_indices[0]] - vertices[frontotemporal_indices[1]])

            # We project the vectors to the measurement plane to avoid that if one eyebrow is raised that this
            # perturbs the angle measurement
            supraorbital_line1 = utils.normalize_vector(measurement_plane.project_vectors_to_plane(
                vertices[glabella_index] - vertices[sa_indices[0]]))
            supraorbital_line2 = utils.normalize_vector(measurement_plane.project_vectors_to_plane(
                vertices[glabella_index] - vertices[sa_indices[1]]))
            supraorbital_angle = np.arccos(supraorbital_line1.dot(supraorbital_line2))

            if use_measurement_plane_for_cranial_volume:
                slice_plane = measurement_plane
            else:
                slice_plane = zero_plane

            # Compute volumes
            registered_mesh_head = slice_plane.slice_mesh(registered_mesh_flattened_ears_no_eyes)
            registered_mesh_head_back_half = frontal_plane.slice_mesh(registered_mesh_head)
            registered_mesh_head_front_half = frontal_plane.slice_mesh(registered_mesh_head, flip_direction=True)

            # head quarters are also tracked, but currently we only save their absolute values, no ratios
            registered_mesh_head_front_left = nasion_plane.slice_mesh(
                registered_mesh_head_front_half)
            registered_mesh_head_front_right = nasion_plane.slice_mesh(
                registered_mesh_head_front_half, flip_direction=True)
            registered_mesh_head_back_left = nasion_plane.slice_mesh(
                registered_mesh_head_back_half)
            registered_mesh_head_back_right = nasion_plane.slice_mesh(
                registered_mesh_head_back_half, flip_direction=True)

            # Get the volumes of the partial cranial meshes
            head_front_left_volume = registered_mesh_head_front_left.get_volume()
            head_front_right_volume = registered_mesh_head_front_right.get_volume()
            head_back_left_volume = registered_mesh_head_back_left.get_volume()
            head_back_right_volume = registered_mesh_head_back_right.get_volume()

            head_front_volume = head_front_left_volume + head_front_right_volume
            head_back_volume = head_back_left_volume + head_back_right_volume

            head_left_volume = head_front_left_volume + head_back_left_volume
            head_right_volume = head_front_right_volume + head_back_right_volume

            measurements.update({
                # Common measurements
                "CVAI": cvai, "CVA": cva, "CI": ci,
                "CVAI_signed": cvai_signed, "CVA_signed": cva_signed,
                "circumference": circumference,
                "TI": head_length / glabella_opisthocranion_perimeter,
                "FI": frontotemporal_dist / head_width, "SA": supraorbital_angle,
                "VNO": np.arccos(utils.dot_vector_list(
                    utils.normalize_vector(nasion_top_vector),
                    utils.normalize_vector(nasion_opisthocranion_vector))) * 180 / np.pi,
                "NFA": np.arccos(utils.dot_vector_list(
                    utils.normalize_vector(nasion_glabella_vector),
                    utils.normalize_vector(nasion_pronasale_vector))) * 180 / np.pi,
                "MI": metopic_length / head_width,

                # Distance ratios
                "TR": tragus_left_nasion_distance / tragus_right_nasion_distance,
                **{f"TR{coord}": tragus_left_nasion_coord_distances[idx] / tragus_right_nasion_coord_distances[idx]
                   for idx, coord in enumerate(["x", "y", "z"])},
                "TRtop": tragus_left_top_distance / tragus_right_top_distance,
                "TSR": tragus_left_subnasale_distance / tragus_right_subnasale_distance,
                "EOR": outer_left_eye_nasion_distance / outer_right_eye_nasion_distance,
                "EIR": inner_left_eye_nasion_distance / inner_right_eye_nasion_distance,
                "EWR": left_eye_width / right_eye_width,

                # just the volumes
                "v_front_left": head_front_left_volume, "v_front_right": head_front_right_volume,
                "v_back_left": head_back_left_volume, "v_back_right": head_back_right_volume,
                "v_tot": head_front_left_volume + head_front_right_volume + head_back_left_volume + head_back_right_volume,

                # volume ratios
                "VR_front_back": head_front_volume / head_back_volume,
                "VR_left_right": head_left_volume / head_right_volume,
                "VR_back_left_right": head_back_left_volume / head_back_right_volume,
                "VR_front_left_right": head_front_left_volume / head_front_right_volume,
                "VR_left_front_back": head_front_left_volume / head_back_left_volume,
                "VR_right_front_back": head_front_right_volume / head_back_right_volume,
            })

            # Add cvai-signed-based inversion of distance and volume ratios
            for mkind in ["TR", "TRtop", "TSR", "EOR", "EIR", "EWR", "VR_left_right",
                          "VR_back_left_right", "VR_front_left_right"]:
                if cvai_signed > 0:
                    measurements[f"{mkind}_side-adjusted"] = measurements[mkind]
                else:
                    measurements[f"{mkind}_side-adjusted"] = 1/measurements[mkind]

        #
        #
        # Above, we computed the cranial measurements. We now also add the cleft measurements.
        #
        #
        # Only compute cleft attributes for INCLEFT model (INCRAN doesn't have them defined)
        if self.model.has_index("palate"):
            vertices = registered_mesh_ears_stand_out_vertices
            def get_cleft_percentage(cleft_pairs: np.ndarray) -> float:
                num_cleft_pairs = len(cleft_pairs)
                first_open_index = num_cleft_pairs
                for num_cleft_pair, cleft_pair in enumerate(cleft_pairs):
                    if np.linalg.norm(vertices[cleft_pair[1]] - vertices[cleft_pair[0]]) < 0.0001:
                        first_open_index = num_cleft_pair
                        break
                return first_open_index / num_cleft_pairs

            palate_indices = get_indices("palate")
            nasolabial_indices = get_indices("nasolabial")
            # If the nasolabial vertices were not registered (probably because only a palate scan
            # was available in this case), we don't measure the lip clefts.
            if not np.all(excluded_mask[nasolabial_indices]):
                right_lip_cleft_pairs = get_indices("right_lip_cleft_pairs")
                left_lip_cleft_pairs = get_indices("left_lip_cleft_pairs")
                right_lip_cleft_percentage = get_cleft_percentage(right_lip_cleft_pairs)
                left_lip_cleft_percentage = get_cleft_percentage(left_lip_cleft_pairs)

                measurements.update({
                    "right_lip_cleft_percentage": right_lip_cleft_percentage,
                    "left_lip_cleft_percentage": left_lip_cleft_percentage,
                })
            # If the palate vertices were not registered (probably because only a face scan was available in this case),
            # we don't measure the alveolar and palatal clefts.
            if not np.all(excluded_mask[palate_indices]):
                # we have defined segment_vomer_right and segment_vomer_left in the INCLEFT indices.
                # During our registrations, the palatal cleft was handled symmetrically in the sense
                # that both segments stayed connected at corresponding vertices to the vomer or not.
                # As such, it doesn't matter which side we take here.
                cleft_palate_pairs = get_indices("segment_vomer_right")
                cleft_palate_percentage = get_cleft_percentage(cleft_palate_pairs)

                right_alveolar_cleft_size = float(np.linalg.norm(
                    vertices[get_indices('premaxilla_ridge_right_to_segment', single_index=True)] -
                    vertices[get_indices('segment_ridge_right_to_premaxilla', single_index=True)]
                ))
                left_alveolar_cleft_size = float(np.linalg.norm(
                    vertices[get_indices('premaxilla_ridge_left_to_segment', single_index=True)] -
                    vertices[get_indices('segment_ridge_left_to_premaxilla', single_index=True)]
                ))
                measurements.update({
                    "cleft_palate_percentage": cleft_palate_percentage,
                    "right_alveolar_cleft_size": right_alveolar_cleft_size,
                    "left_alveolar_cleft_size": left_alveolar_cleft_size
                })

        # Safety check that we don't compute any measurements here that are not yet part of the
        # supported ones.
        available_measurements = self.get_all_measurement_kinds()
        unexplained_measurements = [
            measurement_key for measurement_key in measurements if
            "side-adjusted" not in measurement_key and measurement_key not in available_measurements]
        if len(unexplained_measurements) > 0:
            warnings.warn("It seems you added a new measurement without adding an explanation to "
                          f"get_all_measurement_kinds(). Please add an entry for: {unexplained_measurements} ")
        return measurements

    @staticmethod
    def get_measurements_path(registered_mesh_path: Union[str, Path]) -> Path:
        """
        Returns the path where measured values are saved.
        """
        return Path(registered_mesh_path).with_name(f"{registered_mesh_path.stem}_measurements.json")

    @staticmethod
    def save_measurements(measurements: Dict, registered_mesh_path: Union[str, Path]) -> None:
        # Save the measurements in a json
        with open(CranialAttributes.get_measurements_path(registered_mesh_path), "w") as f:
            json.dump(measurements, f, indent=4)

    def get_measurements(self, registered_mesh_or_path: Union[str, Path, Mesh], recompute: bool = False) -> Dict:
        """
        Get the measurements for the provided mesh.
        :param registered_mesh_or_path: Mesh or path to one. Note that if you provide a Mesh,
                                        the measurements always need to be (re)computed, whereas
                                        if you provide a path, they can be loaded.
        :param recompute: Whether to recompute the measurements if you provided a path for the mesh,
                          and the respective measurements file already exists.
        :return Dictionary with measurements.
        """
        # If user supplied a mesh instead of a path, we don't know where any measurement file would be saved,
        # so we always (re)compute the measurements.
        if isinstance(registered_mesh_or_path, Mesh):
            measurements = self.compute_measurements_on_registered_mesh(registered_mesh_or_path)
        else:
            assert isinstance(registered_mesh_or_path, (str, Path))
            # Define measurement file path
            measurements_path = self.get_measurements_path(registered_mesh_or_path)

            # Load measurements if available and if they don't need to be recomputed
            if measurements_path.exists() and not recompute:
                with open(measurements_path, "r") as f:
                    measurements = json.load(f)

            # Otherwise compute them (again) and return them
            else:
                excluded_regions = RegistrationFileCommunicator.get_excluded_regions_on_registered_mesh(
                    registered_mesh_or_path)[0]
                measurements = self.compute_measurements_on_registered_mesh(
                    Mesh.load(registered_mesh_or_path), excluded_regions=excluded_regions)
                self.save_measurements(measurements, registered_mesh_or_path)
        return measurements


    @staticmethod
    def get_all_measurement_kinds() -> Dict[str, str]:
        """
        This method simply contains the name/kind and explanation for all measurements handled within this class.
        """
        return {
            "CVAI": "Cranial vault asymmetry index. Measures head asymmetry (ratio between the two head diagonals)."
                    "This one is not correlated with the model space; the signed version is used for that.",
            "CVA": "Absolute difference between head diagonals (unlike CVAI, which is relative)."
                   "This one is not correlated with the model space; the signed version is used for that.",
            "CI": "Cranial Index. Measures ratio between head with and length.",
            "CVAI_signed": "Cranial vault asymmetry index. Measures head asymmetry (ratio between the "
                           "two head diagonals), including plus minus.",
            "CVA_signed": "Absolute difference between head diagonals (unlike CVAI, which is relative), "
                          "including plus and minus.",
            "FI": "Frontal Index. Measures how pointy the head is at the front.",
            "SA": "Supraorbital angle -> Angle (radians) between two vectors. "
                  "The first connects the glabella with the SA eyebrow point on the left. "
                  "The second connects the glabella with the SA eyebrow point on the right. "
                  "Note that we project these points to the measurement plane "
                  "to avoid that raising the eyebrow on one side affects the computed angle.",
            "TI": "Head length divided by glabella-opisthocranion perimeter",
            "VNO": "Angle (degrees) between two vectors: the first connects the nasion and the vertex "
                   "(top of the head); the second connects the nasion and the opisthocranion.",
            "circumference": "Head circumference at the circumference plane. "
                             "The circumference plane is defined by including the glabella and opisthocranion "
                             "while being perpendicular to the nasion plane.",
            "NFA": "Nasofrontal Angle -> Angle (degrees) between two vectors: "
                   "the first connects the nasion and the glabella; "
                   "the second connects the nasion and the pronasale (nose tip).",
            "MI": "Metopic Index. Ratio between two distances. "
                  "The first is the distance between two points on the two eyebrows, "
                  "just where the cranium starts to curve toward the back. "
                  "The second distance is the head width, i.e., distance between two eurion points.",

            "TR": "Ratio between the distances between the right tragus and the nasion "
                  "and the left tragus and the nasion.",
            "TRx": "Ratio between the distances between the left tragus and the nasion and the right tragus "
                   "and the nasion; direction in x.",
            "TRy": "Ratio between the distances between the left tragus and the nasion and the right tragus "
                   "and the nasion; direction in y.",
            "TRz": "Ratio between the distances between the left tragus and the nasion and the right tragus "
                   "and the nasion; direction in z.",
            "TRtop": "Ratio between the distances between the right tragus and the top of the "
                     "head and the left tragus and the top of the head.",
            "TSR": "Ratio between the distances between the right tragus and the subnasale "
                   "and the left tragus and the subnasale.",
            "EOR": "Ratio between the distances between the right outer eye corner and the "
                   "nasion and the left outer eye corner and the nasion.",
            "EIR": "Ratio between the distances between the right inner eye corner and the "
                   "nasion and the left inner eye corner and the nasion.",
            "EWR": "Ratio between the right eye width and left eye width "
                   "(eye width is defined as the distance between outer and inner eye corner).",

            "v_front_left": "Volume of anterior left cranial quarter (no ratio, in mm^3)",
            "v_front_right": "Volume of anterior right cranial quarter (no ratio, in mm^3)",
            "v_back_left": "Volume of posterior left cranial quarter (no ratio, in mm^3)",
            "v_back_right": "Volume of posterior right cranial quarter (no ratio, in mm^3)",
            "v_tot": "Total cranial volume (no ratio, in mm^3)",
            "VR_front_back": "Ratio between the anterior and posterior cranial volume halves.",
            "VR_left_right": "Ratio between the left and right cranial volume halves.",
            "VR_back_left_right": "Ratio between the left and right posterior cranial volume quarters.",
            "VR_front_left_right": "Ratio between the left and right anterior cranial volume quarters.",
            "VR_left_front_back": "Ratio between the anterior and posterior left cranial volume quarters.",
            "VR_right_front_back": "Ratio between the anterior and posterior right cranial volume quarters.",

            "*_side-adjusted": "The distance and volume ratios between left and right are "
                               "adjusted here depending on the CVAI_signed. "
                               "For positive CVAI_signed, they stay the way they are, whereas for negative "
                               "ones, the ratio is flipped. These measurements are not correlated with the "
                               "model space, since their sign swapping makes them nonlinear.",

            "age": "Patient age in months",

            "right_lip_cleft_percentage": "How far the right lip cleft is open (percentage between 0 and 1, where "
                                          "0 means closed and 1 means full cleft).",
            "left_lip_cleft_percentage": "How far the left lip cleft is open (percentage between 0 and 1, where "
                                         "0 means closed and 1 means full cleft).",
            "cleft_palate_percentage": "How far the palatal cleft is open (percentage between 0 and 1, where "
                                        "0 means closed and 1 means full cleft).",
            "right_alveolar_cleft_size": "Actual size of right alveolar cleft.",
            "left_alveolar_cleft_size": "Actual size of left alveolar cleft.",
        }

    def get_measurement_kinds(self) -> Dict[str, str]:
        measurement_kinds = self.get_all_measurement_kinds()
        if hasattr(self, "shape_factor_function"):
            measurement_kinds = {k: v for k, v in measurement_kinds.items() if k in self.shape_factor_function}
        return measurement_kinds

    @staticmethod
    def compute_shape_factor_function_ls(
            vertices_projected: np.ndarray, labels: Union[List, np.ndarray], label_name: str,
            limit_components: Union[List, np.ndarray, None] = None,
            balance_pos_weights: bool = False,
            transform: Union[str, None] = None) -> Tuple[np.ndarray, float, float]:
        """
        Fit a linear regressor to the vertex projections to predict the given labels, using regularized least squares.
        Different regularizations are evaluated on a 5-fold cross validation of the data, and the setting with the
        lowest test loss is used.
        This function requires the sklearn library.
        :param vertices_projected: The mesh vertices, projected into a model's latent space.
        :param labels: The label values (list or np array of floats) that the regressor should predict.
        :param label_name: Name of the label; only used for print feedback.
        :param label_name: Name of the label; only used for print feedback.
        :param limit_components: Optionally provide indices for specific latent components to limit the regressor to.
                                 This can be used as additional regularizer to avoid overfitting.
        :param balance_pos_weights: Set True for non-uniform weight for fitting.
                                    Specifically, weigh entries according to zero vs non-zero label frequency.
        :param transform: Optional string that specifies how to transform the labels before training the linear
                          regressor on the transformed labels. Supported transforms:
                          1) 'rootx', where x can be any positive number, so e.g., 'root2', 'root3', 'root1.5'
                          2) 'powx', cf. above
                          3) 'logx' or just 'log', where x can be any positive number like above, excluding 1,
                              and without, e is used.
                          4) 'expx', cf. above
                          5) 'piecewise_linear': Implements a piecewise linear function,
                                                 alternatingly optimized with the linear regressor.
        :return Tuple:
                    1) Numpy weight vector (same size as model latent space)
                    2) label average (this is important, since we first center the labels before fitting the regressor,
                       so the actual label prediction is the weight vector times the projection plus this mean).
                    3) label standard deviation (not so important, but gives more info about the label range).
        """
        from sklearn.linear_model import Ridge
        from sklearn.model_selection import KFold
        from sklearn.isotonic import IsotonicRegression

        def fit_current_labels(current_labels: np.ndarray) -> Tuple[np.ndarray, float, float]:
            m, d = vertices_projected.shape
            assert len(current_labels) == m
            current_labels = np.asarray(current_labels)
            label_average, label_stdev = float(np.mean(current_labels)), float(np.std(current_labels))
            # We center the labels, so we don't need to include an offset in the least squares optimization,
            labels_normed = (current_labels - label_average)

            sample_weight = np.ones_like(current_labels, dtype=float)
            if balance_pos_weights:
                sample_weight[current_labels > 0] = (len(current_labels) - (current_labels > 0).sum()) / max((current_labels > 0).sum(), 1)  # inverse freq

            def get_weight_vector(X_current: np.ndarray, labels_normed_current: np.ndarray,
                                  current_alpha: float, sample_weight_current) -> np.ndarray:
                # We give the user the option to only optimize the weight on a subset of the components of X
                def get_fit(data):
                    model = Ridge(alpha=current_alpha, fit_intercept=False)
                    model.fit(data, labels_normed_current, sample_weight=sample_weight_current)
                    return model.coef_

                if limit_components is not None:
                    w_est_current = np.zeros(X_current.shape[1])
                    w_est_current[limit_components] = get_fit(X_current[:, limit_components])
                else:
                    w_est_current = get_fit(X_current)
                return w_est_current

            label_stats_per_alpha = {}
            # We loop over different regularization settings to find the best ls estimator
            for alpha in [0, 0.01, 0.1, 1, 10, 100, 1000, 10000]:
                # Only for testing the accuracy, we use k-fold cross validation
                kf = KFold(n_splits=5, shuffle=False)
                label_diff = []
                for train_index, test_index in kf.split(vertices_projected):
                    X_train, X_test = vertices_projected[train_index], vertices_projected[test_index]
                    Y_train = labels_normed[train_index]
                    # Get the least-squares weight vector for the current training set with the alpha regularizer
                    w_est_kfold = get_weight_vector(X_train, Y_train, current_alpha=alpha,
                                                    sample_weight_current=sample_weight[train_index])
                    # Evaluate the weight matrix on the test set
                    labels_estimated = np.einsum("i,ji->j", w_est_kfold, X_test) + label_average
                    label_diff.append(current_labels[test_index] - labels_estimated)

                # Keep track of the results over all folds
                label_diff = np.concatenate(label_diff, axis=0)
                label_stats_per_alpha[alpha] = (np.mean(np.abs(label_diff)), np.std(np.abs(label_diff)))

            # Choose the best alpha as the one that achieves the lowest average + stdev test label error
            best_alpha = min(label_stats_per_alpha, key=lambda p: label_stats_per_alpha[p][0] + label_stats_per_alpha[p][1])
            best_label_stats = label_stats_per_alpha[best_alpha]
            print(f"Label {label_name} stats: distribution {label_average:.2f} +- {label_stdev:.2f}. "
                  f"Estimator with {best_alpha=} diff: {best_label_stats[0]:.2f} +- {best_label_stats[1]:.2f}")

            # Get the ls weight vector again with the best alpha trained on the whole dataset
            w_est = get_weight_vector(vertices_projected, labels_normed, best_alpha, sample_weight_current=sample_weight)

            # Return weight, mean, and stdev
            return w_est, label_average, label_stdev

        if transform is None:
            return fit_current_labels(labels)
        else:
            original_mean, original_stdev = np.mean(labels), np.std(labels)
            transform = transform.lower()
            # Simple transforms involve taking the n-
            if "root" in transform or "pow" in transform or "log" in transform or "exp" in transform:
                try:
                    num = float(transform.replace("root", "").replace("power", "").replace("pow", "").replace(
                        "log", "").replace("exp", "").rstrip())
                except ValueError as e:
                    if "pow" in transform or "root" in transform:
                        raise ValueError(f"If you specify a root or power in your transform, "
                                         f"include the number in there as well "
                                         f"and nothing else. E.g., 'root3', 'pow3.5'. "
                                         f"You provided {transform}. Error: {e}")
                    else:
                        num = np.e
                else:
                    if num <= 0:
                        raise ValueError(f"Please choose a transform with a positive number. You chose {transform}.")
                    elif num == 1:
                        if "exp" in transform or "log" in transform:
                            raise ValueError(f"You chose exp/log as transform, where 1 cannot be the base. "
                                             f"Choose any other positive base.")
                        else:
                            warnings.warn(f"You chose pow/root as transform, but 1 as exponent, which is kinda "
                                          f"unnecessary, since this doesn't do any transformation. "
                                          f"Make sure this is what you intended to do.")
                if "pow" in transform or "root" in transform:
                    pow_factor = num if "pow" in transform else 1/num
                    return *fit_current_labels(np.power(labels, pow_factor)), "power", pow_factor, original_mean, original_stdev
                else:
                    if "log" in transform:
                        return *fit_current_labels(np.log(labels) / np.log(num)), "log", num, original_mean, original_stdev
                    else:
                        return *fit_current_labels(np.power(num, labels)), "exp", num, original_mean, original_stdev
            elif "piece" in transform and "lin" in transform:
                # Initialize mapping function
                t = (labels - np.mean(labels)) / np.std(labels)  # todo: maybe remove division by std?
                # repeat until convergence
                while True:
                    w_current, avg_current, stdev_current = fit_current_labels(t)
                    score = np.einsum("i,ji->j", w_current, vertices_projected) + avg_current

                    # 2. Fit monotone function -> morphology-score mapping
                    isotonic = IsotonicRegression(
                        increasing=True,
                        out_of_bounds="clip",
                    ).fit(labels, score)

                    label_knots = isotonic.X_thresholds_.copy()
                    score_knots = isotonic.y_thresholds_.copy()

                    # Isotonic regression can contain flat intervals. Add a very small
                    # positive slope to make the mapping strictly invertible.
                    label_position = (
                            (label_knots - label_knots[0])
                            / (label_knots[-1] - label_knots[0])
                    )
                    score_knots += 1e-5 * label_position

                    # Normalize the learned transformed label coordinate.
                    t_new = CranialAttributes.piecewise_linear_transform(labels, label_knots, score_knots)
                    location = t_new.mean()
                    scale = t_new.std()

                    score_knots = (score_knots - location) / scale
                    t_new = (t_new - location) / scale

                    change = np.sqrt(np.mean((t_new - t) ** 2))
                    t = t_new

                    if change < 1e-5:
                        break
                return w_current, avg_current, stdev_current, label_knots, score_knots, original_mean, original_stdev
            else:
                raise ValueError(f"You provided {transform} as a transform. We only support"
                                 f"roots, powers, and piecewise linear currently.")


    @staticmethod
    def piecewise_linear_transform(label: float, label_knots: np.ndarray, score_knots: np.ndarray) -> float:
        return np.interp(label, label_knots, score_knots)

    @staticmethod
    def piecewise_linear_inverse_transform(label: float, label_knots: np.ndarray, score_knots: np.ndarray) -> float:
        return np.interp(label, score_knots, label_knots)

    def transform_label(self, label: float, measurement_kind: str) -> float:
        if measurement_kind not in self.measurement_mappings:
            return label
        else:
            mappings = self.measurement_mappings[measurement_kind]
            if isinstance(mappings[0], str):
                if "pow" in mappings[0].lower():
                    return np.power(label, mappings[1])
                elif "log" in mappings[0].lower():
                    return np.log(label) / np.log(mappings[1])
                elif "exp" in mappings[0].lower():
                    return np.power(mappings[1], label)
            else:
                return self.piecewise_linear_transform(label, mappings[0], mappings[1])

    def inverse_transform_label(self, label: float, measurement_kind: str) -> float:
        if measurement_kind not in self.measurement_mappings:
            return label
        else:
            mappings = self.measurement_mappings[measurement_kind]
            if isinstance(mappings[0], str):
                if "pow" in mappings[0].lower():
                    return np.power(label, 1/mappings[1])
                elif "log" in mappings[0].lower():
                    return np.power(mappings[1], label)
                elif "exp" in mappings[0].lower():
                    return np.log(label) / np.log(mappings[1])
            else:
                return self.piecewise_linear_inverse_transform(label, mappings[0], mappings[1])

    def transform_labels(self, labels: np.ndarray, measurement_kinds: List[str]) -> np.ndarray:
        return np.asarray([self.transform_label(label, measurement_kind)
                           for label, measurement_kind in zip(labels, measurement_kinds)])

    def inverse_transform_labels(self, labels: np.ndarray, measurement_kinds: List[str]) -> np.ndarray:
        return np.asarray([self.inverse_transform_label(label, measurement_kind)
                           for label, measurement_kind in zip(labels, measurement_kinds)])

    def get_original_mean(self, measurement_kind: str) -> float:
        if measurement_kind in self.measurement_mappings:
            return self.measurement_mappings[measurement_kind][2]
        else:
            return self.measurements_mean[measurement_kind]

    def get_original_stdev(self, measurement_kind: str) -> float:
        if measurement_kind in self.measurement_mappings:
            return self.measurement_mappings[measurement_kind][3]
        else:
            return self.measurements_mean[measurement_kind]

    def get_measurement_vectors(self, *measurement_kinds: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Return the measurement vectors, so weights, mean, and stdev as numpy arrays for the provided measurement kinds.
        :param measurement_kinds: measurement kinds (str) to get the measurement vectors for; order is preserved.
        :return Tuple:
                    1) weight matrix [len(measurement_kinds) x latent_size]
                    2) averages [len(measurement_kinds)]
                    3) standard deviations [len(measurement_kinds)]
        """
        measurement_kinds_filtered = [measurement_kind for measurement_kind in measurement_kinds
                                      if measurement_kind in self.measurements_w]
        measurements_w = np.asarray([self.measurements_w[measurement_kind]
                                     for measurement_kind in measurement_kinds_filtered])
        measurements_mean = np.asarray([self.measurements_mean[measurement_kind]
                                        for measurement_kind in measurement_kinds_filtered])
        measurements_stdev = np.asarray([self.measurements_stdev[measurement_kind]
                                         for measurement_kind in measurement_kinds_filtered])

        return measurements_w, measurements_mean, measurements_stdev

    def has_age(self):
        """
        This function specifically checks the instance's shape factor function includes an age regressor,
        since this measurement needs to be provided by the user when a new shape factor function is trained.
        """
        return "age" in self.shape_factor_function

    def get_measurement_corrected_meshes(
            self, registered_meshes: List[Mesh], measurement_kinds: List[str],
            measurement_targets: Union[List[Union[float, None]], float, None] = None,
            fix_age: bool = False, fix_total_volume: bool = False,
            highlight_difference: bool = True) -> Tuple[List[Mesh], np.ndarray, np.ndarray, np.ndarray]:
        """
        Corrected multiple meshes to corresponding versions with healthy craniums (based on the measurements
        to be provides as arguments).
        :param registered_meshes: List of registered meshes (type Mesh)
        :param measurement_targets: Optionally provide desired values for each measurement that deviate from the average.
                                    This is already a list/array; it needs to be in the order that the keys in the
                                    shape_factor_function are sorted.
        :param fix_age: There is no "optimal" age to adjust the cranium to, but you can choose this arg to avoid that
                        by shifting the space for the other measurements also shifts the age, so you can fix the age
                        while varying the other attributes.
        :param fix_total_volume: Fix the total volume of total cranium, like you can fix the age (cf. fix_age).
                                 This can be used to avoid that the cranium is expanded when correcting the other
                                 attributes, thus serving of a better reference for a doctor showing what can be done
                                 during surgery (the head doesn't grow; the skull can be moved, but not expanded).
        :param highlight_difference: Assign colors to corrected meshes that represent the difference between
                                     the input and the corrected mesh. Set False to disable this.
        :return Tuple
                    1) List of adjusted/corrected meshes, length n
                    2) np float array of signed vertex distance between input and corrected meshes [n x num_vertices]
                    3) np float array of estimated measurements [num_measurements x n]
                    4) np float array of values measurements were adjusted to [num_measurements x n]
        """

        measurement_kind_indices = [i for i, measurement_kind in enumerate(measurement_kinds)
                                    if measurement_kind in self.shape_factor_function]
        measurement_kinds = [measurement_kinds[i] for i in measurement_kind_indices]

        num_measurement_kinds_without_fix = len(measurement_kinds)

        if fix_age and not self.has_age():
            warnings.warn("Age cannot be fixed during shape correction, since the shape factor function wasn't "
                          "trained to estimate the age. This is probably, because you didn't supply the age "
                          "during training.")
            fix_age = False

        if fix_age:
            assert "age" not in measurement_kinds
            measurement_kinds = [*measurement_kinds, "age"]

        if fix_total_volume:
            assert "total_volume" not in measurement_kinds
            measurement_kinds = [*measurement_kinds, "v_tot"]

        # Projected registered meshes into PCA space
        registered_vertices_projected = self.model.get_projected_vertices(
            *registered_meshes, normalize_projection=True)

        # Estimate all measurements for the meshes
        # The estimates are in the original space, not the mapped space (apply_back_mapping=True)
        measurements_estimated = self.get_estimated_measurements(
            registered_vertices_projected=registered_vertices_projected, measurement_kinds=measurement_kinds,
            apply_back_mapping=True)

        _, measurements_mean, *_ = self.get_measurement_vectors(*measurement_kinds)

        # Initialize the target for each measurement, i.e., to which value to correct the meshes for each measurement
        if not isinstance(measurement_targets, (list, np.ndarray)):
            measurement_targets = [measurement_targets]
        measurement_targets = [measurement_targets[i] for i in measurement_kind_indices]

        # We use the optimal target value, where availabel, and for the others
        # we take the average value from our dataset.
        # The targets are also in the original space, not the mapped space
        # (which is also why we take the original mean here).
        measurement_targets = np.asarray([
            measurement_optimum if measurement_optimum is not None else self.get_original_mean(measurement_kind)
            for measurement_optimum, measurement_mean, measurement_kind in
            zip(measurement_targets, measurements_mean, measurement_kinds)])

        # If we fix the age, we add the estimated age to the target
        # (so the corrected latent has the same age as the input latent)
        # Since the estimated measurements are already in the original space,
        # we don't need to invert the value here.
        if fix_age:
            if measurement_targets.ndim == 1:
                measurement_targets = np.expand_dims(measurement_targets, axis=1).repeat(len(registered_meshes), axis=1)
            est_start_ind = num_measurement_kinds_without_fix
            measurement_targets = np.concatenate(
                (measurement_targets, measurements_estimated[est_start_ind:est_start_ind + 1]), axis=0)

        # Same as for age, we also add the total volume to the target
        if fix_total_volume:
            if measurement_targets.ndim == 1:
                measurement_targets = np.expand_dims(measurement_targets, axis=1).repeat(len(registered_meshes), axis=1)
            est_start_ind = num_measurement_kinds_without_fix + 1 if fix_age else num_measurement_kinds_without_fix
            measurement_targets = np.concatenate(
                (measurement_targets, measurements_estimated[est_start_ind:est_start_ind + 1]), axis=0)

        # Do the correction
        vertices_shifted_measurement = self.correct_estimated_measurements(
            measurements_estimated=measurements_estimated, measurement_kinds=measurement_kinds,
            measurement_targets=measurement_targets, registered_vertices_projected=registered_vertices_projected,
            are_measurements_in_original_space=True
        )

        # Create meshes from shifted vertices
        meshes_shifted_measurements = [
            Mesh(vertices=vertices_shifted_measurement_ind, triangles=registered_mesh.get_triangles())
            for vertices_shifted_measurement_ind, registered_mesh in
            zip(vertices_shifted_measurement, registered_meshes)]

        vertex_errors = []
        for mesh_shifted, registered_mesh in zip(meshes_shifted_measurements, registered_meshes):
            vertex_diff = mesh_shifted.get_vertices(copy=False) - registered_mesh.get_vertices(copy=False)
            # Negative distances are determined via the vertex normals of the registered mesh
            vertex_diff_sign = np.sign(utils.dot_vector_list(vertex_diff, registered_mesh.get_vertex_normals()))
            vertex_error = np.linalg.norm(vertex_diff, axis=1) * vertex_diff_sign
            vertex_errors.append(vertex_error)
            # Assign colors for the vertex difference between input and corrected mesh.
            # The error range is -5 units to 5 units.
            if highlight_difference:
                mesh_shifted.set_colors(utils.get_error_heatmap_colors(vertex_error, min_value=-5, max_value=5))
            else:
                if registered_mesh.has_colors():
                    mesh_shifted.set_colors(registered_mesh.get_colors(copy=True))

        # Remove the estimated and target values for the fixed values
        if fix_total_volume:
            measurements_estimated, measurement_targets = measurements_estimated[:-1], measurement_targets[:-1]
        if fix_age:
            measurements_estimated, measurement_targets = measurements_estimated[:-1], measurement_targets[:-1]

        # corrected meshes, each color coded with error, the estimated measurements and what value they were corrected to
        return meshes_shifted_measurements, np.asarray(vertex_errors), measurements_estimated, measurement_targets

    def correct_estimated_measurements(
            self, measurements_estimated: np.ndarray,
            measurement_kinds: List[str], measurement_targets: Union[List, np.ndarray],
            registered_vertices_projected: np.ndarray, are_measurements_in_original_space: bool = True) -> np.ndarray:
        """
        Correct/adjust the 3DMM projections to match desired measurement values.
        Note that the measurement
        :param measurements_estimated: Estimated values for the measurements to be adjusted.
        :param measurement_kinds: List of all measurement kinds (str) corresponding to the provided
                                  measurement estimates and targets.
        :param measurement_targets: Values to adjust the measurements to by changing the latents.
        :param registered_vertices_projected: Latents (projections of vertices into the model space)
                                              [num_meshes x 3DMM_latent_size]
        :param are_measurements_in_original_space: By default, the provided measurement estimates and the targets
                                                   are assumed to not have been mapped yet, so before the latent
                                                   correction, they are first mapped. Set False if you provide
                                                   them already provide them in mapped space.
        :return numpy array [num_meshes x num_vertices x 3]
        """

        measurement_targets = np.asarray(measurement_targets)
        if measurement_targets.ndim == 1:
            measurement_targets = np.expand_dims(measurement_targets, axis=1)
        if are_measurements_in_original_space:
            measurements_estimated = self.transform_labels(measurements_estimated, measurement_kinds)
            measurement_targets = self.transform_labels(measurement_targets, measurement_kinds)

        measurements_w, _, _ = self.get_measurement_vectors(*measurement_kinds)

        # We use ls solving instead of computing the inverse, since the W @ W^T might not be invertible.
        # It's similar to computing the pseudo-inverse.
        x = np.linalg.lstsq(measurements_w @ measurements_w.T, measurements_w, rcond=None)[0]
        r = (measurement_targets - measurements_estimated)
        delta_projection = r.T @ x

        projection_shifted_measurement = self.model.unnormalize_weights(
            registered_vertices_projected + delta_projection)
        vertices_shifted_measurement = self.model.decode(projection_shifted_measurement)
        return np.reshape(vertices_shifted_measurement, (len(vertices_shifted_measurement), -1, 3))

    def get_estimated_measurements(
            self, registered_vertices_projected: np.ndarray, measurement_kinds: List[str],
            apply_back_mapping: bool = True) -> np.ndarray:
        """
        Compute the estimated measurements for the provided vertex projections into model space for each
        of the provided measurement kinds.
        :param registered_vertices_projected: Latents (projections of vertices into the model space)
                                              [num_meshes x 3DMM_latent_size]
        :param measurement_kinds: list of strings representing the measurement kinds
        :param apply_back_mapping: By default, the estimated measurements are mapped back to their original space,
                                   so that they are more interpretable. Set this to False to get the mapped measurments.
        :return estimated measurements [len(measurement_kinds) x num_meshes]
        """
        measurements_w, measurements_mean, _ = self.get_measurement_vectors(*measurement_kinds)

        measurements_estimated = np.einsum(
            "ki,ji->kj", measurements_w, registered_vertices_projected) + np.expand_dims(measurements_mean, axis=1)

        if apply_back_mapping:
            measurements_estimated = self.inverse_transform_labels(measurements_estimated, measurement_kinds)

        return measurements_estimated


    def get_fully_corrected_meshes(
            self, registered_meshes: List[Mesh], fix_age: bool = True,
            fix_total_volume: bool = True, highlight_difference: bool = True) -> Tuple[List[Mesh], np.ndarray, np.ndarray, np.ndarray]:
        """
        Fix cranial meshes based on all measurements, we deemed relevant for medical purposes.
        Cf. docs of get_measurement_corrected_meshes for an explanation of the arguments and return variables.
        """

        measurement_kinds = [measurement_kind for measurement_kind in self.measurements_relevant_for_correction
                             if measurement_kind in self.shape_factor_function]

        measurement_targets = [
            self.measurement_optima[measurement_kind] if measurement_kind in self.measurement_optima else None
            for measurement_kind in measurement_kinds]

        corrected_meshes, vertex_errors, measurements_est, measurements_adjusted = self.get_measurement_corrected_meshes(
            registered_meshes=registered_meshes,
            measurement_kinds=self.measurements_relevant_for_correction, measurement_targets=measurement_targets,
            fix_age=fix_age, fix_total_volume=fix_total_volume, highlight_difference=highlight_difference
        )

        return corrected_meshes, vertex_errors, measurements_est, measurements_adjusted


    def correct_meshes_from_paths(
            self, *registered_mesh_paths: Union[str, Path], fix_age: bool = True, fix_total_volume: bool = True,
            highlight_difference: bool = True, overwrite: bool = False,
            region_measure_errors: Union[List[Union[str, Path]], None] = None,
            mesh_unit: Union[str, None] = None, num_jobs: int = 1) -> None:
        """
        This method differs from get_fully_corrected_meshes() in that it works with file loading and writing.
        You give it an arbitrary number of mesh paths, and it loads each mesh, corrects the cranial shape,
        and then saves the result under "INPUTMESHSTEM_corrected.INPUTMESHTYPE". It additionally saves a file
        "INPUTMESHSTEM_correction_numbers.txt" specifying the estimated values for each measurement and the values
        they were adjusted to; additionally, the difference between the corrected and input meshes are logged.
        :param registered_mesh_paths: Paths to registered meshes.
        :param fix_age: cf. docs of get_measurement_corrected_meshes()
        :param fix_total_volume: cf. docs of get_measurement_corrected_meshes()
        :param highlight_difference: cf. docs of get_measurement_corrected_meshes()
        :param region_measure_errors: Optionally provide one or more model indices or paths to index selections to
                                      measure the difference between the corrected meshes and their input in.
        :param overwrite: Set True to overwrite an existing file for cranial shape correction.
        :param mesh_unit: Optionally provide the mesh unit, such that it is included in the saved text file.
        """
        def correct_ind(registered_mesh_path: Union[str, Path]):
            # Define file output paths using input mesh path as reference
            corrected_mesh_path = registered_mesh_path.with_stem(f"{registered_mesh_path.stem}_corrected")
            corrected_numbers_path = registered_mesh_path.with_name(f"{registered_mesh_path.stem}_correction_numbers.txt")

            measurements = self.get_measurements(registered_mesh_path)

            # Only process the mesh if it hasn't been processed before or if the user chose to overwrite
            if (not corrected_mesh_path.exists() or not corrected_numbers_path.exists()) or overwrite:
                # Load mesh and correct it
                registered_mesh = Mesh.load(registered_mesh_path)
                corrected_meshes, vertex_errors, measurements_est, measurements_adjusted = self.get_fully_corrected_meshes(
                    [registered_mesh], fix_age=fix_age, fix_total_volume=fix_total_volume,
                    highlight_difference=highlight_difference)

                # Export corrected mesh
                corrected_meshes[0].export(corrected_mesh_path)

                # Save the correction numbers into a text file
                with open(corrected_numbers_path, "w") as f:
                    # Here we write for each measurement kind, which value was estimated for it and which
                    # value it was adjusted to
                    for measurement_kind, measurement_est, measurement_adjusted in zip(
                            self.measurements_relevant_for_correction, measurements_est[:, 0], measurements_adjusted[:, 0]):
                        f.write(f"{measurement_kind}: {measurement_est:.2f} -> {measurement_adjusted:.2f}\n")

                    def write_errors_per_region(current_vertex_errors: np.ndarray):
                        f.write(f"Total: {np.mean(np.abs(current_vertex_errors)):.3f} +- "
                                f"{np.std(np.abs(current_vertex_errors)):.3f}\n")
                        if region_measure_errors is not None:
                            # If the user has provided indices, we also log the vertex difference in each of these regions
                            for region_measure_error in region_measure_errors:
                                # A region can be defined as one of the pre-defined model indices
                                if self.model.has_index(str(region_measure_error)):
                                    indices = self.model.get_indices(region_measure_error)
                                    index_name = region_measure_error
                                # or it can be a path to a custom index selection
                                else:
                                    indices = IndicesAndMasks.load(region_measure_error)
                                    index_name = Path(region_measure_error).stem
                                f.write(
                                    f"{index_name}: {np.mean(np.abs(current_vertex_errors[indices])):.3f} +- "
                                    f"{np.std(np.abs(current_vertex_errors[indices])):.3f}\n")

                    # Here we write the mean and standard deviation of the absolute vertex difference
                    # between input and corrected mesh. We don't use the signed distance, because
                    # averaging that over the vertex dimension would only yield values close to 0
                    f.write(f"\nAbsolute difference between input and optimum")
                    f.write(("in the unit of the mesh" if mesh_unit is None else f"[{mesh_unit}]") + "\n")
                    write_errors_per_region(vertex_errors)

                    if "circumference" in measurements:
                        # Here we write the mean and standard deviation of the relative vertex difference
                        # between input and corrected mesh. We don't use the signed distance, because
                        # averaging that over the vertex dimension would only yield values close to 0
                        f.write(f"\nRelative difference between input and optimum "
                                f"(absolute difference divided by head circumference; no unit)\n")
                        write_errors_per_region(vertex_errors / measurements["circumference"])
        utils.parallel_method(correct_ind, iterator=registered_mesh_paths, num_jobs=num_jobs,
                              tqdm_message="Correcting shapes")

    def measure_meshes_and_compute_shape_factor_function(
            self, *registered_mesh_paths: Union[str, Path], shape_factor_function_path: Union[str, Path],
            limit_components: Union[List, np.ndarray, None] = None, overwrite_measurements: bool = False,
            retrain_shape_factor_function: bool = False,
            ages_per_mesh: Union[str, Path, Dict[str, float], None] = None,
            use_nonlinear_mappings: bool = False,
            num_jobs: int = 1) -> None:
        """
        This is also a method that includes file input and output.
        You give it some paths to registered meshes and it computes the cranial measurements/attributes
        for each and saves them in the same folder under the name "INPUTMESHNAME_measurements.json".
        Additionally, it either loads the shape factor function from the provided path or trains a new one.
        This function will also load the shape factor function into the current instance.
        :param registered_mesh_paths: Paths to registered meshes.
        :param shape_factor_function_path: Path to the shape factor function. If the path doesn't exist yet, or
                                           if you set retrain_shape_factor_function True,
                                           a new function will be trained based on the
                                           measurements computed on the registered meshes you provided.
        :param limit_components: Optionally specify which model components to correlate to the measurements.
        :param overwrite_measurements: Set True to overwrite existing measurement files
        :param retrain_shape_factor_function: Set True to retrain shape factor function and overwrite existing file.
        :param ages_per_mesh: To include the age into the measurements file, as well as to train a new shape factor
                              function to estimate the age, you need to provide the ages for the subjects as a
                              dictionary with the keys named after the mesh file names (absolute or just the name)
                              and the values equal to the subject's age (in months). This argument can also be a file
                              path in which case it's loaded either via json or pickle.
        :param use_nonlinear_mappings: Set True for an experimental nonlinear mapping of certain attributes,
                                       cf. class documentation.
        :param num_jobs: Optionally choose number of processors to run computations in parallel.
        :return: None
        """
        # Load ages per mesh if available
        if ages_per_mesh is not None:
            if isinstance(ages_per_mesh, (str, Path)):
                ages_per_mesh_path = Path(ages_per_mesh)
                if ages_per_mesh_path.suffix == ".json":
                    with open(ages_per_mesh_path) as f:
                        ages_per_mesh: Dict[str, float] = json.load(f)
                elif ages_per_mesh_path.suffix.lower() in [".pkl", ".pickle"]:
                    with open(ages_per_mesh_path, "rb") as f:
                        ages_per_mesh: Dict[str, float] = pickle.load(f)
                else:
                    with open(ages_per_mesh_path, "r") as f:
                        ages_per_mesh_lines = f.readlines()
                    ages_per_mesh_lines_split = [line.rstrip().replace(" ", "").split(":")
                                                 for line in ages_per_mesh_lines]
                    if not all([len(line) == 2 for line in ages_per_mesh_lines_split]):
                        raise ValueError("Provided ages file does contain simple 'key: value' pairs.")
                    ages_per_mesh: Dict[str, float] = {
                        line[0]: float(line[1]) for line in ages_per_mesh_lines_split}
                    if not len(ages_per_mesh) == len(ages_per_mesh_lines_split):
                        line_keys = [line[0] for line in ages_per_mesh_lines_split]
                        from collections import Counter
                        duplicates = [x for x, n in Counter(line_keys).items() if n > 1]
                        warnings.warn(f"You have duplicated entries in your ages file: {duplicates}. "
                                      f"You can use a larger part of the path to make sure they're unique.")

        def compute_measurements_local(registered_mesh_path: Path):
            registered_mesh = Mesh.load(registered_mesh_path)

            measurements = self.get_measurements(registered_mesh_path, recompute=overwrite_measurements)

            # Add age to measurements if it can be found in the dictionary
            if ages_per_mesh is not None:
                # We first check if the full path is in the dict, then the path relative to the top folder,
                # then to the second top, etc., going down to the file name, then to the file stem.
                parts = registered_mesh_path.parts
                age = next((ages_per_mesh[key] for key in (*(str(Path(*parts[i:])) for i in range(
                    len(parts))), registered_mesh_path.stem) if key in ages_per_mesh), None)
                if age is not None:
                    measurements["age"] = float(age)

            if overwrite_measurements:
                self.save_measurements(measurements, registered_mesh_path)

            # Keep track of the latents and corresponding computed measurements
            latent = self.model.get_projected_vertices(registered_mesh, normalize_projection=True)[0]
            return latent, measurements

        # Do the actual measurement computation in parallel and save the output per mesh (latent, measurement)
        # in a list
        registered_mesh_projections_and_measurements = utils.parallel_method(
            compute_measurements_local, iterator=registered_mesh_paths, num_jobs=num_jobs,
            tqdm_message="Computing/Loading measurements")

        # Now we load/train the shape factor function (mapping from latents to measurements).
        shape_factor_function_path = Path(shape_factor_function_path).with_suffix(".json")

        # Train shape factor function on the just computed measurement,
        # if the shape factor function is not available yet, or retrain if chosen by user
        if not shape_factor_function_path.exists() or retrain_shape_factor_function:
            # Prepare parameters
            measurement_kinds = CranialAttributes.get_all_measurement_kinds()
            shape_factor_function = {}
            vertices_projected = np.asarray([latent for (latent, _) in registered_mesh_projections_and_measurements])

            # Loop over each measurement kind and train a separate linear regressor for each
            for measurement_kind in measurement_kinds.keys():
                # we skip the ones that cannot be correlated well with the model space.
                if measurement_kind in self.exclude_measurements_from_training:
                    continue
                # measurement mask allows to skip certain meshes for training, e.g.,
                # if age is not available for some of them.
                measurement_mask = [
                    num_mesh for num_mesh, (_, measurements) in
                    enumerate(registered_mesh_projections_and_measurements)
                    if measurement_kind in measurements]
                # Sanity-check availability of the measurement
                if len(measurement_mask) == 0:
                    warnings.warn(f"Couldn't find any meshes for which {measurement_kind} is available.")
                    if measurement_kind == "age":
                        warnings.warn("When training your shape factor function, you should provide a file that "
                                      "specifies the age of each subject in months.")
                    continue

                # Define labels, so the measurements computed for the meshes
                # Unavailable ones are skipped
                labels = np.asarray([
                    measurement[measurement_kind] for (_, measurement) in registered_mesh_projections_and_measurements
                    if measurement_kind in measurement])

                if use_nonlinear_mappings:
                    transform_func = self.nonlinear_mappings.get(measurement_kind, None)
                else:
                    transform_func = None

                # Train the linear regressor on the available projection label pairs
                shape_factor_output = CranialAttributes.compute_shape_factor_function_ls(
                    vertices_projected=vertices_projected[measurement_mask], labels=labels, label_name=measurement_kind,
                    limit_components=limit_components, transform=transform_func,
                    # Clefts are very unevenly distributed and 0 in all healthy cases, so we balance the weighting
                    # for training.
                    balance_pos_weights="cleft" in measurement_kind.lower()
                )

                # Save the weight, average, standard deviation, and a description of the measurement
                # into the shape factor function
                shape_factor_function[measurement_kind] = (
                    list(shape_factor_output[0]), shape_factor_output[1],
                    shape_factor_output[2], measurement_kinds[measurement_kind],
                    *[list(el) if isinstance(el, np.ndarray) else el for el in shape_factor_output[3:]])

            # Save the regressors for all measurements
            with open(shape_factor_function_path, "w") as f:
                json.dump(shape_factor_function, f, indent=4)

        # Load the trained linear regressors into the current instance
        self.load_shape_factor_function(shape_factor_function_path)



def main():
    parser = ArgumentParser()
    parser.add_argument('--path_to_hdf5_file', type=str, required=True,
                        help="Absolute path to autoencoder or PCA model file with .h5 ending.")
    parser.add_argument('--path_to_correlated_attributes', type=str, required=True,
                        help="Absolute path to json file that encodes correlated cranial attributes "
                             "(must fit to provided model). If it doesn't exist, a new correlator"
                             "is trained and saved under this file path.")
    parser.add_argument('--input_files_or_dirs', type=str, nargs="+", required=True,
                        help="Directory of registered dataset, or path(s) to a specific registered file(s).")
    parser.add_argument('--mesh_regex', default=None, type=str, nargs="+",
                        help="Filter the meshes you want to load. "
                             "You can specify one or several regexes, e.g. '0001_*' to only "
                             "load meshes that start with '0001_'.")
    parser.add_argument('--mesh_exclude_regex', default=[], type=str, nargs="+",
                        help="Negative filtering of meshes to load. "
                             "All files that match the regexes you supply here will not be registered.")
    parser.add_argument('--limit_components', default=None, type=str, nargs="+",
                        help="Optionally limit the correlation of the cranial attributes to the model "
                             "to specific components of the model. You can provide each component index here."
                             "If you train the correlation on PCA, you might want to just correlate with just the "
                             "first N components. In that case, provide ':N' as input, e.g., ':64' "
                             "if you want to train on the first 64 components. You may also combine both notations, "
                             "e.g., ':32 48' will include components from 0 to 31 as well as 48.")
    parser.add_argument('--use_nonlinear_mappings', default=False, action="store_true",
                        help="Choose this for an experimental nonlinear mapping of certain attributes when training "
                             "the shape factor function.")
    parser.add_argument('--region_measure_errors', default=None, type=str, nargs="+",
                        help="Optionally provide model indices or paths to custom indices to measure "
                             "the vertex difference between input and corrected meshes in.")
    parser.add_argument('--overwrite', default=False, action="store_true",
                        help="Overwrite existing measurement files as well as corrected shape meshes.")
    parser.add_argument('--skip_correction', default=False, action="store_true",
                        help="Choose this to skip the shape correction, so stop after computing measurements "
                             "and training shape factor function.")
    parser.add_argument('--retrain_shape_factor_function', default=False, action="store_true",
                        help="Retrain the shape factor function on the measurements of the provided registered "
                             "meshes even if the path you provided already exists.")
    parser.add_argument("--ages_per_mesh", type=str, default=None,
                        help="We need a file here that specifies for each mesh/scan the age of the subject in months. "
                             "Note that if you train a new shape factor function without supplying this, "
                             "you may run into issues when evaluating it.")
    parser.add_argument("--mesh_unit", type=str, default=None,
                        help="When correcting a meshes, a text file is created that specifies absolute differences "
                             "between input and 'optimal' shape. You can optionally provide the mesh unit here, "
                             "e.g. 'mm' if you want to include this unit in the text file.")
    parser.add_argument("--num_jobs", type=int, default=1,
                        help="Optionally choose number of processors to run computations in parallel.")
    args = parser.parse_args()

    registered_mesh_paths = Mesh.get_all_mesh_files_in_path(
        args.input_files_or_dirs, regex_names=args.mesh_regex,
        exclude_regex_names=args.mesh_exclude_regex + ["*target_aligned", "*corrected"],
        file_suffixes=".ply")

    overwrite = args.overwrite

    cranial_attributes = CranialAttributes(model_path=args.path_to_hdf5_file)

    limit_components = args.limit_components
    if limit_components is not None:
        limit_components = [comp_ind for comp in limit_components for comp_ind in
                            (range(int(comp.replace(':', ''))) if comp.startswith(":") else [int(comp)])]

    cranial_attributes.measure_meshes_and_compute_shape_factor_function(
        *registered_mesh_paths, shape_factor_function_path=args.path_to_correlated_attributes,
        overwrite_measurements=overwrite, retrain_shape_factor_function=args.retrain_shape_factor_function,
        limit_components=limit_components, ages_per_mesh=args.ages_per_mesh, num_jobs=args.num_jobs,
        use_nonlinear_mappings=args.use_nonlinear_mappings
    )

    if not args.skip_correction:
        cranial_attributes.correct_meshes_from_paths(
            *registered_mesh_paths, overwrite=overwrite, mesh_unit=args.mesh_unit,
            num_jobs=args.num_jobs)




if __name__ == '__main__':
    main()