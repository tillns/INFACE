"""
This file offers the methods to compute all cranial measurements used for cranial shape correction in the paper.
Additionally, methods are included to load the linear regressors for each attribute and to correct
the cranial shape based on one or more of these attributes.

compute_measurements_on_registered_mesh() requires installation of trimesh shapely, and mapbox_earcut.

Author: Till Schnabel (contact till.schnabel@inf.ethz.ch)

MIT License

Copyright (c) 2025 ETH Zurich, Till Schnabel

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
from mesh import Mesh
from typing import Dict, List, Union, Tuple
from morphable_model import MorphableModel
from utils import Plane, Line, normalize_vector, get_rotation_matrix_from_axis_and_angle

import json
import numpy as np


def compute_measurements_on_registered_mesh(registered_mesh: Mesh) -> Dict:
    """
    This method requires the libraries
    trimesh including shapely and mapbox_earcut
    Given a registered cranial mesh (soft tissue), compute all medically relevant measurements on the mesh
    that were used for the paper, using the known correspondences and preselected indices.
    The hereby computed measurements for each mesh from our dataset were used to train the linear regressors
    that can then be used to estimate and adjust the meshes to corresponding meshes with a healthy cranium
    using their 3DMM projections, cf. the methods further below.
    :param registered_mesh: registered cranial soft-tissue mesh of type Mesh.
    :return dictionary containing as keys the names of the measurements and as values the computed scalar floats.
    """
    def import_indices(index_name: str):
        return np.loadtxt(f"cranial_indices/{index_name}.txt", delimiter=",", dtype=int)

    measurement_plane_indices = import_indices("measurement_plane_line")
    frontal_plane_indices = import_indices("frontal_plane_line")
    nasion_plane_indices = import_indices("nasion_plane_line")
    frontotemporal_indices = import_indices("frontotemporal")
    sa_indices = import_indices("supraorbital")

    glabella_possible_indices = import_indices("glabella_possible_indices")
    ophistocranion_possible_indices = import_indices("ophistocranion_possible_indices")
    ear_indices = import_indices("ears")
    template_tragus_vertex_indices = import_indices("tragus")
    template_nasion_vertex_index = int(import_indices("nasion"))
    template_glabella_landmark_vertex_index = int(import_indices("glabella_landmark_index"))
    template_head_top_vertex_index = int(import_indices("head_top"))

    vertical_line_sorted_index_sets = [
        import_indices("vertical_line_sorted_index_sets0"), import_indices("vertical_line_sorted_index_sets1")]

    registered_mesh_ears_stand_out = registered_mesh.get_copy()
    registered_mesh_ears_stand_out_vertices = registered_mesh_ears_stand_out.get_vertices()
    tragus_left_nasion_vector = (registered_mesh_ears_stand_out_vertices[template_tragus_vertex_indices[0]] -
                                 registered_mesh_ears_stand_out_vertices[template_nasion_vertex_index])
    tragus_right_nasion_vector = (registered_mesh_ears_stand_out_vertices[template_tragus_vertex_indices[1]] -
                                 registered_mesh_ears_stand_out_vertices[template_nasion_vertex_index])
    tragus_left_nasion_coord_distances = np.abs(tragus_left_nasion_vector)
    tragus_right_nasion_coord_distances = np.abs(tragus_right_nasion_vector)
    tragus_left_nasion_distance = np.linalg.norm(registered_mesh_ears_stand_out_vertices[template_tragus_vertex_indices[0]] -
                                                 registered_mesh_ears_stand_out_vertices[template_nasion_vertex_index])
    tragus_right_nasion_distance = np.linalg.norm(registered_mesh_ears_stand_out_vertices[template_tragus_vertex_indices[1]] -
                                                 registered_mesh_ears_stand_out_vertices[template_nasion_vertex_index])
    tragus_left_top_distance = np.linalg.norm(registered_mesh_ears_stand_out_vertices[template_tragus_vertex_indices[0]] -
                                              registered_mesh_ears_stand_out_vertices[template_head_top_vertex_index])
    tragus_right_top_distance = np.linalg.norm(registered_mesh_ears_stand_out_vertices[template_tragus_vertex_indices[1]] -
                                               registered_mesh_ears_stand_out_vertices[template_head_top_vertex_index])
    # The ears are in the way for the measurements, so we just flatten them away
    registered_mesh_flattened_ears = registered_mesh_ears_stand_out.smooth_laplacian(num_iterations=1000, indices=ear_indices)
    vertices = registered_mesh_flattened_ears.get_vertices()
    measurement_plane = Plane.fit_plane_to_points(vertices[measurement_plane_indices])
    nasion_plane = Plane.fit_plane_to_points(vertices[nasion_plane_indices])
    frontal_plane = Plane.fit_plane_to_points(vertices[frontal_plane_indices])

    #
    # Find head length
    head_length = -1
    glabella_index, ophistocranion_index = None, None
    # todo: add commentary how glabella and ophistocranion (name?) are found by comparing each point to each other
    for idx in glabella_possible_indices:
        for cf_idx in ophistocranion_possible_indices:
            current_length = np.linalg.norm(vertices[idx] - vertices[cf_idx])
            if current_length > head_length:
                head_length = current_length
                glabella_index, ophistocranion_index = (idx, cf_idx)
    glabella, ophistocranion = vertices[[glabella_index, ophistocranion_index]]
    glabella_ophistocranion_in_between_vertices = None
    for vertical_line_sorted_index_set in vertical_line_sorted_index_sets:
        if glabella_index in vertical_line_sorted_index_set:
            assert ophistocranion_index in vertical_line_sorted_index_set
            glabella_set_index = vertical_line_sorted_index_set.tolist().index(glabella_index)
            ophistocranion_set_index = vertical_line_sorted_index_set.tolist().index(ophistocranion_index)
            glabella_ophistocranion_in_between_vertices = vertical_line_sorted_index_set[
                min(glabella_set_index, ophistocranion_set_index):max(glabella_set_index, ophistocranion_set_index)+1]
            break
    assert glabella_ophistocranion_in_between_vertices is not None
    glabella_ophistocranion_perimeter = sum([
        np.linalg.norm(vertices[vertex_idx0]-vertices[vertex_idx1])
        for vertex_idx0, vertex_idx1 in zip(
            glabella_ophistocranion_in_between_vertices[:-1], glabella_ophistocranion_in_between_vertices[1:])])

    # Define circumference plane as the plane that includes the glabella and the ophistocranion
    # while being perpendicular to the nasion plane. CVAI, CVA, and CI are all computed by points
    # within this plane
    circumference_plane = Plane.get_plane_from_two_points_and_one_vector(
        glabella, ophistocranion, nasion_plane.get_normal())

    # Intersect circumference plane with registered mesh to get the line set on which all other points
    # for CVAI, CVA, and CI are found
    # I noticed the eyes are sometimes also intersected. so we remove them
    circ_intersection_line_segments = circumference_plane.intersect_mesh(
        registered_mesh_flattened_ears.remove_vertices(np.concatenate(
            [import_indices("eye_left"), import_indices("eye_right")])))

    def find_largest_circumference_width_given_direction(current_direction):
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
    # I haven't actually checked if left and right is consistently and correctly defined here,
    # but it doesn't really matter, I think
    head_width, eurion_left, eurion_right = find_largest_circumference_width_given_direction(head_width_direction)

    ci = head_width * 100 / head_length

    # Find diagonal points (similar to eurion points, only the direction is different) for CVA(I) measurements
    rot_axis = circumference_plane.get_normal()
    direction_init = normalize_vector(glabella - ophistocranion)
    rot_mat_plus = get_rotation_matrix_from_axis_and_angle(axis=rot_axis, angle=np.pi/6, homogeneous=False)
    rot_mat_minus = get_rotation_matrix_from_axis_and_angle(axis=rot_axis, angle=-np.pi/6, homogeneous=False)
    diagonal_plus_width_direction = rot_mat_plus @ direction_init
    diagonal_minus_width_direction = rot_mat_minus @ direction_init
    diagonal_width_plus, diagonal_plus_first_point, diagonal_plus_second_point = find_largest_circumference_width_given_direction(diagonal_plus_width_direction)
    diagonal_width_minus, diagonal_minus_first_point, diagonal_minus_second_point = find_largest_circumference_width_given_direction(diagonal_minus_width_direction)

    cva_signed = diagonal_width_plus - diagonal_width_minus
    cvai_signed = cva_signed * 100 / min(diagonal_width_plus, diagonal_width_minus)

    cva = abs(cva_signed)
    cvai = abs(cvai_signed)

    frontotemporal_dist = np.linalg.norm(vertices[frontotemporal_indices[0]] - vertices[frontotemporal_indices[1]])

    # We project the vectors to the measurement plane to avoid that if one eyebrow is raised that this
    # perturbs the angle measurement
    supraorbital_line1 = normalize_vector(measurement_plane.project_vectors_to_plane(
        vertices[template_glabella_landmark_vertex_index] - vertices[sa_indices[0]]))
    supraorbital_line2 = normalize_vector(measurement_plane.project_vectors_to_plane(
        vertices[template_glabella_landmark_vertex_index] - vertices[sa_indices[1]]))
    supraorbital_angle = np.arccos(supraorbital_line1.dot(supraorbital_line2))

    # Compute volumes
    registered_mesh_head = measurement_plane.slice_mesh(registered_mesh_flattened_ears)
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

    head_front_left_volume = registered_mesh_head_front_left.get_volume()
    head_front_right_volume = registered_mesh_head_front_right.get_volume()
    head_back_left_volume = registered_mesh_head_back_left.get_volume()
    head_back_right_volume = registered_mesh_head_back_right.get_volume()

    head_front_volume = head_front_left_volume + head_front_right_volume
    head_back_volume = head_back_left_volume + head_back_right_volume

    head_left_volume = head_front_left_volume + head_back_left_volume
    head_right_volume = head_front_right_volume + head_back_right_volume

    measurements = {
        # 2D measurements
        "CVAI": cvai, "CVA": cva, "CI": ci,
        "CVAI_signed": cvai_signed, "CVA_signed": cva_signed,
        "TR": tragus_left_nasion_distance / tragus_right_nasion_distance,
        **{f"TR{coord}": tragus_left_nasion_coord_distances[idx] / tragus_right_nasion_coord_distances[idx]
           for idx, coord in enumerate(["x", "y", "z"])},
        "TRtop": tragus_left_top_distance / tragus_right_top_distance,
        "TI": head_length / glabella_ophistocranion_perimeter,
        "FI": frontotemporal_dist / head_width,  "SA": supraorbital_angle,
        # just the volumes
        "v_front_left": head_front_left_volume, "v_front_right": head_front_right_volume,
        "v_back_left": head_back_left_volume, "v_back_right": head_back_right_volume,
        # volume ratios
        "VR_front_back": head_front_volume / head_back_volume,
        "VR_left_right": head_left_volume / head_right_volume,
        "VR_back_left_right": head_back_left_volume / head_back_right_volume,
        "VR_front_left_right": head_front_left_volume / head_front_right_volume,
        "VR_left_front_back": head_front_left_volume / head_back_left_volume,
        "VR_right_front_back": head_front_right_volume / head_back_right_volume,
    }
    return measurements



def get_projected_vertices(registered_meshes: List[Mesh], model: MorphableModel) -> np.ndarray:
    """
    Project multiple registered meshes into 3DMM space.
    :param registered_meshes: list of registered meshes (Mesh type)
    :param model: Morphable model used to encode the meshes (their vertices)
    :return np array of projected vertices [num_meshes x 3DMM_latent_size] the weights are normalized
    """
    registered_vertices = np.asarray([
        registered_mesh.get_vertices() for registered_mesh in registered_meshes])

    return model.normalize_weights(np.asarray([
        model.encode(registered_vertices_ind) for registered_vertices_ind in registered_vertices]))


def get_estimated_measurements(registered_vertices_projected: np.ndarray,
        measurements_w: Union[List, np.ndarray], measurements_mean: Union[List, np.ndarray]) -> np.ndarray:
    """
    :param registered_vertices_projected: Latents (projections of vertices into the model space)
                                          [num_meshes x 3DMM_latent_size]
    :param measurements_w: Weight vectors for all measurements [num_measurements x 3DMM_latent_size]
    :param measurements_mean: Mean for each measurement [num_measurements x 1]
    :return estimated measurements [num_measurements x num_meshes]
    """
    measurements_w = np.asarray(measurements_w)
    if measurements_w.ndim == 1:
        measurements_w = np.expand_dims(measurements_w, axis=0)

    measurements_mean = np.asarray(measurements_mean)
    if measurements_mean.ndim == 0:
        measurements_mean = np.expand_dims(measurements_mean, axis=0)

    assert len(measurements_w) == len(measurements_mean)

    measurements_estimated = np.einsum(
        "ki,ji->kj", measurements_w, registered_vertices_projected[:, :measurements_w.shape[1]]) + np.expand_dims(measurements_mean, axis=1)

    return measurements_estimated


def correct_estimated_measurements(
        measurements_estimated: np.ndarray, model: MorphableModel,
        measurements_w: Union[List, np.ndarray], measurement_optima: Union[List, np.ndarray],
        registered_vertices_projected: np.ndarray) -> np.ndarray:
    """
    Correct/adjust the 3DMM projections to match desired measurement values.
    :param measurements_estimated: Estimated values for the measurements to be adjusted.
    :param model: Morphable model used to decode the adjusted latents back to vertex space.
    :param measurements_w: Weight vectors for all measurements [num_measurements x 3DMM_latent_size]
    :param measurement_optima: Values to adjust the measurements to by changing the latents.
    :param registered_vertices_projected: Latents (projections of vertices into the model space)
                                          [num_meshes x 3DMM_latent_size]
    :return numpy array [num_meshes x num_vertices x 3]
    """

    measurement_optima = np.asarray(measurement_optima)
    if measurement_optima.ndim == 1:
        measurement_optima = np.expand_dims(measurement_optima, axis=1)

    measurements_w = np.asarray(measurements_w)
    if measurements_w.ndim == 1:
        measurements_w = np.expand_dims(measurements_w, axis=0)

    delta_projection = np.einsum(
        "ij,jk->ki", measurements_w.T.dot(np.linalg.inv(measurements_w.dot(measurements_w.T))),
        measurement_optima - measurements_estimated)

    if delta_projection.shape[1] < registered_vertices_projected.shape[1]:
        delta_projection = np.concatenate([
            delta_projection, np.zeros((delta_projection.shape[0], registered_vertices_projected.shape[1]-delta_projection.shape[1]))], axis=1)

    projection_shifted_measurement = model.unnormalize_weights(registered_vertices_projected + delta_projection)
    vertices_shifted_measurement = model.decode(projection_shifted_measurement)
    return np.reshape(vertices_shifted_measurement, (len(vertices_shifted_measurement), -1, 3))

def get_measurement_corrected_meshes(
        registered_meshes: List[Mesh], model: MorphableModel,
        shape_factor_function: Dict[str, List], measurement_kinds: List[str],
        measurement_optima: Union[List[Union[float, None]], float, None]=None,
        fix_age: bool = False, fix_total_volume: bool = False) -> Tuple[List[Mesh], np.ndarray, np.ndarray]:
    """
    Corrected multiple meshes to corresponding versions with healthy craniums (based on the measurements
    to be provides as arguments).
    :param registered_meshes: List of registered meshes (type Mesh)
    :param model: Morphable model that is used to adjust the meshes in the model spaces using the linear regression
                  functions also provided as arguments.
    :param shape_factor_function: Dictionary of linear regressors. Entries specify the name of the measurement.
                                  Values are a tuple comprising a numpy array for the regressor weight,
                                  the mean of the measurement (important for normalization),
                                  and the standard deviation (not used by default, but can give additional
                                  information about the range of this measurement).
    :param measurement_optima: Optionally provide desired values for each measurement that deviate from the average.
                               This is already a list/array; it needs to be in the order that the keys in the
                               shape_factor_function are sorted.
    :param fix_age: There is no "optimal" age to adjust the cranium to, but you can choose this arg to avoid that
                    by shifting the space for the other measurements also shifts the age, so you can fix the age
                    while varying the other attributes.
    :param fix_total_volume: Fix the total volume of total cranium, like you can fix the age (cf. fix_age).
                             This can be used to avoid that the cranium is expanded when correcting the other
                             attributes, thus serving of a better reference for a doctor showing what can be done
                             during surgery (the head doesn't grow; the skull can be moved, but not expanded).
    :return Tuple 1) List of adjusted/corrected meshes 2) estimated measurements 3) values measurements were adjusted to
    """

    num_measurement_kinds_without_fix = len(measurement_kinds)
    if fix_age:
        assert "age" not in measurement_kinds
        measurement_kinds = [*measurement_kinds, "age"]

    if fix_total_volume:
        assert "total_volume" not in measurement_kinds
        measurement_kinds = [*measurement_kinds, "v_tot"]

    measurements_w = [np.asarray(shape_factor_function[measurement_kind][0]) for measurement_kind in measurement_kinds]
    measurements_mean = [np.asarray(shape_factor_function[measurement_kind][1]) for measurement_kind in measurement_kinds]

    # Projected registered meshes into PCA space
    registered_vertices_projected = get_projected_vertices(registered_meshes, model)

    # Estimate all measurements for the meshes
    measurements_estimated = get_estimated_measurements(registered_vertices_projected, measurements_w, measurements_mean)

    # Initialize the optimum for each measurement, i.e., to which value to correct the meshes for each measurement
    if not isinstance(measurement_optima, (list, np.ndarray)):
        measurement_optima = [measurement_optima]
    measurement_optima = np.asarray([measurement_optimum if measurement_optimum is not None else measurement_mean
                                     for measurement_optimum, measurement_mean in zip(measurement_optima, measurements_mean)])

    if fix_age:
        if measurement_optima.ndim == 1:
            measurement_optima = np.expand_dims(measurement_optima, axis=1).repeat(len(registered_meshes), axis=1)
        est_start_ind = num_measurement_kinds_without_fix
        measurement_optima = np.concatenate((measurement_optima, measurements_estimated[est_start_ind:est_start_ind+1]), axis=0)

    if fix_total_volume:
        if measurement_optima.ndim == 1:
            measurement_optima = np.expand_dims(measurement_optima, axis=1).repeat(len(registered_meshes), axis=1)
        est_start_ind = num_measurement_kinds_without_fix + 1 if fix_age else num_measurement_kinds_without_fix
        measurement_optima = np.concatenate((measurement_optima, measurements_estimated[est_start_ind:est_start_ind+1]), axis=0)

    # Do the correction
    vertices_shifted_measurement = correct_estimated_measurements(
        measurements_estimated, model, measurements_w, measurement_optima, registered_vertices_projected)

    # Create meshes from shifted vertices
    meshes_shifted_measurements = [
        Mesh(vertices=vertices_shifted_measurement_ind, triangles=registered_mesh.get_triangles())
        for vertices_shifted_measurement_ind, registered_mesh in zip(vertices_shifted_measurement, registered_meshes)]
    # Remove the estimated and optima values for the fixed values
    if fix_total_volume:
        measurements_estimated, measurement_optima = measurements_estimated[:-1], measurement_optima[:-1]
    if fix_age:
        measurements_estimated, measurement_optima = measurements_estimated[:-1], measurement_optima[:-1]

    # corrected meshes, each color coded with error, the estimated measurements and what value they were corrected to
    return meshes_shifted_measurements, measurements_estimated, measurement_optima

def get_fully_corrected_meshes(registered_meshes: List[Mesh], model: MorphableModel,
        shape_factor_function: Dict[str, List],
        fix_age: bool = True, fix_total_volume: bool = True) -> Tuple[List[Mesh], np.ndarray, np.ndarray]:
    """
    Fix cranial meshes based on all measurements, we deemed relevant for medical purposes.
    Cf. docs of get_measurement_corrected_meshes for an explanation of the arguments and return variables.
    """

    measurements_relevant_for_correction = ["FI", "CI", "CVAI_signed", "TRx", "TRy", "TRz", "TRtop",
                                            "VR_back_left_right", "VR_front_left_right", "VR_front_back"]

    # These are optimal values that may be different from the average values in the dataset
    measurement_optima = {"CI": 80, "CVAI_signed": 0, "TR": 1, "TRx": 1, "TRy": 1, "TRz": 1, "TR_top": 1,
                          "VR_left_right": 1, "VR_back_left_right": 1, "VR_front_left_right": 1}

    measurement_optima_arg = [
        measurement_optima[measurement_kind] if measurement_kind in measurement_optima else None
        for measurement_kind in measurements_relevant_for_correction]

    return get_measurement_corrected_meshes(
        registered_meshes=registered_meshes, model=model, shape_factor_function=shape_factor_function,
        measurement_kinds=measurements_relevant_for_correction, measurement_optima=measurement_optima_arg,
        fix_age=fix_age, fix_total_volume=fix_total_volume
    )


def main():
    parser = ArgumentParser()
    parser.add_argument('--path_to_hdf5_file', type=str, required=True,
                        help="Absolute path to autoencoder or PCA model file with .h5 ending.")
    parser.add_argument('--path_to_correlated_attributes', type=str, default=None,
                        help="Absolute path to json file that encodes correlated cranial attributes "
                             "(must fit to provided model).")
    args = parser.parse_args()
    model = MorphableModel.load_correct_morphable_model(args.path_to_hdf5_file)
    sample_meshes = model.sample_meshes(5)
    with open(args.path_to_correlated_attributes, "r") as f:
        shape_factor_function = json.load(f)
    corrected_meshes, measurements_estimated, measurement_optima = get_fully_corrected_meshes(
            registered_meshes=sample_meshes, model=model, shape_factor_function=shape_factor_function,
            fix_age=True, fix_total_volume=True)
    for sample_mesh, corrected_mesh in zip(sample_meshes, corrected_meshes):
        sample_mesh.show_multiple_meshes(sample_mesh, corrected_mesh)


if __name__ == '__main__':
    main()