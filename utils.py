"""
File with supporting methods and classes.

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

import warnings
from typing import Union, Tuple, List
from mesh import Mesh
import numpy as np

def normalize_vector(vec_or_vecs):
    return np.asarray(vec_or_vecs) / np.linalg.norm(vec_or_vecs, axis=-1, keepdims=True)

def get_rotation_matrix_from_axis_and_angle(axis, angle, homogeneous=False):
    """
    Given a 3D axis and an angle, this method returns a 3x3 rotation matrix that can rotate a point
    by angle around axis (should be counter-clockwise, but not sure, maybe check).
    @param axis: 3D numpy array or of type that's convertible to 3D numpy array.
    @param angle: float angle in radian
    @param homogeneous: bool whether to create a homogeneous rotation matrix (4x4 if True, 3x3 if False)
    @return: 3x3 numpy float array
    """
    axis_normalized = normalize_vector(np.asarray(axis))
    a = np.cos(angle/2)
    b, c, d = -axis_normalized*np.sin(angle/2)
    aa, bb, cc, dd = a*a, b*b, c*c, d*d
    bc, ad, ac, ab, bd, cd = b*c, a*d, a*c, a*b, b*d, c*d
    rot_mat = np.identity(4 if homogeneous else 3)
    rot_mat[:3, :3] = np.asarray([[aa+bb-cc-dd, 2*(bc+ad), 2*(bd-ac)],
                                  [2*(bc-ad), aa+cc-bb-dd, 2*(cd+ab)],
                                  [2*(bd+ac), 2*(cd-ab), aa+dd-bb-cc]])
    return rot_mat


class Plane:
    """
    A Plane is defined by a point on the plane (the anchor) and a normal vector that's perpendicular to the plane
    (the direction). This class saves these attributes and offers additional methods to build a plane from various
    inputs, intersect planes, also work with lines (cf. Line class below), and meshes.
    """
    point_on_plane: Union[list, np.ndarray]
    plane_normal: Union[list, np.ndarray]
    half_plane_direction: Union[list, np.ndarray] = None
    is_half_plane: bool = False


    def __init__(self, point_on_plane: Union[list, np.ndarray], plane_normal: Union[list, np.ndarray],
                 half_plane_direction: Union[list, np.ndarray] = None):
        """
        :param point_on_plane: Any point on the plane. Shape (3).
        :param plane_normal: Vector perpendicular to plane (shape (3), no need to normalize it before;
                             it's normalized in the __init__ method).
        """
        self.set_point(point_on_plane)
        self.set_normal(plane_normal)
        if half_plane_direction is not None:
            self.set_half_plane_direction(half_plane_direction)

    @staticmethod
    def get_plane_from_three_points(
            first_point: Union[list, np.ndarray], second_point: Union[list, np.ndarray],
            third_point: Union[list, np.ndarray]) -> "Plane":
        plane_anchor = np.mean(np.asarray([first_point, second_point, third_point]), axis=0)
        plane_normal = np.cross(second_point - first_point, third_point - first_point)
        return Plane(point_on_plane=plane_anchor, plane_normal=plane_normal)

    @staticmethod
    def get_plane_from_two_points_and_one_vector(
            first_point: Union[list, np.ndarray], second_point: Union[list, np.ndarray],
            vector_on_plane: Union[list, np.ndarray]) -> "Plane":
        plane_anchor = np.mean(np.asarray([first_point, second_point]), axis=0)
        plane_normal = np.cross(second_point - first_point, vector_on_plane)
        return Plane(point_on_plane=plane_anchor, plane_normal=plane_normal)

    @staticmethod
    def get_plane_from_one_point_and_two_vectors(
            point_on_plane: Union[list, np.ndarray], first_vector: Union[list, np.ndarray],
            second_vector: Union[list, np.ndarray]) -> "Plane":
        return Plane(point_on_plane=point_on_plane,
                     plane_normal=np.cross(first_vector, second_vector))


    def get_point(self) -> np.ndarray:
        """
        :return: np array representing a point on the plane (the one the Plane instance was initialized with)
        """
        return np.array(self.point_on_plane)

    def set_point(self, point_on_plane: Union[list, np.ndarray]):
        """
        Setter method for plane anchor.
        :param point_on_plane: Any point on the plane. Shape (3).
        :return: None
        """
        self.point_on_plane = np.array(point_on_plane)

    def get_normal(self) -> np.ndarray:
        """
        :return: plane's normal vector as np array, normalized to Euclidean length 1
        """
        return np.array(self.plane_normal)

    def set_normal(self, plane_normal: Union[list, np.ndarray]):
        """
        Setter method for plane direction.
        :param plane_normal: Vector perpendicular to plane (shape (3), no need to normalize it before).
        :return: None
        """
        self.plane_normal = np.asarray(plane_normal)
        self.plane_normal = self.plane_normal / np.linalg.norm(self.plane_normal)

    def flip_normal(self):
        self.set_normal(-self.plane_normal)

    def set_half_plane_direction(self, half_plane_direction: Union[list, np.ndarray]):
        self.half_plane_direction = normalize_vector(np.asarray(half_plane_direction))
        if not np.dot(self.plane_normal, self.half_plane_direction) < 1e-5:
            raise AssertionError("Given half plane direction is not perpendicular to plane normal")
        # make sure the half plane direction really lies inside the plane
        self.half_plane_direction = normalize_vector(self.project_vectors_to_plane(half_plane_direction))
        self.is_half_plane = True

    def get_half_plane_direction(self):
        return self.half_plane_direction

    def get_type(self) -> str:
        if self.is_half_plane:
            return "half-plane"
        return "plane"

    def get_params(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        :return: Tuple(plane anchor, plane normal). The order of object instantiation is kept,
        so you can input these params in that order when creating a new Plane instance.
        """
        return self.get_point(), self.get_normal()

    def get_mesh(self, size: float = 1, geom_class=None, other_centroid: Union[list, np.ndarray] = None):
        """
        Create a plane mesh using the Plane's point_on_plane as centroid unless another is provided.
        :param size: float that defines the half length of the plane's side
        :param geom_class: Output geometry class
        :param other_centroid: Optionally provide another centroid if you want to avoid having a shifted plane
                               (warning is printed if that point isn't on the plane)
        :return: plane mesh (four corner vertices connected by two triangles) of class as specified (default open3d)
        """

        if self.is_half_plane:
            u = self.half_plane_direction
        else:
            # Compute two orthogonal vectors in the plane
            if abs(self.plane_normal.dot([0, 0, 1])) < 1:
                u = np.cross(self.plane_normal, [0, 0, 1])
            else:
                u = np.asarray([1, 0, 0])
        v = np.cross(self.plane_normal, u)
        u = u / np.linalg.norm(u) * size
        v = v / np.linalg.norm(v) * size

        if other_centroid is None:
            plane_centroid = self.point_on_plane
        else:
            plane_centroid = np.ndarray(other_centroid)
            if self.compute_distance_points_to_plane(plane_centroid) > 1e-5:
                warnings.warn("You provided another plane centroid, but the point is actually not on the plane."
                              "Define another plane instance that covers this point to remove this warning.")

        if self.is_half_plane:
            # A half plane has a defined boundary line on one side,
            # so we shift the centroid, such that the plane mesh respects that boundary.
            plane_centroid = plane_centroid + u

        # Compute the four corners of the plane
        corners = np.array([
            plane_centroid + u + v,
            plane_centroid + u - v,
            plane_centroid - u - v,
            plane_centroid - u + v
        ])

        Mesh(vertices=corners, triangles=np.asarray([[0, 1, 2], [2, 3, 0]]))

        return import_mesh({"vertices": corners, "triangles": [[0, 1, 2], [2, 3, 0]]},
                           compute_vertex_normals=True, geom_class=geom_class)

    @staticmethod
    def fit_plane_to_points(points: Union[List, np.ndarray]) -> "Plane":
        """
        Given a set of points, compute a plane that fits the points as closely as possible
        (minimizing squared Euclidean distance).
        In the normal case, the normal of the plane is in a right-hand system with the vectors between
        first and second projected point towards all points' center. This only doesn't apply in edge cases
        where we get linear dependence between these directions, in which case by the point causing the edge
        case is replaced by the next point in order.
        :param points: List or numpy array of 3D points.
        :return: point center (representing a point on the plane, 3D np array), plane normal (normalized, 3D np array)
        """
        # check input
        points = np.asarray(points)
        if len(points.shape) != 2 or points.shape[1] != 3:
            raise TypeError(f"Given points are not a 3D point array: {points}")
        if not len(points) >= 3:
            raise ValueError("You must provide at least three points to fit a plane through.")

        # Compute the centroid of the points
        centroid = np.mean(points, axis=0)

        # Subtract the centroid from the points
        centered_points = points - centroid

        # Compute the covariance matrix
        cov_matrix = np.dot(centered_points.T, centered_points) / centered_points.shape[0]

        # Perform SVD -> The best plane is computed via the vector that minimizes the variance in the points
        # (thus is perpendicular to the plane that has the largest variance in points)
        try:
            _, _, vh = np.linalg.svd(np.asarray(cov_matrix, dtype=float))
        except np.linalg.LinAlgError:
            raise ValueError("The points you provided are all on the same line.")

        # The normal to the plane is the last column of vh (corresponding to the smallest singular value)
        # By construction, this is alreay normalized
        normal = vh[-1]

        initial_plane = Plane(centroid, normal)

        # We adjust the direction of the normal to make it deterministic:
        # We find two distinct directions on the plane and compute their cross product.
        # To find these two directions, we go through the given points in order
        # and compute each point's projected direction to the center.
        # We take the first two directions that are linearly independent.
        for num_point, point in enumerate(points):
            # project first point to plane
            projected_point = initial_plane.project_points(point)
            # We cannot compute the direction to the center if the point's projection falls
            # directly on the center, so we skip it in that case.
            if not np.array_equal(projected_point, centroid):
                # Compute the the difference between center and projected
                # point to give us the first direction in the plane.
                first_direction = normalize_vector(centroid - projected_point)
                # Iterate through remaining points to find the second direction
                for next_point in points[num_point + 1:]:
                    next_projected_point = initial_plane.project_points(next_point)
                    # The second valid point's projection cannot fall on the center of the first point's direction.
                    if (not np.array_equal(next_projected_point, projected_point) and not
                    np.array_equal(next_projected_point, centroid)):
                        second_direction = normalize_vector(centroid - next_projected_point)
                        # The second direction also cannot be linearly dependent on the first direction.
                        # It could be that the two points lie exactly on opposite sides of the center,
                        # in which case their projections would not equal in any way, but the directions
                        # are still not enough to get the normal out.
                        # We check for linear dependence via the dot product with some
                        # threshold for safety/numerical stability.
                        if abs(np.dot(first_direction, second_direction)) < 0.999:
                            # Compute new normal. We sanity-check that it equals the old normal, except for the sign.
                            adjusted_normal = normalize_vector(np.cross(first_direction, second_direction))
                            if abs(np.dot(normal, adjusted_normal)) < 0.999:
                                raise AssertionError("You didn't only change the sign of the normal. Something's off.")
                            return Plane(centroid, adjusted_normal)

        raise AssertionError("I don't think it's possible to reach this point here. Check why it still happened.")

    def compute_distance_points_to_plane(self, points_to_compute_dist_for: Union[List, np.ndarray], compute_absolute: bool = True) -> Union[np.ndarray, float]:
        """
        Project a set of points onto a plane, represented via one point on the plane and the normal.
        :param points_to_compute_dist_for: The input points that the distance to the plane should be computed for;
                                           can also be a single point. Shape (3) or (nx3)
        :param compute_absolute: Set to False to also allow negative distances.
        :return np array of distances, one for each given point (it's a simple float if only one point is given)
        """
        # todo: One could use projection method here, then compute distance, handle nan cases for half-planes.
        # Vector from plane_point to the point
        points_to_plane_points = np.asarray(points_to_compute_dist_for) - self.point_on_plane

        # Distance equals the projection of this connecting vector onto the normal
        distances = np.dot(points_to_plane_points, self.plane_normal)

        if compute_absolute:
            distances = np.abs(distances)

        return distances

    def compute_distance_to_other_plane(self, other_plane: "Plane", compute_absolute: bool = True) -> float:
        """
        Compute distance from this plane to another plane. This doesn't consider half planes currently.
        :param other_plane: The other plane (type Plane) to compute distance to.
        :param compute_absolute: Set this to False to also allow negative distances (this plane's normal defines sign)
        :return: Float representing the distance to the other plane (optionally with sign).
        """
        other_plane_point = other_plane.get_point()
        return self.compute_distance_points_to_plane(other_plane_point, compute_absolute=compute_absolute)

    def project_points(self, points_to_project: np.ndarray) -> np.ndarray:
        """
        Project a set of points onto a plane, represented via one point on the plane and the normal.
        Half-plane structure is not respected yet.
        :param points_to_project: The input points that are to be projected onto the plane; can also be a single point.
                                  Shape (3) or (nx3)
        :return np array of points projected onto plane. Shape same as input, so (3) or (nx3).
        """
        if self.is_half_plane:
            raise NotImplementedError("One could do something similar as for the line here.")
        # Projection of the vector onto the normal
        normal_dot = self.compute_distance_points_to_plane(points_to_project, compute_absolute=False)

        # Distinguish between case where points_to_project is only a single point vs multiple
        if points_to_project.ndim == 2:
            normal_dot = np.expand_dims(normal_dot, axis=-1)
        projections_onto_normal = normal_dot * self.plane_normal

        # Subtract this projection from the original point to get the projection on the plane
        return points_to_project - projections_onto_normal


    def project_vectors_to_plane(self, vectors_to_project: np.ndarray) -> np.ndarray:
        """
        Project a set of vectors onto a plane defined by its normal vector.
        Half-plane structure is not taken into account here, as vectors can be moved in space.
        :param vectors_to_project: The input vectors that are to be projected onto the plane; can also be a single vector.
                                   Shape (3) or (nx3)
        :return: np array of vectors projected onto the plane. Shape same as input, so (3) or (nx3).
                 The vectors are not normalized.
        """
        # Projection of the vector onto the normal
        normal_dot = np.dot(vectors_to_project, self.plane_normal)
        # Distinguish between case where vectors_to_project is only a single vector vs multiple
        if vectors_to_project.ndim == 2:
            normal_dot = np.expand_dims(normal_dot, axis=-1)
        projections_onto_normal = normal_dot * self.plane_normal

        # Subtract this projection from the original vector to get the projection on the plane
        return vectors_to_project - projections_onto_normal


    def check_side_of_plane(self, points_to_check_side_for: np.ndarray) -> np.ndarray:
        """
        Given one or multiple points, compute the side of the provided plane the points are on w.r.t.
        the plane's normal direction. Half-plane not considered here.
        :param points_to_check_side_for: One or multiple 3D points.
        :return: Numpy array with 1 and -1 entry for each provided point indicating positive and negative side on the plane.
        """
        # Determine the side based on the sign of the dot product
        return np.sign(self.compute_distance_points_to_plane(points_to_check_side_for, compute_absolute=False))

    def intersect_plane(self, other_plane: "Plane") -> "Line":
        """
        cf. docs of intersect_planes
        """
        return self.intersect_planes(self, other_plane)


    @staticmethod
    def intersect_planes(plane1: "Plane", plane2: "Plane") -> "Line":
        """
        Compute the line that intersects the given two planes.
        :param plane1: First plane (to be intersected with plane2)
        :param plane2: Second plane (to be intersected with plane1)
        :return: instance of Line that represents the intersection of the two planes
        :raise: ValueError if the two planes are parallel.
        """
        if plane1.is_half_plane or plane2.is_half_plane:
            raise NotImplementedError("Half-plane intersection is not implemented yet.")
        plane1_point, plane1_normal = plane1.get_point(), plane1.get_normal()
        plane2_point, plane2_normal = plane2.get_point(), plane2.get_normal()

        # Normalize the plane normals
        n1, n2 = normalize_vector([plane1_normal, plane2_normal])

        # Check if the planes are parallel by calculating the cross product of the normals
        cross_n1_n2 = np.cross(n1, n2)
        if np.allclose(cross_n1_n2, 0):
            # Planes are parallel
            raise ValueError("The two planes are parallel, so there's either no intersection or the planes are the same.")

        # The direction of the intersecting line is given by the cross product of the normals
        direction = normalize_vector(cross_n1_n2)

        # To find a point on the intersecting line, we solve for a specific point.
        # We can set one of the coordinates to zero and solve the resulting system.
        A = np.array([n1, n2])
        d = np.array([np.dot(n1, plane1_point), np.dot(n2, plane2_point)])

        # Find a point on the intersection line by setting one coordinate to zero and solving
        if not np.allclose(direction[0], 0):
            A_reduced = A[:, 1:]
            point_reduced = np.linalg.solve(A_reduced, d)
            point_on_line = np.array([0, *point_reduced])
        elif not np.allclose(direction[1], 0):
            A_reduced = A[:, [0, 2]]
            point_reduced = np.linalg.solve(A_reduced, d)
            point_on_line = np.array([point_reduced[0], 0, point_reduced[1]])
        else:
            A_reduced = A[:, :2]
            point_reduced = np.linalg.solve(A_reduced, d)
            point_on_line = np.array([*point_reduced, 0])

        # return the line parametrization
        return Line(point_on_line, direction)

    def slice_mesh(self, mesh: Mesh, flip_direction: bool = False, cap: bool = True) -> Mesh:
        """
        Slice mesh through plane and only keep vertices that are above the plane, so plane direction matters.
        Main computation is outsourced to trimesh.intersections.slice_mesh_plane()
        :param mesh: Triangle mesh of type Mesh
        :param flip_direction: If you notice that the direction is wrong, you can flip it by setting this arg to True.
        :param cap: Whether to close the mesh at the plane intersection with flat triangles.
        :return: Sliced mesh of type Mesh.
        """
        import trimesh
        normal = np.array(self.plane_normal)
        if flip_direction:
            normal = -normal

        mesh_trimesh = mesh.get_trimesh_mesh()
        sliced_mesh = trimesh.intersections.slice_mesh_plane(
            mesh_trimesh, plane_origin=self.point_on_plane, plane_normal=normal, cap=cap)
        return mesh.get_from_trimesh_mesh(sliced_mesh)

        #mesh_open3d = mesh.get_open3d_mesh(use_legacy=True)
        #sliced_mesh = mesh_open3d.clip_plane(point=self.point_on_plane.tolist(), normal=normal.tolist())
        #return Mesh.get_from_open3d_mesh(sliced_mesh)

    def intersect_mesh(self, mesh: Mesh) -> List[np.ndarray]:
        """
        Intersect plane with a mesh, resulting in a set of line segments (each segment is one triangle of the
        mesh that is intersected by the plane). Note that there's no order, so it's not given that the second
        line segment's start point equals the first line segment's end point! # todo: Is that also true for open3d?
        Main computation is outsourced to open3d's slice_mesh()
        :param mesh: Triangle mesh of type Mesh
        :return: line segments (List of 3D point 2-pairs)
        """
        mesh_open3d = mesh.get_open3d_mesh(use_legacy=True)
        line_segments_open3d = mesh_open3d.slice_plane(point=self.point_on_plane.tolist(), normal=self.plane_normal.tolist())
        line_segment_positions = line_segments_open3d.point.positions.numpy()
        line_segment_indices = line_segments_open3d.line.indices.numpy()
        line_segments = [line_segment_positions[index_pair] for index_pair in line_segment_indices]
        return line_segments


class Line:
    point_on_line: Union[list, np.ndarray]
    line_direction: Union[list, np.ndarray]
    is_ray: bool

    def __init__(self, point_on_line: Union[list, np.ndarray], line_direction: Union[list, np.ndarray],
                 is_ray: bool = False):
        """
        :param point_on_line: Any point on the line. Shape (3).
        :param line_direction: Vector that goes along the line (shape (3), no need to normalize it before;
                               it's normalized in the __init__ method).
        :param is_ray: If you want to treat the line instead as a ray (so only going into positive direction
                       from point_on_line), set this to True. Class methods' behavior is adjusted accordingly, e.g.,
                       points behind the ray will get np.nan values assigned for the projection,
                       and the distance also takes the ray's end point into account.
        """
        self.set_point(point_on_line)
        self.set_direction(line_direction)
        self.is_ray = is_ray

    def get_point(self) -> np.ndarray:
        """
        :return: np array representing a point on the line (the one the line instance was initialized with)
        """
        return self.point_on_line

    def set_point(self, point_on_line: Union[list, np.ndarray]):
        """
        Setter method for line anchor.
        :param point_on_line: Any point on the line. Shape (3).
        :return: None
        """
        self.point_on_line = np.asarray(point_on_line)

    def get_direction(self) -> np.ndarray:
        """
        :return: line's directional normal vector as np array, normalized to Euclidean length 1
        """
        return self.line_direction

    def set_direction(self, line_direction: Union[list, np.ndarray]):
        """
        Setter method for line direction.
        :param line_direction: Vector that goes along the line (shape (3), no need to normalize it before).
        :return: None
        """
        self.line_direction = np.asarray(line_direction)
        self.line_direction = self.line_direction / np.linalg.norm(self.line_direction)

    def get_params(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        :return: Tuple(line anchor, line direction). The order of object instantiation is kept,
        so you can input these params in that order when creating a new Line instance.
        """
        return self.get_point(), self.get_direction()

    def get_type(self) -> str:
        if self.is_ray:
            return "ray"
        return "line"

    def get_mesh(self, size: float = 1, radius: float = 0.15, other_centroid: Union[list, np.ndarray] = None):
        """
        Create a line mesh using the line's point_on_line as center point unless another is provided.
        Rays are taken into account, unless you mess it up with giving another center point.
        :param size: float that defines half of the length of the line mesh
                     (size goes in both directions from center point)
        :param radius: float that defines the radius of the cylindrical line mesh
        :param other_centroid: Optionally provide another center point if you want to avoid having a shifted line
                             (warning is printed if that point isn't on the line/ray)
        :return: line mesh (cylinder)
        """
        raise NotImplementedError("Due to a lot of dependencies, this method is currently unavailable.")


    @staticmethod
    def fit_line_to_points(points: Union[List, np.ndarray]) -> "Line":
        """
        Given a set of points, compute a line that fits the points as closely as possible
        (minimizing squared Euclidean distance).
        In the normal case, the direction of the line points from the first point towards all points' center.
        This only doesn't apply if the first point is directly on the center, in which case the next point
        decides the sign of the direction.
        :param points: List or numpy array of 3D points.
        :return: Line instance
        """

        # check input
        points = np.asarray(points)
        if len(points.shape) != 2 or points.shape[1] != 3:
            raise TypeError(f"Given points are not a 3D point array: {points}")
        if not len(points) >= 3:
            raise ValueError("You must provide at least three points to fit a plane through.")

        # Compute the centroid of the points
        centroid = np.mean(points, axis=0)

        # Subtract the centroid from the points
        centered_points = points - centroid

        # Compute the covariance matrix
        cov_matrix = np.dot(centered_points.T, centered_points)

        # Perform SVD -> line is computed via the vector that represents the largest variance in the data
        # (first principal component).
        try:
            _, _, vh = np.linalg.svd(np.asarray(cov_matrix, dtype=float))
        except np.linalg.LinAlgError:
            raise ValueError("The points you provided are all on the same point.")

        # The direction of the line is the first column of vh (corresponding to the largest singular value)
        direction = vh[0]

        initial_line = Line(centroid, direction)

        # We update the sign of the direction by projecting the points (in order) onto the line,
        # and the first projection that doesn't fall directly on the center, then defines the sign
        # of the direction -- we subtract the projection from the center.
        # So in the normal case, the direction of the line goes from the first point towards all points' center.
        for point in points:
            projected_point = initial_line.project_points(point)
            if not np.array_equal(projected_point, centroid):
                adjusted_direction = normalize_vector(centroid - projected_point)
                if abs(np.dot(direction, adjusted_direction)) < 0.999:
                    raise AssertionError("Updated line direction is not in line with the original direction. "
                                         "We should only get at most a sign change, but nothing else. Check.")
                return Line(centroid, adjusted_direction, is_ray=False)

        raise AssertionError("Every of the individual points projected to the fitted line equals the center "
                             "of the points. This shouldn't happen unless all points are the same, "
                             "in which case the SVD should have already thrown an error, so this error "
                             "here should never happen. Check why it did. ^^")

    def compute_points_distance_to_line(self, points_to_compute_dist_for: np.ndarray) -> np.ndarray:
        """
        Compute the distance on a line that a set of points have (how far away are the points from the line).
        This method differs from compute_points_distance_on_line() in that it doesn't project the points onto the line
        and then computes their distance to the anchor, but rather it computes the points' actual distance to the line
        (measured orthogonally to the line).
        The ray property is considered here: Points that lie behind the ray have as closest point on the ray the anchor,
        so their distance to the ray is that between the point and the anchor.
        :param points_to_compute_dist_for: The input points that the distance on the line should be computed for;
                                           can also be a single point. Shape (3) or (nx3)
        :return np array of distances. Shape (n). todo: how is single number treated?
        """

        # We project the points onto the line, then compute the distance
        # between the input points and the projected points.
        projected_points = self.project_points(points_to_compute_dist_for)

        # We replace nan values with the ray anchor
        if projected_points.ndim == 2:
            projected_points[np.isnan(projected_points[:, 0])] = self.point_on_line
        else:
            if np.isnan(projected_points[0]):
                projected_points = np.array(self.point_on_line)

        return np.linalg.norm(projected_points - points_to_compute_dist_for, axis=-1)

    def compute_points_distance_on_line(self, points_to_compute_dist_for: np.ndarray) -> np.ndarray:
        """
        Compute the distance on a line that a set of points have (how far along the line from the line's anchor).
        This method differs from compute_points_distance_to_line() in that it projects the points onto the line and
        then computes their distance to the line's anchor, rather than measureing the points' actual distance
        to the line itself.
        Note that the ray property is not considered here (you can just take the negative numbers --
        these would not lie on the ray anymore.
        :param points_to_compute_dist_for: The input points that the distance on the line should be computed for;
                                           can also be a single point. Shape (3) or (nx3)
        :return np array of distances. Shape (n). todo: how is single number treated?
        """
        # Vector from point_on_line to the points to project
        points_to_line_points = points_to_compute_dist_for - self.point_on_line

        # Projection of the vector onto the line direction
        direction_dot = np.dot(points_to_line_points, self.line_direction)

        return direction_dot

    def project_points(self, points_to_project: np.ndarray) -> np.ndarray:
        """
        Project a set of points onto a line, represented via one point on the line and the line direction.
        Note that if the instance is a ray, then points behind the ray will get np.nan as projected point.
        :param points_to_project: The input points that are to be projected onto the line; can also be a single point.
                                  Shape (3) or (nx3)
        :return np array of points projected onto the line. Shape same as input, so (3) or (nx3).
        """
        # Projection of the vector onto the line direction
        direction_dot = self.compute_points_distance_on_line(points_to_project)

        # Points that are behind a ray get nan values assigned. This ensures that they can be
        # distinguished from points being projected directly onto the ray origin.
        if self.is_ray:
            direction_dot[direction_dot < 0] = np.nan

        # Distinguish between case where points_to_project is only a single point vs multiple
        if points_to_project.ndim == 2:
            direction_dot = np.expand_dims(direction_dot, axis=-1)
        projections_onto_direction = direction_dot * self.line_direction

        # Add this projection to the original point on the line to get the projection on the line
        projected_points = self.point_on_line + projections_onto_direction

        return projected_points

    def project_vectors_to_line(self, vectors_to_project: np.ndarray) -> np.ndarray:
        """
        Project a set of vectors onto a line defined by its directional vector.
        :param vectors_to_project: The input vectors that are to be projected onto the line; can also be a single vector.
                                   Shape (3) or (nx3)
        :return: np array of vectors projected onto the line. Shape same as input, so (3) or (nx3).
                 The vectors are not normalized.
        """
        # Projection of the vector onto the normal
        line_dot = np.dot(vectors_to_project, self.line_direction)
        # Distinguish between case where vectors_to_project is only a single vector vs multiple
        if vectors_to_project.ndim == 2:
            line_dot = np.expand_dims(line_dot, axis=-1)

        # Project the vectors onto the line direction
        return line_dot * self.line_direction

    def intersect_line(self, other_line: "Line") -> Tuple[np.ndarray, np.ndarray]:
        """
        cf. docs of intersect_lines()
        """
        return self.intersect_lines(self, other_line)

    @staticmethod
    def intersect_lines(line1: "Line", line2: "Line") -> Tuple[np.ndarray, np.ndarray]:
        """
        Find the intersection of two lines in 3D space.
        Two arbitrary lines never actually intersect each other in 3D space,
        and even if the lines are chosen to be in the same plane,
        numerical inaccuracies may still not find an exact
        intersection, so this method computes the two points that lie on the first and second line respectively and
        that are closest to the second and first line respectively. If the points do actually intersect,
        these two points should be (close to) identical.
        :param line1: First line (to be intersected with line2)
        :param line2: Second line (to be intersected with line1)
        :return: Tuple of two 3D numpy points: 1) point on first line that's closest to second line
                                               2) point on second line that's closest to first line
        :raise ValueError: If the two lines are parallel
        """
        p1, p2 = line1.get_point(), line2.get_point()
        d1, d2 = line1.get_direction(), line2.get_direction()

        # Define vector w from p2 to p1
        w0 = p1 - p2

        # Calculate coefficients of the system
        a = np.dot(d1, d1)
        b = np.dot(d1, d2)
        c = np.dot(d2, d2)
        d = np.dot(d1, w0)
        e = np.dot(d2, w0)

        # Solve the system of equations to find t and s
        denominator = a * c - b * b
        if denominator < 1e-6:
            raise ValueError("The lines are (close to) parallel, so there's no unique closest point.")

        t = (b * e - c * d) / denominator
        s = (a * e - b * d) / denominator

        # Closest point on Line 1
        closest_point_line1 = p1 + t * d1
        # Closest point on Line 2 (for reference, not necessarily needed)
        closest_point_line2 = p2 + s * d2

        return closest_point_line1, closest_point_line2

    def intersect_line_and_line_segments(self, line_segments):
        """
        Intersect a line with a set of line segments. Since it's 3D, the "intersection" will check for the closest point
        on each line segment and check if it's within the segment's limits. There can be from zero up to n intersection
        points, where n would be the number of line segments (although having n intersection points would kinda mean that
        the line segments go zick-zack from one side of the line to the other, or something like that).
        :param line_segments: List of line segments, each segment consisting of two 3D numpy points that
                              define beginning and end point of line segment.
        :return: Tuple: 1) List of 3D numpy intersection points.
                        2) List of line segment indices that correspond to the intersection points.
        """
        # Keep results in list
        intersection_points = []
        intersection_segment_indices = []
        # Loop over each line segment and check individually for intersection with line
        for segment_idx, line_segment in enumerate(line_segments):
            line_segment_point = line_segment[0]
            line_segment_direction = line_segment[1]-line_segment[0]
            # Find point on line segment that's closest to line
            try:
                _, segment_intersection_point = self.intersect_line(Line(line_segment_point, line_segment_direction))
            # If the line segment is (very close) to parallel to the line, we skip it
            except ValueError:
                continue
            intersection_vector = segment_intersection_point - line_segment_point
            # If the intersection point is closer to the first line segment point than the second point is,
            # and if the intersection point is on the positive side of the first segment point, i.e.,
            # on the same side as the second, i.e., the vectors between int point and first point and
            # between second point and first point align, then the intersection points is within the
            # limits of the line segment.
            if np.linalg.norm(intersection_vector) <= np.linalg.norm(line_segment_direction) and np.dot(intersection_vector, line_segment_direction) > 0:
                intersection_points.append(segment_intersection_point)
                intersection_segment_indices.append(segment_idx)
        # Return the 3D intersection points and the corresponding list of line segment indices.
        return intersection_points, intersection_segment_indices
