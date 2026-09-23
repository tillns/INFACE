"""
This script allows to reconstruct 3D geometry of an infant face based on a monocular 2D image, as described in the
INFACE paper. The method is purely landmark based: We pre-defined landmarks on the face models, and we use INFANFACE
to detect respective 2D landmarks on the input images. We then optimize camera and model parameters to best explain
the detected 2D image. There's no deep surface or appearance matching, no metrical guarantees, and no going out of
model space. The script only works with the face models, not with the head models. If you would be interested in
getting a whole head reconstruction from e.g. a video, contact me.

To use this script, call it and provide the path to the face model, the directory that includes the face images,
and an output directory to save the results in. Use --help for an explanation of the arguments.

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

import argparse
import math
from typing import Union, List

import cv2
from pathlib import Path
from copy import deepcopy

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm
from scipy.spatial import ConvexHull

from src import utils
from src.blender.python_interface import render_image
from src.processing.infanface_lm_detector.model import InfFaceLMDetector

from src.objects.mesh import Mesh, CurvilinearFeatures, Landmarks
from src.objects.morphable_model import MorphableModel
from src.processing.registration import get_rot_matrix


def get_align_point_weights(lm_kind):
    """
    We use these weights during initial alignment by focusing on some major points in the face.
    """
    if "eye" in lm_kind and not "eyebrow" in lm_kind:
        return 5
    if lm_kind == "nasion":
        return 1
    if lm_kind.endswith("nasale"):
        return 1
    if lm_kind == "pogonion":
        return 5
    if lm_kind.endswith("nostril"):
        return 0
    if lm_kind.startswith("columella"):
        return 0
    if lm_kind in ["cubids_bow_center", "tubercle", "lower_lip_inner_middle", "lower_lip_outer_middle",
                   "inner_lip_corner_left", "inner_lip_corner_right", "lip_cubids_peak_left", "lip_cubids_peak_right"]:
        return 0
    if lm_kind.startswith("outer_lip_corner"):
        return 0.5
    if lm_kind.startswith("eyebrow"):
        return 0
    raise AssertionError(f"You haven't considered lm_kind {lm_kind}.")



def draw_other_landmarks_on_image(
        image: np.ndarray, point_landmarks: np.ndarray, line_sets: Union[List, np.ndarray],
        save_path: Union[str, Path, None] = None, resize_min_res: int = None, invert_y: bool = False,
        write_numbers: bool = True, write_line_numbers: bool = False) -> None:
    """
    Draw landmarks and contour lines on image and show or save the output.
    :param image: numpy image array
    :param point_landmarks: Provide normal landmarks here as numpy array.
    :param line_sets: We draw lines through the line sets.
    :param save_path: Optionally provide a path where you want to save the image. If you don't provide this,
    the image will be shown.
    :param resize_min_res: integer specifying the minimum size of either side of the image.
                           The image will be upscaled only by integers to have boths sides longer than this minimum.
    :param invert_y: Invert the y-direction of the image (compatibility between PIL and cv2 might require this).
    :param write_numbers: Write numbers on the landmarks. Disable this to just look at the points without numbers.
    :param write_line_numbers: Same as write_numbers but for line_sets.
    :return: None
    """
    copied_image = np.ascontiguousarray(deepcopy(image))
    ih, iw, _ = copied_image.shape
    if resize_min_res is not None:
        resize_factor = max(max(resize_min_res / iw, resize_min_res / ih), 1)
        if resize_factor > 1:
            resize_factor = int(math.ceil(resize_factor))
            copied_image = cv2.resize(copied_image, (resize_factor*iw, resize_factor*ih))
            ih, iw, _ = copied_image.shape

    point_size = max(int(16 * ih/1000), 2)
    line_thickness = max(int(6 * iw/1000), 1)
    text_scale = 0.4 * ih/1000
    line_text_scale = text_scale / 2

    def get_img_coords(current_lm):
        lm_copy = deepcopy(current_lm)
        if invert_y:
            lm_copy[1] = -lm_copy[1]
        return int((lm_copy[0] + 1) / 2 * iw), int((lm_copy[1] + 1) / 2 * ih)

    for num_lm, lm in enumerate(point_landmarks):
        x, y = get_img_coords(lm)
        cv2.circle(copied_image, (x, y), point_size, (0, 255, 0), -1)  # Draw circle


    if line_sets is not None:
        for num_line_set, line_set in enumerate(line_sets):
            lms_xy = [get_img_coords(lm) for lm in line_set]
            if len(lms_xy) > 1:
                for num_line, ((x_start, y_start), (x_end, y_end)) in enumerate(zip(lms_xy[:-1], lms_xy[1:])):
                    cv2.line(copied_image, (x_start, y_start), (x_end, y_end), (255, 0, 0), line_thickness)


    if write_numbers:
        for num_lm, lm in enumerate(point_landmarks):
            x, y = get_img_coords(lm)
            cv2.putText(copied_image, str(num_lm), (x, y), cv2.FONT_HERSHEY_SIMPLEX, text_scale, (0, 0, 255),
                        max(line_thickness//4, 1))  # Adjust text position

    if write_line_numbers and line_sets is not None:
        for num_line_set, line_set in enumerate(line_sets):
            lms_xy = [get_img_coords(lm) for lm in line_set]
            if len(lms_xy) > 1:
                for num_line, ((x_start, y_start), (x_end, y_end)) in enumerate(zip(lms_xy[:-1], lms_xy[1:])):
                    x, y = (x_start + x_end) // 2, (y_start + y_end) // 2
                    cv2.putText(copied_image, f"{num_line_set + 0}_{num_line + 0}",
                                (x, y), cv2.FONT_HERSHEY_SIMPLEX, line_text_scale, (255, 0, 255), 1)  # Adjust text position

    if save_path is None:
        cv2.imshow('image', copied_image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    else:
        if Path(save_path).exists():
            Path(save_path).unlink()
        cv2.imwrite(str(save_path), copied_image)


def get_cam_to_world_matrix(translation, rotation, convert_to_numpy: bool = False, convert_to_open3d: bool = False, device: str = "cpu"):
    translation_matrix = torch.eye(4, device=device)
    translation_matrix[[0, 1, 2], -1] = translation
    rotx = get_rot_matrix(rotation[0], "x", device=device)
    roty = get_rot_matrix(rotation[1], "y", device=device)
    rotz = get_rot_matrix(rotation[2], "z", device=device)
    rot_mat = rotz @ roty @ rotx
    if convert_to_open3d:
        rotx_open3d = get_rot_matrix(torch.tensor(torch.pi), "x", device=device)
        rot_mat = rotx_open3d @ rot_mat
    full_mat = translation_matrix @ rot_mat
    if convert_to_numpy:
        full_mat = full_mat.detach().cpu().numpy()
    return full_mat


def get_world_to_cam_matrix(translation, rotation, device: str = "cpu"):
    return torch.linalg.inv(get_cam_to_world_matrix(translation, rotation, device=device))


def import_image(
        image_or_path: Union[str, Path, np.ndarray, Image.Image], switch_blue_and_red: bool = False,
        return_as_numpy: bool = False, convert_to_float: bool = False,
        keep_alpha_channel: bool = False) -> Union[np.ndarray, Image.Image]:
    """
    Import an image from a path or convert an already given image between PIL and numpy and between float and uint8.
    :param image_or_path: Provide an image as np array or PIL Image, or a path to one.
    :param switch_blue_and_red: Set True to switch the first and third color channel.
    :param return_as_numpy: Set True to get the image back as numpy array, keep False to get a PIL Image.
    :param convert_to_float: Set True to convert the image to float with values in [0, 1], keep False to get uint8.
                             Only applies if return_as_numpy=True.
    :param keep_alpha_channel: Set True if the image as an alpha channel you want to keep.
                               The alpha channel is removed by default.
    :return: numpy array (float [0,1] or uint8) or PIL Image.
    """
    # --- Load image ---
    if isinstance(image_or_path, (str, Path)):
        img = Image.open(image_or_path)
    else:
        img = deepcopy(image_or_path)

    if isinstance(img, Image.Image):
        img = np.asarray(img)

    assert isinstance(img, np.ndarray)

    # Remove alpha channel if present, unless user chose to keep it
    if img.ndim == 3 and img.shape[-1] == 4 and not keep_alpha_channel:
        img = img[..., :3]

    # switch red and blue channel if chosen by user and if not a grayscale image
    if img.ndim == 3 and img.shape[-1] >= 3 and switch_blue_and_red:
        img = img[..., [2, 1, 0, *list(range(3, img.shape[-1]))]]

    if return_as_numpy:
        if convert_to_float:
            if not np.issubdtype(img.dtype, np.floating):
                img = img.astype(np.float32) / 255.0
        else:
            if np.issubdtype(img.dtype, np.floating):
                img = np.clip(img, 0.0, 1.0)
                img = (img * 255.0).round().astype(np.uint8)
        return img

    else:
        # NumPy → PIL
        if img.dtype != np.uint8:
            # assume float [0,1]
            img = np.clip(img, 0.0, 1.0)
            img = (img * 255.0).round().astype(np.uint8)

        if img.ndim == 2:
            # grayscale
            return Image.fromarray(img, mode="L")

        else:
            assert img.ndim == 3
            c = img.shape[-1]
            if c == 3:
                return Image.fromarray(img, mode="RGB")
            else:
                assert c == 4
                return Image.fromarray(img, mode="RGBA")


def get_landmark_names_and_detected_indices() -> dict:
    """
    Just hard-coded matching between our model landmarks,
    and the landmarks predicted by the INFANFACE landmark detector.
    """
    return {
        "columella_inner_left": 34,
        "columella_inner_right": 32,
        "cubids_bow_center": 51,
        "eyebrow_left_inner_endpoint": 22,
        "eyebrow_left_outer_endpoint": 26,
        "eyebrow_right_inner_endpoint": 21,
        "eyebrow_right_outer_endpoint": 17,
        "inner_lip_corner_left": 64,
        "inner_lip_corner_right": 60,
        "left_eye_inner": 42,
        "left_eye_outer": 45,
        "left_nostril": 35,
        "lip_cubids_peak_left": 52,
        "lip_cubids_peak_right": 50,
        "lower_lip_inner_middle": 66,
        "lower_lip_outer_middle": 57,
        "nasion": 27,
        "outer_lip_corner_left": 54,
        "outer_lip_corner_right": 48,
        "pogonion": 8,
        "pronasale": 30,
        "right_eye_inner": 39,
        "right_eye_outer": 36,
        "right_nostril": 31,
        "subnasale": 33,
        "tubercle": 62
    }

def get_curvilinear_feature_names_and_detected_indices() -> dict:
    """
    Just hard-coded matching between our model curvilinear features,
    and the landmarks predicted by the INFANFACE landmark detector.
    """
    return {
        "left_eyebrow": [26, 25, 24, 23, 22],
        "lower_left_eye": [45, 46, 47, 42],
        "lower_lip_inner_left": [66, 65, 64],
        "lower_lip_inner_right": [66, 67, 60],
        "lower_lip_outer_left": [57, 56, 55, 54],
        "lower_lip_outer_right": [57, 58, 59, 48],
        "lower_right_eye": [36, 41, 40, 39],
        "nose_ridge": [27, 28, 29, 30],
        "right_eyebrow": [17, 18, 19, 20, 21],
        "upper_left_eye": [45, 44, 43, 42],
        "upper_lip_inner_left": [62, 63, 64],
        "upper_lip_inner_right": [62, 61, 60],
        "upper_lip_outer_left": [52, 53, 54],
        "upper_lip_outer_right": [50, 49, 48],
        "upper_right_eye": [36, 37, 38, 39],
    }

class Reconstruction:
    """
    Class to reconstruct 3D geometry from monocular image
    """
    def __init__(self, model_path: Union[str, Path]):
        """
        We initialize some parameters here, so that we don't need to redefine them during each iteration
        of the reconstruction.
        """
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        # Load model, use torch for optimization
        self.model = MorphableModel.load_correct_morphable_model(model_path, use_torch=True, device=self.device)

        # Get average model vertices for normalization
        template = self.model.get_average_mesh()
        template_vertices, self.triangles = template.get_vertices(), template.get_triangles()
        self.template_vertices_torch = torch.as_tensor(template_vertices, device=self.device, dtype=torch.float32)

        # We distinguish between normal and detected landmarks. The normal landmarks are the 3D
        # landmarks on our model. The detected landmarks are the 2D landmarks predicted on the
        # image.
        landmark_names_and_detected_indices = get_landmark_names_and_detected_indices()

        curvilinear_feature_names_and_detected_indices = get_curvilinear_feature_names_and_detected_indices()

        self.detected_landmark_point_indices = np.asarray(list(landmark_names_and_detected_indices.values()))
        self.detected_curvilinear_feature_indices = list(curvilinear_feature_names_and_detected_indices.values())
        # this is another hard-coded extraction of the INFANFACE lower contour landmarks,
        # the order is matched with our model landmarks defined further below in
        # self.mesh_face_landmark_contour_indices
        self.detected_landmark_contour_indices = [[16, 15, 14, 13, 12, 11, 10, 9], [0, 1, 2, 3, 4, 5, 6, 7]]

        # We separate the face into left and right to compute the lower contour for each half to avoid overlaps
        self.face_left_indices = self.model.get_indices("face_left")
        self.face_right_indices = self.model.get_indices("face_right")

        self.mesh_face_landmark_point_indices = np.asarray([
            self.model.get_indices(lm_name, make_unique=False)[-1]
            for lm_name in landmark_names_and_detected_indices.keys()])

        self.mesh_face_curvilinear_feature_indices = [
            self.model.get_indices(feature_name, make_unique=False)
            for feature_name in curvilinear_feature_names_and_detected_indices.keys()
        ]

        self.mesh_face_landmark_contour_indices = (
            self.model.get_indices("lower_contour_left", make_unique=False),
            self.model.get_indices("lower_contour_right", make_unique=False))

        self.uniform_point_weights = torch.ones(len(self.mesh_face_landmark_point_indices), device=self.device)
        # non-uniform weight we use for the initial alignment iterations.
        self.align_point_weights = torch.tensor([
            get_align_point_weights(lm_name) for lm_name in landmark_names_and_detected_indices.keys()],
            device=self.device)

        self.uniform_line_set_weights = torch.ones(len(self.mesh_face_curvilinear_feature_indices), device=self.device)

        # We give the contours a high weight, since all other landmarks are only positioned around the
        # face center, so to take a reasonable size, we need the contour.
        self.contour_accentuated_weights = torch.ones(len(self.mesh_face_landmark_contour_indices), device=self.device) * 5

        # Instantiate face landmark detector. We use InfAnFace. The code is copied from their github
        # https://github.com/ostadabbas/Infant-Facial-Landmark-Detection-and-Tracking.
        # The model checkpoints can be downloaded from
        # https://drive.google.com/drive/folders/1sSBXbRmYWVQ3cOF-qNN7aSWRE_L02EYh
        self.lm_detector = InfFaceLMDetector()

    def reconstruct(self, images_or_paths: List[Union[str, Path, np.ndarray, Image.Image]],
                    output_dir: Union[str, Path], overwrite: bool = False):
        """
        Main method that loops over the provided images and produces a 3D reconstruction for each.
        For each input image, a separate folder subfolder is created inside the given output directory,
        in which we save several images and the final mesh.
        :param images_or_paths:
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(exist_ok=True)
        for num_image, image_or_path in enumerate(tqdm(images_or_paths)):
            if isinstance(image_or_path, (str, Path)):
                image_path = Path(image_or_path)
                image_stem = image_path.stem
            else:
                image_stem = str(num_image)

            # Create output subdirectory for the current image
            output_dir_specific = output_dir.joinpath(image_stem)
            output_dir_specific.mkdir(exist_ok=True)
            # Check if mesh has already been computed, and go to next image in that case
            # unless user chose to overwrite
            output_mesh_path = output_dir_specific.joinpath(f"mesh.ply")
            if output_mesh_path.exists() and not overwrite:
                continue
            image = import_image(image_or_path, return_as_numpy=True, convert_to_float=False, switch_blue_and_red=True)
            height, width = image.shape[:2]

            # Predict landmarks
            raw_image = import_image(image, return_as_numpy=True, convert_to_float=True, switch_blue_and_red=True)
            if not height == width:
                # If the image is not square, we perform a center cropping of it along the longer axis.
                diff = np.abs(height - width)
                diff_half = int(diff / 2)
                depth = raw_image.shape[2]
                if height > width:
                    cropped_image = np.concatenate([
                        np.zeros((height, diff_half, depth), dtype=raw_image.dtype), raw_image,
                        np.zeros((height, diff - diff_half, depth), dtype=raw_image.dtype)], axis=1)
                else:
                    cropped_image = np.concatenate([
                        np.zeros((diff_half, width, depth), dtype=raw_image.dtype), raw_image,
                        np.zeros((diff - diff_half, width, depth), dtype=raw_image.dtype)], axis=0)
            else:
                cropped_image = raw_image
            # We resize the square image to 256x256, because this is the size the landmark detector expects.
            resized_image = cv2.resize(cropped_image, (256, 256))
            if resized_image.ndim == 2:
                resized_image = np.expand_dims(resized_image, axis=-1)
            # We rescale and reshift the predicted landmarks to the original image (unscaled and uncropped)
            landmarks = self.lm_detector.forward(resized_image) * height/256
            if not height == width:
                if height > width:
                    landmarks[:, 0] -= diff_half
                else:
                    landmarks[:, 1] -= diff_half

            ih, iw, _ = image.shape
            # input landmarks expected to be in image pixel space, transformed ones are in UV [-1, 1] space
            lms_transformed = landmarks / np.asarray([[iw, ih]]) * 2 - 1
            lms_transformed[:, 1] *= -1
            # Move landmarks to GPU for optimization
            detected_lms_torch = torch.as_tensor(lms_transformed, device=self.device, dtype=torch.float32)
            # Distinguish between normal and contour landmarks
            detected_point_landmarks = detected_lms_torch[self.detected_landmark_point_indices]
            detected_curvilinear_features = CurvilinearFeatures([
                lms_transformed[line_set_ind] for line_set_ind in self.detected_curvilinear_feature_indices])
            detected_contour_features = CurvilinearFeatures([
                lms_transformed[contour_ind] for contour_ind in self.detected_landmark_contour_indices])
            # We save the input image into the target directory
            cv2.imwrite(str(output_dir_specific.joinpath(f"input.png")), image)
            # We save the input image with the detected landmarks drawn on it.
            draw_other_landmarks_on_image(
                image, detected_point_landmarks,
                detected_curvilinear_features.get_line_sets()+detected_contour_features.get_line_sets(),
                invert_y=True, save_path=output_dir_specific.joinpath(f"input_landmarked.png"),
                write_numbers=False)

            # Camera parameters to optimize
            # The unit is mm here, but we only work with their relative factor, so it actually doesn't matter
            camera_focal_length = torch.tensor(50, dtype=torch.float32, requires_grad=True, device=self.device)
            camera_sensor_width = torch.tensor(36, dtype=torch.float32, requires_grad=True, device=self.device)
            camera_aspect_ratio = image.shape[1] / image.shape[0]
            cam_translation = torch.asarray([0, 0, 0.2], dtype=torch.float32, requires_grad=True, device=self.device)
            cam_rotation = torch.asarray([0, 0, 0], dtype=torch.float32, requires_grad=True, device=self.device)

            latent_mean, latent_std = self.model.latent_mean, self.model.latent_std
            # Initialize model parameters to be optimized to mean
            current_latent = torch.zeros_like(latent_mean, requires_grad=True)
            latent_variables = [current_latent]

            def get_current_vertices(return_numpy: bool = False, scale_vertices_to_meters: bool = True):
                """
                Get the vertices reconstructed from the current state of the model parameters, potentially
                rescaled to meters.
                """
                absolute_latent = latent_mean + current_latent * latent_std
                vertices = self.template_vertices_torch + self.model.decode(absolute_latent)
                if return_numpy:
                    vertices = vertices.detach().cpu().numpy()
                # scale down from mm to m.
                if scale_vertices_to_meters:
                    vertices = vertices / 1000
                return vertices

            def get_mesh(any_transform=None, scale_vertices_to_meters: bool = False, mesh: Union[Mesh, None] = None) -> Mesh:
                """
                Get the mesh from the current state of the vertices. You can also provide the mesh, and apply
                rescaling to meters or any other transformation.
                """
                if mesh is not None:
                    vertices, triangles = mesh.get_vertices(copy=True), mesh.get_triangles()
                    if np.any(np.max(vertices, axis=0) - np.min(vertices, axis=0) > 3):
                        if scale_vertices_to_meters:
                            vertices = vertices / 1000
                    else:
                        if not scale_vertices_to_meters:
                            vertices = vertices * 1000
                else:
                    vertices = get_current_vertices(return_numpy=True, scale_vertices_to_meters=scale_vertices_to_meters)
                    triangles = self.triangles
                if any_transform is not None:
                    vertices = transform_vertices(vertices, any_transform)
                return Mesh(vertices=vertices, triangles=triangles)

            def transform_vertices(current_vertices, current_transform):
                """
                Transform the vertices, either as numpy or as torch (keeping torch gradients).
                """
                if isinstance(current_vertices, np.ndarray):
                    if isinstance(current_transform, torch.Tensor):
                        current_transform = current_transform.detach().cpu().numpy()
                    ones = torch.ones(current_vertices.shape[0], 1, device=self.device)
                    vertices_homogeneous = np.concatenate([current_vertices, ones], axis=1)
                    return np.einsum('ab,cb->ca', current_transform, vertices_homogeneous)[:, :3]
                else:
                    ones = torch.ones(current_vertices.shape[0], 1, dtype=torch.float32, device=self.device)
                    vertices_homogeneous = torch.cat([current_vertices, ones], dim=1)
                    return torch.einsum('ab,cb->ca', current_transform, vertices_homogeneous)[:, :3]

            def project_vertices(current_vertices):
                """
                Project the current vertices to camera space
                (keeping torch gradients of the vertices and the camera parameters).
                """
                camera_rel_width = camera_focal_length / camera_sensor_width
                world_to_cam_matrix = get_world_to_cam_matrix(cam_translation, cam_rotation, device=self.device)
                vertices_cam_coord = transform_vertices(current_vertices, world_to_cam_matrix)
                vertices_projected_x = - (vertices_cam_coord[:, 0] / vertices_cam_coord[:, 2] * camera_rel_width)
                vertices_projected_y = - (vertices_cam_coord[:, 1] / vertices_cam_coord[:,
                                                                     2] * camera_rel_width * camera_aspect_ratio)
                return torch.stack([vertices_projected_x, vertices_projected_y], dim=1)

            def compute_dist(generated_vertices, target_points):
                """
                Compute 2D (image space) Euclidean distance between the vertices and their target.
                """
                return torch.linalg.norm(generated_vertices[:, :2] - target_points[:, :2], dim=1)

            def has_converged(losses, rel_error_break, window_length: int = 5):
                """
                Check the past losses for convergence. We compare the last window_length loss entries
                with the window_length loss entries that came before them, and if the average hasn't gone
                down (below rel_error_break), we assume it has converged.
                """
                with torch.no_grad():
                    return len(losses) >= 2 * window_length and torch.mean(
                        torch.stack(losses[-2 * window_length:-window_length])) / torch.mean(torch.stack(losses[-window_length:])) < rel_error_break

            def get_current_projected_point_line_and_contour_vertices(
                    return_numpy: bool = False, scale_vertices_to_meters: bool = True):
                """
                Project the vertices from the current model parameters to the image space.
                Return the points at the relevant point and contour landmarks.
                The contour landmarks are adjusted based on the convex hull of the projected points
                in the image space.
                """

                current_vertices = get_current_vertices(
                    return_numpy=False, scale_vertices_to_meters=scale_vertices_to_meters)
                projected_vertices = project_vertices(current_vertices)
                projected_vertices_numpy = projected_vertices.detach().cpu().numpy()

                # Compute the current lower counter based on the convex hull of the projected vertices
                # We separate the left from the right face to avoid overlapping issues
                hull_left = ConvexHull(projected_vertices_numpy[self.face_left_indices])
                hull_right = ConvexHull(projected_vertices_numpy[self.face_right_indices])
                projected_contour_vertex_indices = [face_indices[hull.vertices[np.argmin(np.linalg.norm(
                    np.expand_dims(hull.points[hull.vertices], axis=0) -
                    np.expand_dims(projected_vertices_numpy[contour_indices], axis=1), axis=-1), axis=-1)]]
                    for contour_indices, hull, face_indices in zip(
                        self.mesh_face_landmark_contour_indices, [hull_left, hull_right],
                        [self.face_left_indices, self.face_right_indices])]

                projected_contour_vertices = [
                    projected_vertices[contour_indices] for contour_indices in projected_contour_vertex_indices]

                # Points
                projected_point_vertices = projected_vertices[self.mesh_face_landmark_point_indices]

                # Line sets
                projected_line_set_vertices = [projected_vertices[line_indices] for line_indices
                                           in self.mesh_face_curvilinear_feature_indices]

                if return_numpy:
                    projected_point_vertices = projected_point_vertices.detach().cpu().numpy()
                    projected_line_set_vertices = [line_verts.detach().cpu().numpy() for line_verts
                                                   in projected_line_set_vertices]
                    projected_contour_vertices = [
                        line_verts.detach().cpu().numpy() for line_verts in projected_contour_vertices]

                return projected_point_vertices, projected_line_set_vertices, projected_contour_vertices

            def get_dist_loss(point_weights, line_set_weights, contour_weights, point_v_line_weight_ratio: float = 1):
                """
                This is the distance loss function, based on which to optimize the model and camera parameters.
                The distance is computed between the model vertices at the point and contour landmark indices,
                projected into image space, and the 2D detected point and contour landmarks.
                The point and contour distances can be weighted differently for the loss.
                """
                projected_point_vertices, projected_line_set_vertices, projected_contour_vertices = (
                    get_current_projected_point_line_and_contour_vertices(return_numpy=False))

                # Point distance computation (simple, because closest points are fixed)
                point_dists = compute_dist(projected_point_vertices, detected_point_landmarks)
                point_dists_average = torch.sum(point_dists * point_weights) / torch.sum(point_weights)



                # Line set distance computation (first need to find the closest points)
                projected_features = CurvilinearFeatures([
                    line_set.detach().cpu().numpy() for line_set in projected_line_set_vertices])
                closest_line_set_points, _ = projected_features.get_closest_points(detected_curvilinear_features)
                line_sets_dists = [
                    compute_dist(torch.tensor(closest_line, device=self.device), projected_line)
                    for projected_line, closest_line in zip(projected_line_set_vertices, closest_line_set_points)]
                line_sets_dists_average = sum([
                    torch.sum(line_set_dists * line_set_weight) / len(line_set_dists)
                    for line_set_weight, line_set_dists in
                    zip(line_set_weights, line_sets_dists)]) / torch.sum(line_set_weights)

                # Contour distance computation (same as for line sets)
                projected_contour_features = CurvilinearFeatures([
                    contour.detach().cpu().numpy() for contour in projected_contour_vertices])
                closest_contour_points, _ = projected_contour_features.get_closest_points(detected_contour_features)
                contours_dists = [
                    compute_dist(torch.tensor(closest_line, device=self.device), projected_line)
                    for projected_line, closest_line in zip(projected_contour_vertices, closest_contour_points)]
                contours_dists_average = sum([
                    torch.sum(contour_dists * contour_weight) / len(contour_dists)
                    for contour_weight, contour_dists in
                    zip(contour_weights, contours_dists)]) / torch.sum(contour_weights)

                # Reweighting of loss.
                point_weight, line_set_weight, contour_weight = point_v_line_weight_ratio, 0.5, 0.5
                return (point_weight / (point_weight + line_set_weight + contour_weight) * point_dists_average +
                        line_set_weight / (point_weight + line_set_weight + contour_weight) * line_sets_dists_average +
                        contour_weight / (point_weight + line_set_weight + contour_weight) * contours_dists_average)

            def change_lr(optimizer, lr):
                """
                Change learning rate for all parameters.
                """
                for param_group in optimizer.param_groups:
                    param_group['lr'] = lr

            def optimize_camera(point_weights, line_set_weights, contour_weights,
                                rel_error_break: Union[float, List],
                                point_v_dist_ratio: float = 1,
                                optimize_camera_intrinsics: bool = True, num_it_max: int = 1000,
                                lr: Union[float, List] = 1e-2, window_length: int = 5):
                """
                This is the method to optimize the camera parameters
                (no optimization of model parameters; these are kept fixed).
                We define separate optimizers for the camera extrinsics and intrinsics, and we loop until
                convergence of the landmark distance loss; we support multiple learning rates, in which
                case we loop until convergence on the first, then switch to the second and loop again, etc.
                You may choose to optimize only the camera extrinsics, so you can align the camera first.
                """
                if not isinstance(lr, list):
                    lr = [lr]
                lr_idx = 0
                if not isinstance(rel_error_break, list):
                    rel_error_break = [rel_error_break]
                if not len(rel_error_break) < len(lr):
                    rel_error_break = [*rel_error_break] + [rel_error_break[-1] for _ in
                                                            range(len(lr) - len(rel_error_break))]
                rel_error_break = rel_error_break[:len(lr)]

                # Define optimizers
                cam_extrinsic_optimizer = torch.optim.Adam([cam_translation, cam_rotation], lr=lr[lr_idx])
                cam_intrinsic_optimizer = torch.optim.Adam([camera_focal_length, camera_sensor_width], lr=lr[lr_idx])
                # Optionally disable gradients of camera intrinsics
                camera_sensor_width.requires_grad = optimize_camera_intrinsics
                camera_focal_length.requires_grad = optimize_camera_intrinsics
                # Disable model optimization
                for latent_variable in latent_variables:
                    latent_variable.requires_grad = False
                # Camera extrinsics are always optimized.
                cam_rotation.requires_grad = True
                cam_translation.requires_grad = True
                losses = []
                # Loop: first compute landmark loss, then optimize camera parameters, lower learning rate upon
                # convergence, and repeat until last learning rate is reached.
                for it in range(num_it_max):
                    camera_loss = get_dist_loss(
                        point_weights, line_set_weights, contour_weights,
                        point_v_line_weight_ratio=point_v_dist_ratio)
                    cam_extrinsic_optimizer.zero_grad()
                    if optimize_camera_intrinsics:
                        cam_intrinsic_optimizer.zero_grad()
                    camera_loss.backward()
                    cam_extrinsic_optimizer.step()
                    if optimize_camera_intrinsics:
                        cam_intrinsic_optimizer.step()
                    losses.append(camera_loss.data)
                    if has_converged(losses, rel_error_break[lr_idx], window_length=window_length):
                        lr_idx += 1
                        if lr_idx == len(lr):
                            break
                        losses.clear()
                        change_lr(cam_extrinsic_optimizer, lr[lr_idx])
                        change_lr(cam_intrinsic_optimizer, lr[lr_idx])

                return torch.mean(torch.stack(losses[-window_length:])).detach().cpu().numpy()

            def optimize_latent(point_weights, line_set_weights, contour_weights, rel_error_break: float, point_v_dist_ratio: float = 1,
                                num_it_max=1000, alpha=0.01,
                                lr: float = 1e-2, window_length: int = 5):
                """
                Here, we keep all camera parameters fixed and only optimize
                the model parameters. We also use a regularization on the size of the model parameters to avoid
                unrealistic results. We don't lower the learning rate here, we just loop until we converge on the
                initial learning rate.
                """
                latent_optimizer = torch.optim.Adam(latent_variables, lr=lr)

                camera_sensor_width.requires_grad = False
                camera_focal_length.requires_grad = False
                cam_rotation.requires_grad = False
                cam_translation.requires_grad = False
                for latent_variable in latent_variables:
                    latent_variable.requires_grad = True
                losses = []
                for it in range(num_it_max):
                    dist_loss = get_dist_loss(point_weights, line_set_weights, contour_weights,
                                              point_v_line_weight_ratio=point_v_dist_ratio)
                    model_reg_loss = torch.mean(torch.cat([torch.square(latent_variable) for latent_variable in latent_variables], dim=-1))
                    loss = dist_loss + alpha * model_reg_loss
                    latent_optimizer.zero_grad()
                    loss.backward()
                    latent_optimizer.step()
                    losses.append(loss.data)
                    if has_converged(losses, rel_error_break=rel_error_break, window_length=window_length):
                        break

                return torch.mean(torch.stack(losses[-window_length:])).detach().cpu().numpy()

            def render_current_mesh(output_name_add: str = None, mesh_name_add: str = None, mesh=None):
                """
                Render the mesh with Blender using the vertices reconstructed from the current model parameters,
                as well as the optimized camera settings.
                """
                # Get the mesh from the current model parameters
                mesh = get_mesh(scale_vertices_to_meters=True, mesh=mesh)

                # Path naming.
                if mesh_name_add is not None:
                    keep_exported_image = True
                    output_image_path = output_dir_specific.joinpath(f"{mesh_name_add}.png")
                else:
                    keep_exported_image = False
                    output_image_path = output_dir.joinpath(f"mesh_rendered_tmp.png")

                # Temporally export the mesh to be rendered to give the path to Blender.
                mesh_tmp_path = output_image_path.with_suffix(".ply")
                mesh.export(mesh_tmp_path)

                # Render with blender.
                render_image(
                    mesh_tmp_path, camera_location=list(cam_translation.detach().cpu().numpy()),
                    camera_rotation_euler_rad=list(cam_rotation.detach().cpu().numpy()),
                    camera_focal_length=float(camera_focal_length.data) / 2,  # no idea why we divide by 2 here
                    camera_sensor_width=float(camera_sensor_width.data), resolution=[iw, ih],
                    output_image_path=output_image_path, switch_sensor_fit=True)

                # remove temporary mesh again
                mesh_tmp_path.unlink()

                # Combine the rendered mesh with the input mesh and save the result
                rendered_image = import_image(output_image_path, return_as_numpy=False, keep_alpha_channel=True)
                actual_image = import_image(image, return_as_numpy=False, switch_blue_and_red=True)
                combined = Image.alpha_composite(actual_image.convert('RGBA'), rendered_image)
                if output_name_add is not None:
                    combined.save(output_dir_specific.joinpath(f"{output_name_add}.png"))
                else:
                    combined.show()
                if not keep_exported_image:
                    output_image_path.unlink()

            # We optimize the camera and model parameters in an alternating fashion, first with large regularization
            # on the model parameters and only camera extrinsics for rough alignment, later more refined optimization.

            # Rough camera alignment
            optimize_camera(self.align_point_weights, self.uniform_line_set_weights, self.contour_accentuated_weights,
                            rel_error_break=[1.01, 1.001, 1.0001],
                            point_v_dist_ratio=0.3,
                            window_length=30, lr=[1e-2, 1e-3, 1e-4], num_it_max=10000,
                            optimize_camera_intrinsics=False)

            # Rough model alignment (first get overall scaling correct)
            optimize_latent(self.align_point_weights, self.uniform_line_set_weights, self.contour_accentuated_weights,
                            point_v_dist_ratio=0.3,
                            rel_error_break=1.001, alpha=0.01)

            # Optimize camera intrinsic as well
            optimize_camera(self.uniform_point_weights, self.uniform_line_set_weights, self.contour_accentuated_weights,
                            rel_error_break=[1.01, 1.001, 1.0001],
                            point_v_dist_ratio=0.5,
                            optimize_camera_intrinsics=True, window_length=30,
                            lr=[1e-2, 1e-3, 1e-4])

            # Optimize model with uniform weights to now also match expression
            optimize_latent(self.uniform_point_weights, self.uniform_line_set_weights, self.contour_accentuated_weights,
                            rel_error_break=1.001, alpha=0.005, point_v_dist_ratio=0.5)

            # Optimize camera again, now also with uniform weights
            final_cam_loss = optimize_camera(
                self.uniform_point_weights, self.uniform_line_set_weights, self.contour_accentuated_weights,
                point_v_dist_ratio=0.5,
                rel_error_break=[1.01, 1.001, 1.0005],
                optimize_camera_intrinsics=True, window_length=50,
                lr=[1e-2, 1e-3, 1e-4])
            print(f"Final Camera loss for {image_stem}: {final_cam_loss:.5f}")

            # Final model optimization
            final_model_loss = optimize_latent(
                self.uniform_point_weights, self.uniform_line_set_weights, self.contour_accentuated_weights,
                rel_error_break=1.001, alpha=0.0003,
                window_length=10, lr=1e-3, point_v_dist_ratio=0.5)
            print(f"Final Model loss for {image_stem}: {final_model_loss:.5f}")

            # Render mesh and overlay it with input image.
            render_current_mesh(output_name_add="combined", mesh_name_add="mesh")

            # Also export the final mesh.
            get_mesh(scale_vertices_to_meters=False).export(output_mesh_path)


def main():
    parser = argparse.ArgumentParser('Arguments')
    parser.add_argument('--model_path', type=str, required=True,
                        help="Absolute path to autoencoder or PCA model file with .h5 ending.")
    parser.add_argument('--image_dir', type=str, required=True)
    parser.add_argument('--output_dir', type=str, required=True)
    parser.add_argument('--overwrite', default=False, action="store_true")
    parser.add_argument('--filter_image_names', default=None, type=str, nargs="+")

    args = parser.parse_args()
    recon_class = Reconstruction(args.model_path)
    image_paths = utils.get_all_files_in_path(
        args.image_dir, file_suffixes=[".png", ".jpg", ".jpeg", ".JPG", ".PNG", ".JPEG"],
        regex_names=args.filter_image_names)

    recon_class.reconstruct(image_paths, overwrite=args.overwrite, output_dir=args.output_dir)



if __name__ == '__main__':
    main()