"""
predict.py — Automated infant face landmark prediction pipeline.

Usage
-----
Single scan:
    python predict.py path/to/scan.obj

All mesh files in a directory:
    python predict.py path/to/scans/

With custom output directory:
    python predict.py path/to/scans/ --output_dir results/

Run:
    python predict.py --help
for more information

Pipeline stages
---------------
1. Pass 1  — soft-argmax (sigma=0.02) on the raw mesh
2. Procrustes alignment + scale normalization
   (based on Pass 1 predictions)
3. Artifact detection: Provides an estimate of which vertices are artifacts.
3. Pass 2  — soft-argmax on the aligned+rescaled mesh, optionally also using artifact probabilities.


Output
------
1) csv file containing the landmarks, including a confidence measure
   (saved under OUTPUT/DIR/MESH_NAME_model.csv).
2) txt file containing the (binary) indices of vertices predicted to be artifacts
   (saved under OUTPUT/DIR/MESH_NAME_exclude-pred.txt).

Author: Till Schnabel, Vera Schubert (contact till.schnabel@inf.ethz.ch);
        parts of the code were copied, cf. further below.


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

import os
import argparse
import warnings
from pathlib import Path
from typing import Union, Tuple, Dict

import numpy as np
import torch
from tqdm import tqdm

from src.objects.indices_and_masks import IndicesAndMasks
from src.objects.mesh import Mesh, Landmarks
from models import models_folder

from src.processing.diffusion_net import (
    normalize_positions,
    compute_rescale_params, apply_rescale,
    build_model,
    run_forward, softargmax, peak_confidences,
)

# ---------------------------------------------------------------------------
# Single-scan pipeline
# ---------------------------------------------------------------------------

def predict_landmarks_and_artifacts(
        mesh_or_path: Union[str, Path, Mesh],
        ckpt_pass1: Union[str, Path, None] = None,
        ckpt_pass2: Union[str, Path, None] = None,
        ckpt_art: Union[str, Path, None] = None,
        lm_ref_target: Union[Landmarks, str, Path, np.ndarray, None] = None,
        device: Union[torch.device, str, None] = None,
        sigma_softmax: float = 0.02,
        conf_align_min: float = 0.9,
        include_artifact_prob_in_landmark_softmax: bool = False,
        op_cache_dir: Union[str, Path, None] = None,
):
    """
    Runs the pipeline on a single mesh. It returns landmark predictions that include also a confidence term, as well
    as a prediction for an artifact mask (binary).

    :param mesh_or_path: Mesh or path to mesh.
    :param ckpt_pass1: Path to checkpoint for first-pass prediction. If None, default path is used.
                       Note that this checkpoint must exist.
    :param ckpt_pass2: Path to checkpoint for second-pass prediction. If None, default path is used.
                       If the checkpoint can't be found, pass1 prediction is returned.
    :param ckpt_art: Path to checkpoint for artifact prediction. If None, default path is used.
                     If the checkpoint can't be found, an empty artifact mask is returned.
    :param lm_ref_target: Reference landmarks used to align mesh after first-pass prediction.
                          Neither artifact nor second-pass prediction is done without these reference landmarks.
    :param device: Device to run the model on. If None, Cuda is used if available.
    :param sigma_softmax: Sigma used for softmax aggregation.
    :param conf_align_min: Confidence threshold used for procrustes alignment on the landmarks.
    :param include_artifact_prob_in_landmark_softmax: Experimental setting. Set True to compute the landmark
                                                      positions based via softmax that not only uses the landmark
                                                      prediction probabilities, but also the artifact  probabilities.
                                                      It is currently not clear whether this improves the results.
    :param op_cache_dir: Directory to cache operators that need to be pre-computed before model prediction.
                         If None, operators are not cached, so if you run this method again with the same mesh,
                         the operators would need to be re-computed.

    :return: Tuple:
                1) Landmarks, including predicted positions, names (based on name entries in reference landmarks),
                   and confidence
                2) Binary mask with True entries where artifacts are predicted.
    """
    # Setup device
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device)

    # Setup paths
    ckpt_base_dir = models_folder.joinpath("diffusion-net", "healthy_faces_label_prediction")

    if ckpt_pass1 is None:
        ckpt_pass1 = ckpt_base_dir.joinpath("pass1.pt")
    if ckpt_pass2 is None:
        ckpt_pass2 = ckpt_base_dir.joinpath("pass2.pt")
    if ckpt_art is None:
        ckpt_art = ckpt_base_dir.joinpath("artifact_seg.pt")

    p1_model, config_pass1 = build_model(ckpt_pass1, device)
    if isinstance(mesh_or_path, Mesh):
        mesh = mesh_or_path
    else:
        mesh = Mesh.load(mesh_or_path, enable_post_processing=True, remove_duplicate_vertices=True)
    max_num_triangles = config_pass1.get("max_num_triangles", 800000)
    if max_num_triangles is not None and mesh.get_num_triangles() > max_num_triangles:
        raw_mesh = mesh.get_copy()
        mesh = mesh.simplify_quadric_decimation(max_num_triangles)
    else:
        raw_mesh = None
    verts_raw, faces = mesh.get_vertices(), mesh.get_faces()

    # Normalization parameters (needed for inverse transform later)
    center_raw  = verts_raw.mean(axis=0)
    centered    = verts_raw - center_raw
    max_rad_raw = float(np.max(np.linalg.norm(centered, axis=1)))
    if max_rad_raw < 1e-12:
        max_rad_raw = 1.0

    faces_tensor = torch.tensor(faces).long().to(device)
    verts_norm   = normalize_positions(verts_raw)
    verts_norm_t = torch.tensor(verts_norm).float().to(device)

    # -------------------------------------------------------------------------
    # Pass 1
    # -------------------------------------------------------------------------
    hm1, operators = run_forward(p1_model, verts_norm_t, faces_tensor, config_pass1.get("k_eig", 128), operators=None,
                                 op_cache_dir=op_cache_dir)
    pred_p1  = softargmax(hm1, verts_norm, sigma=sigma_softmax)
    conf_p1  = peak_confidences(hm1)
    del p1_model

    clean_probability = np.ones(mesh.get_num_vertices()).astype(bool)
    final_pred = pred_p1
    final_conf = conf_p1

    if lm_ref_target is None:
        lm_ref_target = ckpt_base_dir.joinpath("reference_landmarks.csv")
    try:
        lm_ref_target = Landmarks(lm_ref_target)
    except FileNotFoundError as e:
        warnings.warn(f"Ref landmarks {lm_ref_target} could not be found. Remaining prediction steps are skipped. "
                      f"Error: {e}")
        landmark_names = None
    else:
        landmark_names = lm_ref_target.names
        mask_align = conf_p1 >= conf_align_min
        # -------------------------------------------------------------------------
        # Procrustes alignment
        # -------------------------------------------------------------------------
        try:
            aligned_mesh, T = Mesh(verts_norm, faces, landmarks_or_points_array=pred_p1).align_procrustes(
                lm_ref_target, landmark_mask_or_indices=mask_align, scale=False, reflection=False,
                return_transform=True)
        except AssertionError as e:
            warnings.warn(f"Alignment failed. Returning initial prediction. Error: {e}")
        else:
            verts_aligned, faces_aligned = aligned_mesh.get_vertices(), aligned_mesh.get_faces()
            faces_p2_tensor = torch.tensor(faces_aligned).long().to(device)
            aligned_pred_p1 = Landmarks(Landmarks.transform_points(pred_p1, T),
                                        names=landmark_names, confidence=conf_p1)

            def get_scale_and_pivot(current_config: Dict) -> Tuple[float, np.ndarray]:
                # Scaling may vary depending on chosen lm pair. We provide default scalings here
                # based on average landmark distances from our dataset.
                # The interocular distance is our preferred metric, but we also provide
                # the inner eye corner distance and tragus distance as possible alternatives.
                align_pairs = current_config.get(
                    "align_pairs",
                    [("right_eye_outer", "left_eye_outer", 0.3),
                     ("right_eye_inner", "left_eye_inner", 0.12),
                     ("ear_tragus_right", "ear_tragus_left", 0.475),]
                )

                try:
                    scale, pivot, _ = compute_rescale_params(
                        aligned_pred_p1, align_pairs=align_pairs, confidence_lower_threshold=conf_align_min)
                # If no valid landmark pair was found, return scale and pivot that don't change anything
                except ValueError:
                    scale = 1
                    pivot = np.asarray([0, 0, 0])
                return scale, pivot

            def scale_operators(scale: float) -> Tuple:
                """
                The operators are invariant to rigid transformation, but uniform scaling changes them.
                To avoid having to recompute them, we adjust them based on the scaling factor here.
                """
                frames, mass, L, evals, evecs, gradX, gradY = operators
                mass_new = (scale ** 2) * mass
                evals_new = evals / (scale ** 2)
                evecs_new = evecs / scale
                gradX_new = gradX / scale
                gradY_new = gradY / scale
                return frames, mass_new, L, evals_new, evecs_new, gradX_new, gradY_new

            # -------------------------------------------------------------------------
            # Artifact detection
            # -------------------------------------------------------------------------
            if Path(ckpt_art).exists():
                art_model, config_art = build_model(ckpt_art, device)

                scale_art, pivot_art = get_scale_and_pivot(config_art)
                verts_art_tensor = torch.tensor(apply_rescale(verts_aligned, scale_art, pivot_art)).float().to(device)
                art_proba, _ = run_forward(art_model, verts_art_tensor, faces_p2_tensor, config_art.get("k_eig", 128),
                                           operators=scale_operators(scale_art), softmax=True)
                clean_probability = art_proba[:, 1]
                del art_model

            # -------------------------------------------------------------------------
            # Pass 2
            # -------------------------------------------------------------------------
            if Path(ckpt_pass2).exists():
                p2_model, config_pass2 = build_model(ckpt_pass2, device)
                scale_p2, pivot_p2 = get_scale_and_pivot(config_pass2)
                verts_p2_tensor = torch.tensor(apply_rescale(verts_aligned, scale_p2, pivot_p2)).float().to(device)
                verts_p2_np = verts_p2_tensor.detach().cpu().numpy()
                hm2, _ = run_forward(p2_model, verts_p2_tensor, faces_p2_tensor, config_pass2.get("k_eig", 128),
                                     operators=scale_operators(scale_p2))
                # Two options for getting final landmark position and confidence:
                if include_artifact_prob_in_landmark_softmax:
                    # Scale the softmax aggregation by the predicted probability of the vertices being not artifacts.
                    # This avoids that landmarks are placed on artifacts.
                    # todo: test this option. It could also be that landmarks are placed at a very wrong placed
                    #  if there's a big artifact where they would normally be?
                    final_pred  = softargmax(hm2 * np.expand_dims(clean_probability, axis=1),
                                             verts_p2_np, sigma=sigma_softmax)
                else:
                    # Default softmax and peak confidence, without including clean probability
                    final_pred  = softargmax(hm2, verts_p2_np, sigma=sigma_softmax)
                final_conf = peak_confidences(hm2)

                # revert scaling
                final_pred = apply_rescale(final_pred, 1.0 / scale_p2, pivot_p2)
                # revert procrustes alignment
                final_pred = Landmarks.transform_points(final_pred, np.linalg.inv(T))
                del p2_model

    # Undo initial normalization
    pred_raw = final_pred * max_rad_raw + center_raw
    _, pred_vertex_indices = mesh.resnap_landmarks(pred_raw)
    # Landmark confidence is based on the landmark prediction confidence times the probability the closest vertex
    # is NOT an artifact.
    lm_confidence = final_conf * clean_probability[pred_vertex_indices]
    pred_landmarks = Landmarks(pred_raw, names=landmark_names, confidence=lm_confidence)

    # Artifact mask is converted to binary.
    artifact_mask = IndicesAndMasks.invert_mask(clean_probability > 0.5)
    # Transfer artifact mask to raw mesh in case we downscaled the mesh for prediction
    if raw_mesh is not None:
        artifact_mask = mesh.transfer_vertex_indices_to_new_mesh(artifact_mask, raw_mesh)
    return pred_landmarks, artifact_mask


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Automated infant face landmark prediction pipeline."
    )
    parser.add_argument("--input_file_or_dir", type=str, required=True,
        help="Path to a single mesh file or a directory of mesh files.")
    parser.add_argument("--output_dir", default=None,
        help="Directory for prediction csv and txt (default: same location as input).")
    parser.add_argument("--pass1_ckpt",    default=None, type=str)
    parser.add_argument("--pass2_ckpt",    default=None, type=str)
    parser.add_argument("--art_ckpt",      default=None, type=str)
    parser.add_argument("--ref_landmarks", default=None, type=str,
        help="Pre-computed reference Procrustes target (csv landmark file).")
    parser.add_argument("--cache_operators", default=None, type=str,
        help="Write True if you want to cache used operators in input directory, or provide path where to save them.")
    parser.add_argument("--include_artifact_prob_in_landmark_softmax", default=False, action="store_true",
                        help="Experimental setting. Choose this to compute the landmark positions based via softmax "
                             "that not only uses the landmark prediction probabilities, but also the artifact "
                             "probabilities. It is currently not clear whether this improves the results.")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    scan_paths = Mesh.get_all_mesh_files_in_path(args.input_file_or_dir)
    if len(scan_paths) == 0:
        print(f"No mesh files found at: {args.input_file_or_dir}")
        return

    for i, mesh_path in tqdm(enumerate(scan_paths), total=len(scan_paths), desc="Predicting landmarks"):
        print(f"\n[{i+1}/{len(scan_paths)}] {os.path.basename(mesh_path)}")
        if args.output_dir is not None:
            output_dir = Path(args.output_dir)
            output_dir.mkdir(exist_ok=True)
        else:
            output_dir = mesh_path.parent

        if args.cache_operators is None:
            op_cache_dir = None
        else:
            if args.cache_operators.lower() == "true":
                op_cache_dir = output_dir
            else:
                op_cache_dir = Path(args.cache_operators)
                op_cache_dir.mkdir(exist_ok=True)
        landmarks, artifacts = predict_landmarks_and_artifacts(
            mesh_or_path=mesh_path,
            ckpt_pass1=args.pass1_ckpt,
            ckpt_pass2=args.pass2_ckpt,
            ckpt_art=args.art_ckpt,
            lm_ref_target=args.ref_landmarks,
            device=device,
            op_cache_dir=op_cache_dir,
            include_artifact_prob_in_landmark_softmax=args.include_artifact_prob_in_landmark_softmax,
        )

        landmarks.export(Path(output_dir).joinpath(f"{mesh_path.stem}_model.csv"))
        IndicesAndMasks.export(artifacts, Path(output_dir).joinpath(f"{mesh_path.stem}_exclude.txt"))


if __name__ == "__main__":
    main()
