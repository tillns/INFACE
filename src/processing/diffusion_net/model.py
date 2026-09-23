"""
DiffusionNet model construction and inference helpers.

Author: Till Schnabel, Vera Schubert (contact till.schnabel@inf.ethz.ch)

MIT License

Copyright (c) 2026 ETH Zurich, Till Schnabel.

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

The methods were developed mostly by Vera Schubert as part of here Bachelor's thesis supervised by Till Schnabel.
"""
from pathlib import Path
from typing import Union, Tuple

import numpy as np
import torch
from src.processing import diffusion_net

def get_input_dim(input_features: str) -> int:
    """Returns the number of input feature channels for a given feature type."""
    if input_features.startswith("xyz"):
        dim = 3
    elif input_features.startswith("hks"):
        dim = 16
    else:
        raise ValueError(f"Unknown input_features '{input_features}'. Use 'xyz' or 'hks'.")
    if input_features.endswith("rgb"):
        dim += 3
    return dim


def build_model(checkpoint_path: Union[Path, str], device: torch.device):
    """
    Builds and loads a DiffusionNet landmark prediction model.

    config must contain config['model'] (C_width, N_block, outputs_at, k_eig)
    and config['dataset']['num_landmarks'].
    """
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    config = ckpt["config"]
    model = diffusion_net.layers.DiffusionNet(
        C_in=get_input_dim(config.get("input_features", "xyz")),
        C_out=config["C_out"],
        C_width=config["C_width"],
        N_block=config["N_block"],
        last_activation=None,
        outputs_at=config["outputs_at"],
        dropout=config.get("dropout", 0.0),
    ).to(device)

    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model, config


def run_forward(model, verts_tensor: torch.Tensor, faces_tensor: torch.Tensor,
                k_eig: int, operators: Tuple = None, softmax: bool = False,
                op_cache_dir: Union[str, Path, None] = None) -> Tuple[np.ndarray, Tuple]:
    """
    Runs a DiffusionNet forward pass and returns the output heatmap as a
    (V, num_landmarks) numpy array.
    """
    if operators is None:
        operators = diffusion_net.geometry.get_operators(
            verts_tensor, faces_tensor, k_eig=k_eig, op_cache_dir=op_cache_dir)
    _, mass, L, evals, evecs, gradX, gradY = operators
    with torch.no_grad():
        out = model(verts_tensor, mass, L=L, evals=evals, evecs=evecs, gradX=gradX, gradY=gradY)
        if softmax:
            out = torch.softmax(out, dim=1)
    return out.detach().cpu().numpy(), operators


def softargmax(heatmaps: np.ndarray, verts: np.ndarray, sigma: float = 0.02) -> np.ndarray:
    """
    Soft-argmax decoding: computes a weighted average of vertex positions using
    softmax weights derived from the heatmap values.

    Parameters
    ----------
    heatmaps : (V, L) float32 — per-vertex, per-landmark logit scores
    verts    : (V, 3) float32 — vertex positions
    sigma    : temperature (lower = sharper, more argmax-like)

    Returns  : (L, 3) float32 — predicted landmark positions
    """
    num_lm = heatmaps.shape[1]
    preds  = np.zeros((num_lm, 3), dtype=np.float64)
    for i in range(num_lm):
        logits = heatmaps[:, i].astype(np.float64)
        finite = np.isfinite(logits)
        if not finite.any():
            continue
        logits -= logits[finite].max()
        w = np.where(finite, np.exp(logits / sigma), 0.0)
        w_sum = w.sum()
        if w_sum > 0:
            preds[i] = (w[:, None] * verts).sum(axis=0) / w_sum
    return preds.astype(np.float32)


def peak_confidences(heatmaps: np.ndarray) -> np.ndarray:
    """
    Returns the peak heatmap value per landmark — used as a proxy for prediction
    confidence. Shape: (L,) float32.
    """
    finite_hm = np.where(np.isfinite(heatmaps), heatmaps, -1e30)
    pred_idx  = np.argmax(finite_hm, axis=0)
    return heatmaps[pred_idx, np.arange(heatmaps.shape[1])]
