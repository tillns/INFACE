# ------------------------------------------------------------------------------
# Copyright (c) Microsoft
# Licensed under the MIT License.
# Created by Tianheng Cheng(tianhengcheng@gmail.com)
# ------------------------------------------------------------------------------

import numpy as np
from typing import Union
from pathlib import Path

import torch
import torch.backends.cudnn as cudnn
import yaml
from PIL import Image

import src.processing.infanface_lm_detector.models as models
from models import get_model_path
from src.processing.infanface_lm_detector.config import config
from src.processing.infanface_lm_detector.core.evaluation import decode_preds, compute_nme


class InfFaceLMDetector:
    def __init__(self, ckpt: Union[str, Path] = "hrnet-r90jt"):
        model_ckpt_path = get_model_path(Path(ckpt).with_suffix(".pth"))
        yaml_cfg_path = model_ckpt_path.with_suffix(".yaml")
        with open(yaml_cfg_path, 'r') as file:
            yaml_cfg = yaml.safe_load(file)

        cudnn.benchmark = config.CUDNN.BENCHMARK
        cudnn.determinstic = config.CUDNN.DETERMINISTIC
        cudnn.enabled = config.CUDNN.ENABLED

        config.defrost()
        config.MODEL.INIT_WEIGHTS = False
        for cfg_name, model_cfg_adjust in yaml_cfg["MODEL"].items():
            if isinstance(model_cfg_adjust, dict):
                for key, value in model_cfg_adjust.items():
                    if isinstance(value, dict):
                        for k, v in value.items():
                            setattr(getattr(getattr(config.MODEL, cfg_name), key), k, v)
                    else:
                        setattr(getattr(config.MODEL, cfg_name), key, value)
            else:
                setattr(config.MODEL, cfg_name, model_cfg_adjust)
        config.freeze()
        self.model = models.get_face_alignment_net(config)

        # load model
        state_dict = torch.load(str(model_ckpt_path))
        if 'state_dict' in state_dict.keys():
            state_dict = state_dict['state_dict']
        self.model.load_state_dict(state_dict)

    def convert_image(self, image_or_path: Union[str, Path, np.ndarray]) -> torch.Tensor:

        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32)

        if isinstance(image_or_path, np.ndarray):
            img = image_or_path
        else:
            img = Image.open(image_or_path).convert("RGB")
            img = np.asarray(img, dtype=np.float32) / 255.0

        img = img[:, :, :min(img.shape[2], 3)]
        img = (img - mean) / std
        img = img.transpose([2, 0, 1])

        return torch.from_numpy(img)

    def forward(self, images):
        if not isinstance(images, list) and images.ndim == 3:
            images = [images]
            single_image = True
        else:
            single_image = False
        predictions = []
        for image in images:
            with torch.no_grad():
                image_torch = self.convert_image(image)
                output = self.model(image_torch.unsqueeze(0))
            score_map = output.data.cpu()
            preds = decode_preds(score_map, [64, 64])
            predictions.append(preds[0].detach().cpu().numpy())
        if single_image:
            return predictions[0]
        return predictions
