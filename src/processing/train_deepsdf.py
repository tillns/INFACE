"""
Training file for implicit model. Training can be started with
python -m src.processing.train_deepsdf --config config_name.yaml. Add --help to see further arguments.
The file mainly consists of the Trainer class, which is intialized with the folders to write results to,
and then called to start/continue training. Its settings are based on defaults and what's defined in the passed
config file.

torch is required for using this file.

Author: Till Schnabel (contact till.schnabel@inf.ethz.ch); parts of the code were copied, cf. further below.

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

This file was initially copied from https://github.com/maurock/DeepSDF, but it was heavily adjusted since.
Still, it may be subject to the following license:

MIT License

Copyright (c) 2023 Mauro

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
import random
import torch
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import torch.optim as optim
from torch.utils.data import random_split, Subset
from torch.utils.data import DataLoader
import os
import numpy as np
import time
from models import models_folder
from src import utils_deepsdf
from torch.utils.tensorboard import SummaryWriter
import yaml
from deepsdf_configs import deepsdf_configs_path
from pathlib import Path
import shutil
from src.objects.implicit_model import ImplicitShapeModel

# The device is generally defined based on availability.
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(f'Device: {device}')

class Trainer:
    def __init__(self, train_cfg_path: Path, test_run: bool = False, ckpt_path: str = None, name: str = None,
                 overwrite: bool = False):
        # Define folder to save results to.
        # This is hard-coded into a subfolder of the models folder called DeepSDF_runs.
        self.runs_dir = models_folder.joinpath("DeepSDF_runs")
        self.runs_dir.mkdir(exist_ok=True)
        # The actual folder name is based on the name of the config file or a user-speficied name if provided.
        run_dir_name = train_cfg_path.stem if name is None else name
        try:
            relative_config_run_dir = train_cfg_path.relative_to(deepsdf_configs_path)
            relative_config_run_dir = relative_config_run_dir.with_name(run_dir_name)
        except ValueError:
            relative_config_run_dir = Path(run_dir_name)

        # If it's a test run, we add "__testrun" to the folder name.
        if test_run:
            relative_config_run_dir = relative_config_run_dir.with_name(relative_config_run_dir.name + "__testrun")
        self.run_dir = self.runs_dir.joinpath(relative_config_run_dir)
        self.ckpt = None

        # If a checkpoint was provided, set the run dir to that checkpoint and load the checkpoint.
        if ckpt_path is not None:
            if test_run:
                raise AssertionError("test run not supported in combination with loading ckpt.")
            ckpt_path = Path(ckpt_path).with_suffix(".pt")
            if not ckpt_path.is_absolute():
                ckpt_path = self.run_dir.joinpath(ckpt_path)
            if not ckpt_path.exists():
                raise FileNotFoundError(f"Could not locate checkpoint: {ckpt_path}")
            self.ckpt = torch.load(ckpt_path, map_location=device)
            self.run_dir = ckpt_path.parent
        # If no checkpoint was provided, we only overwrite an existing folder if it's a test run or if
        # the user chose to overwrite it.
        elif self.run_dir.exists():
            if test_run or overwrite:
                shutil.rmtree(str(self.run_dir))
            else:
                raise FileExistsError(f"Run folder {self.run_dir} already exists. You may choose to overwrite it.")

        # We also save the config file into the folder.
        self.log_path = self.run_dir.joinpath("settings.yaml")
        if not self.run_dir.exists():
            self.run_dir.mkdir(parents=True)
            shutil.copy(train_cfg_path, self.log_path)

        # Only here do we actually load the config
        with open(self.log_path, 'rb') as f:
            self.train_cfg = yaml.load(f, Loader=yaml.FullLoader)
        self.train_cfg["test_run"] = test_run

    # Start/continue training
    def __call__(self):
        # Logging
        self.writer = SummaryWriter(log_dir=str(self.run_dir))

        # Load the dataset before we define the model
        train_loader, val_loader, self.dataset = self.get_loaders()

        # Define model
        self.model = ImplicitShapeModel(**self.train_cfg)

        # Latent size defines the size of the trainable latent parameters.
        # The number of latents is defined by the dataset size.
        self.latent_size = self.model.lat_dim

        # define gradient clipping for optimizers.
        self.model_grad_clip = self.train_cfg.get("model_grad_clip", 0)
        self.latent_grad_clip = self.train_cfg.get("latent_grad_clip", 0)

        # Model parameters to pass to optimizers.
        # We exclude the correspondence_latents here, because they have their own
        # optimizer.
        self.model_params_to_optimize = [param for param_name, param in self.model.named_parameters()
                                         if param_name != "correspondence_latents"]
        # Generate a unique random latent code for each shape (initialization based on
        # https://github.com/SimonGiebenhain/NPHM#learning-neural-parametric-head-models-nphm)
        self.latent_codes = torch.nn.Embedding(
            self.dataset.__len__()*len(self.dataset.orientations), self.latent_size,
            max_norm=1.0, sparse=True, device=device).float()
        torch.nn.init.normal_(
            self.latent_codes.weight.data,
            0.0,
            0.1 / np.sqrt(self.latent_size),
        )

        # Define the optimizers.
        self.optimizer_latent = optim.SparseAdam(list(self.latent_codes.parameters()), lr=self.train_cfg['lr_latent'])
        if self.model.predict_correspondence_landmarks > 0 and self.model.correspondence_lat_dim > 0:
            self.optimizer_lm_latent = optim.Adam(list(self.model.correspondence_latents.parameters()), lr=self.train_cfg['lr_latent'], weight_decay=0)
        else:
            self.optimizer_lm_latent = None
        weight_decay = self.train_cfg.get("weight_decay", 0)
        if weight_decay > 0:
            self.optimizer_model = optim.AdamW(self.model_params_to_optimize, lr=self.train_cfg['lr_model'],
                                               weight_decay=weight_decay)
        else:
            self.optimizer_model = optim.Adam(self.model_params_to_optimize, lr=self.train_cfg['lr_model'],
                                              weight_decay=0)

        # Define learning rate schedulers. The user is required to define this.
        # They can use fixed scheduling to lower the learning rate very n epochs,
        # or they can specify exactly which epochs to lower the learning rate at.
        # They can also specify to lower the learning rate when the validation loss hits a plateau.
        # We observed that this can be a bit unreliable, especially for small dataset, since
        # the validation is prone to some random fluctuations due to the random orientation and
        # point subsampling.
        if self.train_cfg['lr_scheduler']:
            if isinstance(self.train_cfg['lr_scheduler'], str) and "fix" in self.train_cfg['lr_scheduler'].lower():
                if "lr_decay_step_size" in self.train_cfg:
                    self.scheduler_model =  torch.optim.lr_scheduler.StepLR(
                        self.optimizer_model, step_size=self.train_cfg["lr_decay_step_size"],
                        gamma=self.train_cfg['lr_decay_multiplier'])
                    if self.optimizer_latent is not None:
                        self.scheduler_latent =  torch.optim.lr_scheduler.StepLR(
                            self.optimizer_latent,
                            gamma=self.train_cfg.get("lr_latent_decay_multiplier", self.train_cfg["lr_decay_multiplier"]),
                            step_size=self.train_cfg.get("lr_latent_decay_step_size", self.train_cfg["lr_decay_step_size"]))
                    if self.optimizer_lm_latent is not None:
                        self.scheduler_lm_latent =  torch.optim.lr_scheduler.StepLR(
                            self.optimizer_lm_latent,
                            gamma=self.train_cfg.get("lr_latent_decay_multiplier", self.train_cfg["lr_decay_multiplier"]),
                            step_size=self.train_cfg.get("lr_latent_decay_step_size", self.train_cfg["lr_decay_step_size"]))
                elif "lr_decay_steps" in self.train_cfg:
                    self.scheduler_model =  torch.optim.lr_scheduler.MultiStepLR(
                        self.optimizer_model, milestones=self.train_cfg["lr_decay_steps"],
                        gamma=self.train_cfg['lr_decay_multiplier'])
                    if self.optimizer_latent is not None:
                        self.scheduler_latent =  torch.optim.lr_scheduler.MultiStepLR(
                            self.optimizer_latent,
                            gamma=self.train_cfg.get("lr_latent_decay_multiplier", self.train_cfg["lr_decay_multiplier"]),
                            milestones=self.train_cfg.get("lr_latent_decay_steps", self.train_cfg["lr_decay_steps"]))
                    if self.optimizer_lm_latent is not None:
                        self.scheduler_lm_latent =  torch.optim.lr_scheduler.MultiStepLR(
                            self.optimizer_lm_latent,
                            gamma=self.train_cfg.get("lr_latent_decay_multiplier", self.train_cfg["lr_decay_multiplier"]),
                            milestones=self.train_cfg.get("lr_latent_decay_steps", self.train_cfg["lr_decay_steps"]))
            else:
                self.scheduler_model =  torch.optim.lr_scheduler.ReduceLROnPlateau(
                    self.optimizer_model, mode='min', factor=self.train_cfg['lr_decay_multiplier'],
                    patience=self.train_cfg['lr_decay_patience'],
                    threshold=0.0001, threshold_mode='rel')
                if self.optimizer_latent is not None:
                    self.scheduler_latent =  torch.optim.lr_scheduler.ReduceLROnPlateau(
                        self.optimizer_latent, mode='min',
                        factor=self.train_cfg.get("lr_latent_decay_multiplier", self.train_cfg["lr_decay_multiplier"]),
                        patience=self.train_cfg.get("lr_latent_decay_patience", self.train_cfg["lr_decay_patience"]),
                        threshold=0.0001, threshold_mode='rel')
                if self.optimizer_lm_latent is not None:
                    self.scheduler_lm_latent =  torch.optim.lr_scheduler.ReduceLROnPlateau(
                        self.optimizer_lm_latent, mode='min',
                        factor=self.train_cfg.get("lr_latent_decay_multiplier", self.train_cfg["lr_decay_multiplier"]),
                        patience=self.train_cfg.get("lr_latent_decay_patience", self.train_cfg["lr_decay_patience"]),
                        threshold=0.0001, threshold_mode='rel')
        
        # Load pretrained weights and optimisers to continue training
        if self.ckpt is not None:
            # load pretrained weights
            self.model.load_state_dict(self.ckpt["model_state"])

            # load pretrained optimizers and respective schedulers
            self.optimizer_model.load_state_dict(self.ckpt["optimizer_model_state"])
            self.scheduler_model.load_state_dict(self.ckpt["model_scheduler_state"])
            if self.optimizer_latent is not None:
                self.optimizer_latent.load_state_dict(self.ckpt["optimizer_latent_state"])
                self.scheduler_latent.load_state_dict(self.ckpt["latent_scheduler_state"])
            if self.optimizer_lm_latent is not None:
                self.optimizer_lm_latent.load_state_dict(self.ckpt["optimizer_lm_latent_state"])
                self.scheduler_lm_latent.load_state_dict(self.ckpt["lm_latent_scheduler_state"])

            # retrieve latent codes
            self.latent_codes.load_state_dict(self.ckpt["latent_codes"])
            start_epoch = self.ckpt["epoch"] + 1
            best_checkpoints = self.ckpt["best_checkpoints"]
            past_val_losses = self.ckpt["past_val_losses"]
        else:
            start_epoch = 0
            best_checkpoints = {}
            past_val_losses = []
        num_best_checkpoints_to_keep = self.train_cfg.get("num_best_checkpoints_to_keep", 10)
        start = time.time()

        # Start training loop (start from start_epoch, which can be > 0 if we continue training from a checkpoint).
        for epoch in range(start_epoch, self.train_cfg['epochs']):
            print(f'============================ Epoch {epoch} ============================')
            self.epoch = epoch

            # One training epoch.
            self.model_step(train_loader, mode="train")

            # One validation epoc. We keep track of validation loss to decide which checkpoints to keep,
            # and also for the learning rate schedulers.
            val_loss = self.model_step(val_loader, mode="validation")
            past_val_losses.append(val_loss)
            if len(past_val_losses) > self.train_cfg.get("validation_avg_window", self.train_cfg.get("lr_decay_patience", 20)):
                past_val_losses.pop(0)
            avg_val_loss = np.mean(past_val_losses)

            # Define what to save inside checkpoint.
            def get_checkpoint():
                return_dict = {
                    "epoch": epoch,
                    "val_loss": avg_val_loss,
                    "model_state": self.model.state_dict(),
                    "optimizer_model_state": self.optimizer_model.state_dict(),
                    "optimizer_latent_state": self.optimizer_latent.state_dict() if self.optimizer_latent is not None else None,
                    "optimizer_lm_latent_state": self.optimizer_lm_latent.state_dict() if self.optimizer_lm_latent is not None else None,
                    "model_scheduler_state": self.scheduler_model.state_dict(),
                    "latent_scheduler_state": self.scheduler_latent.state_dict() if self.optimizer_latent is not None else None,
                    "lm_latent_scheduler_state": self.scheduler_lm_latent.state_dict() if self.optimizer_lm_latent is not None else None,
                    "latent_codes": self.latent_codes.state_dict() if isinstance(self.latent_codes, torch.nn.Embedding) else self.latent_codes,
                    "latent_code_names": [s["name"] for orientation in self.dataset.orientations for s in self.dataset.subjects[orientation]],
                    "train_cfg": self.train_cfg,
                    "best_checkpoints": best_checkpoints,
                    "past_val_losses": past_val_losses,
                }
                if hasattr(self.dataset, "avg_landmarks"):
                    return_dict["avg_landmarks"] = self.dataset.avg_landmarks
                return return_dict

            current_ckpt = get_checkpoint()

            # Keep num_best_checkpoints_to_keep best checkpoints saved (delete old ones that are beaten)
            if len(best_checkpoints) < num_best_checkpoints_to_keep or avg_val_loss < max(list(best_checkpoints.values())):
                checkpoint_path = self.run_dir.joinpath(f"checkpoint_ep{epoch}_best_loss{f'{avg_val_loss:.3f}'.replace('.', 'p')}.pt")
                torch.save(current_ckpt, checkpoint_path)
                best_checkpoints[checkpoint_path.stem] = avg_val_loss
                if len(best_checkpoints) > num_best_checkpoints_to_keep:
                    worst_checkpoint = max(best_checkpoints, key=best_checkpoints.get)
                    best_checkpoints.pop(worst_checkpoint)
                    self.run_dir.joinpath(worst_checkpoint).with_suffix(".pt").unlink()

            # Also keep checkpoints in regular intervals saved (don't delete old ones)
            if epoch > 0 and epoch % self.train_cfg.get("save_ckpt_interval", 500) == 0:
                checkpoint_path = self.run_dir.joinpath(f"checkpoint_ep{epoch}_int_loss{f'{avg_val_loss:.3f}'.replace('.', 'p')}.pt")
                torch.save(current_ckpt, checkpoint_path)

            # Also always save the last checkpoint (overwritten each epoch),
            # so that you can always continue training from the latest epoch.
            torch.save(current_ckpt, self.run_dir.joinpath("last.pt"))

            # Learning rate potential update given state of validation loss.
            if self.train_cfg['lr_scheduler']:
                if isinstance(self.scheduler_model, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    self.scheduler_model.step(avg_val_loss)
                else:
                    self.scheduler_model.step()
                self.writer.add_scalar('Learning rate (model)', self.scheduler_model._last_lr[0], epoch)
                if self.optimizer_latent is not None:
                    if isinstance(self.scheduler_latent, torch.optim.lr_scheduler.ReduceLROnPlateau):
                        self.scheduler_latent.step(avg_val_loss)
                    else:
                        self.scheduler_latent.step()
                    self.writer.add_scalar('Learning rate (latent)', self.scheduler_latent._last_lr[0], epoch)
                if self.optimizer_lm_latent is not None:
                    if isinstance(self.scheduler_lm_latent, torch.optim.lr_scheduler.ReduceLROnPlateau):
                        self.scheduler_lm_latent.step(avg_val_loss)
                    else:
                        self.scheduler_lm_latent.step()
                    self.writer.add_scalar('Learning rate (lm latent)', self.scheduler_lm_latent._last_lr[0], epoch)

        # End training (we usually never went here; we just set the number of total epochs very large
        # and then chose the best intermediate checkpoint (early stopping)).
        end = time.time()
        print(f'Time elapsed: {end - start} s')

    # Get the dataset loaders.
    def get_loaders(self):
        # First define the dataset class. This already loads all the mesh data.
        data = utils_deepsdf.SDFDataset(self.train_cfg, device=device)

        # Train/val split. We don't have any test split here. Feel free to add it.
        train_percentage = self.train_cfg.get("train_split", 0.85)

        # Get the subject names.
        names = [data.subjects['normal'][idx]['name'] for idx in range(len(data))]
        # Since our dataset contained substantially more healthy than cleft faces,
        # we make sure that also the small subset of cleft faces is evenly split
        # between training and validation.
        # We detect those subjects by checking which ones have "cleft" in their name.
        # If those are not available, we simply split the whole dataset as usual.
        cleft_indices = [idx for idx, name in enumerate(names) if "cleft" in name.lower()]
        remaining_indices = [idx for idx, name in enumerate(names) if "cleft" not in name.lower()]
        if len(cleft_indices) > 0:
            cleft_train_indices, cleft_val_indices = train_test_split(
                cleft_indices, test_size=1-train_percentage, shuffle=True, random_state=42)
            if len(remaining_indices) > 0:
                remaining_train_indices, remaining_val_indices = train_test_split(
                    remaining_indices, test_size=1-train_percentage, shuffle=True, random_state=42)
            else:
                remaining_train_indices, remaining_val_indices = [], []
            train_data = Subset(data, cleft_train_indices + remaining_train_indices)
            val_data = Subset(data, cleft_val_indices + remaining_val_indices)
        else:
            train_size = int(train_percentage * len(data))
            val_size = len(data) - train_size
            train_data, val_data = random_split(data, [train_size, val_size])

        # We save in two files the names of the subjects we used in training vs in validation.
        with open(self.run_dir.joinpath("train_patients.txt"), "w") as f:
            for patient_idx in sorted(train_data.indices):
                f.write(f"{data.subjects['normal'][patient_idx]['name']}\n")

        with open(self.run_dir.joinpath("val_patients.txt"), "w") as f:
            for patient_idx in sorted(val_data.indices):
                f.write(f"{data.subjects['normal'][patient_idx]['name']}\n")

        # Define the loaders and return them.
        train_loader = DataLoader(
                train_data,
                batch_size=self.train_cfg['batch_size'],
                shuffle=True,
                drop_last=True
            )
        if len(val_data) > 0:
            val_loader = DataLoader(
                val_data,
                batch_size=self.train_cfg['batch_size'],
                shuffle=False,
                drop_last=False
                )
        else:
            val_loader = []
        return train_loader, val_loader, data

    # Return the latent codes for the index subjects given the provided orientation (normal or flipped).
    def get_latent_codes(self, indices, orientations):
        # The flipped latent codes come after all normal latent codes,
        # so for those we simply add the dataset size to the indices.
        orientations_add = torch.asarray([self.dataset.__len__()*self.dataset.orientations.index(orientation) for orientation in orientations])
        if isinstance(self.latent_codes, torch.nn.Embedding):
            return self.latent_codes((indices + orientations_add).to(device))
        else:
            return self.latent_codes[indices+orientations_add]

    # Get the points from the batch, so combine on surface, near surface and off-surface points
    # into a single tensor. Also return an SDF tensor, which has 0 for on surface points,
    # -1 for near surface points, and 1 for off-surface points.
    def get_batch_data(self, batch):
        latent_codes_batch = self.get_latent_codes(batch["idx"], batch["orientation"])
        assert "surface" in batch
        x = torch.cat([
            batch[points_type] for points_type in ["surface", "close", "far"] if points_type in batch
        ], dim=1).clone().detach().requires_grad_()

        sdf = torch.ones((x.size(0), x.size(1)))
        on_surface_length = batch["surface"].size(1)
        sdf[:, :on_surface_length] = 0
        # this has no real meaning; it's just an indicator for us to distinguish the different regions in the loss func
        if "close" in batch:
            near_surface_length = batch["close"].size(1)
            sdf[:, on_surface_length:on_surface_length+near_surface_length] = -1
        return x, sdf, latent_codes_batch.unsqueeze(1).expand(-1, x.size(1), -1)

    # One epoch of training/validation (loop over training loader in batches and compute
    # model passes and update parameters based on loss).
    def model_step(self, data_loader, mode="train"):
        total_loss = 0.0
        loss_dict_tot = {}
        iterations = 0.0
        # Distinguish between training and validation
        if mode == "train":
            self.model.train()
        else:
            self.model.eval()
        mode_word = 'Training' if mode=='train' else 'Validation'
        # Loop over dataset in batches.
        for batch in tqdm(data_loader, desc=mode_word, total=len(data_loader)):
            iterations += 1.0

            # We only zero the model and lm correspondence optimizers' gradients during training, since
            # their parameters are not optimized during validation.
            if mode == "train":
                self.optimizer_model.zero_grad()
                if self.optimizer_lm_latent is not None:
                    self.optimizer_lm_latent.zero_grad()
            # The latents of the validation sampled are optimized during validation, so we always
            # zero the respective optimizer's gradient.
            if self.optimizer_latent is not None:
                self.optimizer_latent.zero_grad()

            # Get model input and ground truth
            x, sdf, latent_codes_batch = self.get_batch_data(batch)
            latent_codes_batch_non_rep = latent_codes_batch[:, 0]
            landmarks_gt = batch["landmarks"]
            landmark_weights = batch["landmark_weights"] if "landmark_weights" in batch else None

            # Model prediction step.
            prediction, correction, deformed_coords, landmarks_deformed, landmarks_corr_pred = self.model(
                x, latent_codes_batch_non_rep,
                landmarks=None if self.model.predict_correspondence_landmarks > 0 else landmarks_gt)
            prediction = prediction.squeeze(-1)

            # Compute loss function.
            loss_value, loss_dict = utils_deepsdf.loss(
                net_input={"key_pts": landmarks_gt, "landmark_weights": landmark_weights, "gt_sdf": sdf,
                           "gt_normals": batch["normals"], "latent": latent_codes_batch_non_rep, "coords": x,
                           "correspondence_latent": self.model.correspondence_latents.weight if hasattr(self.model, "correspondence_latents") else None,
                           "all_key_pts": torch.tensor(self.dataset.avg_landmarks, device=device)},
                net_middle={"correction": correction, "deformed_coords": deformed_coords,
                            "landmarks_deformed": landmarks_deformed,
                            "landmarks_correspondence_pred": landmarks_corr_pred},
                pred_sdf=prediction,
                weights=self.train_cfg["loss_weights"],
                skip_weighting_deformed_landmarks=self.model.predict_correspondence_landmarks > 0
            )

            # Backpropagate loss.
            loss_value.backward()
            total_loss += loss_value.data.cpu().numpy()

            # The latent gradient clipping NPHM proposed does not work with more recent
            # pytorch versions anymore. We therefore implemented our own with ChatGPT,
            # which loops over the individual gradients and clips them.
            if self.optimizer_latent is not None:
                if self.latent_grad_clip > 0:
                    for p in self.latent_codes.parameters():
                        if p.grad is None:
                            continue

                        if p.grad.is_sparse:
                            grad = p.grad.coalesce()
                            values = grad.values()

                            norm = values.norm(2)
                            if norm > self.latent_grad_clip:
                                scale = self.latent_grad_clip / (norm + 1e-6)
                                values.mul_(scale)
                        else:
                            torch.nn.utils.clip_grad_norm_([p], self.latent_grad_clip)
                # Latent update.
                self.optimizer_latent.step()

            # Only if we train do we also update the model and correspondence latents.
            if mode == "train":
                if self.optimizer_lm_latent is not None:
                    if self.latent_grad_clip > 0:
                        torch.nn.utils.clip_grad_norm_(self.model.correspondence_latents.parameters(),
                                                       max_norm=self.latent_grad_clip)
                    self.optimizer_lm_latent.step()
                if self.model_grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(self.model_params_to_optimize, max_norm=self.model_grad_clip)
                self.optimizer_model.step()

            # Keep track of the losses.
            for loss_kind, loss_spec in loss_dict.items():
                if loss_kind not in loss_dict_tot:
                    loss_dict_tot[loss_kind] = 0
                loss_dict_tot[loss_kind] += loss_spec.data.cpu().numpy()

        # print current loss average
        avg_loss = total_loss/iterations
        print(f"{mode_word}: loss {avg_loss}")
        # Also add total and individual losses to tensorboard.
        self.writer.add_scalar(f"{mode_word} loss", avg_loss, self.epoch)
        for loss_kind, loss_spec in loss_dict_tot.items():
            self.writer.add_scalar(f"{mode_word}_{loss_kind}", loss_spec/iterations, self.epoch)

        return avg_loss


# Set all kinds of random seeds for reproducibility.
def set_seed(seed: int = 0) -> None:
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # When running on the CuDNN backend, two further options must be set
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    # for making some tensor operations deterministic
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":16:8"
    torch.use_deterministic_algorithms(True, warn_only=True)
    # Set a fixed value for the hash seed
    os.environ["PYTHONHASHSEED"] = str(seed)
    print(f"Random seed set as {seed}")


def main():
    parser = argparse.ArgumentParser('Arguments')
    parser.add_argument('--config', type=str, help='specify the config file', required=True)
    parser.add_argument('--test_run', default=False, action="store_true", help='Smaller preparation time')
    parser.add_argument('--overwrite', default=False, action="store_true",
                        help='Overwrite folder if it exists already.')
    parser.add_argument('--name', type=str, default=None,
                             help='name of the experiment. It decides where to store samples and models. '
                                  'No need to specify if a config is given, then the config name is simply chosen.')
    parser.add_argument('--ckpt', default=None, type=str,
                        help="Optionally continue training from specified ckpt. "
                             "You may provide either the epoch, the name/path of the ckpt, "
                             "or a keyword like 'last', or 'true' to continue from the last ckpt.")
    parser.add_argument('--seed', default=42, type=int,
                        help="Specify the random seed. Default 42.")
    args = parser.parse_args()

    set_seed(seed=args.seed)

    train_cfg_path = Path(args.config).with_suffix('.yaml')
    if not train_cfg_path.is_absolute():
        train_cfg_path = list(deepsdf_configs_path.parent.glob(f"**/{train_cfg_path.name}"))[0]

    trainer = Trainer(train_cfg_path, test_run=args.test_run, ckpt_path=args.ckpt, name=args.name,
                      overwrite=args.overwrite)
    trainer()


if __name__=='__main__':
    main()
