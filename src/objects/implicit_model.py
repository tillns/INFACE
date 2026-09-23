"""
Class for implicit shape model
Basic functionality includes:
1) Infer latent from input mesh/vertices (optionally with mask, indicating which vertices to exclude)
   via iterative optimization.
2) Decode a latent to SDF values at specific positions, converted to a mesh via marching cubes
3) Sample one or multiple meshes.
4)

This class requires torch.

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

The torch layers and model formulation were copied and adjusted from
https://github.com/MingwuZheng/ImFace/tree/58d3b8eba24dd27eb5722e5253b71c20122f321f
and may be subject to the following license:

MIT License

Copyright (c) 2022 Mingwu Zheng, Haiyu Zhang

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
import math
import re
from argparse import ArgumentParser
from collections import OrderedDict
from pathlib import Path

import numpy as np
from torch import optim
from torch.autograd import grad
from tqdm import tqdm

from models import get_model_path
from src import utils_deepsdf, utils
from src.objects.indices_and_masks import IndicesAndMasks
from src.objects.mesh import Mesh, Landmarks
from typing import Union, Tuple, List, Dict
try:
    import torch
except ImportError as e:
    raise ImportError(f"This class requires PyTorch, please install. Error: {e}")
ArrayLike = Union[np.ndarray, torch.Tensor]
import torch.nn as nn


def get_subdict(dictionary, key=None):
    if dictionary is None:
        return None
    if (key is None) or (key == ''):
        return dictionary
    key_re = re.compile(r'^{0}\.(.+)'.format(re.escape(key)))
    return OrderedDict((key_re.sub(r'\1', k), value) for (k, value)
                       in dictionary.items() if key_re.match(k) is not None)


class MetaModule(nn.Module):
    def meta_named_parameters(self, prefix="", recurse=True):
        """
        Minimal replacement for torchmeta.modules.MetaModule.meta_named_parameters().
        """
        return self.named_parameters(prefix=prefix, recurse=recurse)

    def meta_parameters(self, recurse=True):
        for _, param in self.meta_named_parameters(recurse=recurse):
            yield param


def get_child_parameters(params, key):
    """
    Extract parameters for a child module.

    Example:
        params = {
            "0.weight": ...,
            "0.bias": ...,
            "1.weight": ...
        }

        get_child_parameters(params, "0") -> {
            "weight": ...,
            "bias": ...
        }
    """
    if params is None:
        return None

    prefix = key + "."
    return {
        name[len(prefix):]: value
        for name, value in params.items()
        if name.startswith(prefix)
    }


class MetaSequential(nn.Sequential, MetaModule):
    """
    Minimal replacement for torchmeta.modules.MetaSequential.

    Passes the matching params subdict to children that accept a `params`
    argument.
    """

    def forward(self, input, params=None):
        for name, module in self._modules.items():
            subparams = get_child_parameters(params, name)

            if isinstance(module, MetaModule):
                input = module(input, params=subparams)
            else:
                input = module(input)

        return input


class BatchLinear(nn.Linear, MetaModule):
    '''A linear meta-layer that can deal with batched weight matrices and biases, as for instance output by a
    hypernetwork.
    '''
    __doc__ = nn.Linear.__doc__

    def forward(self, input, params=None):

        if params is None:
            return nn.Linear.forward(self, input)

        else:

            bias = params.get('bias', None)
            weight = params['weight']
            output = input.matmul(weight.permute(*[i for i in range(len(weight.shape) - 2)], -1, -2))
            output += bias.unsqueeze(-2)
            return output


class Sine(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, input):
        return torch.sin(30 * input)


class FCBlock(MetaModule):
    '''A fully connected neural network that also allows swapping out the weights when used with a hypernetwork.
    Can be used just as a normal neural network though, as well.
    '''

    def __init__(self, in_features, out_features, num_hidden_layers, hidden_features,
                 outermost_linear=False, nonlinearity='relu', weight_init=None,
                 use_nphm_model_setting: bool = False):
        super().__init__()

        if use_nphm_model_setting:
            nonlinearity = "softplus"

        self.first_layer_init = None

        # Dictionary that maps nonlinearity name to the respective function, initialization, and, if applicable,
        # special first-layer initialization scheme
        nls_and_inits = {'sine': (Sine(), sine_init, first_layer_sine_init, last_layer_sine_init),
                         'relu': (nn.ReLU(inplace=True), init_weights_normal, None, None),
                         'sigmoid': (nn.Sigmoid(), init_weights_xavier, None, None),
                         'tanh': (nn.Tanh(), init_weights_xavier, None, None),
                         'selu': (nn.SELU(inplace=True), init_weights_selu, None, None),
                         'softplus': (nn.Softplus(beta=100), init_weights_normal, None, None),
                         'elu': (nn.ELU(inplace=True), init_weights_elu, None, None)}

        nl, nl_weight_init, first_layer_init, last_layer_init = nls_and_inits[nonlinearity]

        if weight_init is not None:  # Overwrite weight init if passed
            self.weight_init = weight_init
        else:
            self.weight_init = nl_weight_init

        self.net = []
        self.net.append(MetaSequential(
            BatchLinear(in_features, hidden_features), nl
        ))

        for i in range(num_hidden_layers):
            self.net.append(MetaSequential(
                BatchLinear(hidden_features, hidden_features), nl
            ))

        if outermost_linear:
            self.net.append(MetaSequential(BatchLinear(hidden_features, out_features)))
        else:
            self.net.append(MetaSequential(
                BatchLinear(hidden_features, out_features), nl
            ))

        self.net = MetaSequential(*self.net)
        if use_nphm_model_setting:
            for num_layer, layer in enumerate(self.net):
                torch.nn.init.kaiming_uniform_(layer[0].weight, a=math.sqrt(5))
                if hasattr(layer[0], "bias") and layer[0].bias is not None:
                    fan_in, _ = torch.nn.init._calculate_fan_in_and_fan_out(layer[0].weight)
                    bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
                    torch.nn.init.uniform_(layer[0].bias, -bound, bound)
        else:
            if self.weight_init is not None:
                self.net.apply(self.weight_init)

            if first_layer_init is not None:  # Apply special initialization to first layer, if applicable.
                self.net[0].apply(first_layer_init)

            if last_layer_init is not None:
                self.net[-1].apply(last_layer_init)

    def forward(self, coords, params=None, **kwargs):
        if params is not None:
            params = get_subdict(params, 'net')

        output = self.net(coords, params=params)
        return output


class SingleBVPNet(MetaModule):
    '''A canonical representation network for a BVP.'''

    def __init__(self, out_features=1, in_features=2, hidden_features=256, num_hidden_layers=3):
        super().__init__()
        self.net = FCBlock(in_features=in_features, out_features=out_features, num_hidden_layers=num_hidden_layers,
                           hidden_features=hidden_features, outermost_linear=True,  use_nphm_model_setting=True)
        # print(self)

    def forward(self, model_input, params=None):
        # Enables us to compute gradients w.r.t. coordinates
        coords_org = model_input['coords'].requires_grad_(True)
        # coords = coords_org
        # various input processing methods for different applications
        output = self.net(coords_org, get_subdict(params, 'net'))
        return {'model_in': coords_org, 'model_out': output}


class HyperNetwork(nn.Module):
    def __init__(self, hyper_in_features, hyper_hidden_layers, hyper_hidden_features, hypo_module):
        """
        Args:
            hyper_in_features: In features of hypernetwork
            hyper_hidden_layers: Number of hidden layers in hypernetwork
            hyper_hidden_features: Number of hidden units in hypernetwork
            hypo_module: MetaModule. The module whose parameters are predicted.
        """
        super().__init__()

        hypo_parameters = hypo_module.meta_named_parameters()

        self.names = []
        self.nets = nn.ModuleList()
        self.param_shapes = []
        for name, param in hypo_parameters:
            self.names.append(name)
            self.param_shapes.append(param.size())

            hn = FCBlock(in_features=hyper_in_features,
                         out_features=int(torch.prod(torch.tensor(param.size()))),
                         num_hidden_layers=hyper_hidden_layers, hidden_features=hyper_hidden_features,
                         outermost_linear=True, nonlinearity='relu')
            self.nets.append(hn)

            if 'weight' in name:
                self.nets[-1].net[-1].apply(lambda m: hyper_weight_init(m, param.size()[-1]))
            elif 'bias' in name:
                self.nets[-1].net[-1].apply(lambda m: hyper_bias_init(m))

    def forward(self, z):
        """
        Args:
            z: Embedding. Input to hypernetwork. Could be output of "Autodecoder"
        Returns:
            params: OrderedDict. Can be directly passed as the "params" parameter of a MetaModule.
        """
        params = OrderedDict()
        for name, net, param_shape in zip(self.names, self.nets, self.param_shapes):
            batch_param_shape = (-1,) + param_shape
            params[name] = net(z).reshape(batch_param_shape)
        # print(params)
        return params


################################################
# Initialization schemes
################################################

def hyper_weight_init(m, in_features_main_net):
    if hasattr(m, 'weight'):
        nn.init.kaiming_normal_(m.weight, a=0.0, nonlinearity='relu', mode='fan_in')
        m.weight.data = m.weight.data / 1.e2

    if hasattr(m, 'bias'):
        with torch.no_grad():
            m.bias.uniform_(-1 / in_features_main_net, 1 / in_features_main_net)


def hyper_bias_init(m):
    if hasattr(m, 'weight'):
        nn.init.kaiming_normal_(m.weight, a=0.0, nonlinearity='relu', mode='fan_in')
        m.weight.data = m.weight.data / 1.e2

    if hasattr(m, 'bias'):
        fan_in, _ = nn.init._calculate_fan_in_and_fan_out(m.weight)
        with torch.no_grad():
            m.bias.uniform_(-1 / fan_in, 1 / fan_in)


################################################
# Initialization methods
def _no_grad_trunc_normal_(tensor, mean, std, a, b):
    # For PINNet, Raissi et al. 2019
    # Method based on https://people.sc.fsu.edu/~jburkardt/presentations/truncated_normal.pdf
    # grab from upstream pytorch branch and paste here for now
    def norm_cdf(x):
        # Computes standard normal cumulative distribution function
        return (1. + math.erf(x / math.sqrt(2.))) / 2.

    with torch.no_grad():
        # Values are generated by using a truncated uniform distribution and
        # then using the inverse CDF for the normal distribution.
        # Get upper and lower cdf values
        l = norm_cdf((a - mean) / std)
        u = norm_cdf((b - mean) / std)

        # Uniformly fill tensor with values from [l, u], then translate to
        # [2l-1, 2u-1].
        tensor.uniform_(2 * l - 1, 2 * u - 1)

        # Use inverse cdf transform for normal distribution to get truncated
        # standard normal
        tensor.erfinv_()

        # Transform to proper mean, std
        tensor.mul_(std * math.sqrt(2.))
        tensor.add_(mean)

        # Clamp to ensure it's in the proper range
        tensor.clamp_(min=a, max=b)
        return tensor



def init_weights_normal(m):
    if type(m) == BatchLinear or type(m) == nn.Linear:
        if hasattr(m, 'weight'):
            nn.init.kaiming_normal_(m.weight, a=0.0, nonlinearity='relu', mode='fan_in')


def init_weights_selu(m):
    if type(m) == BatchLinear or type(m) == nn.Linear:
        if hasattr(m, 'weight'):
            num_input = m.weight.size(-1)
            nn.init.normal_(m.weight, std=1 / math.sqrt(num_input))


def init_weights_elu(m):
    if type(m) == BatchLinear or type(m) == nn.Linear:
        if hasattr(m, 'weight'):
            num_input = m.weight.size(-1)
            nn.init.normal_(m.weight, std=math.sqrt(1.5505188080679277) / math.sqrt(num_input))


def init_weights_xavier(m):
    if type(m) == BatchLinear or type(m) == nn.Linear:
        if hasattr(m, 'weight'):
            nn.init.xavier_normal_(m.weight)


def sine_init(m):
    with torch.no_grad():
        if hasattr(m, 'weight'):
            num_input = m.weight.size(-1)
            # See supplement Sec. 1.5 for discussion of factor 30
            m.weight.uniform_(-np.sqrt(6 / num_input) / 30, np.sqrt(6 / num_input) / 30)


def first_layer_sine_init(m):
    with torch.no_grad():
        if hasattr(m, 'weight'):
            num_input = m.weight.size(-1)
            # See paper sec. 3.2, final paragraph, and supplement Sec. 1.5 for discussion of factor 30
            m.weight.uniform_(-1 / num_input, 1 / num_input)


def last_layer_sine_init(m):
    with torch.no_grad():
        if hasattr(m, 'weight'):
            nn.init.zeros_(m.weight)
            nn.init.zeros_(m.bias)

class MiniNet(nn.Module):
    def __init__(self, code_dim, in_features, out_features, hidden_features=32, num_hidden_layers=3,
                 hyper_hidden_layers=1, hyper_hidden_features=32):
        super().__init__()
        self.mini_net = SingleBVPNet(
            hidden_features=hidden_features,
            num_hidden_layers=num_hidden_layers,
            in_features=in_features,
            out_features=out_features
        )
        if code_dim is not None:
            self.mini_hypernet = HyperNetwork(
                hyper_in_features=code_dim,
                hyper_hidden_layers=hyper_hidden_layers,
                hyper_hidden_features=hyper_hidden_features,
                hypo_module=self.mini_net
            )

    def forward(self, x_relative, code=None):
        """
        :param x: (B,N,3*5)
        :param code: (B,N)
        :return: f(x-p)
        """
        if code is None:
            return self.mini_net({'coords': x_relative})['model_out']
        else:
            return self.mini_net({'coords': x_relative}, params=self.mini_hypernet(code))['model_out']

class FusionNet(nn.Module):
    def __init__(self, fusion_number, condition_dim, weight_feature_dim=128):
        super().__init__()
        self.fusion_number = fusion_number
        self.weights = nn.Sequential(
            nn.Linear(condition_dim, weight_feature_dim),
            nn.LayerNorm(weight_feature_dim),
            nn.LeakyReLU(inplace=True),
            nn.Linear(weight_feature_dim, weight_feature_dim),
            nn.LayerNorm(weight_feature_dim),
            nn.LeakyReLU(inplace=True),
            nn.Linear(weight_feature_dim, fusion_number),
            nn.LayerNorm(fusion_number),
            nn.Softmax(dim=2)
        )

    def forward(self, condition, inputs):
        """
        :param condition: (B,N,3)
        :param inputs: (B,N,K,O)
        """
        B, N, _ = condition.size()
        weights = self.weights(condition)  # (B,N,K)
        return torch.sum(weights[..., None] * inputs, dim=2)  # (B,N,O)


class MiniNets(nn.Module):
    def __init__(self, embedding_dim, kpt_num, in_features, out_features, hidden_features=128, num_hidden_layers=3,
                 hyper_hidden_layers=1, hyper_hidden_features=128, use_global_net: bool = False):
        super().__init__()
        self.kpt_num = kpt_num
        self.use_global_net = use_global_net

        if self.kpt_num == 0:
            self.num_mini_net = 1
        else:
            self.num_mini_net = self.kpt_num
            if self.use_global_net:
                self.num_mini_net += 1

        self.mini_nets = MiniNet(embedding_dim, in_features=in_features * self.num_mini_net,
                                 out_features=out_features * self.num_mini_net,
                                 hidden_features=hidden_features, num_hidden_layers=num_hidden_layers,
                                 hyper_hidden_layers=hyper_hidden_layers, hyper_hidden_features=hyper_hidden_features)

        if self.num_mini_net > 1:
            self.fusion = FusionNet(self.num_mini_net, in_features)

    def forward(self, xyz, keypoints=None, code=None):
        """
        :param keypoints: (B,K,3) or (K,3)
        :param xyz: (B,N,3) or (N,3)
        :param code: (B,D)
        """
        if len(xyz.size()) == 2:
            xyz = xyz.unsqueeze(0)
        B, N, _ = xyz.size()
        if keypoints is None:
            assert self.kpt_num == 0
            k_xyz = xyz
            output = self.mini_nets(k_xyz, code)
        else:
            if len(keypoints.size()) == 2:
                keypoints = keypoints.unsqueeze(0).repeat(B, 1, 1)
            K = keypoints.size(1)
            assert K == self.kpt_num

            k_xyz = xyz.unsqueeze(-2) - keypoints.unsqueeze(1).expand(-1, N, -1, -1)  # (B, N, K, 3)
            if self.use_global_net:
                k_xyz = torch.cat([k_xyz, xyz.unsqueeze(-2)], dim=-2)
            assert k_xyz.shape[-2] == self.num_mini_net
            k_xyz = k_xyz.flatten(start_dim=2)  # (B, N, K*3)
            outputs = self.mini_nets(k_xyz, code).view(B, N, self.num_mini_net, -1)
            if self.num_mini_net > 1:
                output = self.fusion(xyz, outputs)
            else:
                output = outputs.squeeze(-2)
        return output


class ImplicitShapeModel(nn.Module):
    def __init__(self, **config) -> None:
        """
        Inheriting class should accept some path to load its weights, and additionally
        initialize the latent_mean, latent_std, and triangles attributes (possibly also from file)
        """
        super().__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() and config.get("use_gpu", True) else "cpu")
        self.float_dtype = torch.float64 if config.get("double_precision", False) else torch.float32  # todo: maybe don't allow this at all?
        self.int_dtype = torch.int64

        self.lat_dim = int(config.get("lat_dim", config["latent_size"]))

        self.warp_type = config.get("warp_type", "se3")
        self.num_deform_features = 3 if self.warp_type == 'translation' else 6
        self.use_sdf_correction = config.get("use_sdf_correction", True)

        hidden_dim = config.get("hidden_dim", 128)
        num_hidden_layers = config.get("num_hidden_layers", 3)
        hyper_num_hidden_layers = config.get("hyper_num_hidden_layers", 1)
        hyper_hidden_dim = config.get("hyper_hidden_dim", 128)

        self.predict_correspondence_landmarks = int(config.get("predict_correspondence_landmarks", 0))
        self.correspondence_lat_dim = int(config.get("correspondence_lat_dim", 128))

        if self.predict_correspondence_landmarks > 0:
            self.correspondence_latents = torch.nn.Embedding(
                self.predict_correspondence_landmarks, self.correspondence_lat_dim,
                max_norm=1.0).float()
            torch.nn.init.normal_(
                self.correspondence_latents.weight.data,
                0.0, 0.1 / np.sqrt(self.correspondence_lat_dim))
            self.lm_correspondence_net = nn.Sequential(
                nn.Linear(self.lat_dim + self.correspondence_lat_dim, 256),
                nn.LeakyReLU(inplace=True),
                nn.Linear(256, 256),
                nn.LeakyReLU(inplace=True),
                nn.Linear(256, 3))


        # deformation network
        deform_net_num_outfeatures = self.num_deform_features
        if self.use_sdf_correction:
            deform_net_num_outfeatures += 1
        self.deform_net = MiniNets(
            self.lat_dim, kpt_num=0, in_features=3, out_features=deform_net_num_outfeatures,
            hidden_features=hidden_dim,
            num_hidden_layers=num_hidden_layers, hyper_hidden_layers=hyper_num_hidden_layers,
            hyper_hidden_features=hyper_hidden_dim
        )

        # template network
        self.template_net = MiniNets(
            None, kpt_num=0, in_features=3, out_features=1, hidden_features=hidden_dim,
            num_hidden_layers=num_hidden_layers)
        template_landmarks: Union[torch.Tensor, np.ndarray, None] = config.get("template_landmarks", None)
        if isinstance(template_landmarks, torch.Tensor):
            template_landmarks = template_landmarks.detach().cpu().numpy()
        if template_landmarks is not None:
            self.template_landmarks = Landmarks(template_landmarks, names=config.get("template_landmarks_names", None))
        else:
            self.template_landmarks = None

        # This is how the latents were initialized. If not provided by the checkpoint, we assume
        # the model stuck to this distribution.
        self.latent_mean = torch.zeros(self.lat_dim, dtype=self.float_dtype, device=self.device)
        self.latent_std = torch.ones(self.lat_dim, dtype=self.float_dtype, device=self.device) * 0.1 / np.sqrt(self.lat_dim)
        self.latent_pca = {}

        # The trained models may include some correlated attributes, i.e., latent directions that represent the change
        # of a specific attribute, which were found by correlating this attribute with the latent space.
        self.correlated_attributes = {}

        self.to(self.device)

    def load(self, ckpt_or_path: Union[str, Path, Dict]) -> None:
        if isinstance(ckpt_or_path, dict):
            ckpt = ckpt_or_path
        else:
            ckpt = torch.load(get_model_path(ckpt_or_path), map_location=self.device, weights_only=False)
        self.load_state_dict(ckpt["model_state"])
        self.latent_mean = ckpt["latent_mean"]
        self.latent_std = ckpt["latent_std"]
        self.latent_pca = ckpt["latent_pca"]
        self.correlated_attributes = ckpt.get("correlated_attributes", {})

    @staticmethod
    def load_static(ckpt_path: Union[str, Path], device: Union[str, torch.device, None] = None) -> "ImplicitShapeModel":
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        ckpt = torch.load(get_model_path(ckpt_path), map_location=device, weights_only=False)
        config = ckpt["config"]
        model = ImplicitShapeModel(**config)
        model.load(ckpt)
        return model

    def get_tensor(self, some_list_or_array: Union[List, ArrayLike], copy: bool = False, dtype=None) -> torch.Tensor:
        if not isinstance(some_list_or_array, torch.Tensor):
            current_tensor = torch.as_tensor(some_list_or_array)
        else:
            current_tensor = some_list_or_array
        current_tensor = current_tensor.to(dtype=dtype, device=self.device)
        return current_tensor.clone() if copy else current_tensor

    def forward(self, xyz: torch.Tensor, latent: torch.Tensor, landmarks: Union[None, torch.Tensor] = None,
                skip_correction: bool = False):
        if xyz.ndim == 2:
            xyz = xyz.unsqueeze(0)
        coords_ori = xyz

        batch_size = xyz.size(0)

        deform_net_output_coords = self.deform_net(coords_ori, None, latent)
        deformation_coords_for_warp = deform_net_output_coords[:, :, :self.num_deform_features]
        deformed_coords = utils_deepsdf.warp(coords_ori, deformation_coords_for_warp, self.warp_type)

        sdf_template = self.template_net(deformed_coords, None)
        sdf = sdf_template
        if self.use_sdf_correction and not skip_correction:
            sdf_correction = deform_net_output_coords[:, :, -1:]
            sdf = sdf + sdf_correction
        else:
            sdf_correction = None

        if landmarks is not None or self.predict_correspondence_landmarks > 0:
            if landmarks is not None:
                current_landmarks_pred = None
                current_landmarks = landmarks
            else:
                latent_expanded = latent.unsqueeze(1).expand(
                    batch_size, self.predict_correspondence_landmarks, self.lat_dim)
                lm_lat_expanded = self.correspondence_latents.weight.unsqueeze(0).expand(
                    batch_size, self.predict_correspondence_landmarks, self.correspondence_lat_dim)
                combined_latents = torch.cat([latent_expanded, lm_lat_expanded], dim=-1)

                current_landmarks_pred = self.lm_correspondence_net(combined_latents).view(
                    batch_size, self.predict_correspondence_landmarks, 3)

                current_landmarks = current_landmarks_pred.detach()
            deform_net_output_landmarks = self.deform_net(current_landmarks, None, latent)
            deformation_landmarks_for_warp = deform_net_output_landmarks[:, :, :self.num_deform_features]

            deformed_landmarks = utils_deepsdf.warp(current_landmarks, deformation_landmarks_for_warp, self.warp_type)
            return sdf, sdf_correction, deformed_coords, deformed_landmarks, current_landmarks_pred
        else:
            return sdf, sdf_correction, deformed_coords

    def normalize_weights(self, latent_weights: ArrayLike) -> ArrayLike:
        """
        Normalize the given weights, such that they are scaled relative to the latent distribution of the
        morphable model. The entries then represent how many standard deviations each value is away from the mean.
        :param latent_weights: Unnormalized weights. A vector that's shorter than the model's components is supported;
                               we simply shorten the latent mean and stdev vectors by cutting of the end. Note that
                               this probably doesn't make sense for an autoencoder. If the vector is too long, it will
                               raise an assertion error.
        :return: Numpy array with normalized weights. Same length as latent_weights.
        """
        latent_weights_array = self.get_tensor(latent_weights, copy=False, dtype=self.float_dtype)
        assert latent_weights_array.shape[-1] == len(self.latent_mean)
        return (latent_weights_array - self.latent_mean) / self.latent_std

    def unnormalize_weights(self, latent_weights: ArrayLike) -> ArrayLike:
        """
        Unnormalize the given weights, such that they are absolute, not relatively scaled. This method scales the
        weights by the standard deviation and adds the mean.
        :param latent_weights: Normalized weights, representing how many standard deviations each value is away from
                               the mean. A vector that's shorter than the model's components is supported;
                               we simply shorten the latent mean and stdev vectors by cutting of the end. Note that
                               this probably doesn't make sense for an autoencoder. If the vector is too long, it will
                               raise an assertion error.
        :return: Numpy array with unnormalized weights. Same length as latent_weights.
        """
        latent_weights_array = self.get_tensor(latent_weights, copy=False, dtype=self.float_dtype)
        assert latent_weights_array.shape[-1] <= len(self.latent_mean)
        return latent_weights_array * self.latent_std + self.latent_mean

    def encode(self, surface_points: ArrayLike, unknown_vertex_mask: Union[List[int], np.ndarray, None] = None,
               invert_mask: bool = False, landmarks: Union[ArrayLike, Landmarks, None] = None,
               surface_normals: Union[ArrayLike, None] = None, **configs) -> torch.Tensor:
        """
        Return latent code corresponding to the vertices given as input to this method.
        Since this is an auto-decoder model, the latent is inferred via iterative optimization.
        The latent is defined as a torch parameter, initialized as zero (average), then during each iteration,
        the model predicts the SDF for the provided surface_points, given the current latent parameter,
        and we penalize deviations from 0, then the latent code is updated based on this loss and some regularization.
        Note that this is a standard framework, and we haven't thoroughly experimented with it.
        Although like most work, we use ADAM by default for optimization, we additionally offer the option to
        use LBFGS instead, since it might converge faster and/or more robustly.
        If you run into issues, you might want to test some of the following:
        1) Add normals and/or landmarks to better constrain the optimization
        2) Set use_squared_sdf_loss=True to use l = sdf.square() instead of l = sdf.abs()
        3) Instead of using a fixed number of iterations, maybe use a while True loop that lowers the learning rate
           a few times whenever the loss converges, and the stops at a lower threshold for the learning rate.
           We haven't implemented this.
        4) We haven't thoroughly experimented with the losses. Adjust the scaling and scheduling.
        5) Try LBFGS instead of ADAM (optimizer="lbfgs" as argument)
        :param surface_points: nx3 numpy/torch array of points on the surface (where SDF should be 0).
        :param unknown_vertex_mask: Optionally provide a numpy mask array indicating which of the vertices
                                    inside the provided array should not be considered for encoding.
        :param invert_mask: If your array for unknown_vertex_mask actually contains the vertices that should be
                            considered, use this argument to invert the array.
        :param landmarks: Optionally provide landmarks, in which case correspondence of those landmarks to the
                          template is added as additional constraint.
        :param surface_normals: Optionally provide surface normals to be incorporated in the loss.
        :param configs: cf code.
        :return: latent code representing the input vertices in the morphable model's compact space.
                 Should be a 1D numpy/torch array of the same shape as latent_mean and latent_std.
        """
        # Set up inputs
        surface_points = self.get_tensor(surface_points, copy=False, dtype=self.float_dtype)
        if unknown_vertex_mask is not None:
            if invert_mask:
                unknown_vertex_mask = IndicesAndMasks.invert(unknown_vertex_mask, len(surface_points))
            surface_points = surface_points[IndicesAndMasks.invert(unknown_vertex_mask, len(surface_points))]
        num_points_per_observation = configs.get('num_points_per_observation', 5000)
        step_scale = configs.get('step_scale', 1)
        learning_rate = configs.get("lr", 0.01)

        # This is the parameter we optimize
        lat_rep_shape = torch.zeros([1, self.lat_dim], device=surface_points.device)
        lat_rep_shape.requires_grad = True


        # The ADAM setting is adopted from previous work
        if "adam" in configs.get("optimizer", "adam").lower():
            opt = optim.Adam(params=[lat_rep_shape], lr=learning_rate)
            n_steps = configs.get("n_steps", 1000)
            use_lbfgs = False
        # The LBFGS setting should take roughly the same time as the ADAM setting with the
        # default number of iterations.
        else:
            opt = optim.LBFGS(params=[lat_rep_shape], lr=learning_rate)
            n_steps = configs.get("n_steps", 20)
            use_lbfgs = True
        # This scheduling is adopted from previous work. It currently doesn't include
        # any adjustment of the landmark or normal loss. Feel free to experiment more with it.
        scheduling = {
            'lr': {int(i * n_steps / 5): 2 for i in range(1, 5)},
            'reg': {n_steps // 5: 3, int(3 * n_steps / 5): 10},
        }

        # This is the loss setting NPHM uses during training
        lambdas = {
            "surface": configs.get("surface_loss", 2.0),
            "reg": configs.get("reg", 0.01),
            "landmark": configs.get("landmark_loss", 1),
            "normal": configs.get("normal_loss", 0.3),
        }

        # Set up landmarks if provided
        if landmarks is None:
            landmarks_torch = landmarks
        else:
            assert self.template_landmarks is not None
            if isinstance(landmarks, Landmarks):
                landmarks, template_landmarks = Landmarks.get_corresponding_landmarks(landmarks, self.template_landmarks)
                landmarks = landmarks.get_points()
                template_landmarks = template_landmarks.get_points()
            else:
                assert self.template_landmarks.get_num_points() == len(landmarks)
                template_landmarks = self.template_landmarks.get_points()

            landmarks_torch = self.get_tensor(landmarks, copy=False, dtype=self.float_dtype)
            template_landmarks_torch = self.get_tensor(template_landmarks, copy=False, dtype=self.float_dtype)

        # Set up normals if provided
        if surface_normals is None:
            surface_normals_torch = surface_normals
        else:
            surface_normals_torch = self.get_tensor(surface_normals, copy=False, dtype=self.float_dtype)

        for j in tqdm(range(int(n_steps * step_scale)), desc="Optimizing latent"):
            # eye-balled scheduling of learning rate and weighing of losses (from previous work)
            j_step = int(j / step_scale)
            for schd_kind, sched_div in scheduling.items():
                if not j_step in sched_div:
                    continue
                if schd_kind == "lr":
                    for param_group in opt.param_groups:
                        param_group[schd_kind] /= sched_div[j_step]
                else:
                    lambdas[schd_kind] /= sched_div[j_step]

            # Random subsampling of surface points during each iteration
            idx = torch.multinomial(torch.ones(len(surface_points)),
                                    min(num_points_per_observation, len(surface_points)), replacement=False)
            obs = surface_points[idx].unsqueeze(0)

            # Optimization is defined in a function for compatibility with LBFGS
            # (LBFGS performs multiple passes to estimate second derivative to make a more informed parameter update)
            def closure():
                opt.zero_grad()

                sdf, *remaining_output = self(obs, lat_rep_shape, landmarks=landmarks_torch)

                # Surface loss
                l = sdf.square() if configs.get("use_squared_sdf_loss", False) else sdf.abs()

                # eye-balled schedule for loss clamping (based on previous work)
                l = l[l < 0.1]
                if j > int(n_steps / 4 * step_scale):
                    l = l[l < 0.05]
                if j > int(n_steps / 2 * step_scale):
                    l = l[l < 0.0075]

                # Surface and regularization losses are always used.
                loss_dict = {
                    'surface': l.mean(),
                    "reg": (torch.norm(lat_rep_shape, dim=-1) ** 2).mean()}

                # Landmark loss is computed only if landmarks are available.
                if landmarks_torch is not None:
                    deformed_landmarks = remaining_output[2]
                    loss_dict["landmark"] = torch.mean(torch.linalg.norm(
                        deformed_landmarks - template_landmarks_torch, dim=-1) ** 2)

                # Normal loss is computed only if normals are available.
                if surface_normals_torch is not None:
                    # The gradient of the SDF should align with the normals.
                    pred_grad = grad(
                        outputs=sdf, inputs=obs,
                        grad_outputs=torch.ones_like(sdf, requires_grad=False, device=self.device),
                        create_graph=True, retain_graph=True,
                        only_inputs=True, allow_unused=True)[0][..., -3:]
                    loss_dict["normal"] = torch.mean(torch.linalg.norm(pred_grad - surface_normals_torch[idx], dim=-1))

                # Compute loss as weighted sum of individual losses.
                loss = 0
                for loss_key, loss_value in loss_dict.items():
                    if lambdas[loss_key] > 0:
                        loss += lambdas[loss_key] * loss_value
                loss.backward()
                return loss

            if use_lbfgs:
                opt.step(closure)
            else:
                closure()
                opt.step()

        return lat_rep_shape[0]

    def decode(self, latent: ArrayLike, resolution: int = 256, split_batch: int = 50000,
               skip_sdf_correction: bool = False, highlight_correspondences: bool = False,
               correspondence_number_or_radius: Union[float, int] = 0.02) -> Mesh:
        """
        Given 1D latent vector reconstruct the vertices using the morphable model.
        Note that if you normalized the weights, you need to unnormalize them before
        giving them to this method.
        :param latent: 1D numpy array of the same shape as latent_mean and latent_std.
        :param resolution: number of steps per axis, so resolution^3 3D positions are uniformly distributed in the
                           [-1,1] cube to evaluate the SDF and reconstruct the mesh via marching cubes.
        :param split_batch: How many points to evaluate per batch. Tuned for relatively small GPUs (11GB of memory).
                            Increase if you have a larger GPU.
        :param skip_sdf_correction: Set True to skip the SDF correction of the implicit model.
        :param highlight_correspondences: Set True to mark the correspondences to the template landmarks
                                          on the decoded mesh. You can choose how to highlight them with
                                          correspondence_number_or_radius.
        :param correspondence_number_or_radius: Choose either how many of the closest points to each template
                                                landmark you want to mark by choosing an int => 1, or you can choose
                                                the radius around each template landmark to mark on the decoded mesh.
                                                Keep in mind that you can choose 1 to get the one closest
                                                correspondence, but since the cleft correspondences can map two or
                                                three cleft borders to one healthy point, you might be missing some.
        :return: nx3 numpy array of vertices.
        """
        latent = self.get_tensor(latent)
        coords = utils_deepsdf.get_volume_coords(resolution, device=self.device)
        # Split coords into batches because of memory limitations
        coords_batches = torch.split(coords, split_batch)
        sdf = []
        self.eval()
        with torch.no_grad():
            for coords_ind in coords_batches:
                sdf_batch, *_ = self(
                    coords_ind, latent.unsqueeze(0), skip_correction=skip_sdf_correction)
                sdf_batch = sdf_batch.squeeze(0)[..., :1]
                sdf.append(sdf_batch)
            sdf = torch.cat(sdf, dim=0)
        vertices, faces = utils_deepsdf.extract_mesh(resolution, sdf)
        recon_mesh = Mesh(vertices=vertices, triangles=faces)
        # Optionally highlight template correspondences on decoded mesh
        if highlight_correspondences:
            assert self.template_landmarks is not None
            template_landmarks_torch = self.get_tensor(self.template_landmarks.get_points(), dtype=self.float_dtype)
            # Convert mesh vertices to torch
            vertices_torch = self.get_tensor(vertices, dtype=self.float_dtype)
            # Deform the mesh vertices back into template space
            _, _, deformed_vertices_torch, *_ = self(
                vertices_torch, latent.unsqueeze(0), skip_correction=skip_sdf_correction)
            # Compute the distance of the deformed vertices to the template landmarks
            vertex_distances_torch = torch.linalg.norm(
                deformed_vertices_torch - template_landmarks_torch.unsqueeze(1), dim=-1)
            vertex_distances = vertex_distances_torch.detach().cpu().numpy()
            # We choose a different color for each landmark to mark as correspondence on the mesh
            error_colors = self.template_landmarks.get_landmark_colors()
            # The base color of the mesh used for all non-corresponding vertices
            base_color = np.asarray([0.7, 0.7, 0.7])
            # We either mark the vertices closer than this number
            if correspondence_number_or_radius < 1:
                # We assign a gradient that goes from the defined color to the base color for each landmark,
                # assigning the defined color to the vertices very close to the template landmark, and hitting
                # the base color at correspondence_number_or_radius of distance.
                vertex_colors_all = np.asarray([utils.get_error_heatmap_colors(
                    vertex_dist, min_value=0, max_value=correspondence_number_or_radius,
                    heatmap_colors=[error_color, [0.7, 0.7, 0.7]])
                    for vertex_dist, error_color in zip(vertex_distances, error_colors)])
                vertex_colors_combined = np.sum(vertex_colors_all - base_color, axis=0) + base_color
            # Or we choose a fixed number of closest vertices.
            else:
                # Here we assign the same color to all correspondence_number nearest vertices
                correspondence_number = int(correspondence_number_or_radius)
                vertex_colors_combined = np.ones((len(vertices), 3)) * base_color
                for vertex_dist, error_color in zip(vertex_distances, error_colors):
                    indices = np.argsort(vertex_dist)[:correspondence_number]
                    vertex_colors_combined[indices] = error_color
            recon_mesh.set_colors(vertex_colors_combined)
        return recon_mesh

    @staticmethod
    def get_mesh_normalization_transform_from_registered_mesh(
            registered_mesh: Mesh, indices_to_norm_to: Union[np.ndarray, None] = None) -> np.ndarray:
        """
        Compute and return the transformation matrix used to normalize a mesh.
        Since our models were trained on the normalizations computed form the registered meshes,
        which were already well aligned, we ask for a registered mesh here. The method
        won't fail if you provide another mesh, but the resulting mesh will probably
        not be aligned well enough with the implicit model space.
        :param registered_mesh: Mesh registered by the explicit INCLEFT model.
        :param indices_to_norm_to: Indices to normalize the mesh to, such that one part of the mesh can
                                   fill the full unit cube.
        :return 4x4 homogeneous transformation matrix (np array).
        """
        registered_vertices = registered_mesh.get_vertices()
        if indices_to_norm_to is not None:
            registered_vertices = registered_vertices[indices_to_norm_to]
        rv_min = np.min(registered_vertices, axis=0)
        rv_max = np.max(registered_vertices, axis=0)
        center = 0.5 * (rv_min + rv_max)  # bbox center
        half_extents = 0.5 * (rv_max - rv_min)
        scale = np.max(half_extents)
        scale *= 1.005  # we increase the scale very slightly to avoid unwanted rounding
        return utils.get_transformation_matrix(translate=-center, scale=1/scale)

    def get_number_of_components(self) -> int:
        return self.lat_dim

    def sample_latent(self, deviation_factor: float = 1):
        return torch.normal(self.latent_mean, deviation_factor*self.latent_std)

    def sample(self, deviation_factor: float = 1, resolution: int = 256, split_batch: int = 50000,
               skip_sdf_correction: bool = False) -> Mesh:
        """
        Sample model output (mesh).
        :param deviation_factor: How far to deviate from average during sampling relative to standard deviation.
        :param resolution: cf. decode()
        :param split_batch: cf. decode()
        :param skip_sdf_correction: cf. decode()
        :return: A set of vertices, randomly sampled by the morphable model.
        """
        latent = self.sample_latent(deviation_factor=deviation_factor)
        return self.decode(latent, resolution=resolution, split_batch=split_batch,
                           skip_sdf_correction=skip_sdf_correction)

    def sample_multiple(self, num_samples: int, deviation_factor: float = 1, resolution: int = 256,
                        split_batch: int = 50000, skip_sdf_correction: bool = False) -> List[Mesh]:
        """
        Sample multiple meshes from the implicit model..
        :param num_samples: Number of vertex sets to be randomly sampled (int).
        :param deviation_factor: How far to deviate from average during sampling relative to standard deviation.
        :param resolution: cf. decode()
        :param split_batch: cf. decode()
        :param skip_sdf_correction: cf. decode()
        :return: list of randomly sampled vertices (each entry has nx3 vertices).
        """
        return [self.sample(deviation_factor=deviation_factor, resolution=resolution, split_batch=split_batch,
                            skip_sdf_correction=skip_sdf_correction) for _ in range(num_samples)]

    def adjust_latent_along_attribute(
            self, attribute_name: str, attribute_extent: float, latent_init: Union[ArrayLike, None] = None,
            fix_other_attributes: Union[bool, List[str]] = False) -> torch.Tensor:
        if attribute_name not in self.correlated_attributes:
            raise ValueError(f"{attribute_name} not part of the defined correlated attributes. Available are: "
                             f"{list(self.correlated_attributes.keys())}.")
        latent = self.get_tensor(latent_init) if latent_init is not None else self.latent_mean
        if (isinstance(fix_other_attributes, bool) and fix_other_attributes) or (isinstance(fix_other_attributes, list) and len(fix_other_attributes) > 0):
            if isinstance(fix_other_attributes, bool):
                other_attributes = [attr for attr in self.correlated_attributes.keys() if attr != attribute_name]
            else:
                other_attributes = [attr for attr in fix_other_attributes if attr in self.correlated_attributes]
            w_mat = torch.stack([self.correlated_attributes[attr] for attr in [attribute_name, *other_attributes]], axis=1)
            one_hot_vector = torch.zeros(w_mat.shape[1], dtype=self.float_dtype, device=self.device)
            one_hot_vector[0] = 1
            attribute_direction = w_mat @ torch.linalg.inv(torch.transpose(w_mat, 0, 1) @ w_mat) @ one_hot_vector
        else:
            attribute_direction = self.correlated_attributes[attribute_name]
        latent += attribute_extent * (attribute_direction / torch.linalg.norm(attribute_direction))
        return latent

    def get_average_mesh(self, resolution: int = 256, split_batch: int = 50000,
               skip_sdf_correction: bool = False) -> Mesh:
        """
        Returns the average mesh of the implicit model. It simply calls decode with the average latent.
        Cf. docs of decode() function.
        """
        return self.decode(self.latent_mean, resolution=resolution, split_batch=split_batch,
                           skip_sdf_correction=skip_sdf_correction)

    def reconstruct_mesh(
            self, mesh: Mesh, unknown_vertex_mask: Union[None, np.ndarray] = None, invert_mask: bool = False,
            return_latent: bool = False, resolution: int = 256, split_batch: int = 50000,
            skip_sdf_correction: bool = False,  highlight_correspondences: bool = False,
            correspondence_number_or_radius: Union[float, int] = 0.02,
            **encode_configs) -> Union[Mesh, Tuple[Mesh, torch.Tensor]]:
        """
        Basically encode and decode a mesh again, optionally only using parts of the vertices.
        :param mesh: Mesh to be reconstructed by implicit model.
        :param unknown_vertex_mask: Optionally provide a mask of vertices that are unknown. The model will reconstruct
                                    the mesh based only on the information of the remaining vertices.
        :param invert_mask: If the mask you provided specifies the known vertices instead of the unknown, set this True.
        :param return_latent: Set True to return not only the reconstructed mesh, but also its latent projection.
        :param resolution: cf. decode()
        :param split_batch: cf. decode()
        :param skip_sdf_correction: cf. decode()
        :param highlight_correspondences: cf. decode()
        :param correspondence_number_or_radius: cf. decode()
        :param encode_configs: Arguments used for encoding the mesh.
        :return Either only the reconstructed Mesh, or a tuple with the reconstructed Mesh and the latent
                if return_latent=True.
        """
        mesh_vertices = mesh.get_vertices()
        mesh_normals = mesh.get_vertex_normals()
        latent = self.encode(
            mesh_vertices, unknown_vertex_mask=unknown_vertex_mask, invert_mask=invert_mask,
            surface_normals=mesh_normals, landmarks=mesh.landmarks, **encode_configs
        )
        reconstructed_mesh = self.decode(
            latent, resolution=resolution, split_batch=split_batch, skip_sdf_correction=skip_sdf_correction,
            highlight_correspondences=highlight_correspondences,
            correspondence_number_or_radius=correspondence_number_or_radius
        )
        if return_latent:
            return reconstructed_mesh, latent
        else:
            return reconstructed_mesh







def main():
    parser = ArgumentParser()
    parser.add_argument('--ckpt_path', type=str, required=True,
                        help="torch ckpt file path")
    parser.add_argument('--mesh_to_reconstruct_path', type=str, required=True,
                        help="torch ckpt file path")
    args = parser.parse_args()
    model = ImplicitShapeModel.load_static(args.ckpt_path)
    mesh_to_reconstruct = Mesh.load(args.mesh_to_reconstruct_path)
    reconstructed_mesh = model.reconstruct_mesh(mesh_to_reconstruct, return_latent=False)
    Mesh.show_multiple_meshes(mesh_to_reconstruct, reconstructed_mesh)


if __name__ == '__main__':
    main()
