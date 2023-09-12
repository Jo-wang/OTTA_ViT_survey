# Copyright 2020-2021 Evgenia Rusak, Steffen Schneider, George Pachitariu
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# ---
# This licence notice applies to all originally written code by the
# authors. Code taken from other open-source projects is indicated.
# See NOTICE for a list of all third-party licences used in the project.

"""Batch norm variants
AlphaBatchNorm builds upon: https://github.com/bethgelab/robustness/blob/main/robusta/batchnorm/bn.py
"""

import torch
from torch import nn
from torch.nn import functional as F

class AlphaLayerNorm(nn.Module):
    """ Use the source statistics as a prior on the target statistics """

    @staticmethod
    def find_lns(parent, alpha):
        replace_mods = []
        if parent is None:
            return []
        for name, child in parent.named_children():
            if isinstance(child, LayerNormWithStoredStats):  # Replace nn.BatchNorm2d with nn.LayerNorm
                module = AlphaLayerNorm(child, alpha)
                replace_mods.append((parent, name, module))
            else:
                replace_mods.extend(AlphaLayerNorm.find_lns(child, alpha))

        return replace_mods

    @staticmethod
    def adapt_model(model, alpha):
        replace_mods = AlphaLayerNorm.find_lns(model, alpha)
        print(f"| Found {len(replace_mods)} modules to be replaced.")
        for (parent, name, child) in replace_mods:
            setattr(parent, name, child)
        return model

    def __init__(self, layer, alpha):
        assert alpha >= 0 and alpha <= 1

        super().__init__()
        self.layer = layer
        self.layer.eval()
        self.alpha = alpha

        self.norm = LayerNormWithStoredStats(self.layer.num_features, elementwise_affine=False)

    def forward(self, input):
        
        self.norm(input)

        running_mean = ((1 - self.alpha) * self.layer.stored_means + self.alpha * self.norm.mean)
        running_var = ((1 - self.alpha) * self.layer.var + self.alpha * self.norm.var)

        return F.layer_norm(
            input,
            running_mean,
            running_var,
            self.layer.weight,
            self.layer.bias,
            False,
            0,
            self.layer.eps,
        )


    
 
class LayerNormWithStoredStats(nn.LayerNorm):
    def __init__(self, normalized_shape, eps=1e-5, elementwise_affine=True):
        super(LayerNormWithStoredStats, self).__init__(normalized_shape, eps, elementwise_affine)
        
        # Initialize lists to store mean and variance statistics
        self.stored_means = []
        self.stored_vars = []

    def forward(self, input):
        # Compute mean and variance for the current input
        mean = input.mean(-1, keepdim=True)
        var = input.var(-1, keepdim=True, unbiased=False)
        
        # Store the computed statistics
        self.stored_means.append(mean)
        self.stored_vars.append(var)
        
        # Call the original LayerNorm forward method
        return super(LayerNormWithStoredStats, self).forward(input)

# # Example usage
# layer_norm = LayerNormWithStoredStats(512)
# input_tensor = torch.randn(10, 512)  # Example tensor with 10 samples, each with 512 features
# output = layer_norm(input_tensor)

# # Get the stored statistics
# stored_means = layer_norm.stored_means[-1]
# stored_vars = layer_norm.stored_vars[-1]

# stored_means, stored_vars
   



