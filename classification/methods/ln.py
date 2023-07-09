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
#
# ---
# This licence notice applies to all originally written code by the
# authors. Code taken from other open-source projects is indicated.
# See NOTICE for a list of all third-party licences used in the project.

"""Batch norm variants
AlphaBatchNorm builds upon: https://github.com/bethgelab/robustness/blob/main/robusta/batchnorm/bn.py
"""

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
            if isinstance(child, nn.LayerNorm):
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

        self.norm = nn.LayerNorm(self.layer.num_features, elementwise_affine=False)

    def forward(self, input):
        self.norm(input)

        running_mean = ((1 - self.alpha) * self.layer.running_mean + self.alpha * self.norm.running_mean)
        running_var = ((1 - self.alpha) * self.layer.running_var + self.alpha * self.norm.running_var)

        return F.batch_norm(
            input,
            running_mean,
            running_var,
            self.layer.weight,
            self.layer.bias,
            False,
            0,
            self.layer.eps,
        )


class EMALayerNorm(nn.Module):
    @staticmethod
    def adapt_model(model):
        model = EMALayerNorm(model)
        return model

    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x):
        # store statistics, but discard result
        self.model.train()
        self.model(x)
        # store statistics, use the stored stats
        self.model.eval()
        return self.model(x)
