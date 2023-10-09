"""
Builds upon: https://github.com/zhangmarvin/memo
Corresponding paper: https://arxiv.org/abs/2110.09506
"""
import os
import numpy as np
from PIL import Image
import torchvision.transforms as transforms
import torch
from torch import nn as nn
import torch.jit

from methods.base import TTAMethod
from augmentations.transforms_memo_cifar import aug_cifar
from augmentations.transforms_memo_imagenet import aug_imagenet


def tta(image, n_augmentations, aug):

    image = np.clip(image[0].cpu().numpy() * 255., 0, 255).astype(np.uint8).transpose(1, 2, 0)
    inputs = [aug(Image.fromarray(image)) for _ in range(n_augmentations)]
    # output_dir = "/home/uqzxwang/code/test-time-adaptation/classification/img_output-w/"
    # if not os.path.exists(output_dir):
    #     os.makedirs(output_dir)
    # im = Image.fromarray(image.astype('uint8'), 'RGB')
    # im.save(output_dir+'/org.png')
    # for idx, img in enumerate(inputs):
    #     img_path = os.path.join(output_dir, f"augmented_image_{idx+1}.png")
    #     img = transforms.ToPILImage()(img).convert("RGB")
    #     img.save(img_path)
    inputs = torch.stack(inputs).cuda()
    return inputs


class MEMO(TTAMethod):
    def __init__(self, cfg, model, num_classes):
        super().__init__(cfg, model, num_classes)

        self.n_augmentations = cfg.TEST.N_AUGMENTATIONS
        self.augmentations = aug_cifar if "cifar" in self.dataset_name else aug_imagenet

    def forward(self, x):
        # if self.episodic:
        #     self.reset()

        for _ in range(self.steps):
            x_aug = tta(x, self.n_augmentations, aug=self.augmentations)
            _ = self.forward_and_adapt(x_aug)

        return self.model(x)

    @torch.enable_grad()  # ensure grads in possible no grad context for testing
    def forward_and_adapt(self, x):
        """Forward and adapt model on batch of data.
        Measure entropy of the model prediction, take gradients, and update params.
        """
        self.optimizer.zero_grad()
        outputs = self.model(x)
        loss, _ = marginal_entropy(outputs)
        loss.backward()
        self.optimizer.step()
        
        return outputs

    # NOTE default one
    def configure_model(self):
        self.model.train()


    def collect_params(self):
        """Collect the affine scale + shift parameters from batch norms.

        Walk the model's modules and collect all batch normalization parameters.
        Return the parameters and their names.

        Note: other choices of parameterization are possible!
        """
        params = []
        names = []
        for nm, m in self.model.named_modules():
            # if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.LayerNorm, nn.GroupNorm)):
            for np, p in m.named_parameters():
                if np in ['weight', 'bias']:  # weight is scale, bias is shift
                    params.append(p)
                    names.append(f"{nm}.{np}")
        return params, names

    # # NOTE not default setting in MEMO, please comment out this function
    # def configure_model(self):
    #     """Configure model for use with tent."""
        
    #     self.model.eval()  # eval mode to avoid stochastic depth in swin. test-time normalization is still applied
    #     # disable grad, to (re-)enable only what tent updates
    #     self.model.requires_grad_(False)
    #     # configure norm for tent updates: enable grad + force batch statisics
    #     for m in self.model.modules():
    #         if isinstance(m, (nn.LayerNorm, nn.GroupNorm)):
    #             m.requires_grad_(True)
                
def marginal_entropy(outputs):
    logits = outputs - outputs.logsumexp(dim=-1, keepdim=True)
    avg_logits = logits.logsumexp(dim=0) - np.log(logits.shape[0])
    min_real = torch.finfo(avg_logits.dtype).min
    avg_logits = torch.clamp(avg_logits, min=min_real)
    return -(avg_logits * torch.exp(avg_logits)).sum(dim=-1), avg_logits
