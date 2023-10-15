import torch
import torch.nn as nn
import torch.nn.functional as F

from methods.base import TTAMethod

import torch.jit
# from gpu_mem_track import MemTracker


class ConjugatePL(TTAMethod):
    def __init__(self, cfg, model, num_classes):
        super().__init__(cfg, model, num_classes)


    def collect_params(self):
        """Collect the affine scale + shift parameters from batch norms.

        Walk the model's modules and collect all batch normalization parameters.
        Return the parameters and their names.

        Note: other choices of parameterization are possible!
        """
        params = []
        names = []
        for nm, m in self.model.named_modules():
            if isinstance(m, (nn.LayerNorm)):
                for np, p in m.named_parameters():
                    if np in ['weight', 'bias']:  # weight is scale, bias is shift
                        params.append(p)
                        names.append(f"{nm}.{np}")
        return params, names

    def configure_model(self):
        """Configure model for use with tent."""
        # train mode, because tent optimizes the model to minimize entropy
        self.model.eval()
        # disable grad, to (re-)enable only what tent updates
        self.model.requires_grad_(False)
        # configure norm for tent updates: enable grad + force batch statisics
        for m in self.model.modules():
            if isinstance(m, nn.LayerNorm):
                m.requires_grad_(True)
  

    
    # NOTE: change num_class to 1000 if imagenet-c
    @torch.enable_grad()  # ensure grads in possible no grad context for testing
    def forward_and_adapt(self, x, n_inner_iter=1, adaptive=True, use_test_bn=True, num_classes=100, source_poly=False, temp=2.0, eps=1e-6):
        img_test = x[0]
        outputs = self.model(img_test)
        outputs = outputs / temp
        
        for _ in range(n_inner_iter):
            if source_poly:
                softmax_prob = F.softmax(outputs, dim=1)
                eps = eps
                smax_inp = softmax_prob 
                eye = torch.eye(num_classes).to(outputs.device)
                eye = eye.reshape((1, num_classes, num_classes))
                eye = eye.repeat(outputs.shape[0], 1, 1)
                t2 = eps * torch.diag_embed(smax_inp)
                smax_inp = torch.unsqueeze(smax_inp, 2)
                t3 = eps*torch.bmm(smax_inp, torch.transpose(smax_inp, 1, 2))
                matrix = eye + t2 - t3
                y_star = torch.linalg.solve(matrix, smax_inp)
                y_star = torch.squeeze(y_star)
                pseudo_prob = y_star
                tta_loss = torch.logsumexp(outputs, dim=1) - (pseudo_prob * outputs - eps * pseudo_prob *(1-softmax_prob)).sum(dim=1)
                tta_loss = tta_loss.mean()
                self.optimizer.zero_grad()
                tta_loss.backward()
                self.optimizer.step()
            else:
                loss = softmax_entropy(outputs).mean(0)
                self.optimizer.zero_grad()
                loss.backward()
                # MemTracker.track('After backward')
                self.optimizer.step()
                
        outputs_new = self.model(img_test)
        # NOTE need to output output_new
        return outputs_new
    

@torch.jit.script
def softmax_entropy(x: torch.Tensor) -> torch.Tensor:
    """Entropy of softmax distribution from logits."""
    return -(x.softmax(1) * x.log_softmax(1)).sum(1)