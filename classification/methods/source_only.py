import torch.jit
from methods.base import TTAMethod
import torch.nn as nn

class SO(TTAMethod):

    def __init__(self, cfg, model, num_classes):
        super().__init__(cfg, model, num_classes)

    # @torch.enable_grad()  # ensure grads in possible no grad context for testing
    def forward_and_adapt(self, x):
        with torch.no_grad():
            imgs_test = x[0]
            outputs = self.model(imgs_test)
        
            return outputs

    def collect_params(self):
        params = []
        names = []
        for nm, m in self.model.named_modules():
            for np, p in m.named_parameters():
                if np in ['weight', 'bias']:  # weight is scale, bias is shift
                    params.append(p)
                    names.append(f"{nm}.{np}")
        return params, names
    
    def configure_model(self):
        self.model.eval() 