import logging
import timm
import torch
import torch.nn as nn

from robustbench.model_zoo.architectures.utils_architectures import normalize_model, ImageNormalizer

# from timm.models.vision_transformer import generate_default_cfgs
logger = logging.getLogger(__name__)


def get_timm_model(model_name, cls):
    """
    Restore a pre-trained model from timm: https://github.com/huggingface/pytorch-image-models/tree/main/timm
    Quickstart: https://huggingface.co/docs/timm/quickstart
    :param model_name: name of the model to create and initialize with pre-trained weights
    :return: pre-trained model
    """
    # check if the defined model name is supported as pre-trained model
    available_models = timm.list_models(pretrained=True)
    if not model_name in available_models:
        raise ValueError(f"Model '{model_name}' is not available. Choose from: {available_models}")

    # setup pre-trained model
    model = timm.create_model(model_name, pretrained=True, num_classes=cls)
    logger.info(f"Successfully restored the weights of '{model_name}' from timm.")

    # add the corresponding input normalization to the model
    if hasattr(model, "pretrained_cfg"):
        logger.info(f"General model information: {model.pretrained_cfg}")
        logger.info(f"Adding input normalization to the model using: mean={model.pretrained_cfg['mean']} \t std={model.pretrained_cfg['std']}")
        model = normalize_model(model, mean=model.pretrained_cfg["mean"], std=model.pretrained_cfg["std"])
    # TODO: need to check if the input normalization is needed
    elif hasattr(model, "default_cfg"):
        logger.info(f"General model information: {model.default_cfg}")
        logger.info(f"Adding input normalization to the model using: mean={model.default_cfg['mean']} \t std={model.default_cfg['std']}")
        model = normalize_model(model, mean=model.default_cfg["mean"], std=model.default_cfg["std"])
    else:
        raise AttributeError(f"Attribute 'pretrained_cfg' is missing for model '{model_name}' from timm."
                             f" This prevents adding the correct input normalization to the model!")

    return model



class BaseModel(torch.nn.Module):
    """
    Change the model structure to perform the adaptation "AdaContrast" for other datasets
    """
    def __init__(self, model, arch_name, dataset_name):
        super().__init__()

        self.encoder, self.fc = split_up_model(model, arch_name=arch_name, dataset_name=dataset_name)
        if isinstance(self.fc, nn.Sequential):
            for module in self.fc.modules():
                if isinstance(module, nn.Linear):
                    self._num_classes = module.out_features
                    self._output_dim = module.in_features
        elif isinstance(self.fc, nn.Linear):
            self._num_classes = self.fc.out_features
            self._output_dim = self.fc.in_features
        else:
            raise ValueError("Unable to detect output dimensions")

    def forward(self, x, return_feats=False):
        # 1) encoder feature
        feat = self.encoder(x)
        feat = torch.flatten(feat, 1)

        logits = self.fc(feat)

        if return_feats:
            return feat, logits
        return logits

    @property
    def num_classes(self):
        return self._num_classes

    @property
    def output_dim(self):
        return self._output_dim



class TransformerWrapper(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.__dict__ = model.__dict__.copy()

    def forward(self, x):
        # Reshape and permute the input tensor
        x = self.normalize(x)
        x = self.model._process_input(x)
        n = x.shape[0]

        # Expand the class token to the full batch
        batch_class_token = self.model.class_token.expand(n, -1, -1)
        x = torch.cat([batch_class_token, x], dim=1)

        x = self.model.encoder(x)

        # Classifier "token" as used by standard language architectures
        x = x[:, 0]
        return x


def get_model(cfg):
# load model from timm
    base_model = get_timm_model(cfg.MODEL.ARCH, cfg.CORRUPTION.CLS)

    return base_model.cuda()


def split_up_model(model, arch_name, dataset_name):
    """
    Split up the model into an encoder and a classifier.
    This is required for methods like RMT and AdaContrast
    :param model: model to be split up
    :param arch_name: name of the network
    :param dataset_name: name of the dataset
    :return: encoder and classifier
    """
    if "vit_" in arch_name and not "maxvit_" in arch_name:
        encoder = TransformerWrapper(model)
        classifier = model.model.heads.head
    elif "swin_" in arch_name:
        encoder = nn.Sequential(model.normalize, model.model.features, model.model.norm, model.model.permute, model.model.avgpool, model.model.flatten)
        classifier = model.model.head
    elif arch_name == "mobilenet_v2":
        encoder = nn.Sequential(model.normalize, model.model.features, nn.AdaptiveAvgPool2d((1, 1)), nn.Flatten())
        classifier = model.model.classifier
    else:
        raise ValueError(f"The model architecture '{arch_name}' is not supported for dataset '{dataset_name}'.")

    return encoder, classifier
