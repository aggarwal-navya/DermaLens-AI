"""Models: a timm image backbone, optionally fused with patient metadata.

DermaNet = backbone (image -> feature vector)
         + small MLP   (21-dim metadata -> 64-dim vector)
         + head        (concatenated features -> class logits)

All backbones start from pretrained weights (ImageNet-1k/22k or DINOv2).
"""

from __future__ import annotations

from . import metadata as meta

# Friendly name -> (timm model name, native input size, family)
MODEL_ZOO = {
    "resnet18": ("resnet18", 224, "cnn"),                                    # quick checks only
    "resnet50": ("resnet50.a1_in1k", 224, "cnn"),
    "efficientnet_b0": ("efficientnet_b0", 224, "cnn"),
    "efficientnetv2_s": ("tf_efficientnetv2_s.in21k_ft_in1k", 288, "cnn"),
    "convnext_tiny": ("convnext_tiny.fb_in22k_ft_in1k", 224, "cnn"),
    "convnext_small": ("convnext_small.fb_in22k_ft_in1k_384", 288, "cnn"),
    "swin_tiny": ("swin_tiny_patch4_window7_224", 224, "swin"),
    "vit_small": ("vit_small_patch16_224.augreg_in21k_ft_in1k", 224, "vit"),
    "vit_base_dinov2": ("vit_base_patch14_dinov2.lvd142m", 224, "vit"),
}


def zoo_entry(name: str) -> tuple[str, int, str]:
    if name in MODEL_ZOO:
        return MODEL_ZOO[name]
    family = "vit" if name.startswith(("vit", "deit")) else "swin" if name.startswith("swin") else "cnn"
    return name, 224, family


def _torch_module():
    import torch.nn as nn
    return nn.Module


class DermaNet(_torch_module()):
    """Image backbone + optional metadata branch + classification head."""

    def __init__(self, name: str, num_classes: int, use_metadata: bool = True, pretrained: bool = True,
                 drop_rate: float = 0.3, drop_path_rate: float = 0.1, image_size: int | None = None):
        import timm
        import torch.nn as nn

        super().__init__()
        timm_name, native_size, self.family = zoo_entry(name)
        self.name = name
        self.use_metadata = use_metadata
        kwargs = {"pretrained": pretrained, "num_classes": 0, "drop_path_rate": drop_path_rate}
        if self.family in ("vit", "swin"):
            kwargs["img_size"] = image_size or native_size     # timm resizes position embeddings
        try:
            self.backbone = timm.create_model(timm_name, **kwargs)
        except TypeError:                                       # some models reject drop_path_rate
            kwargs.pop("drop_path_rate")
            self.backbone = timm.create_model(timm_name, **kwargs)
        width = self.backbone.num_features
        meta_width = 64 if use_metadata else 0
        if use_metadata:
            self.meta_net = nn.Sequential(nn.Linear(meta.META_DIM, 64), nn.BatchNorm1d(64), nn.SiLU(),
                                          nn.Dropout(0.2), nn.Linear(64, 64), nn.SiLU())
        self.head = nn.Sequential(nn.Dropout(drop_rate), nn.Linear(width + meta_width, num_classes))

    def forward(self, images, meta_vectors=None):
        import torch

        features = self.backbone(images)
        if self.use_metadata:
            if meta_vectors is None:
                meta_vectors = torch.as_tensor(meta.encode(None, None, []), device=images.device)
                meta_vectors = meta_vectors.unsqueeze(0).expand(images.shape[0], -1)
            features = torch.cat([features, self.meta_net(meta_vectors.float())], dim=1)
        return self.head(features)

    def head_parameters(self):
        params = list(self.head.parameters())
        if self.use_metadata:
            params += list(self.meta_net.parameters())
        return params


def build_model(name: str, num_classes: int, pretrained: bool = True, use_metadata: bool = True,
                drop_rate: float = 0.3, image_size: int | None = None) -> DermaNet:
    return DermaNet(name, num_classes, use_metadata, pretrained, drop_rate, image_size=image_size)


def set_backbone_trainable(model: DermaNet, trainable: bool) -> None:
    """Freeze/unfreeze the backbone; the head and metadata branch always train."""
    for param in model.backbone.parameters():
        param.requires_grad = trainable
    for param in model.head_parameters():
        param.requires_grad = True


def count_parameters(model) -> tuple[int, int]:
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


def gradcam_target_layer(model: DermaNet):
    """Layer whose activations Grad-CAM explains, per backbone family."""
    backbone = model.backbone
    if model.family == "vit":
        return backbone.blocks[-1].norm1
    if model.family == "swin":
        return backbone.layers[-1].blocks[-1].norm1
    for attr in ("conv_head", "stages", "layer4"):
        if hasattr(backbone, attr):
            layer = getattr(backbone, attr)
            return layer[-1] if attr in ("stages", "layer4") else layer
    raise ValueError(f"No Grad-CAM layer known for {model.name}")
