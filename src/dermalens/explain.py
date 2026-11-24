"""Grad-CAM: which parts of the photo pushed the model toward its answer.

How it works, in four steps:
    1. Run the image forward and keep the last conv block's feature maps A^k.
    2. Back-propagate the score of the chosen class to get dScore/dA^k.
    3. Average each gradient map to one number alpha_k (how important map k is).
    4. Heatmap = ReLU(sum_k alpha_k * A^k), resized to the image and scaled to 0-1.

Red areas in the overlay = evidence for the predicted class.
"""

from __future__ import annotations

import numpy as np
from PIL import Image


class GradCAM:
    """Usage:
        cam = GradCAM(model, gradcam_target_layer(model, "resnet50"))
        heatmap, class_index, probabilities = cam(image_tensor)   # tensor shaped [3, H, W]
        cam.remove()
    """

    def __init__(self, model, target_layer):
        self.model = model.eval()
        self._activations = None
        self._gradients = None
        self._hooks = [target_layer.register_forward_hook(self._capture)]

    def _capture(self, module, inputs, output):
        """Keep the feature maps, and ask PyTorch to hand us their gradient on backward()."""
        if not output.requires_grad:          # a normal no-grad prediction: nothing to explain
            return
        self._activations = output.detach()
        output.register_hook(self._save_gradients)

    def _save_gradients(self, gradient):
        self._gradients = gradient.detach()

    def _to_nchw(self, tensor):
        """Bring activations to [B, C, H, W]: ViT tokens and Swin's channels-last are reshaped."""
        family = getattr(self.model, "family", "cnn")
        if tensor.dim() == 3:                                            # ViT: [B, tokens, C]
            prefix = getattr(self.model.backbone, "num_prefix_tokens", 1)
            tokens = tensor[:, prefix:, :]
            side = int(round(tokens.shape[1] ** 0.5))
            return tokens.reshape(tokens.shape[0], side, side, -1).permute(0, 3, 1, 2)
        if family == "swin":                                             # Swin: [B, H, W, C]
            return tensor.permute(0, 3, 1, 2)
        return tensor

    def __call__(self, image_tensor, class_index: int | None = None, meta_vector=None):
        """Return (heatmap HxW in 0-1, explained class index, class probabilities)."""
        import torch

        device = next(self.model.parameters()).device
        batch = image_tensor.unsqueeze(0).to(device).requires_grad_(True)
        metas = None if meta_vector is None else torch.as_tensor(meta_vector, device=device).float()[None]
        self.model.zero_grad(set_to_none=True)
        logits = self.model(batch, metas)
        probabilities = torch.softmax(logits, dim=1)[0].detach().cpu().numpy()
        if class_index is None:
            class_index = int(logits.argmax(1))
        logits[0, class_index].backward()

        activations, gradients = self._to_nchw(self._activations), self._to_nchw(self._gradients)
        weights = gradients.mean(dim=(2, 3), keepdim=True)              # alpha_k
        cam = torch.relu((weights * activations).sum(dim=1))[0]         # weighted sum of maps
        cam = torch.nn.functional.interpolate(cam[None, None], size=image_tensor.shape[1:],
                                              mode="bilinear", align_corners=False)[0, 0]
        cam = cam - cam.min()
        cam = cam / cam.max().clamp(min=1e-8)
        return cam.cpu().numpy(), class_index, probabilities

    def remove(self) -> None:
        """Detach the hooks so the model behaves normally again."""
        for hook in self._hooks:
            hook.remove()


def overlay_heatmap(image: Image.Image, heatmap: np.ndarray, alpha: float = 0.45) -> Image.Image:
    """Blend a 0-1 heatmap (blue = low, red = high) over the original photo."""
    import matplotlib

    colours = (matplotlib.colormaps["jet"](heatmap)[..., :3] * 255).astype(np.uint8)
    colour_image = Image.fromarray(colours).resize(image.size, Image.Resampling.BILINEAR)
    return Image.blend(image.convert("RGB"), colour_image, alpha)
