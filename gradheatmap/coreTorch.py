"""PyTorch Grad-CAM implementation."""
from __future__ import annotations

import os
from typing import Any, Callable, Dict, Optional, Tuple, Union

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from ._utils import build_overlay, load_image_bgr, save_heatmap_image


class HeatMapPyTorch:
    """
    Grad-CAM heatmap generator for PyTorch classification models.

    Supports:
    - Binary classification (sigmoid single output)
    - Binary classification (two-class softmax)
    - Multiclass classification (softmax)
    - Automatic last Conv2d layer detection
    - ImageNet normalization for standard backbones

    Parameters
    ----------
    model:
        A ``torch.nn.Module`` already moved to the desired device and set to eval mode.
        Pass a string path to get a descriptive error — load the checkpoint yourself
        before calling this class.
    img_path:
        Path to the input image.
    class_names:
        Human-readable class labels.  Used only for display.
    target_class:
        Class index to explain.  ``None`` (default) uses the predicted class.
    preprocess:
        Preprocessing override.  Options:

        - ``None`` (default) — no ImageNet normalization; pixels scaled to [0, 1]
        - A callable ``fn(img_uint8_rgb: np.ndarray) -> torch.Tensor`` that returns
          a float tensor of shape ``(1, C, H, W)``
        - A string key from the supported backbone list (applies ImageNet
          mean/std normalization): ``"resnet"``, ``"vgg16"``, ``"efficientnet"``, etc.
    image_size:
        ``(height, width)`` to resize to before inference.  Default ``(224, 224)``.
    device:
        Torch device.  Auto-detected (CUDA if available, else CPU) when ``None``.
    """

    _IMAGENET_KEYS = frozenset({
        "vgg16", "vgg19", "resnet", "resnet50", "resnet101", "resnet152",
        "mobilenet", "mobilenetv2", "mobilenetv3",
        "efficientnet", "efficientnetv2",
        "densenet", "inception", "xception", "convnext", "regnet",
    })

    def __init__(
        self,
        model: nn.Module,
        img_path: str,
        class_names: Optional[list] = None,
        target_class: Optional[int] = None,
        preprocess: Optional[Union[str, Callable]] = None,
        image_size: Optional[Tuple[int, int]] = None,
        device: Optional[torch.device] = None,
    ) -> None:
        if isinstance(model, str):
            raise ValueError(
                "Pass a torch.nn.Module (already built and weights loaded). "
                "If you have a .pth/.pt checkpoint, load it into a model first:\n"
                "  model.load_state_dict(torch.load('ckpt.pth')['model_state_dict'])\n"
                "Then pass the model object here."
            )

        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = model.to(self.device).eval()

        self.img_path = img_path
        self.class_names = class_names if class_names is not None else []
        self.target_class = target_class
        self.user_preprocess = preprocess
        self.image_size: Tuple[int, int] = image_size if image_size is not None else (224, 224)

        # Hook buffers
        self._gradients: Optional[torch.Tensor] = None
        self._activations: Optional[torch.Tensor] = None

        self.target_layer, self.target_layer_name = self._find_last_conv()
        print(f"Targeting layer : {self.target_layer_name}")
        print(f"Image size      : {self.image_size}")

    # ------------------------------------------------------------------
    # Layer detection
    # ------------------------------------------------------------------

    def _find_last_conv(self) -> Tuple[nn.Module, str]:
        """Walk named_modules in order; return the last Conv2d found."""
        last_conv: Optional[nn.Module] = None
        last_name: Optional[str] = None
        for name, module in self.model.named_modules():
            if isinstance(module, nn.Conv2d):
                last_conv = module
                last_name = name
        if last_conv is None:
            raise ValueError(
                "No Conv2d layer found in the model. "
                "Grad-CAM requires at least one convolutional layer."
            )
        return last_conv, last_name

    # ------------------------------------------------------------------
    # Forward / backward hooks
    # ------------------------------------------------------------------

    def _forward_hook(
        self, module: nn.Module, inputs: Any, output: torch.Tensor
    ) -> None:
        self._activations = output

    def _backward_hook(
        self, module: nn.Module, grad_input: Any, grad_output: Tuple[torch.Tensor, ...]
    ) -> None:
        self._gradients = grad_output[0]

    # ------------------------------------------------------------------
    # Preprocessing
    # ------------------------------------------------------------------

    def _imagenet_norm(self, x: torch.Tensor) -> torch.Tensor:
        """Apply ImageNet mean/std normalization to a [0,1] tensor (1,3,H,W)."""
        mean = torch.tensor([0.485, 0.456, 0.406], device=x.device).view(1, 3, 1, 1)
        std  = torch.tensor([0.229, 0.224, 0.225], device=x.device).view(1, 3, 1, 1)
        return (x - mean) / std

    def _resolve_preprocess(self) -> Optional[Union[str, Callable]]:
        """
        Return ``"imagenet"``, a callable, or ``None``.

        - User callable → returned as-is
        - User string that maps to a known backbone → ``"imagenet"``
        - None → no special normalization
        """
        p = self.user_preprocess

        if callable(p):
            print("Using user-supplied preprocessing callable.")
            return p

        if isinstance(p, str):
            key = p.lower().strip()
            if key not in self._IMAGENET_KEYS:
                raise ValueError(
                    f"Unknown preprocess='{p}'. "
                    f"Valid string keys: {sorted(self._IMAGENET_KEYS)}"
                )
            print(f"Using ImageNet normalization (preprocess='{p}').")
            return "imagenet"

        return None

    def preprocess_image(self) -> torch.Tensor:
        """
        Load the input image and convert it to a model-ready tensor.

        Returns
        -------
        torch.Tensor
            Shape ``(1, C, H, W)``, float32, on ``self.device``.
        """
        img_bgr = load_image_bgr(self.img_path)

        # Detect input channels from the first Conv2d
        in_channels = 3
        for m in self.model.modules():
            if isinstance(m, nn.Conv2d):
                in_channels = m.in_channels
                break

        H, W = self.image_size
        p = self._resolve_preprocess()

        if in_channels == 1:
            img_gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
            img_gray = cv2.resize(img_gray, (W, H))
            x = torch.from_numpy(img_gray).float().unsqueeze(0).unsqueeze(0) / 255.0
            return x.to(self.device)

        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        img_rgb = cv2.resize(img_rgb, (W, H))

        if callable(p):
            out = p(img_rgb)
            if not isinstance(out, torch.Tensor):
                raise TypeError(
                    "Your preprocess callable must return a torch.Tensor, "
                    f"got {type(out).__name__}."
                )
            if out.ndim != 4:
                raise ValueError(
                    f"Your preprocess callable must return shape (1, C, H, W), "
                    f"got {tuple(out.shape)}."
                )
            return out.to(self.device)

        x = torch.from_numpy(img_rgb).float().permute(2, 0, 1).unsqueeze(0) / 255.0
        x = x.to(self.device)
        if p == "imagenet":
            x = self._imagenet_norm(x)
        return x

    # ------------------------------------------------------------------
    # Prediction
    # ------------------------------------------------------------------

    def predict(
        self, x: torch.Tensor
    ) -> Tuple[int, str, float, Dict[str, float], torch.Tensor]:
        """
        Forward pass; return (class_index, name, confidence, prob_dict, logits).

        Handles sigmoid (output dim 1) and softmax (output dim ≥ 2).

        .. note::
            Gradients must flow through this call — do **not** wrap with
            ``torch.no_grad()``.
        """
        logits = self.model(x)
        if logits.ndim == 1:
            logits = logits.unsqueeze(0)

        if logits.shape[-1] == 1:
            prob_pos = torch.sigmoid(logits.detach()).view(-1)[0].item()
            idx = 1 if prob_pos >= 0.5 else 0
            name = self.class_names[idx] if len(self.class_names) > idx else str(idx)
            conf = prob_pos if idx == 1 else 1.0 - prob_pos
            probs: Dict[str, float] = {
                (self.class_names[0] if len(self.class_names) > 0 else "0"): round(1.0 - prob_pos, 4),
                (self.class_names[1] if len(self.class_names) > 1 else "1"): round(prob_pos, 4),
            }
        else:
            probs_t = torch.softmax(logits.detach(), dim=1)[0]
            idx = int(torch.argmax(probs_t).item())
            conf = float(probs_t[idx].item())
            name = self.class_names[idx] if idx < len(self.class_names) else str(idx)
            probs = {
                (self.class_names[i] if i < len(self.class_names) else str(i)):
                round(float(probs_t[i].item()), 4)
                for i in range(probs_t.shape[0])
            }

        print(f"Predicted: {idx} ({name})  Confidence: {conf * 100:.2f}%")
        return idx, name, conf, probs, logits

    # ------------------------------------------------------------------
    # Grad-CAM
    # ------------------------------------------------------------------

    def compute_gradcam(self) -> np.ndarray:
        """
        Run Grad-CAM and return the raw 2-D heatmap.

        Returns
        -------
        np.ndarray
            2-D float array with values in [0, 1], shape (H_cam, W_cam).
        """
        x = self.preprocess_image()
        x.requires_grad_(True)

        f_hook = self.target_layer.register_forward_hook(self._forward_hook)
        try:
            b_hook = self.target_layer.register_full_backward_hook(self._backward_hook)
        except AttributeError:
            b_hook = self.target_layer.register_backward_hook(self._backward_hook)

        try:
            idx, _name, _conf, _probs, logits = self.predict(x)

            if logits.shape[-1] == 1:
                explain_class = int(self.target_class) if self.target_class is not None else idx
                score = logits[:, 0] if explain_class == 1 else -logits[:, 0]
            else:
                class_index = (
                    int(self.target_class)
                    if self.target_class is not None
                    else int(torch.argmax(logits, dim=1).item())
                )
                score = logits[:, class_index]

            self.model.zero_grad(set_to_none=True)
            score.backward(retain_graph=False)

            if self._gradients is None or self._activations is None:
                raise ValueError(
                    "Gradients or activations are None after backward pass. "
                    "The targeted layer may be disconnected from the output. "
                    "Try a different layer or check the model architecture."
                )

            # Global-average-pool gradients over spatial dims → (C,)
            pooled = torch.mean(self._gradients, dim=(0, 2, 3))

            # Weight feature maps and average → (H_cam, W_cam)
            weighted = self._activations.detach().clone()
            for c in range(weighted.shape[1]):
                weighted[:, c, :, :] *= pooled[c]

            heatmap = torch.mean(weighted, dim=1).squeeze(0)
            heatmap = F.relu(heatmap)

            maxv = torch.max(heatmap)
            if maxv > 0:
                heatmap = heatmap / maxv

            return heatmap.detach().cpu().numpy()

        finally:
            f_hook.remove()
            b_hook.remove()
            # Reset hook buffers so the object can be reused
            self._gradients = None
            self._activations = None

    def overlay_heatmap(self, alpha: float = 0.4) -> np.ndarray:
        """
        Compute Grad-CAM and return a BGR overlay image.

        Parameters
        ----------
        alpha:
            Weight of the original image (0 = heatmap only, 1 = original only).

        Returns
        -------
        np.ndarray
            BGR uint8 image with the heatmap blended over the original.
        """
        heatmap = self.compute_gradcam()
        img_bgr = load_image_bgr(self.img_path)
        return build_overlay(heatmap, img_bgr, alpha=alpha)

    def save_heat_img(self, name: str, output_img: np.ndarray, folder: str = "heatmap") -> str:
        """
        Save *output_img* to ``{folder}/{name}``.

        Parameters
        ----------
        name:
            Filename.  ``.jpg`` appended if no extension is given.
        output_img:
            BGR image returned by :meth:`overlay_heatmap`.
        folder:
            Output directory (default ``"heatmap"``).

        Returns
        -------
        str
            Full path written.
        """
        return save_heatmap_image(output_img, name, folder=folder)
