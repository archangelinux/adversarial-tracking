"""Gradient-based adversarial attacks for YOLO detection models.

Implements FGSM and PGD attacks against the YOLO detection head. Bypasses
NMS (not differentiable) by calling the inner nn.Module directly and
optimizing against the sum of detection confidences across all anchor
positions, keeping the full computation graph differentiable.

Supports both YOLOv8 and YOLO26. YOLO26 uses an end-to-end NMS-free design
whose one-to-one head explicitly detaches gradients. The wrapper disables
end2end mode so it falls back to the one-to-many head, which produces raw
predictions in the same format as YOLOv8 (batch, 4+nc, num_anchors) with
gradients intact.

Custom implementation rather than torchattacks because torchattacks targets
classifiers, not multi-anchor detection models.

References:
    Goodfellow et al., "Explaining and Harnessing Adversarial Examples" (FGSM)
    Madry et al., "Towards Deep Learning Models Resistant to Adversarial Attacks" (PGD)
"""

import cv2
import numpy as np
import torch
import torch.nn as nn


def numpy_to_tensor(image: np.ndarray, device: str = 'cpu') -> torch.Tensor:
    """Convert BGR uint8 image to normalized RGB float32 tensor (1,3,H,W).

    Pads to a multiple of 32 for YOLO stride compatibility.
    """
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    tensor = torch.from_numpy(rgb).permute(2, 0, 1).float() / 255.0
    tensor = tensor.unsqueeze(0).to(device)
    _, _, h, w = tensor.shape
    pad_h = (32 - h % 32) % 32
    pad_w = (32 - w % 32) % 32
    if pad_h > 0 or pad_w > 0:
        tensor = torch.nn.functional.pad(tensor, (0, pad_w, 0, pad_h))
    return tensor


def tensor_to_numpy(tensor: torch.Tensor) -> np.ndarray:
    """Convert normalized RGB float32 tensor back to BGR uint8 image.

    Inverse of numpy_to_tensor. Clamps to [0, 1] to handle perturbation overflow.
    """
    img = tensor.squeeze(0).detach().cpu()
    img = torch.clamp(img, 0.0, 1.0)
    img = (img.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
    return cv2.cvtColor(img, cv2.COLOR_RGB2BGR)


class YOLOAttackWrapper(nn.Module):
    """Differentiable wrapper around YOLO models for gradient-based attacks.

    Bypasses NMS by calling the inner nn.Module directly and returning
    the sum of max-class confidences across all anchor positions as a
    scalar loss. Subtracting the gradient of this loss w.r.t. input
    pixels produces perturbations that suppress detections.

    Works with both YOLOv8 and YOLO26. YOLO26's default end-to-end head
    detaches gradients internally, so we disable it to use the one-to-many
    head which preserves the gradient graph.
    """

    def __init__(self, yolo_model):
        super().__init__()
        self.inner_model = yolo_model.model

        # YOLO26 end2end mode detaches gradients on the one-to-one head.
        # Disable it so we get raw one-to-many predictions with gradients.
        head = self.inner_model.model[-1]
        if hasattr(head, 'end2end'):
            head.end2end = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return sum of max-class confidences across all anchor positions.

        Raw predictions shape: [batch, 4 + num_classes, num_anchors].
        """
        raw_preds = self.inner_model(x)

        if isinstance(raw_preds, (list, tuple)):
            preds = raw_preds[0]
        else:
            preds = raw_preds

        class_confs = preds[:, 4:, :]
        max_confs = class_confs.max(dim=1).values
        return max_confs.sum()


def fgsm_attack(
    image: np.ndarray,
    model,
    epsilon: float = 4 / 255,
    device: str = 'cpu',
) -> np.ndarray:
    """Single-step FGSM attack. Perturbs each pixel by epsilon in the
    direction that minimizes detection confidence.

    Args:
        image: BGR uint8 input image.
        model: Ultralytics YOLO model object.
        epsilon: L-inf perturbation budget (4/255 = imperceptible, 16/255 = barely visible).
        device: PyTorch device.

    Returns:
        Adversarially perturbed BGR uint8 image.
    """
    orig_h, orig_w = image.shape[:2]
    wrapper = YOLOAttackWrapper(model)
    wrapper.eval()

    x = numpy_to_tensor(image, device)
    x.requires_grad_(True)

    confidence = wrapper(x)
    confidence.backward()

    # Perturb in the direction that decreases confidence
    x_adv = x.detach() - epsilon * x.grad.sign()
    x_adv = torch.clamp(x_adv, 0.0, 1.0)
    x_adv = x_adv[:, :, :orig_h, :orig_w]

    return tensor_to_numpy(x_adv)


def pgd_attack(
    image: np.ndarray,
    model,
    epsilon: float = 4 / 255,
    alpha: float = 1 / 255,
    steps: int = 10,
    device: str = 'cpu',
) -> np.ndarray:
    """Iterative PGD attack. Takes multiple gradient steps of size alpha,
    projecting back into the L-inf epsilon-ball after each step.

    Args:
        image: BGR uint8 input image.
        model: Ultralytics YOLO model object.
        epsilon: L-inf perturbation budget.
        alpha: Step size per iteration.
        steps: Number of gradient steps.
        device: PyTorch device.

    Returns:
        Adversarially perturbed BGR uint8 image.
    """
    orig_h, orig_w = image.shape[:2]
    wrapper = YOLOAttackWrapper(model)
    wrapper.eval()

    x_orig = numpy_to_tensor(image, device)
    x_adv = x_orig.clone().detach()

    for _ in range(steps):
        x_adv.requires_grad_(True)
        confidence = wrapper(x_adv)
        confidence.backward()

        x_adv = x_adv.detach() - alpha * x_adv.grad.sign()

        # Project back into epsilon-ball
        delta = torch.clamp(x_adv - x_orig, -epsilon, epsilon)
        x_adv = torch.clamp(x_orig + delta, 0.0, 1.0)

    x_adv = x_adv[:, :, :orig_h, :orig_w]
    return tensor_to_numpy(x_adv)


GRADIENT_ATTACK_CONFIGS = {
    'fgsm_light':  {'method': 'fgsm', 'epsilon': 4 / 255},
    'fgsm_heavy':  {'method': 'fgsm', 'epsilon': 16 / 255},
    'pgd_light':   {'method': 'pgd', 'epsilon': 4 / 255, 'alpha': 1 / 255, 'steps': 10},
    'pgd_heavy':   {'method': 'pgd', 'epsilon': 16 / 255, 'alpha': 2 / 255, 'steps': 20},
}


def apply_gradient_attack(
    image: np.ndarray,
    attack_type: str,
    model,
    device: str = 'cpu',
    **kwargs,
) -> np.ndarray:
    """Apply a gradient-based adversarial attack to an image.

    Dispatcher that follows the same pattern as apply_degradation()
    and apply_adversarial_attack(). Looks up the attack config from
    GRADIENT_ATTACK_CONFIGS and calls the appropriate function.

    Args:
        image: BGR uint8 input image.
        attack_type: One of "fgsm_light", "fgsm_heavy", "pgd_light", "pgd_heavy".
        model: Ultralytics YOLO model object (needed for gradient computation).
        device: PyTorch device string.
        **kwargs: Override any config values (epsilon, alpha, steps).

    Returns:
        Adversarially perturbed BGR uint8 image.

    Raises:
        ValueError: If attack_type is not recognized.
    """
    if attack_type not in GRADIENT_ATTACK_CONFIGS:
        raise ValueError(
            f'Unknown gradient attack: {attack_type}. '
            f'Available: {list(GRADIENT_ATTACK_CONFIGS.keys())}'
        )

    config = {**GRADIENT_ATTACK_CONFIGS[attack_type], **kwargs}
    method = config.pop('method')

    if method == 'fgsm':
        return fgsm_attack(image, model, device=device, **config)
    elif method == 'pgd':
        return pgd_attack(image, model, device=device, **config)
    else:
        raise ValueError(f'Unknown gradient method: {method}')
