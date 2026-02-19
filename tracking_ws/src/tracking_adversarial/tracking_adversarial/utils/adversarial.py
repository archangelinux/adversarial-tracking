"""Adversarial attack utilities for perception testing.

Implements adversarial patches, geometric patterns, and occlusion attacks
that target object detection and tracking systems. These are applied to
specific bounding box regions to simulate physically realizable attacks.
"""

import cv2
import numpy as np
from typing import List, Optional, Tuple


def generate_adversarial_patch(
    patch_size: int = 50,
    pattern: str = 'noise',
) -> np.ndarray:
    """Generate an adversarial patch image.

    Creates a high-frequency, high-contrast patch designed to confuse
    neural network feature extractors. These simulate physically
    printable adversarial patches.

    Args:
        patch_size: Width and height of the square patch in pixels.
        pattern: Patch pattern type. One of "noise", "gradient", "spiral".

    Returns:
        BGR patch image of shape (patch_size, patch_size, 3).
    """
    if pattern == 'noise':
        # High-frequency noise — mimics universal adversarial perturbations
        patch = np.random.randint(0, 256, (patch_size, patch_size, 3), dtype=np.uint8)
        # Add structured high-contrast edges
        for i in range(0, patch_size, 5):
            color = [np.random.randint(0, 256) for _ in range(3)]
            cv2.line(patch, (i, 0), (patch_size - i, patch_size), color, 2)
    elif pattern == 'gradient':
        # Multi-directional gradient
        patch = np.zeros((patch_size, patch_size, 3), dtype=np.uint8)
        for c in range(3):
            grad = np.linspace(0, 255, patch_size).astype(np.uint8)
            if c == 0:
                patch[:, :, c] = np.tile(grad, (patch_size, 1))
            elif c == 1:
                patch[:, :, c] = np.tile(grad.reshape(-1, 1), (1, patch_size))
            else:
                diag = np.fromfunction(
                    lambda i, j: ((i + j) / (2 * patch_size) * 255),
                    (patch_size, patch_size),
                ).astype(np.uint8)
                patch[:, :, c] = diag
    elif pattern == 'spiral':
        # Spiral pattern — high spatial frequency
        patch = np.zeros((patch_size, patch_size, 3), dtype=np.uint8)
        center = patch_size // 2
        for y in range(patch_size):
            for x in range(patch_size):
                dx = x - center
                dy = y - center
                angle = np.arctan2(dy, dx)
                dist = np.sqrt(dx * dx + dy * dy)
                val = int(127 + 127 * np.sin(angle * 8 + dist * 0.3))
                patch[y, x] = [val, 255 - val, (val + 128) % 256]
    else:
        patch = np.random.randint(0, 256, (patch_size, patch_size, 3), dtype=np.uint8)

    return patch


def apply_adversarial_patch(
    image: np.ndarray,
    bbox: Tuple[int, int, int, int],
    patch: np.ndarray,
    alpha: float = 0.8,
) -> np.ndarray:
    """Place an adversarial patch on a detected object.

    Args:
        image: Input BGR image.
        bbox: Bounding box as (x1, y1, x2, y2).
        patch: Patch image to apply.
        alpha: Blending factor for the patch overlay.

    Returns:
        Image with the adversarial patch applied.
    """
    result = image.copy()
    x1, y1, x2, y2 = bbox
    h, w = image.shape[:2]

    # Clamp bbox to image bounds
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w, x2), min(h, y2)
    bw, bh = x2 - x1, y2 - y1
    if bw <= 0 or bh <= 0:
        return result

    # Resize patch to fit the bounding box
    resized_patch = cv2.resize(patch, (bw, bh), interpolation=cv2.INTER_LINEAR)

    # Blend patch onto image
    roi = result[y1:y2, x1:x2]
    blended = cv2.addWeighted(roi, 1.0 - alpha, resized_patch, alpha, 0)
    result[y1:y2, x1:x2] = blended

    return result


def apply_stripe_pattern(
    image: np.ndarray,
    bbox: Tuple[int, int, int, int],
    stripe_width: int = 10,
    color1: Tuple[int, int, int] = (0, 0, 0),
    color2: Tuple[int, int, int] = (255, 255, 255),
) -> np.ndarray:
    """Apply alternating stripe pattern to a bounding box region.

    Stripes are known to confuse HOG-based and some CNN-based detectors
    by disrupting gradient histograms.

    Args:
        image: Input BGR image.
        bbox: Bounding box as (x1, y1, x2, y2).
        stripe_width: Width of each stripe in pixels.
        color1: First stripe color (BGR).
        color2: Second stripe color (BGR).

    Returns:
        Image with stripe pattern applied.
    """
    result = image.copy()
    x1, y1, x2, y2 = bbox
    h, w = image.shape[:2]
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w, x2), min(h, y2)

    if x2 - x1 <= 0 or y2 - y1 <= 0:
        return result

    # Create stripe pattern
    roi = result[y1:y2, x1:x2].copy()
    for i in range(0, roi.shape[1], stripe_width * 2):
        end = min(i + stripe_width, roi.shape[1])
        roi[:, i:end] = color1
        end2 = min(i + stripe_width * 2, roi.shape[1])
        roi[:, end:end2] = color2

    # Blend with original at 70% opacity
    result[y1:y2, x1:x2] = cv2.addWeighted(
        result[y1:y2, x1:x2], 0.3, roi, 0.7, 0
    )

    return result


def apply_checkerboard_pattern(
    image: np.ndarray,
    bbox: Tuple[int, int, int, int],
    square_size: int = 15,
) -> np.ndarray:
    """Apply checkerboard pattern to a bounding box region.

    Checkerboard patterns create high-frequency features that can
    overwhelm detection feature maps.

    Args:
        image: Input BGR image.
        bbox: Bounding box as (x1, y1, x2, y2).
        square_size: Size of each checkerboard square.

    Returns:
        Image with checkerboard pattern applied.
    """
    result = image.copy()
    x1, y1, x2, y2 = bbox
    h, w = image.shape[:2]
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w, x2), min(h, y2)

    bw, bh = x2 - x1, y2 - y1
    if bw <= 0 or bh <= 0:
        return result

    # Create checkerboard
    checker = np.zeros((bh, bw, 3), dtype=np.uint8)
    for r in range(0, bh, square_size):
        for c in range(0, bw, square_size):
            if ((r // square_size) + (c // square_size)) % 2 == 0:
                r_end = min(r + square_size, bh)
                c_end = min(c + square_size, bw)
                checker[r:r_end, c:c_end] = (255, 255, 255)

    # Blend with original
    result[y1:y2, x1:x2] = cv2.addWeighted(
        result[y1:y2, x1:x2], 0.4, checker, 0.6, 0
    )

    return result


def simulate_occlusion(
    image: np.ndarray,
    bbox: Tuple[int, int, int, int],
    occlusion_ratio: float = 0.3,
    occlusion_type: str = 'solid',
) -> np.ndarray:
    """Simulate partial occlusion of an object.

    Covers a portion of the bounding box to simulate physical occlusion
    (e.g., behind a pole, another object, or a sign).

    Args:
        image: Input BGR image.
        bbox: Bounding box as (x1, y1, x2, y2).
        occlusion_ratio: Fraction of bbox to occlude [0.0, 1.0].
        occlusion_type: "solid" for solid color, "blur" for heavy blur,
            "noise" for random noise.

    Returns:
        Image with simulated occlusion.
    """
    result = image.copy()
    x1, y1, x2, y2 = bbox
    h, w = image.shape[:2]
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w, x2), min(h, y2)

    bw, bh = x2 - x1, y2 - y1
    if bw <= 0 or bh <= 0:
        return result

    # Determine occlusion region (random side)
    occ_w = int(bw * occlusion_ratio)
    occ_h = bh
    side = np.random.choice(['left', 'right', 'top', 'bottom'])

    if side == 'left':
        ox1, oy1 = x1, y1
        ox2, oy2 = x1 + occ_w, y2
    elif side == 'right':
        ox1, oy1 = x2 - occ_w, y1
        ox2, oy2 = x2, y2
    elif side == 'top':
        occ_h = int(bh * occlusion_ratio)
        ox1, oy1 = x1, y1
        ox2, oy2 = x2, y1 + occ_h
    else:
        occ_h = int(bh * occlusion_ratio)
        ox1, oy1 = x1, y2 - occ_h
        ox2, oy2 = x2, y2

    # Apply occlusion
    if occlusion_type == 'solid':
        # Dark gray solid occluder
        result[oy1:oy2, ox1:ox2] = (64, 64, 64)
    elif occlusion_type == 'blur':
        roi = result[oy1:oy2, ox1:ox2]
        if roi.size > 0:
            result[oy1:oy2, ox1:ox2] = cv2.GaussianBlur(roi, (0, 0), 20)
    elif occlusion_type == 'noise':
        noise = np.random.randint(0, 256, (oy2 - oy1, ox2 - ox1, 3), dtype=np.uint8)
        result[oy1:oy2, ox1:ox2] = noise

    return result


def apply_adversarial_attack(
    image: np.ndarray,
    attack_type: str,
    bboxes: Optional[List[Tuple[int, int, int, int]]] = None,
    intensity: float = 0.5,
    **kwargs,
) -> np.ndarray:
    """Apply an adversarial attack to an image.

    Convenience dispatcher for all adversarial attack functions.

    Args:
        image: Input BGR image.
        attack_type: One of "patch", "stripe", "checkerboard", "occlusion".
        bboxes: List of bounding boxes (x1, y1, x2, y2) to attack.
            If None, attacks are applied globally.
        intensity: Attack intensity in [0.0, 1.0].
        **kwargs: Additional arguments passed to specific attack functions.

    Returns:
        Attacked image.

    Raises:
        ValueError: If attack_type is not recognized.
    """
    result = image.copy()

    if bboxes is None or len(bboxes) == 0:
        # Apply globally if no bboxes specified
        h, w = image.shape[:2]
        bboxes = [(0, 0, w, h)]

    patch_size = kwargs.get('patch_size', 50)
    stripe_width = kwargs.get('stripe_width', 10)
    checkerboard_size = kwargs.get('checkerboard_size', 15)
    occlusion_ratio = kwargs.get('occlusion_ratio', 0.3)

    for bbox in bboxes:
        if attack_type == 'patch':
            patch = generate_adversarial_patch(patch_size, pattern='noise')
            result = apply_adversarial_patch(result, bbox, patch, alpha=intensity)
        elif attack_type == 'stripe':
            result = apply_stripe_pattern(result, bbox, stripe_width)
        elif attack_type == 'checkerboard':
            result = apply_checkerboard_pattern(result, bbox, checkerboard_size)
        elif attack_type == 'occlusion':
            result = simulate_occlusion(result, bbox, occlusion_ratio)
        else:
            raise ValueError(
                f'Unknown attack type: {attack_type}. '
                f'Available: patch, stripe, checkerboard, occlusion'
            )

    return result
