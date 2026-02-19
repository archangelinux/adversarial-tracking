"""Environmental degradation utilities for perception testing.

Simulates real-world environmental conditions that degrade object
detection and tracking performance: rain, fog, motion blur, lens
contamination, low light, and contrast reduction.
"""

import cv2
import numpy as np


def add_rain(image: np.ndarray, intensity: float = 0.5) -> np.ndarray:
    """Simulate rain streaks on an image.

    Args:
        image: Input BGR image.
        intensity: Rain intensity in [0.0, 1.0]. Higher values produce
            more and longer rain streaks.

    Returns:
        Image with simulated rain overlay.
    """
    h, w = image.shape[:2]
    rain_layer = np.zeros_like(image)

    # Number of rain drops scales with intensity
    num_drops = int(intensity * 1500)
    # Streak length scales with intensity
    streak_length = int(10 + intensity * 30)

    for _ in range(num_drops):
        x = np.random.randint(0, w)
        y = np.random.randint(0, h)
        # Rain falls mostly downward with slight wind drift
        dx = np.random.randint(-2, 3)
        dy = np.random.randint(streak_length // 2, streak_length)
        brightness = np.random.randint(180, 255)
        cv2.line(
            rain_layer,
            (x, y),
            (x + dx, min(y + dy, h - 1)),
            (brightness, brightness, brightness),
            1,
        )

    # Blur the rain layer for realism
    rain_layer = cv2.GaussianBlur(rain_layer, (3, 3), 0)

    # Blend with original
    alpha = 0.3 + intensity * 0.3
    result = cv2.addWeighted(image, 1.0, rain_layer, alpha, 0)

    # Slight overall darkening and blue tint for overcast sky
    darken = np.full_like(image, (30, 15, 10), dtype=np.uint8)
    darken_amount = intensity * 0.2
    result = cv2.addWeighted(result, 1.0 - darken_amount, darken, darken_amount, 0)

    return result


def add_fog(image: np.ndarray, intensity: float = 0.5) -> np.ndarray:
    """Simulate fog/haze on an image.

    Uses a depth-dependent fog model where distant objects (assumed at
    the top of the image) are more obscured.

    Args:
        image: Input BGR image.
        intensity: Fog density in [0.0, 1.0].

    Returns:
        Image with simulated fog.
    """
    h, w = image.shape[:2]

    # Create a depth-like gradient (top = far = more fog)
    depth_map = np.linspace(1.0, 0.3, h).reshape(h, 1)
    depth_map = np.broadcast_to(depth_map, (h, w))

    # Fog color (light gray)
    fog_color = np.full_like(image, (220, 220, 230), dtype=np.uint8)

    # Transmission map based on intensity and depth
    beta = intensity * 3.0  # Extinction coefficient
    transmission = np.exp(-beta * depth_map)
    transmission = np.clip(transmission, 0.1, 1.0)
    transmission = transmission[:, :, np.newaxis]

    # Apply atmospheric scattering model: I = I_0 * t + fog * (1 - t)
    result = (image.astype(np.float32) * transmission +
              fog_color.astype(np.float32) * (1.0 - transmission))

    # Add slight noise for texture
    noise = np.random.normal(0, intensity * 5, image.shape).astype(np.float32)
    result = np.clip(result + noise, 0, 255).astype(np.uint8)

    return result


def add_motion_blur(image: np.ndarray, kernel_size: int = 15) -> np.ndarray:
    """Apply directional motion blur.

    Args:
        image: Input BGR image.
        kernel_size: Size of the motion blur kernel. Larger values produce
            stronger blur. Must be odd.

    Returns:
        Motion-blurred image.
    """
    kernel_size = max(3, kernel_size | 1)  # Ensure odd

    # Horizontal motion blur kernel
    kernel = np.zeros((kernel_size, kernel_size))
    kernel[kernel_size // 2, :] = 1.0 / kernel_size

    # Add slight angle variation
    angle = np.random.uniform(-15, 15)
    center = (kernel_size // 2, kernel_size // 2)
    rot_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
    kernel = cv2.warpAffine(
        kernel, rot_matrix, (kernel_size, kernel_size),
        flags=cv2.INTER_LINEAR, borderValue=0,
    )

    # Normalize
    kernel_sum = kernel.sum()
    if kernel_sum > 0:
        kernel /= kernel_sum

    return cv2.filter2D(image, -1, kernel)


def add_lens_dirt(image: np.ndarray, num_spots: int = 10) -> np.ndarray:
    """Simulate lens contamination with dirt/water spots.

    Args:
        image: Input BGR image.
        num_spots: Number of dirt/smudge spots.

    Returns:
        Image with simulated lens contamination.
    """
    h, w = image.shape[:2]
    result = image.copy()

    for _ in range(num_spots):
        cx = np.random.randint(w // 6, 5 * w // 6)
        cy = np.random.randint(h // 6, 5 * h // 6)
        radius = np.random.randint(20, min(w, h) // 6)

        # Create a soft circular mask
        mask = np.zeros((h, w), dtype=np.float32)
        cv2.circle(mask, (cx, cy), radius, 1.0, -1)
        mask = cv2.GaussianBlur(mask, (0, 0), radius * 0.4)

        # Blend: blur the affected region and mix with original
        blurred = cv2.GaussianBlur(result, (0, 0), radius * 0.3)

        # Random tint for different types of contamination
        tint = np.random.choice(['brown', 'water', 'smudge'])
        if tint == 'brown':
            color_shift = np.array([10, 20, 40], dtype=np.float32)
        elif tint == 'water':
            color_shift = np.array([20, 15, 5], dtype=np.float32)
        else:
            color_shift = np.array([10, 10, 10], dtype=np.float32)

        mask_3c = mask[:, :, np.newaxis]
        result = (result.astype(np.float32) * (1.0 - mask_3c * 0.5) +
                  blurred.astype(np.float32) * mask_3c * 0.3 +
                  color_shift * mask_3c * 0.2)
        result = np.clip(result, 0, 255).astype(np.uint8)

    return result


def reduce_contrast(image: np.ndarray, factor: float = 0.5) -> np.ndarray:
    """Reduce image contrast.

    Args:
        image: Input BGR image.
        factor: Contrast reduction factor in [0.0, 1.0]. Lower values
            produce lower contrast.

    Returns:
        Contrast-reduced image.
    """
    factor = np.clip(factor, 0.0, 1.0)
    mean = np.mean(image, axis=(0, 1), keepdims=True).astype(np.float32)
    result = mean + factor * (image.astype(np.float32) - mean)
    return np.clip(result, 0, 255).astype(np.uint8)


def simulate_low_light(image: np.ndarray, intensity: float = 0.5) -> np.ndarray:
    """Simulate low-light (nighttime) conditions.

    Reduces brightness, adds sensor noise, and shifts color balance
    toward blue.

    Args:
        image: Input BGR image.
        intensity: Darkness intensity in [0.0, 1.0]. Higher values are
            darker.

    Returns:
        Low-light simulated image.
    """
    # Darken the image
    brightness_factor = max(0.05, 1.0 - intensity * 0.85)
    dark = (image.astype(np.float32) * brightness_factor)

    # Add sensor noise (more visible in dark conditions)
    noise_sigma = 5 + intensity * 30
    noise = np.random.normal(0, noise_sigma, image.shape).astype(np.float32)
    dark += noise

    # Blue color shift for night
    dark[:, :, 0] *= 1.0 + intensity * 0.15  # Blue channel boost
    dark[:, :, 2] *= 1.0 - intensity * 0.1   # Red channel reduce

    # Gamma correction to simulate camera auto-exposure
    gamma = 1.0 + intensity * 0.5
    dark = np.clip(dark, 0, 255)
    dark = 255.0 * (dark / 255.0) ** gamma

    return np.clip(dark, 0, 255).astype(np.uint8)


def apply_degradation(
    image: np.ndarray,
    degradation_type: str,
    intensity: float = 0.5,
    **kwargs,
) -> np.ndarray:
    """Apply a named degradation to an image.

    Convenience dispatcher for all degradation functions.

    Args:
        image: Input BGR image.
        degradation_type: One of "rain", "fog", "blur", "lens_dirt",
            "contrast", "low_light".
        intensity: Degradation strength in [0.0, 1.0].
        **kwargs: Additional keyword arguments passed to the specific function.

    Returns:
        Degraded image.

    Raises:
        ValueError: If degradation_type is not recognized.
    """
    dispatch = {
        'rain': lambda img: add_rain(img, intensity),
        'fog': lambda img: add_fog(img, intensity),
        'blur': lambda img: add_motion_blur(img, int(5 + intensity * 25)),
        'lens_dirt': lambda img: add_lens_dirt(img, int(3 + intensity * 15)),
        'contrast': lambda img: reduce_contrast(img, 1.0 - intensity),
        'low_light': lambda img: simulate_low_light(img, intensity),
    }

    if degradation_type not in dispatch:
        raise ValueError(
            f'Unknown degradation type: {degradation_type}. '
            f'Available: {list(dispatch.keys())}'
        )

    return dispatch[degradation_type](image)
