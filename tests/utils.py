"""Utility functions for testing VITools.

This module provides helper functions for creating phantoms and other test data
used across the test suite.
"""
import numpy as np


def create_circle_phantom(
    image_size=512,
    large_radius_ratio=0.85,  # Ratio of image_size / 2
    inner_radius_ratio=0.55,  # Ratio of large_radius
    small_radius_ratio=0.15,  # Ratio of large_radius
    num_small_circles=6,
    bg_value=0.0,         # Background intensity
    large_circle_value=0.5,  # Main circle intensity
    small_circle_values=None,  # List/array of intensities for small circles
    start_angle_offset_deg=0
):
    """Generates a NumPy array representing a circular phantom.

    Creates a large circle containing smaller circles placed equidistantly
    on an inner radius. This is useful for creating consistent, simple phantoms
    for testing purposes.

    Args:
        image_size (int): The width and height of the output image in pixels.
        large_radius_ratio (float): Radius of the large circle as a fraction
            of half the image size.
        inner_radius_ratio (float): Radius for placing small circle centers,
            as a fraction of the large circle's radius.
        small_radius_ratio (float or list or np.ndarray, optional): Radius of
            the small circles, as a fraction of the large circle's radius.
            If float, a constant radius is used. If list, it must have a
            length equal to `num_small_circles`.
        num_small_circles (int): The number of smaller circles inside.
        bg_value (float): The intensity value for the background.
        large_circle_value (float): The intensity value for the large circle.
        small_circle_values (list or np.ndarray, optional):
            A list or array of intensity values for the small circles.
            The length must match `num_small_circles`. If None, defaults
            to a range of values from 0.1 to 0.9.
        start_angle_offset_deg (float): Rotates the starting position of the
            first small circle.

    Returns:
        np.ndarray: A 2D NumPy array representing the generated image.

    Raises:
        ValueError: If the length of `small_circle_values` or
            `small_radius_ratio` does not match `num_small_circles`.
    """
    # --- Parameter Validation and Setup ---
    if small_circle_values is None:
        small_circle_values = np.linspace(0.1, 0.9, num_small_circles)
    elif len(small_circle_values) != num_small_circles:
        raise ValueError(f"Length of small_circle_values ({len(small_circle_values)}) "
                         f"must match num_small_circles ({num_small_circles})")
    if isinstance(small_radius_ratio, float):
        small_radius_ratio = [small_radius_ratio] * num_small_circles
    elif len(small_radius_ratio) != num_small_circles:
        raise ValueError(f"Length of small_radius_ratio ({len(small_radius_ratio)}) "
                         f"must match num_small_circles ({num_small_circles})")

    # --- Calculate Dimensions ---
    center_x, center_y = image_size / 2, image_size / 2
    large_radius = large_radius_ratio * (image_size / 2)
    inner_radius = inner_radius_ratio * large_radius

    # --- Create Coordinate Grid ---
    x = np.arange(0, image_size)
    y = np.arange(0, image_size)
    xx, yy = np.meshgrid(x, y)
    dist_from_center = np.sqrt((xx - center_x)**2 + (yy - center_y)**2)

    # --- Initialize Image ---
    image = np.full((image_size, image_size), bg_value, dtype=np.float32)

    # --- Draw Large Circle ---
    large_circle_mask = dist_from_center <= large_radius
    image[large_circle_mask] = large_circle_value

    # --- Draw Small Circles ---
    angles = np.linspace(0, 2 * np.pi, num_small_circles, endpoint=False)
    angles += np.deg2rad(start_angle_offset_deg)

    for i, angle in enumerate(angles):
        small_cx = center_x + inner_radius * np.cos(angle)
        small_cy = center_y + inner_radius * np.sin(angle)
        dist_from_small_center = np.sqrt((xx - small_cx)**2 + (yy - small_cy)**2)
        small_radius = small_radius_ratio[i] * large_radius
        small_circle_mask = dist_from_small_center <= small_radius
        image[small_circle_mask & large_circle_mask] = small_circle_values[i]

    return image