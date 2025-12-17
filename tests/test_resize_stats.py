
import numpy as np
import pytest
from VITools.phantom import resize

def test_resize_shape_longest():
    """Test that resize correctly scales the longest dimension."""
    # 100x50x25
    img = np.zeros((100, 50, 25))
    target_max = 50
    # Expected scale 0.5 -> 50x25x12 (roughly)
    resized = resize(img, (target_max, target_max, target_max))
    
    assert resized.shape[0] == target_max
    # Allow small rounding differences
    assert 24 <= resized.shape[1] <= 26
    assert 11 <= resized.shape[2] <= 13

def test_resize_preserves_range():
    """Test that resizing preserves pixel intensity range (min/max)."""
    img = np.random.rand(50, 50, 50) * 1000
    # Add outliers
    img[0,0,0] = -1000
    img[49,49,49] = 2000
    
    # Resize with default (linear)
    resized = resize(img, (25, 25, 25), order=1)
    
    # Linear interpolation normally stays within convex hull of neighbors, 
    # but can slightly overshoot with cubic. 
    # Checking if it's "close enough".
    
    # Tolerances might need adjustment depending on order.
    # Linear should be strictly bounded? Actually bilinear can be bounded if 0-1 coeff sum to 1.
    assert resized.min() >= img.min() - 10 # small epsilon
    assert resized.max() <= img.max() + 10

def test_resize_statistics():
    """Test that resizing doe not greatly impact pixel statistics."""
    # Uniform block
    img = np.ones((100, 100, 100)) * 100
    resized = resize(img, (50, 50, 50))
    
    assert np.isclose(resized.mean(), 100, atol=1)
    assert np.isclose(resized.std(), 0, atol=1)
    
    # Gradient
    x, y, z = np.indices((100, 100, 100))
    img = x.astype(float)
    resized = resize(img, (50, 50, 50))
    
    # Use loose bounds as downsampling changes distribution slightly
    assert abs(resized.mean() - img.mean()) < 5
    
def test_resize_nearest_mode():
    """Test mode='nearest' uses nearest neighbor interpolation."""
    img = np.zeros((10, 10, 10))
    img[0:5, :, :] = 0
    img[5:, :, :] = 100
    
    # Resize, should remain binary-ish if nearest, no 50s
    resized = resize(img, (5, 5, 5), mode='nearest')
    
    unique_vals = np.unique(resized)
    assert np.all(np.isin(unique_vals, [0, 100]))
    
