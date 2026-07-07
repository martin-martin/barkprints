"""Tests for feature extraction."""

import numpy as np

from barkprints.feature_extractor import ImageFeatureExtractor


def test_feature_extraction(sample_image):
    """Test that features can be extracted from an image."""
    extractor = ImageFeatureExtractor(sample_image)
    features = extractor.extract_features()
    
    # Check that features is a numpy array
    assert isinstance(features, np.ndarray)
    
    # Check default dimension (384)
    assert len(features) == 384
    
    # Check that all features are in [-1, 1] range
    assert np.all(features >= -1.0)
    assert np.all(features <= 1.0)
    
    # Check that all features are finite numbers
    assert np.all(np.isfinite(features))


def test_feature_extraction_custom_dim(sample_image):
    """Test feature extraction with custom dimensions."""
    extractor = ImageFeatureExtractor(sample_image)
    
    # Test different dimensions
    for dim in [128, 384, 768]:
        features = extractor.extract_features(target_dim=dim)
        assert len(features) == dim
        assert np.all(np.isfinite(features))


def test_deterministic_features(sample_image):
    """Test that the same image produces the same features."""
    extractor1 = ImageFeatureExtractor(sample_image)
    features1 = extractor1.extract_features()
    
    extractor2 = ImageFeatureExtractor(sample_image)
    features2 = extractor2.extract_features()
    
    np.testing.assert_array_equal(features1, features2)


def test_different_images_different_features(sample_image, sample_image_2):
    """Test that different images produce different features."""
    extractor1 = ImageFeatureExtractor(sample_image)
    features1 = extractor1.extract_features()

    extractor2 = ImageFeatureExtractor(sample_image_2)
    features2 = extractor2.extract_features()

    # Features should be different (with very high probability)
    assert not np.array_equal(features1, features2)


def test_extract_spatial_grid_shape_and_range(sample_image):
    """Test the per-cell spatial grid has the expected shape and is well-formed."""
    extractor = ImageFeatureExtractor(sample_image)
    grid = extractor.extract_spatial_grid()

    assert grid.shape == (100, 4)
    assert np.all(np.isfinite(grid))
    # std (col 1) and edge_density (col 3) can't be negative.
    assert np.all(grid[:, 1] >= 0)
    assert np.all(grid[:, 3] >= 0)


def test_extract_spatial_grid_flat_image(sample_image_2):
    """A uniform (zero-variance) image shouldn't produce NaN/inf from the guard."""
    extractor = ImageFeatureExtractor(sample_image_2)
    grid = extractor.extract_spatial_grid()

    assert np.all(np.isfinite(grid))


def test_extract_spatial_grid_different_images_differ(sample_image, sample_image_2):
    """Different images should produce different spatial grids."""
    grid1 = ImageFeatureExtractor(sample_image).extract_spatial_grid()
    grid2 = ImageFeatureExtractor(sample_image_2).extract_spatial_grid()

    assert not np.array_equal(grid1, grid2)


def test_extract_spatial_grid_deterministic(sample_image):
    """Same image should always produce the same spatial grid."""
    grid1 = ImageFeatureExtractor(sample_image).extract_spatial_grid()
    grid2 = ImageFeatureExtractor(sample_image).extract_spatial_grid()

    np.testing.assert_array_equal(grid1, grid2)
