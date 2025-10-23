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
