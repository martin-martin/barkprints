"""Tests for feature extraction."""

import numpy as np

from barkprints.feature_extractor import ImageFeatureExtractor


def test_feature_extraction(sample_image):
    """Test that features can be extracted from an image."""
    extractor = ImageFeatureExtractor(sample_image)
    features = extractor.extract_features()
    
    # Check that features is a numpy array
    assert isinstance(features, np.ndarray)
    
    # Check that we got a reasonable number of features
    assert len(features) > 0
    
    # Check that all features are finite numbers
    assert np.all(np.isfinite(features))


def test_deterministic_seed(sample_image):
    """Test that the same image produces the same seed."""
    extractor1 = ImageFeatureExtractor(sample_image)
    seed1 = extractor1.get_deterministic_seed()
    
    extractor2 = ImageFeatureExtractor(sample_image)
    seed2 = extractor2.get_deterministic_seed()
    
    assert seed1 == seed2
    assert isinstance(seed1, int)


def test_different_images_different_seeds(sample_image, sample_image_2):
    """Test that different images produce different seeds."""
    extractor1 = ImageFeatureExtractor(sample_image)
    seed1 = extractor1.get_deterministic_seed()
    
    extractor2 = ImageFeatureExtractor(sample_image_2)
    seed2 = extractor2.get_deterministic_seed()
    
    # Different images should (almost certainly) produce different seeds
    assert seed1 != seed2


def test_feature_vector_consistency(sample_image):
    """Test that feature extraction is consistent across multiple calls."""
    extractor = ImageFeatureExtractor(sample_image)
    
    features1 = extractor.extract_features()
    features2 = extractor.extract_features()
    
    np.testing.assert_array_equal(features1, features2)

