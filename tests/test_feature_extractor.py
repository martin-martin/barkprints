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


def test_transect_features_shape_and_determinism(sample_image):
    """Transect extraction returns one finite vector per step, reproducibly."""
    extractor = ImageFeatureExtractor(sample_image)

    transect = extractor.extract_transect_features(7, target_dim=384)
    assert transect.shape == (7, 384)
    assert np.all(np.isfinite(transect))

    again = ImageFeatureExtractor(sample_image).extract_transect_features(
        7, target_dim=384
    )
    np.testing.assert_array_equal(transect, again)


def test_transect_windows_differ_across_image(temp_dir):
    """Left and right windows of a left/right-split image give different features."""
    from PIL import Image
    import numpy as np

    # Left half dark and smooth, right half bright noise.
    array = np.zeros((120, 240, 3), dtype=np.uint8)
    array[:, 120:] = np.random.RandomState(0).randint(
        0, 255, (120, 120, 3), dtype=np.uint8
    )
    path = temp_dir / "split.png"
    Image.fromarray(array, mode="RGB").save(path)

    transect = ImageFeatureExtractor(path).extract_transect_features(5)
    assert not np.array_equal(transect[0], transect[-1])


def test_transect_single_step(sample_image):
    """steps=1 yields a single centered window."""
    transect = ImageFeatureExtractor(sample_image).extract_transect_features(1)
    assert transect.shape == (1, 384)
