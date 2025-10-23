"""Extract numerical features from images for deterministic text generation."""

import hashlib
from pathlib import Path

import numpy as np
from PIL import Image
from scipy import ndimage


class ImageFeatureExtractor:
    """Extract deterministic numerical features from images."""

    def __init__(self, image_path: str | Path):
        """Initialize with an image path.
        
        Args:
            image_path: Path to the image file
        """
        self.image_path = Path(image_path)
        self.image = Image.open(self.image_path)
        
    def extract_features(self) -> np.ndarray:
        """Extract comprehensive numerical features from the image.
        
        Returns:
            Feature vector as numpy array
        """
        # Convert to RGB if needed
        img = self.image.convert('RGB')
        img_array = np.array(img)
        
        features = []
        
        # Color statistics for each channel
        for channel in range(3):
            channel_data = img_array[:, :, channel].astype(np.float64)
            features.extend([
                np.mean(channel_data),
                np.std(channel_data),
                np.median(channel_data),
                np.percentile(channel_data, 25),
                np.percentile(channel_data, 75),
            ])
        
        # Grayscale for texture analysis
        gray = np.array(img.convert('L')).astype(np.float64)
        
        # Texture features using gradients
        grad_x = ndimage.sobel(gray, axis=0)
        grad_y = ndimage.sobel(gray, axis=1)
        gradient_magnitude = np.hypot(grad_x, grad_y)
        
        features.extend([
            np.mean(gradient_magnitude),
            np.std(gradient_magnitude),
            np.max(gradient_magnitude),
        ])
        
        # Statistical measures
        features.extend([
            np.mean(gray),
            np.std(gray),
            np.var(gray),
        ])
        
        # Histogram features
        hist, _ = np.histogram(gray, bins=16, range=(0, 256))
        hist_normalized = hist / hist.sum()
        features.extend(hist_normalized.tolist())
        
        return np.array(features)
    
    def get_deterministic_seed(self) -> int:
        """Generate a deterministic seed from image features.
        
        Returns:
            Integer seed for random number generator
        """
        features = self.extract_features()
        
        # Convert features to bytes and hash
        feature_bytes = features.tobytes()
        hash_obj = hashlib.sha256(feature_bytes)
        
        # Convert first 8 bytes of hash to integer
        seed = int.from_bytes(hash_obj.digest()[:8], byteorder='big')
        
        return seed

