"""Extract 768-dimensional feature vectors from images for embedding matching."""

from pathlib import Path

import numpy as np
from PIL import Image
from scipy import ndimage
from scipy.fftpack import dct


class ImageFeatureExtractor:
    """Extract feature vectors from images matching sentence-transformer dimensions."""

    TARGET_DIM = 384  # Match sentence-transformer embedding size (all-MiniLM-L6-v2)

    def __init__(self, image_path: str | Path):
        """Initialize with an image path.
        
        Args:
            image_path: Path to the image file
        """
        self.image_path = Path(image_path)
        self.image = Image.open(self.image_path)
        
    def extract_features(self, target_dim: int | None = None) -> np.ndarray:
        """Extract feature vector from the image.
        
        Args:
            target_dim: Target dimensionality (default: self.TARGET_DIM)
        
        Returns:
            Feature vector normalized to [-1, 1] range
        """
        if target_dim is None:
            target_dim = self.TARGET_DIM
        # Convert to RGB if needed
        img = self.image.convert('RGB')
        img_array = np.array(img)
        
        features = []
        
        # === COLOR FEATURES (180 features) ===
        # Color histograms for each channel (60 bins × 3 channels)
        for channel in range(3):
            channel_data = img_array[:, :, channel]
            hist, _ = np.histogram(channel_data, bins=60, range=(0, 256))
            hist_normalized = hist / (hist.sum() + 1e-10)
            features.extend(hist_normalized)
        
        # === TEXTURE FEATURES (150 features) ===
        gray = np.array(img.convert('L')).astype(np.float64)
        
        # Gradient features at multiple scales
        for sigma in [1, 2, 4]:
            smoothed = ndimage.gaussian_filter(gray, sigma=sigma)
            grad_x = ndimage.sobel(smoothed, axis=0)
            grad_y = ndimage.sobel(smoothed, axis=1)
            gradient_magnitude = np.hypot(grad_x, grad_y)
            
            # Histogram of gradient magnitudes (10 bins per scale)
            hist, _ = np.histogram(gradient_magnitude, bins=10)
            hist_normalized = hist / (hist.sum() + 1e-10)
            features.extend(hist_normalized)
            
            # Gradient direction histogram (10 bins per scale)
            grad_direction = np.arctan2(grad_y, grad_x)
            hist, _ = np.histogram(grad_direction, bins=10, range=(-np.pi, np.pi))
            hist_normalized = hist / (hist.sum() + 1e-10)
            features.extend(hist_normalized)
        
        # Edge density at different thresholds
        for threshold in [10, 30, 50, 70, 90]:
            edges = gradient_magnitude > threshold
            features.append(edges.mean())
        
        # === SPATIAL STATISTICS (100 features) ===
        # Divide image into 10×10 grid and compute statistics per region
        h, w = gray.shape
        grid_h, grid_w = 10, 10
        cell_h, cell_w = h // grid_h, w // grid_w
        
        for i in range(grid_h):
            for j in range(grid_w):
                cell = gray[i*cell_h:(i+1)*cell_h, j*cell_w:(j+1)*cell_w]
                features.append(cell.mean())
        
        # === FREQUENCY FEATURES (200 features) ===
        # DCT coefficients for texture analysis
        # Resize to standard size for consistent DCT
        gray_resized = np.array(Image.fromarray(gray.astype(np.uint8)).resize((64, 64)))
        dct_coeffs = dct(dct(gray_resized.T, norm='ortho').T, norm='ortho')
        
        # Take low-frequency coefficients (top-left 14×14 region, avoiding DC)
        dct_features = []
        for i in range(14):
            for j in range(14):
                if i == 0 and j == 0:  # Skip DC component
                    continue
                dct_features.append(dct_coeffs[i, j])
        features.extend(dct_features[:200])
        
        # === STATISTICAL FEATURES (138 features) ===
        # Overall image statistics
        for channel in range(3):
            channel_data = img_array[:, :, channel].astype(np.float64)
            features.extend([
                np.mean(channel_data),
                np.std(channel_data),
                np.var(channel_data),
                np.median(channel_data),
                np.percentile(channel_data, 10),
                np.percentile(channel_data, 25),
                np.percentile(channel_data, 75),
                np.percentile(channel_data, 90),
                np.min(channel_data),
                np.max(channel_data),
            ])
        
        # Color correlation features
        r, g, b = img_array[:, :, 0], img_array[:, :, 1], img_array[:, :, 2]
        features.extend([
            np.corrcoef(r.flatten(), g.flatten())[0, 1],
            np.corrcoef(r.flatten(), b.flatten())[0, 1],
            np.corrcoef(g.flatten(), b.flatten())[0, 1],
        ])
        
        # Grayscale statistics
        features.extend([
            np.mean(gray),
            np.std(gray),
            np.var(gray),
            np.median(gray),
            np.percentile(gray, 10),
            np.percentile(gray, 25),
            np.percentile(gray, 75),
            np.percentile(gray, 90),
            np.min(gray),
            np.max(gray),
        ])
        
        # Texture complexity measures
        features.extend([
            np.std(grad_x),
            np.std(grad_y),
            np.mean(np.abs(grad_x)),
            np.mean(np.abs(grad_y)),
            np.max(gradient_magnitude),
        ])
        
        # Pad or truncate to target dimensions
        features = np.array(features)
        if len(features) < target_dim:
            # Pad with zeros
            padding = np.zeros(target_dim - len(features))
            features = np.concatenate([features, padding])
        elif len(features) > target_dim:
            # Truncate
            features = features[:target_dim]
        
        # Normalize to embedding-like range [-1, 1]
        # Use robust normalization to handle outliers
        median = np.median(features)
        mad = np.median(np.abs(features - median))
        if mad > 0:
            features = (features - median) / (mad * 1.4826)  # Scale factor for normal distribution
        
        # Clip to [-1, 1] range
        features = np.clip(features, -1, 1)
        
        return features
