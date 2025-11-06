"""
Image preprocessing and enhancement utilities.

This module provides comprehensive image preprocessing capabilities including:
- Edge detection (Canny, Sobel, Laplacian, etc.)
- Morphological operations (erosion, dilation, opening, closing)
- Filters (Gaussian, bilateral, median, etc.)
- Color space conversions
- Normalization and standardization
- Preprocessing pipelines
"""

from .enhancer import (
    ImageEnhancer,
    PreprocessingPipeline,
    edge_detection,
    morphological_operation,
    apply_filter,
    normalize_image,
)
from .mixin import PreprocessingMixin

__all__ = [
    # Main classes
    "ImageEnhancer",
    "PreprocessingPipeline",
    "PreprocessingMixin",
    # Functional API
    "edge_detection",
    "morphological_operation",
    "apply_filter",
    "normalize_image",
]
