"""
Dataset loading utilities for standard anomaly detection benchmarks.

Provides easy-to-use loaders for popular datasets:
- MVTec AD
- BTAD
- VisA
- Custom datasets

Example:
    >>> from pyimgano.utils.datasets import MVTecDataset
    >>> dataset = MVTecDataset(root='./mvtec_ad', category='bottle')
    >>> train_data = dataset.get_train_data()
    >>> test_data, test_labels = dataset.get_test_data()
"""

import os
from pathlib import Path
from typing import List, Optional, Tuple, Dict
import numpy as np
from numpy.typing import NDArray
import cv2
from dataclasses import dataclass


@dataclass
class DatasetInfo:
    """Dataset information."""
    name: str
    categories: List[str]
    num_train: int
    num_test: int
    image_size: Tuple[int, int]
    description: str


class BaseDataset:
    """Base class for anomaly detection datasets."""

    def __init__(self, root: str, category: Optional[str] = None):
        self.root = Path(root)
        self.category = category

        if not self.root.exists():
            raise FileNotFoundError(f"Dataset root not found: {root}")

    def get_train_data(self) -> NDArray:
        """Get training data (normal only).

        Returns:
            Training images [N, H, W, C]
        """
        raise NotImplementedError

    def get_test_data(self) -> Tuple[NDArray, NDArray, Optional[NDArray]]:
        """Get test data with labels and masks.

        Returns:
            test_images: Test images [N, H, W, C]
            test_labels: Binary labels [N] (0=normal, 1=anomaly)
            test_masks: Ground truth masks [N, H, W] or None
        """
        raise NotImplementedError

    def get_info(self) -> DatasetInfo:
        """Get dataset information.

        Returns:
            Dataset information
        """
        raise NotImplementedError


class MVTecDataset(BaseDataset):
    """MVTec AD dataset loader.

    MVTec AD is a widely-used benchmark for industrial anomaly detection.
    Contains 15 categories with texture and object classes.

    Categories:
        Textures: carpet, grid, leather, tile, wood
        Objects: bottle, cable, capsule, hazelnut, metal_nut,
                 pill, screw, toothbrush, transistor, zipper

    Args:
        root: Path to MVTec AD dataset root
        category: Category name (e.g., 'bottle', 'carpet')
        resize: Target size for images (H, W)
        load_masks: Whether to load ground truth masks

    Example:
        >>> dataset = MVTecDataset(
        ...     root='./mvtec_ad',
        ...     category='bottle',
        ...     resize=(256, 256)
        ... )
        >>> train_imgs = dataset.get_train_data()
        >>> test_imgs, labels, masks = dataset.get_test_data()
    """

    CATEGORIES = [
        'bottle', 'cable', 'capsule', 'carpet', 'grid',
        'hazelnut', 'leather', 'metal_nut', 'pill', 'screw',
        'tile', 'toothbrush', 'transistor', 'wood', 'zipper'
    ]

    def __init__(
        self,
        root: str,
        category: str,
        resize: Optional[Tuple[int, int]] = None,
        load_masks: bool = True
    ):
        super().__init__(root, category)

        if category not in self.CATEGORIES:
            raise ValueError(f"Invalid category. Choose from: {self.CATEGORIES}")

        self.resize = resize
        self.load_masks = load_masks
        self.category_path = self.root / category

    def _load_images(self, path: Path) -> List[NDArray]:
        """Load all images from a directory."""
        images = []

        for img_path in sorted(path.glob('*.png')):
            img = cv2.imread(str(img_path))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            if self.resize is not None:
                img = cv2.resize(img, (self.resize[1], self.resize[0]))

            images.append(img)

        return images

    def get_train_data(self) -> NDArray:
        """Get training data (normal images only)."""
        train_path = self.category_path / 'train' / 'good'

        if not train_path.exists():
            raise FileNotFoundError(f"Training data not found: {train_path}")

        images = self._load_images(train_path)
        return np.array(images)

    def get_test_data(self) -> Tuple[NDArray, NDArray, Optional[NDArray]]:
        """Get test data with labels and optionally masks."""
        test_path = self.category_path / 'test'
        ground_truth_path = self.category_path / 'ground_truth'

        if not test_path.exists():
            raise FileNotFoundError(f"Test data not found: {test_path}")

        images = []
        labels = []
        masks = [] if self.load_masks else None

        # Load normal test images
        normal_path = test_path / 'good'
        if normal_path.exists():
            normal_imgs = self._load_images(normal_path)
            images.extend(normal_imgs)
            labels.extend([0] * len(normal_imgs))

            if self.load_masks:
                # Normal images have no masks (all zeros)
                for img in normal_imgs:
                    masks.append(np.zeros(img.shape[:2], dtype=np.uint8))

        # Load anomaly test images
        for defect_dir in sorted(test_path.iterdir()):
            if defect_dir.name == 'good':
                continue

            if not defect_dir.is_dir():
                continue

            defect_imgs = self._load_images(defect_dir)
            images.extend(defect_imgs)
            labels.extend([1] * len(defect_imgs))

            # Load masks if requested
            if self.load_masks and ground_truth_path.exists():
                mask_dir = ground_truth_path / defect_dir.name
                if mask_dir.exists():
                    for img_path in sorted(defect_dir.glob('*.png')):
                        mask_path = mask_dir / f"{img_path.stem}_mask.png"
                        if mask_path.exists():
                            mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
                            if self.resize is not None:
                                mask = cv2.resize(mask, (self.resize[1], self.resize[0]))
                            # Binary mask
                            mask = (mask > 127).astype(np.uint8)
                            masks.append(mask)
                        else:
                            # If mask not found, create zero mask
                            masks.append(np.zeros(self.resize or defect_imgs[0].shape[:2], dtype=np.uint8))

        images = np.array(images)
        labels = np.array(labels)
        masks = np.array(masks) if self.load_masks else None

        return images, labels, masks

    def get_info(self) -> DatasetInfo:
        """Get dataset information."""
        train_data = self.get_train_data()
        test_data, test_labels, _ = self.get_test_data()

        return DatasetInfo(
            name='MVTec AD',
            categories=self.CATEGORIES,
            num_train=len(train_data),
            num_test=len(test_data),
            image_size=train_data[0].shape[:2],
            description=f'MVTec AD - {self.category} category'
        )

    @staticmethod
    def list_categories() -> List[str]:
        """List all available categories."""
        return MVTecDataset.CATEGORIES


class BTADDataset(BaseDataset):
    """BTAD (BeanTech Anomaly Detection) dataset loader.

    BTAD contains 3 industrial product categories.

    Categories:
        01: Industrial product 1
        02: Industrial product 2
        03: Industrial product 3

    Args:
        root: Path to BTAD dataset root
        category: Category name ('01', '02', or '03')
        resize: Target size for images (H, W)

    Example:
        >>> dataset = BTADDataset(root='./btad', category='01')
        >>> train_imgs = dataset.get_train_data()
    """

    CATEGORIES = ['01', '02', '03']

    def __init__(
        self,
        root: str,
        category: str,
        resize: Optional[Tuple[int, int]] = None
    ):
        super().__init__(root, category)

        if category not in self.CATEGORIES:
            raise ValueError(f"Invalid category. Choose from: {self.CATEGORIES}")

        self.resize = resize
        self.category_path = self.root / category

    def _load_images(self, path: Path) -> List[NDArray]:
        """Load all images from a directory."""
        images = []

        for ext in ['*.png', '*.jpg', '*.bmp']:
            for img_path in sorted(path.glob(ext)):
                img = cv2.imread(str(img_path))
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

                if self.resize is not None:
                    img = cv2.resize(img, (self.resize[1], self.resize[0]))

                images.append(img)

        return images

    def get_train_data(self) -> NDArray:
        """Get training data (normal images only)."""
        train_path = self.category_path / 'train' / 'ok'

        if not train_path.exists():
            raise FileNotFoundError(f"Training data not found: {train_path}")

        images = self._load_images(train_path)
        return np.array(images)

    def get_test_data(self) -> Tuple[NDArray, NDArray, Optional[NDArray]]:
        """Get test data with labels."""
        test_path = self.category_path / 'test'

        if not test_path.exists():
            raise FileNotFoundError(f"Test data not found: {test_path}")

        images = []
        labels = []

        # Load normal test images
        normal_path = test_path / 'ok'
        if normal_path.exists():
            normal_imgs = self._load_images(normal_path)
            images.extend(normal_imgs)
            labels.extend([0] * len(normal_imgs))

        # Load anomaly test images
        defect_path = test_path / 'ko'
        if defect_path.exists():
            defect_imgs = self._load_images(defect_path)
            images.extend(defect_imgs)
            labels.extend([1] * len(defect_imgs))

        images = np.array(images)
        labels = np.array(labels)

        return images, labels, None

    def get_info(self) -> DatasetInfo:
        """Get dataset information."""
        train_data = self.get_train_data()
        test_data, test_labels, _ = self.get_test_data()

        return DatasetInfo(
            name='BTAD',
            categories=self.CATEGORIES,
            num_train=len(train_data),
            num_test=len(test_data),
            image_size=train_data[0].shape[:2],
            description=f'BTAD - Category {self.category}'
        )


class CustomDataset(BaseDataset):
    """Custom dataset loader for user-defined datasets.

    Expected directory structure:
        root/
            train/
                normal/
                    img1.png
                    img2.png
                    ...
            test/
                normal/
                    img1.png
                    ...
                anomaly/
                    img1.png
                    ...
            (optional) ground_truth/
                anomaly/
                    img1_mask.png
                    ...

    Args:
        root: Path to dataset root
        resize: Target size for images (H, W)
        load_masks: Whether to load ground truth masks

    Example:
        >>> dataset = CustomDataset(root='./my_dataset')
        >>> train_imgs = dataset.get_train_data()
    """

    def __init__(
        self,
        root: str,
        resize: Optional[Tuple[int, int]] = None,
        load_masks: bool = False
    ):
        super().__init__(root)
        self.resize = resize
        self.load_masks = load_masks

    def _load_images(self, path: Path) -> List[NDArray]:
        """Load all images from a directory."""
        images = []

        for ext in ['*.png', '*.jpg', '*.jpeg', '*.bmp']:
            for img_path in sorted(path.glob(ext)):
                img = cv2.imread(str(img_path))
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

                if self.resize is not None:
                    img = cv2.resize(img, (self.resize[1], self.resize[0]))

                images.append(img)

        return images

    def get_train_data(self) -> NDArray:
        """Get training data."""
        train_path = self.root / 'train' / 'normal'

        if not train_path.exists():
            raise FileNotFoundError(f"Training data not found: {train_path}")

        images = self._load_images(train_path)
        return np.array(images)

    def get_test_data(self) -> Tuple[NDArray, NDArray, Optional[NDArray]]:
        """Get test data with labels and optionally masks."""
        test_path = self.root / 'test'

        if not test_path.exists():
            raise FileNotFoundError(f"Test data not found: {test_path}")

        images = []
        labels = []
        masks = [] if self.load_masks else None

        # Load normal test images
        normal_path = test_path / 'normal'
        if normal_path.exists():
            normal_imgs = self._load_images(normal_path)
            images.extend(normal_imgs)
            labels.extend([0] * len(normal_imgs))

            if self.load_masks:
                for img in normal_imgs:
                    masks.append(np.zeros(img.shape[:2], dtype=np.uint8))

        # Load anomaly test images
        anomaly_path = test_path / 'anomaly'
        if anomaly_path.exists():
            anomaly_imgs = self._load_images(anomaly_path)
            images.extend(anomaly_imgs)
            labels.extend([1] * len(anomaly_imgs))

            # Load masks if requested
            if self.load_masks:
                gt_path = self.root / 'ground_truth' / 'anomaly'
                if gt_path.exists():
                    for img_path in sorted(anomaly_path.glob('*.png')):
                        mask_path = gt_path / f"{img_path.stem}_mask.png"
                        if mask_path.exists():
                            mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
                            if self.resize is not None:
                                mask = cv2.resize(mask, (self.resize[1], self.resize[0]))
                            mask = (mask > 127).astype(np.uint8)
                            masks.append(mask)
                        else:
                            masks.append(np.zeros(self.resize or anomaly_imgs[0].shape[:2], dtype=np.uint8))
                else:
                    # No masks available
                    for img in anomaly_imgs:
                        masks.append(np.zeros(img.shape[:2], dtype=np.uint8))

        images = np.array(images)
        labels = np.array(labels)
        masks = np.array(masks) if self.load_masks else None

        return images, labels, masks

    def get_info(self) -> DatasetInfo:
        """Get dataset information."""
        train_data = self.get_train_data()
        test_data, test_labels, _ = self.get_test_data()

        return DatasetInfo(
            name='Custom Dataset',
            categories=['custom'],
            num_train=len(train_data),
            num_test=len(test_data),
            image_size=train_data[0].shape[:2],
            description='User-defined custom dataset'
        )


def load_dataset(
    name: str,
    root: str,
    category: Optional[str] = None,
    **kwargs
) -> BaseDataset:
    """Factory function to load datasets.

    Args:
        name: Dataset name ('mvtec', 'btad', 'custom')
        root: Path to dataset root
        category: Category name (required for mvtec and btad)
        **kwargs: Additional arguments for dataset

    Returns:
        Dataset instance

    Example:
        >>> dataset = load_dataset('mvtec', './mvtec_ad', category='bottle')
        >>> train_data = dataset.get_train_data()
    """
    datasets = {
        'mvtec': MVTecDataset,
        'mvtec_ad': MVTecDataset,
        'btad': BTADDataset,
        'custom': CustomDataset,
    }

    name_lower = name.lower()
    if name_lower not in datasets:
        raise ValueError(f"Unknown dataset: {name}. Choose from: {list(datasets.keys())}")

    dataset_class = datasets[name_lower]

    if name_lower in ['mvtec', 'mvtec_ad', 'btad']:
        if category is None:
            raise ValueError(f"Category is required for {name} dataset")
        return dataset_class(root=root, category=category, **kwargs)
    else:
        return dataset_class(root=root, **kwargs)
