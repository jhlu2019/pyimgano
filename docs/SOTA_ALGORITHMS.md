# State-of-the-Art Algorithms in PyImgAno

PyImgAno includes the latest state-of-the-art (SOTA) algorithms from top computer vision conferences. This document provides an overview of the cutting-edge methods available.

## 🏆 Latest SOTA Algorithms (2023-2024)

### WinCLIP (CVPR 2023) ⭐⭐⭐

**Paper**: "WinCLIP: Zero-/Few-Shot Anomaly Classification and Segmentation"
**Key Innovation**: Zero-shot anomaly detection using CLIP's visual-language understanding

**Highlights**:
- ✅ **Zero-shot capability** - No anomaly samples needed for training
- ✅ **Few-shot learning** - Works with minimal normal samples
- ✅ **Strong localization** - Window-based attention for precise anomaly maps
- ✅ **No fine-tuning** - Leverages pre-trained CLIP directly

**When to use**:
- When you have very limited training data
- For rapid prototyping without training
- When you can describe anomalies in text
- For multi-class anomaly detection

**Example**:
```python
from pyimgano.models import create_model

# Zero-shot detection
detector = create_model(
    "winclip",
    clip_model="ViT-B/32",
    k_shot=0  # Zero-shot
)

detector.set_class_name("screw")  # Describe the object
scores = detector.predict_proba(test_images)

# With anomaly localization
anomaly_maps = detector.predict_anomaly_map(test_images)
```

### SimpleNet (CVPR 2023) ⭐⭐⭐

**Paper**: "SimpleNet: A Simple Network for Image Anomaly Detection and Localization"
**Key Innovation**: Ultra-fast one-stage detection with comparable accuracy to complex methods

**Highlights**:
- ⚡ **Ultra-fast** - 100+ FPS on single GPU
- 🎯 **High accuracy** - Matches PatchCore performance
- 💾 **Memory efficient** - Small model size
- 🚀 **Easy to train** - Simple architecture, fast convergence

**When to use**:
- Real-time applications
- Resource-constrained environments
- When speed is critical
- Industrial inspection systems

### DifferNet (WACV 2023) ⭐⭐

**Paper**: "Same Same But DifferNet: Semi-Supervised Defect Detection with Normalizing Flows"
**Key Innovation**: Learns to detect anomalies via learnable difference with k-NN

**Highlights**:
- 🧠 **Learnable differences** - Trains a network to compute meaningful differences
- 🎯 **k-NN augmented** - Combines k-NN with deep learning
- 📍 **Good localization** - Multi-scale feature comparison
- 🔧 **Flexible** - Works with various backbones

**When to use**:
- When you need both detection and localization
- For subtle anomalies
- When you have sufficient normal samples
- For fine-grained defect detection

**Example**:
```python
detector = create_model(
    "differnet",
    backbone="wide_resnet50",
    k_neighbors=5,
    train_difference=True  # Learn difference module
)

detector.fit(normal_images)
scores = detector.predict_proba(test_images)
```

## 📅 Recent SOTA Algorithms (2021-2022)

### CutPaste (CVPR 2021) ⭐⭐

**Paper**: "CutPaste: Self-Supervised Learning for Anomaly Detection and Localization"
**Key Innovation**: Self-supervised learning via cutting and pasting image patches

**Highlights**:
- 🎨 **Self-supervised** - No anomaly samples needed
- 🔄 **Simple augmentation** - Easy to implement and understand
- 🎯 **Effective** - Strong performance on industrial datasets
- 🚀 **Fast training** - Converges quickly

**When to use**:
- When you only have normal samples
- For texture-based anomaly detection
- When you want interpretable augmentations
- For defect detection in manufacturing

**Example**:
```python
detector = create_model(
    "cutpaste",
    backbone="resnet18",
    augment_type="normal",  # or "scar", "3way"
    epochs=256
)

detector.fit(normal_images)
scores = detector.predict_proba(test_images)
```

### PatchCore (CVPR 2022) ⭐⭐⭐

**Paper**: "Towards Total Recall in Industrial Anomaly Detection"
**Key Innovation**: Coreset-based memory bank for efficient feature matching

**Highlights**:
- 🏆 **SOTA accuracy** - Best performance on MVTec AD
- 💾 **Memory efficient** - Coreset reduces memory footprint
- 📍 **Precise localization** - Patch-level anomaly maps
- ⚡ **Fast inference** - Efficient nearest neighbor search

**When to use**:
- When accuracy is critical
- For pixel-level anomaly localization
- Industrial quality control
- Benchmark comparisons

### STFPM (BMVC 2021) ⭐⭐

**Paper**: "Student-Teacher Feature Pyramid Matching"
**Key Innovation**: Multi-scale student-teacher knowledge distillation

**Highlights**:
- 🎓 **Knowledge distillation** - Teacher-student framework
- 🔍 **Multi-scale** - Feature pyramid for different scales
- 🎯 **Strong localization** - Pixel-wise anomaly maps
- 🚀 **End-to-end training** - Simple optimization

**When to use**:
- For multi-scale anomaly detection
- When you need detailed localization
- For defects of varying sizes
- Educational/research purposes

### DRAEM (ICCV 2021) ⭐⭐

**Paper**: "DRAEM: A Discriminatively Trained Reconstruction Embedding"
**Key Innovation**: Discriminative reconstruction with synthetic anomalies

**Highlights**:
- 🎭 **Synthetic anomalies** - Generates realistic defects
- 🔍 **Discriminative** - Not just reconstruction-based
- 📍 **Segmentation** - Pixel-level anomaly maps
- 🎯 **Robust** - Works well on various datasets

**When to use**:
- When you need segmentation maps
- For texture-based anomalies
- When reconstruction alone isn't enough
- Research and comparison

### CFlow-AD (WACV 2022) ⭐⭐

**Paper**: "CFLOW-AD: Real-Time Unsupervised Anomaly Detection"
**Key Innovation**: Conditional normalizing flows for anomaly scoring

**Highlights**:
- ⚡ **Real-time** - Fast inference
- 📊 **Probabilistic** - Principled likelihood-based scoring
- 🔄 **Flexible** - Normalizing flows are expressive
- 🎯 **Good performance** - Competitive with PatchCore

**When to use**:
- Real-time applications
- When you want probabilistic scores
- For research on normalizing flows
- When speed and accuracy both matter

### FastFlow (AAAI 2022) ⭐⭐

**Paper**: "FastFlow: Unsupervised Anomaly Detection via 2D Normalizing Flows"
**Key Innovation**: 2D normalizing flows for fast and accurate detection

**Highlights**:
- ⚡ **Very fast** - Faster than PatchCore
- 🎯 **High accuracy** - Near SOTA performance
- 💡 **Innovative** - 2D flows for spatial modeling
- 🚀 **Easy to train** - Stable optimization

**When to use**:
- When you need speed without sacrificing accuracy
- For large-scale deployment
- Industrial real-time inspection
- Research on flows

## 📊 Algorithm Comparison

### Performance on MVTec AD

| Algorithm | AUC-ROC (Image) | AUC-ROC (Pixel) | Speed (FPS) | Year |
|-----------|----------------|-----------------|-------------|------|
| **WinCLIP** | ~95% | ~98% | 5-10 | 2023 |
| **SimpleNet** | ~99% | ~98% | 100+ | 2023 |
| **PatchCore** | **99.6%** | **98.7%** | 30-50 | 2022 |
| **DifferNet** | ~97% | ~97% | 20-40 | 2023 |
| **CutPaste** | ~96% | N/A | 50+ | 2021 |
| **STFPM** | ~97% | ~98% | 40-60 | 2021 |
| **DRAEM** | ~98% | ~98% | 30-50 | 2021 |
| **FastFlow** | ~99% | ~98% | 60-80 | 2022 |
| **CFlow-AD** | ~98% | ~97% | 50-70 | 2022 |

*Note: Performance varies by category and implementation*

### Speed vs Accuracy Trade-off

```
High Accuracy, Slower:
├── PatchCore (99.6% AUC, 30-50 FPS)
├── DRAEM (98% AUC, 30-50 FPS)
└── FastFlow (99% AUC, 60-80 FPS)

Balanced:
├── SimpleNet (99% AUC, 100+ FPS) ⭐ Best balance
├── STFPM (97% AUC, 40-60 FPS)
└── DifferNet (97% AUC, 20-40 FPS)

Fast, Good Accuracy:
├── CFlow-AD (98% AUC, 50-70 FPS)
└── CutPaste (96% AUC, 50+ FPS)

Special:
└── WinCLIP (95% AUC, 5-10 FPS) - Zero-shot capable
```

### When to Use Each Algorithm

**For maximum accuracy**:
→ PatchCore, FastFlow

**For real-time (>50 FPS)**:
→ SimpleNet, CutPaste

**For zero-shot/few-shot**:
→ WinCLIP

**For self-supervised learning**:
→ CutPaste, DRAEM

**For pixel-level localization**:
→ PatchCore, STFPM, DRAEM

**For research/education**:
→ DifferNet, CFlow-AD, FastFlow

## 🔬 Algorithm Deep Dive

### CutPaste: Self-Supervised Learning

**How it works**:
1. Takes normal images
2. Cuts random rectangular patches
3. Pastes them at random locations (optionally rotated)
4. Trains classifier to distinguish original vs augmented
5. At test time, uses learned features for anomaly detection

**Variations**:
- **Normal CutPaste**: Regular rectangular patches
- **Scar CutPaste**: Thin elongated patches (for scratch-like defects)
- **3-way CutPaste**: Three-class classification (normal, normal cutpaste, scar cutpaste)

**Pros**:
- Simple and interpretable
- No anomaly data needed
- Fast training
- Good for texture anomalies

**Cons**:
- May not capture all anomaly types
- Depends on augmentation quality
- Limited localization

### WinCLIP: Zero-Shot Detection

**How it works**:
1. Uses pre-trained CLIP model
2. Defines text prompts for "normal" and "anomaly"
3. Extracts image and text features
4. Compares similarity using sliding windows
5. Anomaly score based on relative similarity

**Variations**:
- **Zero-shot**: Uses text prompts only
- **Few-shot**: Learns from k normal examples
- **Multi-scale**: Applies at different resolutions

**Pros**:
- No training required (zero-shot)
- Works with minimal data (few-shot)
- Flexible text-based control
- Strong localization

**Cons**:
- Requires CLIP installation
- Slower inference
- Depends on text prompt quality
- May not work for all defect types

### DifferNet: Learnable Differences

**How it works**:
1. Builds memory bank of normal features
2. For test image, finds k-nearest neighbors
3. Learns a difference module to compare features
4. Computes anomaly score from learned differences

**Key Components**:
- **Feature Extractor**: Pre-trained backbone (ResNet, Wide ResNet)
- **Memory Bank**: Stored normal features with k-D tree
- **Difference Module**: Learnable CNN for feature comparison
- **Multi-scale**: Uses multiple feature layers

**Pros**:
- Learns meaningful differences
- Good localization
- Flexible backbone
- Combines k-NN with deep learning

**Cons**:
- Requires training difference module
- Memory intensive (stores features)
- Slower than simple methods

## 📚 Usage Examples

### Complete Workflow with CutPaste

```python
import numpy as np
from pyimgano.models import create_model
from sklearn.metrics import roc_auc_score

# 1. Load normal training data
normal_images = load_normal_images("train/good/")  # (N, H, W, 3)

# 2. Create CutPaste detector
detector = create_model(
    "cutpaste",
    backbone="resnet18",
    augment_type="3way",  # Use 3-way classification
    pretrained=True,
    epochs=256,
    batch_size=96,
    learning_rate=0.03,
)

# 3. Train
detector.fit(normal_images)

# 4. Test
test_images = load_test_images("test/")
test_labels = load_test_labels("test/")

# 5. Predict
scores = detector.predict_proba(test_images)
predictions = detector.predict(test_images)

# 6. Evaluate
auc = roc_auc_score(test_labels, scores)
print(f"AUC-ROC: {auc:.4f}")
```

### Zero-Shot Detection with WinCLIP

```python
from pyimgano.models import create_model

# 1. Create WinCLIP detector (no training needed!)
detector = create_model(
    "winclip",
    clip_model="ViT-B/32",
    window_size=224,
    k_shot=0  # Zero-shot
)

# 2. Set class name for text prompts
detector.set_class_name("screw")

# 3. Predict directly (no training!)
scores = detector.predict_proba(test_images)

# 4. Get pixel-level anomaly maps
anomaly_maps = detector.predict_anomaly_map(test_images)

# 5. Visualize
import matplotlib.pyplot as plt

plt.figure(figsize=(12, 4))
plt.subplot(131)
plt.imshow(test_images[0])
plt.title("Original")

plt.subplot(132)
plt.imshow(anomaly_maps[0], cmap='hot')
plt.title("Anomaly Map")

plt.subplot(133)
overlay = test_images[0] * 0.5 + anomaly_maps[0][:,:,None] * 0.5
plt.imshow(overlay.astype(np.uint8))
plt.title("Overlay")
plt.show()
```

### Multi-Algorithm Ensemble

```python
from pyimgano.models import create_model
import numpy as np

# Create multiple detectors
detectors = {
    "cutpaste": create_model("cutpaste", backbone="resnet18"),
    "differnet": create_model("differnet", backbone="resnet18"),
    "simplenet": create_model("simplenet"),
}

# Train all
for name, detector in detectors.items():
    print(f"Training {name}...")
    detector.fit(normal_images)

# Ensemble prediction
all_scores = []
for name, detector in detectors.items():
    scores = detector.predict_proba(test_images)
    all_scores.append(scores)

# Average ensemble
ensemble_scores = np.mean(all_scores, axis=0)

# Weighted ensemble (if you know which works better)
weights = [0.4, 0.3, 0.3]  # CutPaste, DifferNet, SimpleNet
weighted_scores = np.average(all_scores, axis=0, weights=weights)
```

## 🎯 Best Practices

### For Production Deployment

1. **Choose the right algorithm**:
   - Real-time: SimpleNet, FastFlow
   - Maximum accuracy: PatchCore, FastFlow
   - Limited data: WinCLIP (zero/few-shot), CutPaste

2. **Optimize for your use case**:
   - Adjust image resolution (smaller = faster)
   - Use appropriate backbone (ResNet18 vs ResNet50)
   - Enable GPU for deep learning methods

3. **Validate thoroughly**:
   - Test on representative data
   - Check edge cases
   - Measure actual throughput

### For Research

1. **Benchmark properly**:
   - Use standard datasets (MVTec AD, BTAD)
   - Report multiple metrics (AUC-ROC image, AUC-ROC pixel, F1)
   - Include timing and memory usage

2. **Compare fairly**:
   - Use same data preprocessing
   - Same evaluation protocol
   - Report confidence intervals

3. **Ablation studies**:
   - Test different backbones
   - Vary hyperparameters
   - Compare components

## 📖 References

### Papers

1. **WinCLIP**: Jeong et al. "WinCLIP: Zero-/Few-Shot Anomaly Classification and Segmentation." CVPR 2023.

2. **SimpleNet**: Liu et al. "SimpleNet: A Simple Network for Image Anomaly Detection and Localization." CVPR 2023.

3. **DifferNet**: Rudolph et al. "Same Same But DifferNet: Semi-Supervised Defect Detection with Normalizing Flows." WACV 2021.

4. **CutPaste**: Li et al. "CutPaste: Self-Supervised Learning for Anomaly Detection and Localization." CVPR 2021.

5. **PatchCore**: Roth et al. "Towards Total Recall in Industrial Anomaly Detection." CVPR 2022.

6. **STFPM**: Wang et al. "Student-Teacher Feature Pyramid Matching for Anomaly Detection." BMVC 2021.

7. **DRAEM**: Zavrtanik et al. "DRAEM: A Discriminatively Trained Reconstruction Embedding for Surface Anomaly Detection." ICCV 2021.

8. **FastFlow**: Yu et al. "FastFlow: Unsupervised Anomaly Detection and Localization via 2D Normalizing Flows." AAAI 2022.

9. **CFlow-AD**: Gudovskiy et al. "CFLOW-AD: Real-Time Unsupervised Anomaly Detection with Localization via Conditional Normalizing Flows." WACV 2022.

### Datasets

- **MVTec AD**: https://www.mvtec.com/company/research/datasets/mvtec-ad
- **BTAD**: https://avires.dimi.uniud.it/papers/btad/btad.zip
- **VisA**: https://github.com/amazon-science/spot-diff

## 🚀 Future Additions

We plan to add more SOTA algorithms:

- **RegAD** (CVPR 2024): Registration-based anomaly detection
- **UniAD** (NeurIPS 2022): Unified anomaly detection framework
- **PyramidFlow**: Multi-scale flow models
- **MemSeg**: Memory-guided semantic segmentation
- **APRIL-GAN**: Adversarial prior based anomaly detection

Stay tuned for updates!

## 💬 Contributing

Have a SOTA algorithm you'd like to see in PyImgAno?

- Open an issue on GitHub
- Submit a pull request
- Check our [Contributing Guide](../CONTRIBUTING.md)

We welcome implementations of new algorithms!
