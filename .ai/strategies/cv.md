# Strategy Reference: Computer Vision Competitions

Image classification, object detection, segmentation, and other vision tasks. This is reference material — read it, then develop your own approach in your workspace STRATEGY.md.

## Approach Order (start simple, add complexity)

### 1. Baseline (first hour)
- Understand the images: sizes, channels, class distribution, quality
- Fine-tune a pretrained model: EfficientNet-B0 or ResNet-50
- Use standard augmentations: flip, rotate, normalize
- Submit — establish floor score

### 2. Model Selection
**Classification:**
- EfficientNet family (B0-B7): good accuracy/speed trade-off
- ConvNeXt: modern CNN, competitive with transformers
- Vision Transformers (ViT, Swin): strong with enough data
- EVA-02, BEiT: latest state-of-the-art

**Detection:**
- YOLOv8/v9: fast and accurate
- DINO/DETRs: transformer-based detectors
- EfficientDet: good for competitions

**Segmentation:**
- U-Net with various backbones
- Mask R-CNN for instance segmentation
- SegFormer for semantic segmentation

### 3. Augmentation (critical for CV)
- Basic: horizontal flip, vertical flip, rotation, scale
- Color: brightness, contrast, saturation, hue jitter
- Advanced: CutMix, MixUp, Mosaic, GridMask
- Domain-specific: whatever transformations preserve the label
- Use albumentations library for efficient augmentation pipelines

### 4. Training Optimization
- Learning rate: cosine annealing with warmup
- Optimizer: AdamW with weight decay
- Mixed precision (fp16/bf16)
- Progressive resizing: train on small images first, then increase
- Test-time augmentation (TTA): flip, rotate at inference, average predictions

### 5. Ensembling
- Ensemble different architectures (CNN + ViT)
- Ensemble different image sizes
- Ensemble different augmentation strategies
- Multi-scale TTA

## Implementation Guidelines

- **Framework**: PyTorch + timm (PyTorch Image Models)
- **CV Strategy**: Stratified K-Fold on label distribution
- **Image size**: Start small (224), increase later if it helps
- **GPU memory**: Monitor with `nvidia-smi`, adjust batch size accordingly

## Common Pitfalls

- Not using pretrained weights (training from scratch rarely wins)
- Too-aggressive augmentation for the domain
- Training on wrong image size (check competition's image dimensions)
- Ignoring image quality issues (corrupt files, wrong labels)
- Not using TTA at inference time (free accuracy boost)
- Overfitting to small datasets without strong augmentation

## What Worked in Past Competitions

- EfficientNet + heavy augmentation for classification
- YOLOv8 for detection competitions
- Progressive resizing for training efficiency
- Pseudo-labeling with teacher-student framework
