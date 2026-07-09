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

**TAO Pre-trained Architecture Quick Reference (GPU-accelerated training):**

| Task | Architecture | TAO Skill | Notes |
|------|-------------|-----------|-------|
| Classification | EfficientNet-B0 to B7 | `tao-train-efficientnet` | Best accuracy/speed trade-off |
| Classification | FAN (Fully Attentional Network) | `tao-train-fan` | Strong on fine-grained |
| Detection | RT-DETR | `tao-train-rt-detr` | Real-time transformer detector |
| Detection | DINO | `tao-train-dino` | State-of-the-art detector |
| Detection | Grounding DINO | `tao-train-grounding-dino` | Open-vocabulary detection |
| Detection | CenterPose | `tao-train-centerpose` | 6-DoF pose estimation |
| Segmentation | SegFormer | `tao-train-segformer` | Efficient semantic seg |
| Segmentation | Mask2Former | `tao-train-mask2former` | Universal segmentation |
| Segmentation | Mask Auto Labeler | `tao-train-mal` | Auto-labeling for segmentation |
| Instance Seg | Mask R-CNN | `tao-train-mask-rcnn` | Standard instance seg |
| Keypoints | Pose Classification | `tao-train-pose-classification` | Action recognition |
| Re-ID | Re-Identification | `tao-train-re-identification` | Object re-identification |
| OCR | OCRNet | `tao-train-ocrnet` | Scene text recognition |

Install any: `npx skills add nvidia/skills/<skill-name>`

### 3. Augmentation (critical for CV)
- Basic: horizontal flip, vertical flip, rotation, scale
- Color: brightness, contrast, saturation, hue jitter
- Advanced: CutMix, MixUp, Mosaic, GridMask
- Domain-specific: whatever transformations preserve the label
- Use albumentations library for efficient augmentation pipelines
- **NVIDIA DALI** for GPU-accelerated data loading and augmentation pipeline. DALI moves augmentation to GPU, eliminating CPU bottleneck. Install: `npx skills add nvidia/skills/dali-dynamic-mode`. Use DALI when CPU augmentation is the bottleneck (check with profiler).
- **Data Designer** for synthetic image generation. Can generate realistic synthetic training images to augment small datasets. Especially useful for rare classes or domain-specific imagery. Install: `npx skills add nvidia/skills/data-designer`.

### 4. Training Optimization
- Learning rate: cosine annealing with warmup
- Optimizer: AdamW with weight decay
- Mixed precision (fp16/bf16)
- Progressive resizing: train on small images first, then increase
- Test-time augmentation (TTA): flip, rotate at inference, average predictions
- **TAO AutoML** for automated hyperparameter optimization with WandB experiment tracking. Install: `npx skills add nvidia/skills/tao-run-automl`.

### 5. Ensembling
- Ensemble different architectures (CNN + ViT)
- Ensemble different image sizes
- Ensemble different augmentation strategies
- Multi-scale TTA

## Implementation Guidelines

- **Framework**: PyTorch + timm (PyTorch Image Models)
- **NVIDIA TAO**: Pre-built training pipelines for 20+ architectures — use when rapid experimentation across architectures is needed
- **DALI**: GPU data loading pipeline — use when CPU augmentation is the training bottleneck
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
