# Strategy Reference: NLP Competitions

Text classification, NER, question answering, summarization, and other natural language tasks. This is reference material — read it, then develop your own approach in your workspace STRATEGY.md.

## Approach Order (start simple, add complexity)

### 1. Baseline (first hour)
- Understand the text data: lengths, languages, domains, label distribution
- TF-IDF + Logistic Regression baseline — surprisingly strong, submit immediately
- Check if pretrained models are allowed by competition rules

### 2. Pretrained Transformers (primary approach)
- Fine-tune a pretrained model on the competition data
- Start with: DeBERTa-v3-base (best accuracy/speed trade-off for English)
- Try: RoBERTa, BERT, DistilBERT for diversity
- For multilingual: XLM-RoBERTa, mBERT

### 3. Training Optimization
- Learning rate: 1e-5 to 5e-5 range for transformers
- Warmup: 5-10% of total steps
- Epochs: 3-5 typically (watch for overfitting)
- Batch size: as large as GPU memory allows
- Mixed precision (fp16) for speed
- Gradient accumulation for effective larger batch sizes

### 4. Data Augmentation
- Back-translation (translate to another language and back)
- Synonym replacement
- Random deletion/insertion
- Mixup at embedding level
- Pseudo-labeling on unlabeled data (if available)

### 5. Ensembling
- Ensemble different model architectures (DeBERTa + RoBERTa + BERT)
- Ensemble different training seeds
- Ensemble different folds
- Stacking with a simple meta-learner

## Implementation Guidelines

- **Framework**: Hugging Face Transformers + PyTorch
- **CV Strategy**: Stratified K-Fold on label distribution
- **Tokenization**: Use the tokenizer that matches your pretrained model
- **Max length**: Analyze text length distribution, don't truncate important content

## Common Pitfalls

- Not cleaning text data (HTML tags, special characters, encoding issues)
- Wrong tokenizer for the model
- Training too many epochs (transformers overfit quickly on small datasets)
- Ignoring class imbalance
- Not using mixed precision (halves training time)
- Leaking validation data through data augmentation

## What Worked in Past Competitions

- DeBERTa-v3 dominates recent NLP competitions
- Multi-sample dropout for regularization
- Adversarial weight perturbation (AWP) for robustness
- Pseudo-labeling with high-confidence predictions
