# Strategy Reference: Tabular ML Competitions

Classification and regression on structured/tabular data. This is reference material — read it, then develop your own approach in your workspace STRATEGY.md.

## Approach Order (start simple, add complexity)

### 0. Similar Winner Upsolve
- Before heavy feature engineering, find similar past regression/classification competitions
- Match by metric, split type, data shape, target distribution, and leakage risk
- Prefer top 3 winner writeups when available and record whether they converge on the same load-bearing pattern
- Extract winner patterns as hypotheses: CV scheme, feature families, model families, ensembling, calibration, target transforms, postprocessing
- Convert each pattern into a small validation experiment before adding it to STRATEGY.md
- Use prospective prediction/postmortem loops from `.ai/checklists/kaggle-upsolve.md` and `.ai/checklists/postmortem.md`

### 1. EDA + Baseline (first hour)
- Understand the data: dtypes, missing values, distributions, correlations
- Identify target variable and evaluation metric
- Train a single LightGBM/XGBoost with default parameters
- Submit this baseline — establish your floor

### 2. Feature Engineering (core work)
- Domain-specific features based on problem understanding
- Aggregation features (group-by statistics)
- Interaction features (ratios, products of important features)
- Target encoding (with proper CV to avoid leakage)
- Date/time features if applicable
- Text features (TF-IDF, embeddings) if text columns exist

### 3. Model Selection
**Gradient Boosting (primary):**
- LightGBM: fastest, good default choice
- XGBoost: slightly different bias, good for ensembling with LightGBM
- CatBoost: handles categorical features natively

**Other models (for diversity in ensembles):**
- Neural networks (TabNet, simple MLP)
- Linear models (Logistic/Ridge with good features)
- k-NN (sometimes adds ensemble diversity)

### 4. Hyperparameter Tuning
- Use Optuna for Bayesian optimization
- Tune on CV, not on single split
- Key LightGBM params: learning_rate, num_leaves, max_depth, min_child_samples, subsample, colsample_bytree, reg_alpha, reg_lambda
- Don't over-tune — diminishing returns after ~100 trials

### 5. Ensembling
- Blend diverse models (GBM + NN + linear)
- Stacking: use out-of-fold predictions as features for a meta-learner
- Weight optimization: find optimal blend weights via CV (not public LB)
- Diversity matters more than individual model quality

## Implementation Guidelines

- **Language**: Python with scikit-learn, LightGBM, XGBoost, CatBoost
- **CV Strategy**: Stratified K-Fold (k=5 or k=10) for classification. Group K-Fold if data has groups. Time-series split if temporal.
- **Reproducibility**: Set random seeds everywhere
- **Memory**: Use appropriate dtypes (int8/16/32 instead of int64, float32 instead of float64)

## Common Pitfalls

- Target leakage through features derived from the target
- Not using proper CV (especially with time-series or grouped data)
- Over-tuning to public LB (classic shake-up victim)
- Ignoring feature importance — adding features blindly
- Training on all data without holdout for final validation
- Ensemble of similar models (doesn't add diversity)

## What Worked in Past Competitions

- Most Kaggle tabular winners: LightGBM + XGBoost + CatBoost ensemble
- Playground Series: heavy feature engineering + stacking
- Feature selection matters as much as model choice
- Treat these as starting hypotheses. Durable claims should be backed by memory cards with sources, scope, evidence, and counter-evidence.
