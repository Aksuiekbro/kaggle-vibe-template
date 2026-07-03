# Competition: [Name]

> Fill this file when setting up a new competition. Every agent reads this.

## Overview
- **URL**: https://www.kaggle.com/competitions/[slug]
- **Type**: [optimization | tabular | nlp | cv | code]
- **Deadline**: [YYYY-MM-DD HH:MM UTC]
- **Daily Submission Limit**: [number, typically 100 for optimization]
- **Team Size**: [number]
- **Prize**: [amount or "knowledge"]

## Problem Description

[Plain language description of what needs to be solved. Copy from competition page + your own understanding. Be specific about inputs, outputs, and constraints.]

## Evaluation Metric
- **Metric**: [e.g., normalized_area, RMSE, F1, AUC-ROC]
- **Direction**: [minimize | maximize]
- **CV Variance Threshold**: [e.g., 0.005 — triggers overfitting warning if CV std exceeds this]

## Data Description
- **Files**: [list each data file and what it contains]
- **Size**: [approximate total size]
- **Format**: [CSV, images, text, parquet, etc.]
- **Key Features**: [important columns/fields that matter for the solution]

## Submission Format
- **File**: [e.g., submission.csv]
- **Columns**: [required columns and their types]
- **Rows**: [how many, what they represent]
- **Example**:
```
id,prediction
1,0.5
2,0.3
```

## Known Constraints
[Hardware limits, time limits, external data rules, specific competition rules. Include anything from the rules page that could affect strategy.]

## Initial Research
[Approaches found in discussions, papers, similar past competitions. Each agent should add their findings here as they research.]

## Baseline Scores
- **Naive baseline**: [score of simplest possible submission, e.g., all zeros, mean prediction]
- **Public best**: [current #1 on public LB if known]
- **Medal thresholds**: [bronze/silver/gold cutoffs if known or estimable]
