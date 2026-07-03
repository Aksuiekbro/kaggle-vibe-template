# Solution Scoring Rubric

Used during cross-review. Score each solution 0-10 based on these criteria.

## Scoring Bands

| Score | Status | Meaning |
|-------|--------|---------|
| 9.0-10.0 | SOLID | Strong approach, minor tweaks only. Likely to survive private LB. |
| 7.0-8.9 | IMPROVE | Good foundation with specific improvements available. |
| 5.0-6.9 | IMPROVE | Fundamental issues but contains salvageable ideas. |
| 0.0-4.9 | IMPROVE | Approach unlikely to be competitive. Consider abandoning. |

## Scoring Dimensions

### Correctness (0-2 points)
- 2: No bugs, handles edge cases, produces valid submissions
- 1: Works but has minor issues or unhandled edge cases
- 0: Contains bugs that affect the score or produces invalid output

### Competitiveness (0-3 points)
- 3: Uses a proven competitive approach, well-tuned
- 2: Reasonable approach but missing key optimizations
- 1: Basic approach unlikely to medal
- 0: Naive or fundamentally wrong approach

### Robustness (0-2 points)
- 2: Consistent across CV folds / random seeds, likely to hold on private LB
- 1: Some variance across folds, moderate overfitting risk
- 0: High variance, likely overfit to public LB

### Improvability (0-2 points)
- 2: Clear next steps that could meaningfully improve score
- 1: Some improvement potential but hitting diminishing returns
- 0: Saturated approach, no clear path forward

### Innovation (0-1 point)
- 1: Uses a novel technique or creative combination
- 0: Standard approach (not a penalty, just no bonus)

## Hard Caps

- Any correctness bug that produces invalid submissions → caps at 4.9
- Evidence of data leakage → caps at 3.9
- Submission is near-identical to a previous one → caps at 5.9
