Submit a solution through the score-gated pipeline.

Usage: /submit <file-path> <description>

Run:
```
python tools/submit.py --agent claude --file $ARGUMENTS
```

The tool will:
1. Evaluate locally (checks format, CV stats, overfitting flags)
2. Compare against your current best score
3. Reject if not an improvement
4. Submit via MCP first, fall back to kaggle CLI
5. Log to registry with provenance

If submitting derivative work (based on another agent's submission during a sharing round), add --based-on <submission-id>.
