# Sharing Round Protocol

## Before Starting (Human Operator)

1. Run `python tools/share.py start`
2. This copies each agent's current best submission + workspace code to `shared/best/<agent>/`
3. Announce to all agents that a sharing round is active

## During Sharing Round (Agents)

### What You Can Do
- Read any file in `shared/best/`
- Study other agents' approaches, code, and submissions
- Build on or improve another agent's work in YOUR workspace
- Submit derivative work through the normal score-gated process

### What You Must Do
- Declare provenance on any derivative work: "based on <agent>'s submission <id>"
- Any cross-review verdicts from `.ai/reviews/` should inform your improvements
- Document what you took, what you changed, and why in your PROGRESS.md

### What You Cannot Do
- Modify files in `shared/best/` (read-only)
- Modify files in other agents' workspaces
- Submit another agent's work unchanged (must add value)

## Ending a Sharing Round (Human Operator)

1. Run `python tools/share.py end`
2. This locks `shared/best/` and logs all derivatives produced
3. Agents return to isolation mode — no more reading shared/best/

## Best Practices

- Time sharing rounds strategically — after cross-reviews identify combinable work
- Don't run sharing rounds too often — isolation drives diversity
- Review the provenance chain in registry.json after each round
