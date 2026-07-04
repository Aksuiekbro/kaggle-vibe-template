# Skill Library

Executable, verified procedures — the strongest form of memory this template has.
A skill is code, not prose: it carries a `META` dict (falsifiable claim + scope +
provenance + status) and a `self_test()` that re-verifies it on every
`python tools/skills.py test` run. A skill that cannot be self-tested is a note,
not a skill — put it in `patterns/` instead.

## Format

```python
META = {
    "id": "kebab-case-id",
    "claim": "One falsifiable sentence about when this code helps.",
    "scope": {"task_type": "tabular", "metric_family": "any"},
    "status": "candidate",   # candidate | validated | rejected
    "provenance": "where this came from (writeup URL, postmortem, agent)",
    "created": "YYYY-MM-DD",
}

def the_skill(...):
    ...

def self_test():
    # must raise on failure
    ...
```

## Lifecycle

- New skills enter as `candidate` with a passing `self_test()`.
- `python tools/skills.py log-use --skill ID --competition SLUG --outcome win|neutral|loss`
  after each real use — this builds the cross-competition win-rate.
- Win-rate < 0.4 over >= 3 uses flags the skill for demotion at consolidation
  (`python tools/memory_cli.py amend-proposals`).
- Promotion to `validated` follows the same cross-review rule as memory cards (C8).

Skills are re-verified continuously, scoped, and demoted on real-world losses —
verification is a regression suite, not a one-time event.
