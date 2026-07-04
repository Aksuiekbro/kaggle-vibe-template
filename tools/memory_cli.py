#!/usr/bin/env python3
"""Curated-memory engine: scope-matched, trust-weighted card retrieval with a
predicted-vs-actual write-back loop.

Trust is computed from each card's own prediction track record, not from an
LLM's opinion of importance. Cards that keep missing get downgraded
automatically; counter-evidence always rides along with retrieval.

Usage:
  python tools/memory_cli.py retrieve  [--task-type t] [--metric-family m] [--modality mo]
                                       [--split-risk s] [--anchor NAME] [--top N] [--loose]
  python tools/memory_cli.py writeback --card ID --competition SLUG --result hit|partial|miss
                                       [--predicted-delta X] [--actual-delta Y] [--note "..."]
  python tools/memory_cli.py promote   --card ID --reviewer NAME [--verdict "..."]
  python tools/memory_cli.py demote    --card ID --to candidate|stale|rejected --reason "..."
  python tools/memory_cli.py log-miss  --card ID --competition SLUG --reason "why it should have surfaced"
  python tools/memory_cli.py sweep     [--dry-run]
  python tools/memory_cli.py revalidation-due --agent <name>
  python tools/memory_cli.py stats
  python tools/memory_cli.py validate
  python tools/memory_cli.py amend-proposals
  python tools/memory_cli.py dedup
  python tools/memory_cli.py new       --id slug --claim "..." [--folder patterns]
"""

import argparse
import json
import os
import re
import shlex
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from discipline import (
    CONSTITUTION_LEDGER,
    CONSTITUTION_PATH,
    MEMORY_ROOT,
    PROJECT_ROOT,
    append_jsonl,
    ledger_event,
    now_iso,
    read_jsonl,
)

CARD_FOLDERS = ["patterns", "failures", "competitions"]
STATUSES = ["candidate", "validated", "rejected", "superseded", "stale"]
STATUS_WEIGHT = {"validated": 1.0, "candidate": 0.6, "stale": 0.4}
SCOPE_DIMS = ["task_type", "metric_family", "modality", "split_risk"]
STALE_AFTER_COMPETITIONS = 5
CONSECUTIVE_MISSES_TO_DOWNGRADE = 3
RETRIEVAL_MISSES = MEMORY_ROOT / "RETRIEVAL_MISSES.jsonl"
DEDUP_STOPWORDS = {"the", "a", "an", "of", "to", "in", "on", "for", "and", "or", "is", "are", "with", "by"}

ANCHORS = {
    "classify": "Competition just classified — retrieve applicable strategy patterns.",
    "stuck": "Stuck / pivoting — retrieve untested candidate ideas and known failures.",
    "pre-submit": "About to submit — retrieve leakage, overfit, and shake-up failure cards.",
    "experiment-failed": "An experiment failed — retrieve related failure cards.",
    "postmortem": "Postmortem — list cards awaiting predicted-vs-actual write-back.",
}


# ---------------------------------------------------------------------------
# Minimal YAML subset parser/serializer (card schema only; no external deps)
# ---------------------------------------------------------------------------

def _parse_value(raw):
    raw = raw.strip()
    if raw in ("", "~", "null"):
        return ""
    if raw == "[]":
        return []
    if raw == "{}":
        return {}
    if len(raw) >= 2 and raw[0] == raw[-1] and raw[0] in "\"'":
        return raw[1:-1]
    return raw


def parse_yaml_block(text):
    """Parse the card-schema YAML subset: nested dicts, lists of scalars/dicts."""
    lines = [l for l in text.split("\n") if l.strip() and not l.strip().startswith("#")]
    pos = [0]

    def parse_block(indent):
        result = None
        while pos[0] < len(lines):
            line = lines[pos[0]]
            cur_indent = len(line) - len(line.lstrip())
            stripped = line.strip()
            if cur_indent < indent:
                break
            if cur_indent > indent:
                # unexpected deeper line without a key — skip defensively
                pos[0] += 1
                continue
            if stripped.startswith("- "):
                if result is None:
                    result = []
                if not isinstance(result, list):
                    break
                item_text = stripped[2:]
                pos[0] += 1
                if ":" in item_text:
                    key, _, val = item_text.partition(":")
                    item = {key.strip(): _parse_value(val)}
                    cont = parse_block(indent + 2)
                    if isinstance(cont, dict):
                        item.update(cont)
                    result.append(item)
                else:
                    result.append(_parse_value(item_text))
            elif ":" in stripped:
                if result is None:
                    result = {}
                if not isinstance(result, dict):
                    break
                key, _, val = stripped.partition(":")
                pos[0] += 1
                if val.strip():
                    result[key.strip()] = _parse_value(val)
                else:
                    child = parse_block(indent + 2)
                    result[key.strip()] = child if child is not None else ""
            else:
                pos[0] += 1
        return result

    parsed = parse_block(0)
    return parsed if isinstance(parsed, dict) else {}


def dump_yaml_block(data, indent=0):
    lines = []
    pad = " " * indent
    for key, value in data.items():
        if isinstance(value, dict):
            if value:
                lines.append(f"{pad}{key}:")
                lines.append(dump_yaml_block(value, indent + 2))
            else:
                lines.append(f"{pad}{key}: {{}}")
        elif isinstance(value, list):
            if not value:
                lines.append(f"{pad}{key}: []")
            else:
                lines.append(f"{pad}{key}:")
                for item in value:
                    if isinstance(item, dict):
                        first = True
                        for k, v in item.items():
                            if first:
                                lines.append(f"{pad}  - {k}: {v}")
                                first = False
                            else:
                                lines.append(f"{pad}    {k}: {v}")
                    else:
                        lines.append(f"{pad}  - {item}")
        else:
            lines.append(f"{pad}{key}: {value}")
    return "\n".join(lines)


YAML_FENCE_RE = re.compile(r"```yaml\n(.*?)```", re.DOTALL)


def load_card(path):
    text = path.read_text()
    match = YAML_FENCE_RE.search(text)
    if not match:
        return None
    card = parse_yaml_block(match.group(1))
    card["_path"] = path
    card["_folder"] = path.parent.name
    return card


def save_card(card):
    path = card.pop("_path")
    card.pop("_folder", None)
    text = path.read_text()
    new_yaml = dump_yaml_block(card) + "\n"
    new_text = YAML_FENCE_RE.sub("```yaml\n" + new_yaml + "```", text, count=1)
    path.write_text(new_text)


def all_cards():
    cards = []
    for folder in CARD_FOLDERS:
        d = MEMORY_ROOT / folder
        if not d.exists():
            continue
        for path in sorted(d.glob("*.md")):
            if path.name == "README.md":
                continue
            card = load_card(path)
            if card:
                cards.append(card)
    return cards


def find_card(card_id):
    for card in all_cards():
        if card.get("id") == card_id or card["_path"].stem == card_id:
            return card
    return None


# ---------------------------------------------------------------------------
# Scoring: trust from track record, recency from competition count
# ---------------------------------------------------------------------------

def prediction_results(card):
    preds = card.get("predictions", [])
    if not isinstance(preds, list):
        return []
    return [p.get("result", "").lower() for p in preds if isinstance(p, dict) and p.get("result")]


def trust_score(card):
    """Laplace-smoothed hit rate over the card's own predicted-vs-actual record."""
    results = prediction_results(card)
    hits = sum(1 for r in results if r == "hit")
    partials = sum(1 for r in results if r == "partial")
    return (1.0 + hits + 0.5 * partials) / (2.0 + len(results))


def competitions_since_validation(card):
    """Staleness measured in competitions, not wall-clock: how many postmortem
    files landed in memory/competitions/ after this card was last validated."""
    comp_dir = MEMORY_ROOT / "competitions"
    if not comp_dir.exists():
        return 0
    stamp = card.get("last_validated") or card.get("created") or ""
    try:
        ref = datetime.fromisoformat(str(stamp).replace("Z", "+00:00"))
        if ref.tzinfo is None:
            ref = ref.replace(tzinfo=timezone.utc)
    except ValueError:
        return 0
    count = 0
    for f in comp_dir.glob("*.md"):
        if f.name == "README.md":
            continue
        mtime = datetime.fromtimestamp(f.stat().st_mtime, tz=timezone.utc)
        if mtime > ref:
            count += 1
    return count


def recency_factor(card):
    return max(0.3, 1.0 - 0.15 * competitions_since_validation(card))


def scope_match(card, query, loose=False):
    """Product over queried scope dims. Mismatch excludes unless --loose."""
    scope = card.get("scope", {}) or {}
    score = 1.0
    for dim in SCOPE_DIMS:
        want = query.get(dim)
        if not want:
            continue
        have = str(scope.get(dim, "")).lower().strip()
        want = want.lower().strip()
        if not have:
            score *= 0.5
        elif have in ("any", "*"):
            score *= 0.75
        elif want in have or have in want:
            score *= 1.0
        else:
            if not loose:
                return 0.0
            score *= 0.2
    return score


def retrieval_score(card, query, loose=False):
    status = str(card.get("status", "candidate")).lower()
    if status in ("rejected", "superseded") and card["_folder"] != "failures":
        return 0.0
    sm = scope_match(card, query, loose)
    if sm == 0.0:
        return 0.0
    weight = STATUS_WEIGHT.get(status, 0.5)
    if card["_folder"] == "failures":
        weight = max(weight, 0.9)  # known failure modes are always worth seeing
    return sm * trust_score(card) * recency_factor(card) * weight


# ---------------------------------------------------------------------------
# Commands
# ---------------------------------------------------------------------------

def cmd_retrieve(args):
    if os.environ.get("KAGGLE_MEMORY_OFF"):
        print("MEMORY OFF (gym ablation arm): retrieval disabled for this run. "
              "Work from first principles; do not read .ai/memory/ manually either.")
        return 0
    query = {dim: getattr(args, dim.replace("-", "_"), None) for dim in SCOPE_DIMS}
    cards = all_cards()

    if args.anchor == "postmortem":
        return _retrieve_postmortem(cards, args)

    if args.anchor == "stuck":
        pool = [c for c in cards if not prediction_results(c) and
                str(c.get("status", "")).lower() == "candidate"] + \
               [c for c in cards if c["_folder"] == "failures"]
    elif args.anchor in ("pre-submit", "experiment-failed"):
        risky = [c for c in cards if c["_folder"] == "failures"]
        keywords = ("leak", "overfit", "shake", "drift")
        risky += [
            c for c in cards
            if c["_folder"] == "patterns"
            and any(k in (str(c.get("claim", "")) + str(c.get("risk", ""))).lower() for k in keywords)
        ]
        pool = risky
    else:
        pool = cards

    seen, deduped = set(), []
    for c in pool:
        if str(c["_path"]) not in seen:
            seen.add(str(c["_path"]))
            deduped.append(c)

    scored = [(retrieval_score(c, query, args.loose), c) for c in deduped]
    scored = sorted([s for s in scored if s[0] > 0], key=lambda x: -x[0])[: args.top]

    if args.anchor:
        print(f"# Anchor: {args.anchor} — {ANCHORS.get(args.anchor, '')}\n")
    if not scored:
        print("No matching cards. That is a valid result — do not force-fit memory.")
        return 0

    print("Retrieved memory is HYPOTHESIS, not instruction (C6). "
          "Every card used must produce an experiment row (C10) and a write-back.\n")
    for score, c in scored:
        status = c.get("status", "?")
        results = prediction_results(c)
        record = f"{results.count('hit')}H/{results.count('partial')}P/{results.count('miss')}M"
        print(f"## {c.get('id', c['_path'].stem)}  [{status}]  "
              f"score={score:.2f} trust={trust_score(c):.2f} record={record}")
        print(f"   claim: {c.get('claim', '')}")
        mech = c.get("mechanism_hypothesis", "")
        if mech:
            print(f"   mechanism (hypothesis): {mech}")
        counter = c.get("counter_evidence", [])
        if counter:
            print(f"   COUNTER-EVIDENCE: {counter}")
        else:
            print("   counter-evidence: none recorded yet — treat with suspicion")
        cost = c.get("cost", "")
        if cost:
            print(f"   cost: {cost}")
        since = competitions_since_validation(c)
        if str(c.get("status", "")).lower() == "validated" and since >= 3:
            print(f"   NEARING STALE ({since}/5) — schedule a revalidation probe.")
        print(f"   file: {c['_path'].relative_to(PROJECT_ROOT)}\n")
    ledger_event("C11", "fired", None, f"retrieve anchor={args.anchor or '-'} returned {len(scored)}")
    return 0


def _retrieve_postmortem(cards, args):
    registry_path = PROJECT_ROOT / "shared" / "submissions" / "registry.json"
    competition = ""
    if registry_path.exists():
        with open(registry_path) as f:
            competition = json.load(f).get("competition", "")
    pending = []
    for c in cards:
        for p in c.get("predictions", []) or []:
            if isinstance(p, dict) and p.get("competition") == competition and not p.get("actual_delta"):
                pending.append((c, p))
    if not pending:
        print(f"No cards awaiting write-back for competition '{competition}'.")
        return 0
    print(f"Cards awaiting predicted-vs-actual write-back for '{competition}':\n")
    for c, p in pending:
        print(f"  {c.get('id')}: predicted {p.get('predicted_delta', '?')} — "
              f"run: python tools/memory_cli.py writeback --card {c.get('id')} "
              f"--competition {competition} --result <hit|partial|miss> --actual-delta <x>")
    return 0


def cmd_writeback(args):
    card = find_card(args.card)
    if not card:
        print(f"ERROR: card not found: {args.card}")
        return 1
    preds = card.get("predictions")
    if not isinstance(preds, list):
        preds = []
    # drop blank template placeholder entries
    preds = [p for p in preds if not (isinstance(p, dict) and not p.get("result") and not p.get("competition"))]

    entry = {
        "date": now_iso()[:10],
        "competition": args.competition,
        "predicted_delta": args.predicted_delta or "",
        "actual_delta": args.actual_delta or "",
        "result": args.result,
    }
    if args.note:
        entry["note"] = args.note
    preds.append(entry)
    card["predictions"] = preds

    if args.result in ("hit", "partial"):
        card["last_validated"] = now_iso()[:10]
    else:
        counter = card.get("counter_evidence")
        if not isinstance(counter, list):
            counter = []
        counter.append(f"{args.competition}: predicted {args.predicted_delta or '?'}, "
                       f"got {args.actual_delta or 'miss'}")
        card["counter_evidence"] = counter

    results = prediction_results(card)
    downgraded = ""
    if (len(results) >= CONSECUTIVE_MISSES_TO_DOWNGRADE
            and all(r == "miss" for r in results[-CONSECUTIVE_MISSES_TO_DOWNGRADE:])):
        old = str(card.get("status", "candidate")).lower()
        new = "candidate" if old == "validated" else "rejected"
        card["status"] = new
        downgraded = f" — {CONSECUTIVE_MISSES_TO_DOWNGRADE} consecutive misses, auto-downgraded {old} -> {new}"

    save_card(dict(card))
    ledger_event("C10", "fired", None, f"writeback {args.card} {args.result}{downgraded}")
    print(f"Write-back recorded on {args.card}: {args.result}{downgraded}")
    print(f"Trust is now {trust_score(find_card(args.card)):.2f}")
    return 0


def cmd_promote(args):
    card = find_card(args.card)
    if not card:
        print(f"ERROR: card not found: {args.card}")
        return 1
    evidence = card.get("evidence", [])
    has_measurement = any(
        isinstance(e, dict) and (e.get("our_cv_delta") or e.get("our_lb_delta"))
        for e in (evidence if isinstance(evidence, list) else [])
    ) or any(r in ("hit", "partial") for r in prediction_results(card))
    if not has_measurement:
        print("REJECTED (C8): cannot promote without a measured delta "
              "(evidence with our_cv_delta/our_lb_delta, or a hit/partial write-back).")
        ledger_event("C8", "prevented", args.reviewer, f"promote blocked for {args.card}")
        return 1
    if not args.reviewer:
        print("REJECTED (C8): promotion requires --reviewer (cross-review, not self-review).")
        return 1

    card["status"] = "validated"
    card["last_validated"] = now_iso()[:10]
    card["review"] = {
        "reviewer": args.reviewer,
        "date": now_iso()[:10],
        "verdict": args.verdict or "approved",
    }
    save_card(dict(card))
    ledger_event("C8", "fired", args.reviewer, f"promoted {args.card} to validated")
    print(f"PROMOTED: {args.card} -> validated (reviewer: {args.reviewer})")
    return 0


def cmd_demote(args):
    card = find_card(args.card)
    if not card:
        print(f"ERROR: card not found: {args.card}")
        return 1
    card["status"] = args.to
    counter = card.get("counter_evidence")
    if not isinstance(counter, list):
        counter = []
    counter.append(f"demoted to {args.to} on {now_iso()[:10]}: {args.reason}")
    card["counter_evidence"] = counter
    save_card(dict(card))
    ledger_event("C12", "fired", None, f"demoted {args.card} to {args.to}")
    print(f"DEMOTED: {args.card} -> {args.to}")
    return 0


def cmd_log_miss(args):
    card = find_card(args.card)
    if not card:
        print(f"ERROR: card not found: {args.card}")
        return 1
    record = {
        "ts": now_iso(),
        "card": card.get("id", args.card),
        "competition": args.competition,
        "reason": args.reason,
    }
    append_jsonl(RETRIEVAL_MISSES, record)
    print(f"Logged retrieval miss: {record['card']} on {args.competition}")
    return 0


def cmd_sweep(args):
    changed = 0
    for card in all_cards():
        if str(card.get("status", "")).lower() != "validated":
            continue
        since = competitions_since_validation(card)
        if since >= STALE_AFTER_COMPETITIONS:
            print(f"STALE: {card.get('id')} — {since} competitions since last validation")
            if not args.dry_run:
                card["status"] = "stale"
                save_card(dict(card))
                ledger_event("C12", "fired", None, f"sweep staled {card.get('id')}")
            changed += 1
    print(f"Sweep complete: {changed} card(s) {'would be' if args.dry_run else ''} marked stale.")
    return 0


def _last_hit_delta(card):
    for pred in reversed(card.get("predictions", []) or []):
        if not isinstance(pred, dict) or pred.get("result") != "hit":
            continue
        try:
            return float(pred.get("actual_delta"))
        except (TypeError, ValueError):
            return 0.001
    return 0.001


def cmd_revalidation_due(args):
    due = [
        c for c in all_cards()
        if str(c.get("status", "")).lower() == "validated"
        and competitions_since_validation(c) >= 3
    ]
    if not due:
        print("No validated cards are nearing stale.")
        return 0
    for card in sorted(due, key=lambda c: -competitions_since_validation(c)):
        cid = card.get("id", card["_path"].stem)
        since = competitions_since_validation(card)
        claim = str(card.get("claim", "")).strip()
        idea = f"revalidate: {claim[:60]}"
        print(f"# {cid} — NEARING STALE ({since}/5)")
        print(
            "python tools/scheduler.py add "
            f"--agent {args.agent} "
            f"--idea {shlex.quote(idea)} "
            f"--predicted-delta {_last_hit_delta(card)} "
            "--cost-hours 0.5 "
            f"--source card:{cid}"
        )
    return 0


def cmd_stats(args):
    cards = all_cards()
    if not cards:
        print("No memory cards yet.")
        return 0
    print(f"{'id':<40} {'status':<11} {'trust':<6} {'record':<10} {'stale-in':<9} folder")
    print("-" * 90)
    for c in sorted(cards, key=lambda c: -trust_score(c)):
        results = prediction_results(c)
        record = f"{results.count('hit')}H/{results.count('partial')}P/{results.count('miss')}M"
        remaining = max(0, STALE_AFTER_COMPETITIONS - competitions_since_validation(c))
        print(f"{str(c.get('id', c['_path'].stem)):<40} {str(c.get('status', '?')):<11} "
              f"{trust_score(c):<6.2f} {record:<10} {remaining:<9} {c['_folder']}")
    return 0


def cmd_validate(args):
    errors = []
    for card in all_cards():
        cid = card.get("id", card["_path"].stem)
        rel = card["_path"].relative_to(PROJECT_ROOT)
        if not card.get("id"):
            errors.append(f"{rel}: missing id")
        if not card.get("claim"):
            errors.append(f"{rel}: missing claim — a card is a falsifiable claim, not a note")
        if str(card.get("status", "")).lower() not in STATUSES:
            errors.append(f"{rel}: invalid status '{card.get('status')}'")
        scope = card.get("scope", {})
        if not isinstance(scope, dict) or not scope.get("task_type"):
            errors.append(f"{rel}: scope.task_type is required (applicability predicate)")
        if "counter_evidence" not in card:
            errors.append(f"{rel}: counter_evidence field missing (must exist even when empty)")
    if errors:
        print(f"Card validation: {len(errors)} error(s)")
        for e in errors:
            print(f"  - {e}")
        return 1
    print(f"Card validation: all {len(all_cards())} card(s) pass schema.")
    return 0


def cmd_amend_proposals(args):
    """Consolidation-time proposals: promote proven cards toward the constitution
    or a gate; demote constitution rules that stopped paying rent."""
    print("# Amendment Proposals\n")

    promoted = []
    for card in all_cards():
        if str(card.get("status", "")).lower() != "validated":
            continue
        evidence = card.get("evidence", [])
        comps = {e.get("competition") for e in evidence if isinstance(e, dict) and e.get("competition")}
        comps |= {p.get("competition") for p in card.get("predictions", []) or []
                  if isinstance(p, dict) and p.get("result") in ("hit", "partial")}
        if len(comps) >= 2 and trust_score(card) >= 0.7:
            promoted.append((card, comps))
    if promoted:
        print("## Promote toward constitution / gate")
        for card, comps in promoted:
            print(f"  - {card.get('id')}: trust={trust_score(card):.2f}, "
                  f"evidence across {len(comps)} competitions — "
                  "consider a constitution rule, a lint check, or a gate")
    else:
        print("## Promote toward constitution / gate\n  (none yet — needs validated cards "
              "with evidence across >=2 competitions and trust >= 0.7)")

    print("\n## Constitution rent check")
    rule_ids = []
    if CONSTITUTION_PATH.exists():
        rule_ids = re.findall(r"^### (C\d+)", CONSTITUTION_PATH.read_text(), re.MULTILINE)
    events = read_jsonl(CONSTITUTION_LEDGER)
    fired = {}
    for e in events:
        fired.setdefault(e.get("rule"), []).append(e)
    for rule in rule_ids:
        n = len(fired.get(rule, []))
        marker = "NEVER FIRED — demotion candidate" if n == 0 else f"{n} event(s)"
        print(f"  - {rule}: {marker}")
    if not rule_ids:
        print("  (no constitution found)")

    print("\n## Skill demotion check")
    usage = read_jsonl(MEMORY_ROOT / "skills" / "USAGE.jsonl")
    by_skill = {}
    for u in usage:
        by_skill.setdefault(u.get("skill"), []).append(u.get("outcome"))
    flagged = False
    for skill, outcomes in by_skill.items():
        wins = outcomes.count("win")
        if len(outcomes) >= 3 and wins / len(outcomes) < 0.4:
            print(f"  - {skill}: win-rate {wins}/{len(outcomes)} — demotion candidate")
            flagged = True
    if not flagged:
        print("  (no skills below the win-rate floor)")

    print("\n## Retrieval health")
    misses = read_jsonl(RETRIEVAL_MISSES)
    recent = misses
    run_root = PROJECT_ROOT / ".ai" / "runs"
    consolidation_dirs = []
    if run_root.exists():
        consolidation_dirs = sorted(
            [p for p in run_root.glob("*/consolidation") if p.is_dir()],
            key=lambda p: p.parent.name,
        )
    if len(consolidation_dirs) >= 2:
        cutoff_name = consolidation_dirs[-2].parent.name
        try:
            cutoff = datetime.strptime(cutoff_name, "%Y%m%dT%H%M%SZ").replace(tzinfo=timezone.utc)
            recent = []
            for miss in misses:
                try:
                    ts = datetime.fromisoformat(str(miss.get("ts", "")).replace("Z", "+00:00"))
                    if ts.tzinfo is None:
                        ts = ts.replace(tzinfo=timezone.utc)
                except ValueError:
                    continue
                if ts >= cutoff:
                    recent.append(miss)
        except ValueError:
            recent = misses
    print(f"  Total misses: {len(misses)}")
    print(f"  Misses in last 2 consolidations: {len(recent)}")
    print("  ≥3 misses across 2 consolidations, or corpus >100 cards → "
          "evaluate a derived retrieval index (see plan Task 5).")
    return 0


def _claim_tokens(claim):
    words = re.findall(r"[a-z0-9]+", str(claim).lower())
    return {w for w in words if w and w not in DEDUP_STOPWORDS}


def _jaccard(a, b):
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def cmd_dedup(args):
    cards = all_cards()
    flagged = []
    for i, left in enumerate(cards):
        left_tokens = _claim_tokens(left.get("claim", ""))
        for right in cards[i + 1:]:
            score = _jaccard(left_tokens, _claim_tokens(right.get("claim", "")))
            if score >= 0.5:
                flagged.append((score, left, right))

    if not flagged:
        print("No near-duplicate memory claims found.")
        return 0

    for score, left, right in sorted(flagged, key=lambda x: -x[0]):
        left_id = left.get("id", left["_path"].stem)
        right_id = right.get("id", right["_path"].stem)
        print(f"{left_id} <-> {right_id}: similarity={score:.2f}")
        print(f"  {left_id}: {left.get('claim', '')}")
        print(f"  {right_id}: {right.get('claim', '')}")
        print("  merge evidence into the older card, mark the newer `superseded` "
              "(`demote --to superseded`), set `superseded_by`.")
    return 1


def cmd_new(args):
    template = MEMORY_ROOT / "templates" / "pattern-card.md"
    folder = MEMORY_ROOT / args.folder
    folder.mkdir(parents=True, exist_ok=True)
    dest = folder / f"{args.id}.md"
    if dest.exists():
        print(f"ERROR: {dest} already exists")
        return 1
    text = template.read_text() if template.exists() else "# Pattern Card\n\n```yaml\nid:\n```\n"
    text = text.replace("id:", f"id: {args.id}", 1)
    text = text.replace("created:", f"created: {now_iso()[:10]}", 1)
    text = text.replace("claim:", f"claim: {args.claim}", 1)
    dest.write_text(text)
    print(f"Created {dest.relative_to(PROJECT_ROOT)} (status: candidate). "
          "Fill scope, mechanism_hypothesis, and evidence.")
    return 0


def main():
    parser = argparse.ArgumentParser(description="Curated-memory engine")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("retrieve")
    p.add_argument("--task-type")
    p.add_argument("--metric-family")
    p.add_argument("--modality")
    p.add_argument("--split-risk")
    p.add_argument("--anchor", choices=sorted(ANCHORS))
    p.add_argument("--top", type=int, default=5)
    p.add_argument("--loose", action="store_true")
    p.set_defaults(func=cmd_retrieve)

    p = sub.add_parser("writeback")
    p.add_argument("--card", required=True)
    p.add_argument("--competition", required=True)
    p.add_argument("--result", required=True, choices=["hit", "partial", "miss"])
    p.add_argument("--predicted-delta")
    p.add_argument("--actual-delta")
    p.add_argument("--note")
    p.set_defaults(func=cmd_writeback)

    p = sub.add_parser("promote")
    p.add_argument("--card", required=True)
    p.add_argument("--reviewer", required=True)
    p.add_argument("--verdict")
    p.set_defaults(func=cmd_promote)

    p = sub.add_parser("demote")
    p.add_argument("--card", required=True)
    p.add_argument("--to", required=True, choices=["candidate", "stale", "rejected", "superseded"])
    p.add_argument("--reason", required=True)
    p.set_defaults(func=cmd_demote)

    p = sub.add_parser("log-miss")
    p.add_argument("--card", required=True)
    p.add_argument("--competition", required=True)
    p.add_argument("--reason", required=True)
    p.set_defaults(func=cmd_log_miss)

    p = sub.add_parser("sweep")
    p.add_argument("--dry-run", action="store_true")
    p.set_defaults(func=cmd_sweep)

    p = sub.add_parser("revalidation-due")
    p.add_argument("--agent", required=True, choices=["claude", "codex", "gemini"])
    p.set_defaults(func=cmd_revalidation_due)

    sub.add_parser("stats").set_defaults(func=cmd_stats)
    sub.add_parser("validate").set_defaults(func=cmd_validate)
    sub.add_parser("amend-proposals").set_defaults(func=cmd_amend_proposals)
    sub.add_parser("dedup").set_defaults(func=cmd_dedup)

    p = sub.add_parser("new")
    p.add_argument("--id", required=True)
    p.add_argument("--claim", required=True)
    p.add_argument("--folder", default="patterns", choices=CARD_FOLDERS)
    p.set_defaults(func=cmd_new)

    args = parser.parse_args()
    sys.exit(args.func(args))


if __name__ == "__main__":
    main()
