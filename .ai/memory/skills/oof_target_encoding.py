"""Leak-free out-of-fold target encoding, pure Python (no deps)."""

META = {
    "id": "oof-target-encoding",
    "claim": "Target encoding computed strictly out-of-fold (with smoothing toward the "
             "out-of-fold prior) adds signal on mid/high-cardinality categoricals "
             "without leaking the row's own target into its feature.",
    "scope": {"task_type": "tabular", "metric_family": "any", "modality": "tabular"},
    "status": "candidate",
    "provenance": "standard Kaggle tabular practice; seeded at template build (2026-07-04)",
    "created": "2026-07-04",
}


def oof_target_encode(categories, targets, fold_ids, smoothing=10.0):
    """Encode each row's category using target statistics from the OTHER folds only.

    encoded[i] = (n_c * mean_c + smoothing * prior) / (n_c + smoothing)
    where n_c, mean_c, and prior are computed excluding row i's fold.
    Unseen categories fall back to the out-of-fold prior.
    """
    n = len(categories)
    if not (n == len(targets) == len(fold_ids)):
        raise ValueError("categories, targets, fold_ids must have equal length")

    folds = sorted(set(fold_ids))
    encoded = [0.0] * n
    for fold in folds:
        sums, counts, total, count_total = {}, {}, 0.0, 0
        for c, t, f in zip(categories, targets, fold_ids):
            if f == fold:
                continue
            sums[c] = sums.get(c, 0.0) + t
            counts[c] = counts.get(c, 0) + 1
            total += t
            count_total += 1
        if count_total == 0:
            raise ValueError(f"fold {fold} covers all rows — need >= 2 folds")
        prior = total / count_total
        for i, (c, f) in enumerate(zip(categories, fold_ids)):
            if f != fold:
                continue
            n_c = counts.get(c, 0)
            mean_c = sums[c] / n_c if n_c else prior
            encoded[i] = (n_c * mean_c + smoothing * prior) / (n_c + smoothing)
    return encoded


def self_test():
    cats = ["a", "a", "b", "b", "a", "b", "c"]
    tgts = [1.0, 0.0, 1.0, 0.0, 1.0, 1.0, 0.0]
    folds = [0, 0, 1, 1, 2, 2, 2]

    enc = oof_target_encode(cats, tgts, folds, smoothing=2.0)
    assert len(enc) == len(cats)

    # Leak check: flipping row 0's target must NOT change encodings of any row
    # in row 0's own fold (their stats exclude fold 0), but MUST change at
    # least one row in another fold (their stats include fold 0).
    tgts2 = list(tgts)
    tgts2[0] = 0.0
    enc2 = oof_target_encode(cats, tgts2, folds, smoothing=2.0)
    assert enc[0] == enc2[0] and enc[1] == enc2[1], "own-fold encoding leaked own target"
    assert any(abs(a - b) > 1e-12 for a, b in zip(enc[2:], enc2[2:])), \
        "other folds should see the changed target"

    # Unseen category falls back to the out-of-fold prior.
    prior_fold2 = sum(t for t, f in zip(tgts, folds) if f != 2) / 4
    assert abs(enc[6] - prior_fold2) < 1e-12, "unseen category must get the OOF prior"

    # Smoothing pulls small categories toward the prior.
    enc_hi = oof_target_encode(cats, tgts, folds, smoothing=1000.0)
    assert abs(enc_hi[0] - prior_fold_of(tgts, folds, 0)) < 0.01, \
        "high smoothing should approach the prior"


def prior_fold_of(tgts, folds, fold):
    vals = [t for t, f in zip(tgts, folds) if f != fold]
    return sum(vals) / len(vals)


if __name__ == "__main__":
    self_test()
    print("oof-target-encoding: self_test passed")
