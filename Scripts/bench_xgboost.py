# Copyright (c) 2026 DevOnBike.
# Reference XGBoost benchmark — establishes the TARGET a future pure-managed Overfit GBDT
# would aim at. We have no GBDT yet, so this is a one-sided baseline ("where we aim"),
# not an A/B. Focus = INFERENCE (our moat: zero-alloc managed inference), training secondary.
#
# Run:  python Scripts/bench_xgboost.py
# Deps: xgboost, numpy (no sklearn/pandas needed — data is synthetic).

import time
import json
import os
import tempfile
import numpy as np
import xgboost as xgb


def make_tabular(n_rows, n_features, seed):
    # Synthetic binary-classification tabular task with a genuine non-linear signal
    # over a handful of informative features (the rest are noise) — representative of
    # real structured data where GBDTs shine.
    rng = np.random.default_rng(seed)
    x = rng.standard_normal((n_rows, n_features)).astype(np.float32)
    # Non-linear interaction signal on features 0..4.
    logit = (
        1.3 * x[:, 0] * x[:, 1]
        - 1.1 * np.sin(2.0 * x[:, 2])
        + 0.9 * (x[:, 3] ** 2 - 1.0)
        + 0.7 * x[:, 4]
        + 0.5 * x[:, 0]
    )
    p = 1.0 / (1.0 + np.exp(-logit))
    y = (rng.random(n_rows) < p).astype(np.float32)
    return x, y


def best_of(fn, repeats, warmup=2):
    for _ in range(warmup):
        fn()
    times = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        fn()
        times.append(time.perf_counter() - t0)
    return min(times), float(np.median(times))


def main():
    n_train, n_test, n_feat = 100_000, 50_000, 50
    x_tr, y_tr = make_tabular(n_train, n_feat, seed=1)
    x_te, y_te = make_tabular(n_test, n_feat, seed=2)

    n_estimators, max_depth = 300, 6
    dtr = xgb.DMatrix(x_tr, label=y_tr)
    dte = xgb.DMatrix(x_te, label=y_te)

    params = {
        "objective": "binary:logistic",
        "max_depth": max_depth,
        "eta": 0.1,
        "tree_method": "hist",
        "eval_metric": "logloss",
        "nthread": 0,  # all cores for training
    }

    # ── Train (reference, multi-thread) ────────────────────────────────────────
    t0 = time.perf_counter()
    booster = xgb.train(params, dtr, num_boost_round=n_estimators)
    train_s = time.perf_counter() - t0

    # ── Accuracy ───────────────────────────────────────────────────────────────
    proba = booster.predict(dte)
    pred = (proba >= 0.5).astype(np.float32)
    acc = float(np.mean(pred == y_te))
    # AUC (rank-based, no sklearn).
    order = np.argsort(proba)
    ranks = np.empty_like(order, dtype=np.float64)
    ranks[order] = np.arange(1, len(proba) + 1)
    pos = y_te == 1.0
    n_pos, n_neg = int(pos.sum()), int((~pos).sum())
    auc = (ranks[pos].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)

    # ── Model characteristics (the "work" our predictor must do) ───────────────
    dump = booster.get_dump(with_stats=False)
    total_nodes = sum(d.count("\n") + 1 for d in dump)
    leaf_nodes = sum(d.count("leaf=") for d in dump)
    internal_nodes = total_nodes - leaf_nodes
    with tempfile.TemporaryDirectory() as td:
        mp = os.path.join(td, "m.json")
        booster.save_model(mp)
        model_bytes = os.path.getsize(mp)

    # ── Inference: clean single-core baseline (our managed target) ─────────────
    booster.set_param({"nthread": 1})
    x_one = x_te[:1]
    d_one = xgb.DMatrix(x_one)

    # Online latency — single row, includes inplace_predict path (no DMatrix build).
    one_min, one_med = best_of(lambda: booster.inplace_predict(x_one), repeats=200)
    # Batch throughput — all test rows at once, single thread.
    batch_min, batch_med = best_of(lambda: booster.inplace_predict(x_te), repeats=20)

    # ── Inference: default multi-thread, for completeness ──────────────────────
    booster.set_param({"nthread": 0})
    batch_mt_min, _ = best_of(lambda: booster.inplace_predict(x_te), repeats=20)

    print("=" * 64)
    print("XGBoost reference benchmark — TARGET for a pure-managed Overfit GBDT")
    print("=" * 64)
    print(f"xgboost {xgb.__version__}   cores={os.cpu_count()}")
    print(f"task     binary classification, non-linear signal")
    print(f"data     train={n_train:,}  test={n_test:,}  features={n_feat}")
    print(f"model    {n_estimators} trees x max_depth {max_depth}")
    print("-" * 64)
    print(f"accuracy {acc*100:.2f}%    AUC {auc:.4f}")
    print(f"trees    internal_nodes={internal_nodes:,}  leaves={leaf_nodes:,}")
    print(f"size     {model_bytes/1024:.1f} KB (JSON)")
    print("-" * 64)
    print(f"TRAIN    {train_s*1000:.0f} ms  ({n_estimators} rounds, all cores)")
    print("-" * 64)
    print("INFERENCE (single-thread = clean per-core target)")
    print(f"  online  1 row : {one_min*1e6:8.1f} us/row (best)   {one_med*1e6:8.1f} us (median)")
    rps = n_test / batch_min
    print(f"  batch   {n_test:,} rows : {batch_min*1e3:7.2f} ms  -> {rps:,.0f} rows/s  ({batch_min/n_test*1e9:.0f} ns/row)")
    print("INFERENCE (all cores)")
    rps_mt = n_test / batch_mt_min
    print(f"  batch   {n_test:,} rows : {batch_mt_min*1e3:7.2f} ms  -> {rps_mt:,.0f} rows/s  ({batch_mt_min/n_test*1e9:.0f} ns/row)")
    print("=" * 64)


if __name__ == "__main__":
    main()
