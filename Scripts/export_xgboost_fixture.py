# Copyright (c) 2026 DevOnBike.
# Exports small XGBoost models + I/O fixtures for the pure-managed predictor's parity tests.
# Three objectives exercise the three output transforms: logistic / softmax / identity.
# Some feature values are set to NaN to validate XGBoost's "default direction" (missing) handling.
#
# Run:  python Scripts/export_xgboost_fixture.py
# Out:  Tests/test_fixtures/xgboost/{clf,multi,reg}_model.json + _io.json

import json
import os
import numpy as np
import xgboost as xgb

OUT = os.path.join("Tests", "test_fixtures", "xgboost")
os.makedirs(OUT, exist_ok=True)


def make_x(n, f, seed):
    rng = np.random.default_rng(seed)
    x = rng.standard_normal((n, f)).astype(np.float32)
    # sprinkle missing values to test default-direction traversal
    mask = rng.random((n, f)) < 0.05
    x[mask] = np.nan
    return x


def signal(x):
    z = np.nan_to_num(x, nan=0.0)
    return (1.3 * z[:, 0] * z[:, 1] - 1.1 * np.sin(2.0 * z[:, 2])
            + 0.9 * (z[:, 3] ** 2 - 1.0) + 0.7 * z[:, 4])


def dump(task, booster, x_test, margin, pred, base_score, objective):
    booster.save_model(os.path.join(OUT, f"{task}_model.json"))
    io = {
        "objective": objective,
        "base_score": base_score,
        "features": [[None if np.isnan(v) else float(v) for v in row] for row in x_test],
        "margin": margin.tolist(),   # raw (output_margin=True), shape [n, groups]
        "pred": pred.tolist(),       # transformed, shape [n, groups]
    }
    with open(os.path.join(OUT, f"{task}_io.json"), "w") as fp:
        json.dump(io, fp)
    # diagnostics
    leaves0 = booster.predict(xgb.DMatrix(x_test[:1]), output_margin=True)
    print(f"[{task}] obj={objective} base_score={base_score} "
          f"n_trees={len(booster.get_dump())} margin[0]={np.ravel(margin[0])} pred[0]={np.ravel(pred[0])}")


def as2d(a):
    a = np.asarray(a)
    return a.reshape(a.shape[0], -1)


def base_of(booster):
    bs = json.loads(booster.save_config())["learner"]["learner_model_param"]["base_score"]
    # XGBoost ≥2 encodes this as a JSON-array string, e.g. "[5.2375E-1]".
    s = bs.strip()
    if s.startswith("["):
        return float(json.loads(s)[0])
    return float(s)


def main():
    n_tr, n_te, f = 4000, 200, 12

    # ── binary:logistic ────────────────────────────────────────────────────────
    x = make_x(n_tr, f, 1); y = (signal(x) + 0.3 > 0).astype(np.float32)
    xt = make_x(n_te, f, 99)
    b = xgb.train({"objective": "binary:logistic", "max_depth": 4, "eta": 0.2,
                   "tree_method": "hist"}, xgb.DMatrix(x, label=y), num_boost_round=20)
    m = as2d(b.predict(xgb.DMatrix(xt), output_margin=True))
    p = as2d(b.predict(xgb.DMatrix(xt)))
    dump("clf", b, xt, m, p, base_of(b), "binary:logistic")

    # ── multi:softprob (3 classes) ──────────────────────────────────────────────
    x = make_x(n_tr, f, 2)
    yc = (np.digitize(signal(x), bins=[-1.0, 1.0])).astype(np.float32)  # 0/1/2
    xt = make_x(n_te, f, 98)
    b = xgb.train({"objective": "multi:softprob", "num_class": 3, "max_depth": 4,
                   "eta": 0.2, "tree_method": "hist"}, xgb.DMatrix(x, label=yc), num_boost_round=20)
    m = as2d(b.predict(xgb.DMatrix(xt), output_margin=True))
    p = as2d(b.predict(xgb.DMatrix(xt)))
    dump("multi", b, xt, m, p, base_of(b), "multi:softprob")

    # ── reg:squarederror ────────────────────────────────────────────────────────
    x = make_x(n_tr, f, 3); yr = signal(x).astype(np.float32)
    xt = make_x(n_te, f, 97)
    b = xgb.train({"objective": "reg:squarederror", "max_depth": 4, "eta": 0.2,
                   "tree_method": "hist"}, xgb.DMatrix(x, label=yr), num_boost_round=20)
    m = as2d(b.predict(xgb.DMatrix(xt), output_margin=True))
    p = as2d(b.predict(xgb.DMatrix(xt)))
    dump("reg", b, xt, m, p, base_of(b), "reg:squarederror")

    # ── structure probe (confirm field names for the C# loader) ─────────────────
    with open(os.path.join(OUT, "clf_model.json")) as fp:
        j = json.load(fp)
    learner = j["learner"]
    gb = learner["gradient_booster"]
    tree0 = gb["model"]["trees"][0]
    print("learner_model_param keys:", list(learner["learner_model_param"].keys()))
    print("gbtree_model_param:", gb["model"]["gbtree_model_param"])
    print("tree[0] keys:", list(tree0.keys()))
    print("tree[0] num_nodes:", tree0["tree_param"]["num_nodes"],
          "left[:5]:", tree0["left_children"][:5],
          "split_idx[:5]:", tree0["split_indices"][:5],
          "split_cond[:5]:", [round(c, 4) for c in tree0["split_conditions"][:5]],
          "default_left[:5]:", tree0["default_left"][:5])
    print("tree_info[:8]:", gb["model"].get("tree_info", [])[:8])


if __name__ == "__main__":
    main()

