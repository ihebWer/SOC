#!/usr/bin/env python3
import json
import joblib
from pathlib import Path

import numpy as np
import pandas as pd
import lightgbm as lgb

from sklearn.metrics import (
    roc_auc_score, average_precision_score,
    precision_recall_curve,
    precision_score, recall_score, f1_score,
    accuracy_score, confusion_matrix
)

INPUT = Path("data/wazuh_native_features_v3.parquet")
OUT_DIR = Path("outputs_wazuh_native_v3_cracking")
MODEL_DIR = OUT_DIR / "model"
OUT_DIR.mkdir(exist_ok=True)
MODEL_DIR.mkdir(exist_ok=True)

RANDOM_STATE = 42

CRITICAL_CLASSES = {
    "cracking",
    "privilege_escalation",
    "webshell",
    "reverse_shell",
}

TRAIN_TARGETS = {
    "false_positive": 30000,
    "dirb": 30000,
    "wpscan": 15000,
    "dnsteal": 8000,
    "cracking": 8000,
    "network_scans": 3000,
    "service_scans": 3000,
    "privilege_escalation": 3000,
    "webshell": 3000,
    "reverse_shell": 3000,
    "service_stop": 1000,
}

LGBM_PARAMS = {
    "n_estimators": 500,
    "max_depth": 7,
    "learning_rate": 0.05,
    "subsample": 0.85,
    "colsample_bytree": 0.85,
    "min_child_samples": 8,
    "random_state": RANDOM_STATE,
    "verbose": -1,
    "n_jobs": -1,
}

DROP_COLS = {
    "scenario", "time", "time_label", "target",
    "src_ip", "agent_name", "rule_id",
    "suricata_signature_id",
}

LEAKY_COLS = {
    "rule_global_count",
    "rule_global_rate",
    "rule_global_rarity",
    "suricata_signature_global_count",
    "suricata_signature_global_rarity",
}


def make_target(labels):
    return labels.isin(CRITICAL_CLASSES).astype(int).values


def get_feature_cols(df):
    cols = []
    for c in df.columns:
        if c in DROP_COLS:
            continue
        if c in LEAKY_COLS:
            continue
        if df[c].dtype == "object":
            continue
        cols.append(c)
    return cols


def balance_train(train_df, feature_cols, rng):
    parts = []

    for label, sub in train_df.groupby("time_label", observed=True):
        label = str(label)
        target = TRAIN_TARGETS.get(label)

        if target is None:
            parts.append(sub)
            continue

        n = len(sub)

        if n >= target:
            idx = rng.choice(sub.index, size=target, replace=False)
            parts.append(sub.loc[idx])
        else:
            parts.append(sub)
            need = target - n
            reps = (need // max(n, 1)) + 1

            X = sub[feature_cols].astype(np.float32).values
            synths = []

            for _ in range(reps):
                noise = rng.normal(0, 0.03, size=X.shape).astype(np.float32)
                Xs = np.nan_to_num(X * (1.0 + noise), nan=0.0, posinf=0.0, neginf=0.0)
                Xs = np.clip(Xs, 0, None)

                s = sub.copy()
                for i, col in enumerate(feature_cols):
                    s[col] = Xs[:, i]

                synths.append(s)

            synth = pd.concat(synths, ignore_index=True).iloc[:need]
            parts.append(synth)

    out = pd.concat(parts, ignore_index=True)
    return out.sample(frac=1, random_state=RANDOM_STATE).reset_index(drop=True)


def best_threshold_from_validation(y_true, scores):
    p, r, th = precision_recall_curve(y_true, scores)

    if len(th) == 0:
        return 0.5

    f1s = 2 * p[:-1] * r[:-1] / (p[:-1] + r[:-1] + 1e-9)
    return float(th[int(np.argmax(f1s))])


def soc_topk(y_true, scores, k_pct):
    order = np.argsort(-scores)
    k = max(1, int(len(y_true) * k_pct / 100))
    selected = y_true[order[:k]]

    precision_k = selected.mean()
    recall_k = selected.sum() / max(y_true.sum(), 1)
    lift_k = precision_k / max(y_true.mean(), 1e-12)

    return precision_k, recall_k, lift_k


def evaluate(y_true, scores, threshold):
    y_pred = (scores >= threshold).astype(int)

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()

    p1, r1, l1 = soc_topk(y_true, scores, 1)
    p5, r5, l5 = soc_topk(y_true, scores, 5)
    p10, r10, l10 = soc_topk(y_true, scores, 10)

    return {
        "auc": float(roc_auc_score(y_true, scores)),
        "ap": float(average_precision_score(y_true, scores)),
        "threshold": float(threshold),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "tp": int(tp),
        "fp": int(fp),
        "tn": int(tn),
        "fn": int(fn),
        "precision_at_1": float(p1),
        "recall_at_1": float(r1),
        "lift_at_1": float(l1),
        "precision_at_5": float(p5),
        "recall_at_5": float(r5),
        "lift_at_5": float(l5),
        "precision_at_10": float(p10),
        "recall_at_10": float(r10),
        "lift_at_10": float(l10),
    }


def train_one_fold(train_df, test_df, feature_cols, rng):
    train_bal = balance_train(train_df, feature_cols, rng)

    y = make_target(train_bal["time_label"])

    pos = np.where(y == 1)[0]
    neg = np.where(y == 0)[0]

    rng.shuffle(pos)
    rng.shuffle(neg)

    n_val_pos = max(2, int(len(pos) * 0.10))
    n_val_neg = max(2, int(len(neg) * 0.10))

    val_idx = np.concatenate([pos[:n_val_pos], neg[:n_val_neg]])
    tr_idx = np.concatenate([pos[n_val_pos:], neg[n_val_neg:]])

    X_train = train_bal.iloc[tr_idx][feature_cols]
    y_train = y[tr_idx]

    X_val = train_bal.iloc[val_idx][feature_cols]
    y_val = y[val_idx]

    spw = (len(y_train) - y_train.sum()) / max(y_train.sum(), 1)

    model = lgb.LGBMClassifier(scale_pos_weight=spw, **LGBM_PARAMS)
    model.fit(X_train, y_train)

    val_scores = model.predict_proba(X_val)[:, 1]
    threshold = best_threshold_from_validation(y_val, val_scores)

    y_test = make_target(test_df["time_label"])
    test_scores = model.predict_proba(test_df[feature_cols])[:, 1]

    metrics = evaluate(y_test, test_scores, threshold)

    scores_df = pd.DataFrame({
        "scenario": test_df["scenario"].values,
        "time": test_df["time"].values,
        "time_label": test_df["time_label"].values,
        "y_true": y_test,
        "score": test_scores,
        "y_pred": (test_scores >= threshold).astype(int),
        "fold_threshold": threshold,
    })

    return model, metrics, scores_df


def main():
    print("=" * 90)
    print("TRAIN WAZUH-NATIVE V2 CLEAN — LOSO")
    print("=" * 90)

    df = pd.read_parquet(INPUT)
    feature_cols = get_feature_cols(df)
    scenarios = sorted(df["scenario"].unique())

    print("[load]", df.shape)
    print("[features]", len(feature_cols))
    print("[target]")
    print(df["target"].value_counts())
    print("[labels]")
    print(df["time_label"].value_counts())

    results = []
    score_parts = []

    for scenario in scenarios:
        print("\n" + "=" * 90)
        print("[LOSO] test scenario:", scenario)
        print("=" * 90)

        train_df = df[df["scenario"] != scenario].reset_index(drop=True)
        test_df = df[df["scenario"] == scenario].reset_index(drop=True)

        rng = np.random.RandomState(RANDOM_STATE)

        _, metrics, scores_df = train_one_fold(train_df, test_df, feature_cols, rng)

        row = {
            "scenario": scenario,
            "n_test": int(len(test_df)),
            "n_attack": int(test_df["target"].sum()),
            "n_features": int(len(feature_cols)),
            **metrics,
        }

        results.append(row)
        score_parts.append(scores_df)

        print(row)

    summary = pd.DataFrame(results)
    all_scores = pd.concat(score_parts, ignore_index=True)

    summary.to_csv(OUT_DIR / "loso_wazuh_native_summary.csv", index=False)
    all_scores.to_parquet(OUT_DIR / "loso_wazuh_native_scores.parquet", index=False)

    metrics_cols = [
        "auc", "ap", "accuracy", "precision", "recall", "f1",
        "precision_at_1", "recall_at_1", "lift_at_1",
        "precision_at_5", "recall_at_5", "lift_at_5",
        "precision_at_10", "recall_at_10", "lift_at_10",
    ]

    print("\n" + "=" * 90)
    print("RESULTATS MOYENS LOSO")
    print("=" * 90)

    for c in metrics_cols:
        print(f"{c:<18}: {summary[c].mean():.4f} ± {summary[c].std():.4f}")

    print("\n[train final model]")
    rng = np.random.RandomState(RANDOM_STATE)
    train_bal = balance_train(df, feature_cols, rng)
    y_final = make_target(train_bal["time_label"])

    spw = (len(y_final) - y_final.sum()) / max(y_final.sum(), 1)

    final_model = lgb.LGBMClassifier(scale_pos_weight=spw, **LGBM_PARAMS)
    final_model.fit(train_bal[feature_cols], y_final)

    payload = {
        "model": final_model,
        "feature_cols": feature_cols,
        "critical_classes": sorted(CRITICAL_CLASSES),
        "train_targets": TRAIN_TARGETS,
        "lgbm_params": LGBM_PARAMS,
        "validation": "LOSO strict",
        "input_file": str(INPUT),
        "purpose": "Wazuh-native SOC alert prioritization",
    }

    joblib.dump(payload, MODEL_DIR / "wazuh_native_prioritizer_v3_cracking.joblib.joblib")

    with open(MODEL_DIR / "wazuh_native_features_v2_clean.txt2", "w") as f:
        for col in feature_cols:
            f.write(col + "\n")

    metadata = {
        "input": str(INPUT),
        "n_alerts": int(len(df)),
        "n_features": int(len(feature_cols)),
        "scenarios": scenarios,
        "critical_classes": sorted(CRITICAL_CLASSES),
        "train_targets": TRAIN_TARGETS,
        "excluded_leaky_cols": sorted(LEAKY_COLS),
        "metrics_mean": {c: float(summary[c].mean()) for c in metrics_cols},
        "metrics_std": {c: float(summary[c].std()) for c in metrics_cols},
    }

    with open(MODEL_DIR / "wazuh_native_metadata_v2_clean.json", "w") as f:
        json.dump(metadata, f, indent=2)

    print("\nSaved:")
    print(" -", OUT_DIR / "loso_wazuh_native_summary.csv")
    print(" -", OUT_DIR / "loso_wazuh_native_scores.parquet")
    print(" -", MODEL_DIR / "wazuh_native_prioritizer_v2_clean.joblib")


if __name__ == "__main__":
    main()