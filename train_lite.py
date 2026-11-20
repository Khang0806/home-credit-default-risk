"""Train the deployable "lite" model: application-form fields only.

    python train_lite.py --data-dir "../Dataset/CSV"              # lite model -> models/
    python train_lite.py --data-dir "../Dataset/CSV" --benchmark  # + all-application-columns model, for comparison

5-fold stratified CV (same protocol as 07_HC_Final) gives the OOF AUC; a final model is then
refit on all labeled rows with the average best iteration and saved for serving.
"""
import argparse, json, re
from datetime import datetime
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

from src.features import RAW_INPUTS, CATEGORICAL_INPUTS, NUMERIC_INPUTS, build_features

MODEL_DIR = Path(__file__).parent / "models"
PARAMS = dict(
    objective="binary", data_sample_strategy="goss", top_rate=0.2, other_rate=0.1,
    n_estimators=20000, learning_rate=0.02, num_leaves=31, max_depth=8,
    min_child_samples=120, reg_alpha=2.5, reg_lambda=3.5,
    colsample_bytree=0.5,          # 07 used 0.15 for 584 features; too few per tree for ~90
    random_state=42, verbosity=-1, n_jobs=-1,
)


def cross_validate(X, y, params, label):
    oof, best_iters, fold_aucs = np.zeros(len(X)), [], []
    for fold, (tr, va) in enumerate(StratifiedKFold(5, shuffle=True, random_state=42).split(X, y), 1):
        model = lgb.LGBMClassifier(**params)
        model.fit(X.iloc[tr], y.iloc[tr], eval_set=[(X.iloc[va], y.iloc[va])], eval_metric="auc",
                  callbacks=[lgb.early_stopping(200, verbose=False)])
        oof[va] = model.predict_proba(X.iloc[va])[:, 1]
        best_iters.append(model.best_iteration_)
        fold_aucs.append(roc_auc_score(y.iloc[va], oof[va]))
        print(f"  [{label}] fold {fold}: AUC={fold_aucs[-1]:.5f}  best_iter={model.best_iteration_}")
    auc = roc_auc_score(y, oof)
    print(f"  [{label}] OOF AUC = {auc:.5f}  ({X.shape[1]} features)")
    return auc, fold_aucs, best_iters


def benchmark(data_dir):
    """All columns from notebook 01's output, preprocessed like 07 (no bureau/previous/etc.)."""
    df = pd.read_csv(data_dir / "application_train_test_final_agg.csv")
    df = df[df["TARGET"].notna()]
    for col in df.select_dtypes(exclude="number").columns:
        df[col] = df[col].astype("category")
    for col in ["APPLICATION_DAYS_BIRTH", "APPLICATION_DAYS_EMPLOYED",
                "APPLICATION_DAYS_REGISTRATION", "APPLICATION_DAYS_ID_PUBLISH"]:
        df[col] = df[col].abs()
    X = df.drop(columns=["SK_ID_CURR", "TARGET"]).replace([np.inf, -np.inf], np.nan)
    X.columns = [re.sub(r"[^A-Za-z0-9_]", "_", c) for c in X.columns]
    auc, _, _ = cross_validate(X, df["TARGET"].astype(int), {**PARAMS, "colsample_bytree": 0.3}, "app-all")
    return {"oof_auc": round(auc, 5), "n_features": X.shape[1]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="data", type=Path)
    ap.add_argument("--benchmark", action="store_true")
    args = ap.parse_args()

    raw = pd.read_csv(args.data_dir / "application_train.csv", usecols=RAW_INPUTS + ["TARGET"])
    X, y = build_features(raw), raw["TARGET"].astype(int)
    print(f"Lite: {len(X):,} applicants, {X.shape[1]} features from {len(RAW_INPUTS)} form fields")

    auc, fold_aucs, best_iters = cross_validate(X, y, PARAMS, "lite")
    n_estimators = int(np.mean(best_iters) * 1.1)   # a bit more data than each fold saw
    final = lgb.LGBMClassifier(**{**PARAMS, "n_estimators": n_estimators}).fit(X, y)

    # Global importance via LightGBM's exact TreeSHAP (same as notebook 07)
    sample = X.sample(20000, random_state=42)
    contrib = final.booster_.predict(sample, pred_contrib=True)[:, :-1]
    top = pd.Series(np.abs(contrib).mean(axis=0), index=X.columns).sort_values(ascending=False)

    MODEL_DIR.mkdir(exist_ok=True)
    final.booster_.save_model(MODEL_DIR / "lite_model.txt")
    meta = {
        "trained_at": datetime.now().isoformat(timespec="seconds"),
        "n_train": len(X), "default_rate": round(float(y.mean()), 4),
        "features": list(X.columns), "n_estimators": n_estimators,
        "cv": {"oof_auc": round(auc, 5), "fold_aucs": [round(a, 5) for a in fold_aucs]},
        "top_features_mean_abs_shap": {k: round(float(v), 4) for k, v in top.head(20).items()},
        # For the demo form: dropdown options and sensible defaults
        "category_options": {c: sorted(raw[c].dropna().unique().tolist()) for c in CATEGORICAL_INPUTS},
        "numeric_medians": {c: float(raw[c].median()) for c in NUMERIC_INPUTS},
    }
    if args.benchmark:
        meta["benchmark_application_all_columns"] = benchmark(args.data_dir)
    (MODEL_DIR / "metadata.json").write_text(json.dumps(meta, indent=2))
    print(f"\nSaved {MODEL_DIR / 'lite_model.txt'} (OOF AUC {auc:.5f}, {n_estimators} trees)")


if __name__ == "__main__":
    main()
