"""IHACDP - trains + benchmarks a classifier per disease, exports model + metadata + metrics."""
from __future__ import annotations
import json, time, warnings
from pathlib import Path
import numpy as np, pandas as pd, joblib

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OrdinalEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (roc_auc_score, accuracy_score, f1_score, precision_score,
                             recall_score, brier_score_loss, roc_curve, confusion_matrix)
from xgboost import XGBClassifier

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[1]
RAW, ART = ROOT / "data" / "raw", ROOT / "artifacts"
ART.mkdir(parents=True, exist_ok=True)
SEED = 42

DISEASES = {
    "diabetes": {"label": "Type 2 Diabetes", "desc": "Metabolic risk from CDC BRFSS health indicators",
                 "source": "UCI #891 - CDC Diabetes Health Indicators (BRFSS)"},
    "heart":    {"label": "Coronary Heart Disease", "desc": "Cardiac risk from resting vitals and stress-test findings",
                 "source": "UCI #45 - Cleveland Heart Disease"},
    "kidney":   {"label": "Chronic Kidney Disease", "desc": "Renal impairment from serum chemistry and urinalysis",
                 "source": "UCI #336 - Chronic Kidney Disease"},
    "liver":    {"label": "Chronic Liver Disease", "desc": "Hepatic injury from liver function tests",
                 "source": "UCI #225 - Indian Liver Patient Dataset"},
    "breast":   {"label": "Breast Cancer (Malignancy)", "desc": "Malignancy from FNA cytology nuclear morphometry",
                 "source": "UCI #17 - Wisconsin Diagnostic Breast Cancer"},
}

def build_pipeline(estimator, num_cols, cat_cols):
    num = Pipeline([("imp", SimpleImputer(strategy="median")), ("sc", StandardScaler())])
    cat = Pipeline([("imp", SimpleImputer(strategy="most_frequent")),
                    ("enc", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1))])
    pre = ColumnTransformer([("num", num, num_cols), ("cat", cat, cat_cols)], remainder="drop")
    return Pipeline([("pre", pre), ("clf", estimator)])


def candidates(n_pos, n_tot):
    spw = max(1.0, (n_tot - n_pos) / max(n_pos, 1))
    return {
        "LogisticRegression": LogisticRegression(max_iter=2000, class_weight="balanced", random_state=SEED),
        "RandomForest": RandomForestClassifier(n_estimators=300, min_samples_leaf=2, class_weight="balanced",
                                               n_jobs=-1, random_state=SEED),
        "XGBoost": XGBClassifier(n_estimators=400, max_depth=5, learning_rate=0.08, subsample=0.9,
                                 colsample_bytree=0.9, scale_pos_weight=spw, eval_metric="logloss",
                                 tree_method="hist", n_jobs=-1, random_state=SEED),
    }


def run(name: str) -> dict:
    df = pd.read_csv(RAW / f"{name}.csv")
    y = df["target"].astype(int).values
    X = df.drop(columns=["target"])
    for c in X.columns:
        conv = pd.to_numeric(X[c], errors="coerce")
        if conv.notna().mean() > 0.80:
            X[c] = conv
    num_cols = [c for c in X.columns if pd.api.types.is_numeric_dtype(X[c])]
    cat_cols = [c for c in X.columns if c not in num_cols]

    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, stratify=y, random_state=SEED)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)

    leaderboard, best, best_auc = [], None, -1.0
    for algo, est in candidates(int(y.sum()), len(y)).items():
        t0 = time.time()
        pipe = build_pipeline(est, num_cols, cat_cols)
        cv_auc = cross_val_score(pipe, Xtr, ytr, cv=cv, scoring="roc_auc", n_jobs=-1)
        pipe.fit(Xtr, ytr)
        proba = pipe.predict_proba(Xte)[:, 1]
        pred = (proba >= 0.5).astype(int)
        row = {
            "algorithm": algo,
            "cv_auc_mean": round(float(cv_auc.mean()), 4), "cv_auc_std": round(float(cv_auc.std()), 4),
            "test_auc": round(float(roc_auc_score(yte, proba)), 4),
            "accuracy": round(float(accuracy_score(yte, pred)), 4),
            "precision": round(float(precision_score(yte, pred, zero_division=0)), 4),
            "recall": round(float(recall_score(yte, pred, zero_division=0)), 4),
            "f1": round(float(f1_score(yte, pred, zero_division=0)), 4),
            "brier": round(float(brier_score_loss(yte, proba)), 4),
            "fit_seconds": round(time.time() - t0, 2),
        }
        leaderboard.append(row)
        print(f"    {algo:<20} cv_auc={row['cv_auc_mean']:.4f}±{row['cv_auc_std']:.3f}  "
              f"test_auc={row['test_auc']:.4f}  f1={row['f1']:.4f}  ({row['fit_seconds']}s)")
        if row["cv_auc_mean"] > best_auc:
            best_auc, best = row["cv_auc_mean"], (algo, pipe, proba, pred)

    algo, pipe, proba, pred = best
    fpr, tpr, _ = roc_curve(yte, proba)
    step = max(1, len(fpr) // 120)
    tn, fp, fn, tp = confusion_matrix(yte, pred).ravel()

    joblib.dump(pipe, ART / f"{name}_model.joblib")

    feats = []
    for c in X.columns:
        if c in num_cols:
            s = pd.to_numeric(X[c], errors="coerce")
            feats.append({"name": c, "type": "number",
                          "min": float(np.nanpercentile(s, 1)), "max": float(np.nanpercentile(s, 99)),
                          "median": float(np.nanmedian(s)),
                          "integer": bool(s.dropna().mod(1).eq(0).all())})
        else:
            vals = sorted(X[c].dropna().astype(str).unique().tolist())[:12]
            feats.append({"name": c, "type": "category", "options": vals,
                          "median": X[c].mode().astype(str).iloc[0] if not X[c].mode().empty else vals[0]})

    meta = {
        "key": name, **DISEASES[name],
        "rows": int(len(df)), "n_features": int(X.shape[1]),
        "prevalence": round(float(y.mean()), 4),
        "chosen_algorithm": algo,
        "metrics": next(r for r in leaderboard if r["algorithm"] == algo),
        "leaderboard": sorted(leaderboard, key=lambda r: -r["cv_auc_mean"]),
        "roc": {"fpr": [round(float(v), 4) for v in fpr[::step]],
                "tpr": [round(float(v), 4) for v in tpr[::step]]},
        "confusion": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
        "features": feats,
        "numeric_cols": num_cols, "categorical_cols": cat_cols,
    }
    (ART / f"{name}_meta.json").write_text(json.dumps(meta, indent=2))
    X.sample(min(200, len(X)), random_state=SEED).to_csv(ART / f"{name}_background.csv", index=False)
    return meta


if __name__ == "__main__":
    t0, summary = time.time(), []
    for name in DISEASES:
        print(f"\n[train] {name}  ({DISEASES[name]['label']})")
        m = run(name)
        print(f"  => BEST: {m['chosen_algorithm']}  test_auc={m['metrics']['test_auc']:.4f}  "
              f"acc={m['metrics']['accuracy']:.4f}")
        summary.append({k: m[k] for k in ("key", "label", "rows", "n_features", "chosen_algorithm", "metrics", "source")})
    (ART / "metrics_summary.json").write_text(json.dumps(summary, indent=2))
    print(f"\n{'='*70}\nTrained {len(summary)} models in {time.time()-t0:.1f}s")
    print(f"{'Disease':<26}{'Algorithm':<18}{'AUC':<9}{'Acc':<9}{'F1'}")
    for s in summary:
        print(f"{s['label']:<26}{s['chosen_algorithm']:<18}{s['metrics']['test_auc']:<9.4f}"
              f"{s['metrics']['accuracy']:<9.4f}{s['metrics']['f1']:.4f}")
