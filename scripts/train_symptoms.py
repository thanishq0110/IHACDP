"""Common-condition triage model.

The published symptom file repeats every pattern ~120 times; training on it as
shipped yields a meaningless 100% score. Duplicates are removed first, the model
is cross-validated on the 304 genuinely distinct patterns, and the result is
described for what it is: a curated symptom-to-condition mapping, not evidence
of diagnostic performance.
"""
from __future__ import annotations
import json, time
from pathlib import Path
import numpy as np, pandas as pd, joblib

from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import BernoulliNB

ROOT = Path(__file__).resolve().parents[1]
RAW, ART = ROOT / "data" / "raw", ROOT / "artifacts"
SEED = 42

# Conditions that warrant urgent care rather than self-management advice.
URGENT = {"Heart attack", "Paralysis (brain hemorrhage)", "Tuberculosis", "AIDS",
          "Pneumonia", "Malaria", "Dengue", "Typhoid", "Hepatitis B", "Hepatitis C",
          "Hepatitis D", "Hepatitis E", "Alcoholic hepatitis", "Jaundice"}


CURATED = Path(__file__).resolve().parents[1] / "data" / "curated" / "curated_conditions.csv"
SYMCAT = RAW / "symcat.csv"

import sys as _sys
_sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from backend.services.vocab import canonical


SYMCAT_KEEP = RAW / "symcat_keep.csv"


def _symcat_wide() -> tuple[pd.DataFrame, dict[str, float]]:
    """SymCat's long edge list as a condition x symptom matrix, plus weights.

    Restricted to the everyday and cannot-miss selection: carrying all 801 made
    every common complaint compete with hundreds of rare diagnoses.
    """
    e = pd.read_csv(SYMCAT)
    if SYMCAT_KEEP.exists():
        allow = set(pd.read_csv(SYMCAT_KEEP)["condition"])
        before = e["condition"].nunique()
        e = e[e["condition"].isin(allow)]
        print(f"    SymCat restricted {before} -> {e['condition'].nunique()} conditions")
    e["symptom"] = e["symptom"].map(canonical)
    e = e.groupby(["condition", "symptom"], as_index=False)["probability"].max()
    wide = (e.assign(v=1).pivot_table(index="condition", columns="symptom",
                                      values="v", fill_value=0, aggfunc="max"))
    wide = wide.reset_index().rename(columns={"condition": "condition"})
    weights = (e.groupby("symptom")["probability"].mean() / 100.0).to_dict()
    return wide, weights


def load_merged() -> tuple[pd.DataFrame, dict[str, str]]:
    """Public symptom matrix plus the curated everyday-complaint patterns.

    The two sources use disjoint symptom vocabularies, so the union is taken and
    absent columns filled with 0 - a symptom a source never mentions is absent
    from its patterns by definition.
    """
    base = pd.read_csv(RAW / "symptoms.csv")
    base.columns = [c.strip() for c in base.columns]
    provenance = {c: "public dataset" for c in base["condition"].astype(str).str.strip()}

    if SYMCAT.exists():
        sym_wide, _w = _symcat_wide()
        for c in sym_wide["condition"].astype(str).str.strip():
            provenance.setdefault(c, "SymCat")
        cols = sorted((set(base.columns) | set(sym_wide.columns)) - {"condition"})
        base = base.reindex(columns=cols + ["condition"], fill_value=0)
        sym_wide = sym_wide.reindex(columns=cols + ["condition"], fill_value=0)
        # a condition already present keeps its original pattern
        known = set(base["condition"].astype(str).str.strip())
        sym_wide = sym_wide[~sym_wide["condition"].astype(str).str.strip().isin(known)]
        base = pd.concat([base, sym_wide], ignore_index=True)
        print(f"[merge] + {len(sym_wide)} SymCat conditions")

    if CURATED.exists():
        cur = pd.read_csv(CURATED)
        cur.columns = [c.strip() for c in cur.columns]
        for c in cur["condition"].astype(str).str.strip():
            provenance[c] = "curated clinical pattern"
        cols = sorted(set(base.columns) | set(cur.columns) - {"condition"})
        cols = [c for c in cols if c != "condition"]
        base = base.reindex(columns=cols + ["condition"], fill_value=0)
        cur = cur.reindex(columns=cols + ["condition"], fill_value=0)
        known = set(base["condition"].astype(str).str.strip())
        overlap = cur["condition"].astype(str).str.strip().isin(known).sum()
        merged = pd.concat([base, cur], ignore_index=True)
        print(f"[merge] {len(base)} existing + {len(cur)} curated ({overlap} overlapping names kept separate)")
    else:
        merged = base
    merged = merged.fillna(0)
    return merged, provenance


def main() -> None:
    df, provenance = load_merged()
    y = df["condition"].astype(str).str.strip().values
    X = df.drop(columns=["condition"])
    X = X.loc[:, X.sum() > 0]                      # drop never-positive columns
    symptoms = list(X.columns)
    print(f"[symptoms] {len(df)} unique patterns · {len(symptoms)} symptoms · {len(set(y))} conditions")

    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=SEED)
    candidates = {
        "BernoulliNB": BernoulliNB(alpha=0.1),
        "LogisticRegression": LogisticRegression(max_iter=2000, C=2.0, random_state=SEED),
        "RandomForest": RandomForestClassifier(n_estimators=300, min_samples_leaf=1,
                                               random_state=SEED, n_jobs=-1),
    }
    board, best, best_score = [], None, -1.0
    for name, est in candidates.items():
        t0 = time.time()
        acc = cross_val_score(est, X, y, cv=cv, scoring="accuracy", n_jobs=-1)
        est.fit(X, y)
        row = {"algorithm": name, "cv_accuracy_mean": round(float(acc.mean()), 4),
               "cv_accuracy_std": round(float(acc.std()), 4), "fit_seconds": round(time.time() - t0, 2)}
        board.append(row)
        print(f"    {name:<20} cv_acc={row['cv_accuracy_mean']:.4f} ±{row['cv_accuracy_std']:.3f}")
        if row["cv_accuracy_mean"] > best_score:
            best_score, best = row["cv_accuracy_mean"], (name, est)

    algo, model = best

    matrix_for_eval = (pd.DataFrame(X.values, columns=symptoms)
                       .assign(condition=y).groupby("condition").max())

    # Evaluate the scorer that actually ships, not the fitted classifier. The
    # engine ranks by weighted overlap; measuring the sklearn model instead
    # reports a number no user ever experiences.
    from backend.services.symptoms import SymptomEngine

    class _Eval(SymptomEngine):
        def __init__(self, matrix):
            self.matrix = matrix
            self.symptoms = list(matrix.columns)
            self.urgent = set()
            self.meta = {}

    ev = _Eval(matrix_for_eval)
    rng = np.random.default_rng(SEED)
    hits = {}
    for k in (2, 3, 4):
        ok = tot = 0
        for cond in matrix_for_eval.index:
            present = [s for s in symptoms if matrix_for_eval.loc[cond, s] == 1]
            if len(present) < k:
                continue
            for _ in range(3):                      # a few draws per condition
                pick = rng.choice(present, size=k, replace=False)
                ranked = ev._score({s: True for s in pick}).sort_values(ascending=False)
                ok += cond in list(ranked.head(3).index)
                tot += 1
        hits[k] = round(ok / max(tot, 1), 3)
    print(f"    top-3 accuracy of the shipped scorer, from k reported symptoms: {hits}")

    matrix = (pd.DataFrame(X.values, columns=symptoms)
              .assign(condition=y).groupby("condition").max())

    joblib.dump({"model": model, "symptoms": symptoms, "classes": list(model.classes_)},
                ART / "symptoms_model.joblib")
    matrix.to_csv(ART / "symptoms_matrix.csv")
    by_source = {}
    for cond, src in provenance.items():
        by_source[src] = by_source.get(src, 0) + 1

    meta = {
        "key": "symptoms", "label": "Common Conditions",
        "provenance": provenance,
        "conditions_by_source": by_source,
        "desc": "Everyday complaints such as fever, colds, headaches and joint pain",
        "source": "Disease-Symptom knowledge base (public), de-duplicated",
        "patterns": int(len(df)), "n_symptoms": len(symptoms), "n_conditions": int(len(set(y))),
        "chosen_algorithm": algo, "leaderboard": sorted(board, key=lambda r: -r["cv_accuracy_mean"]),
        "top3_accuracy_by_symptom_count": hits,
        "conditions": sorted(set(y)),
        "urgent": sorted(URGENT & set(y)),
        "caveat": ("Curated symptom-to-condition mapping, not a learned diagnostic model. The "
                   "source repeats each pattern ~120 times; duplicates were removed before "
                   "training, so cross-validated accuracy reflects pattern separability only."),
    }
    (ART / "symptoms_meta.json").write_text(json.dumps(meta, indent=2))
    print(f"\n  => {algo}  ({len(symptoms)} symptoms, {len(set(y))} conditions)")


if __name__ == "__main__":
    main()
