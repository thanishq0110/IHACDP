"""Disease risk engine: calibrated probability + SHAP attribution per disease.

Critically, every contribution is tagged observed/imputed. The LLM layer is
instructed to reason only over OBSERVED features, so it can never narrate a
finding that was actually a training-set median filling a blank.
"""
from __future__ import annotations
import json, threading
from pathlib import Path
from typing import Any

import numpy as np, pandas as pd, joblib, shap

from backend.config import ARTIFACTS, band, MIN_COVERAGE
from backend.services import clinical

_LOCK = threading.Lock()


class DiseaseModel:
    def __init__(self, key: str):
        self.key = key
        self.meta = json.loads((ARTIFACTS / f"{key}_meta.json").read_text())
        self.pipe = joblib.load(ARTIFACTS / f"{key}_model.joblib")
        self.pre = self.pipe.named_steps["pre"]
        self.clf = self.pipe.named_steps["clf"]
        self.columns = self.meta["numeric_cols"] + self.meta["categorical_cols"]
        try:
            self.feat_names = [n.split("__", 1)[-1] for n in self.pre.get_feature_names_out()]
        except Exception:
            self.feat_names = list(self.columns)

        bg = pd.read_csv(ARTIFACTS / f"{key}_background.csv")
        self._bg_t = self.pre.transform(bg[self.columns])
        self._explainer = self._build_explainer()

    def _build_explainer(self):
        name = type(self.clf).__name__
        try:
            if name in ("XGBClassifier", "RandomForestClassifier"):
                return shap.TreeExplainer(self.clf)
            return shap.LinearExplainer(self.clf, self._bg_t, max_samples=len(self._bg_t))
        except Exception:
            return None

    def _frame(self, mapped: dict) -> pd.DataFrame:
        row = {c: mapped.get(c) for c in self.columns}
        return pd.DataFrame([row], columns=self.columns)

    def _shap_row(self, Xt: np.ndarray) -> np.ndarray | None:
        if self._explainer is None:
            return None
        try:
            with _LOCK:
                sv = self._explainer.shap_values(Xt)
            sv = np.array(sv)
            if sv.ndim == 3:                 # (n, features, classes) or (classes, n, features)
                sv = sv[..., -1] if sv.shape[-1] == 2 else sv[-1]
            return np.array(sv).reshape(-1)[: len(self.feat_names)]
        except Exception:
            return None

    def predict(self, record: dict) -> dict[str, Any]:
        mapped = clinical.MAPPERS[self.key](record)
        observed = {k for k, v in mapped.items() if v is not None}
        cov = clinical.coverage(record, self.key)

        if cov < MIN_COVERAGE:
            return {
                "disease": self.key, "label": self.meta["label"], "status": "insufficient_data",
                "coverage": round(cov, 2), "probability": None, "risk_band": None,
                "contributions": [], "missing": clinical.missing_key_fields(record, self.key),
                "model": self.meta["chosen_algorithm"], "auc": self.meta["metrics"]["test_auc"],
            }

        X = self._frame(mapped)
        Xt = self.pre.transform(X)
        prob = float(self.pipe.predict_proba(X)[0, 1])
        sv = self._shap_row(Xt)

        contribs = []
        if sv is not None:
            order = np.argsort(-np.abs(sv))[:8]
            for i in order:
                fname = self.feat_names[i]
                raw = mapped.get(fname)
                contribs.append({
                    "feature": fname,
                    "label": clinical.human(fname),
                    "value": (round(raw, 3) if isinstance(raw, (int, float)) else raw),
                    "shap": round(float(sv[i]), 4),
                    "direction": "increases" if sv[i] > 0 else "decreases",
                    "observed": fname in observed,
                })

        return {
            "disease": self.key, "label": self.meta["label"], "status": "ok",
            "coverage": round(cov, 2), "probability": round(prob, 4), "risk_band": band(prob),
            "contributions": contribs,
            "missing": [clinical.human(m) for m in clinical.missing_key_fields(record, self.key)],
            "model": self.meta["chosen_algorithm"], "auc": self.meta["metrics"]["test_auc"],
            "source": self.meta["source"],
        }


class RiskEngine:
    def __init__(self):
        self.models: dict[str, DiseaseModel] = {}
        for p in sorted(ARTIFACTS.glob("*_meta.json")):
            key = p.name.replace("_meta.json", "")
            try:
                self.models[key] = DiseaseModel(key)
            except Exception as e:
                print(f"[risk-engine] skipped {key}: {type(e).__name__}: {e}")
        print(f"[risk-engine] loaded {len(self.models)} models: {list(self.models)}")

    def assess(self, record: dict) -> dict[str, Any]:
        rec = clinical.derive(record)
        results = [m.predict(rec) for m in self.models.values()]
        ranked = sorted(results, key=lambda r: -(r["probability"] or -1))
        actionable = [r for r in ranked if r["status"] == "ok"]
        return {
            "record": rec,
            "results": ranked,
            "assessed": len(actionable),
            "top": actionable[0] if actionable else None,
            "catalogue": [
                {"key": m.key, "label": m.meta["label"], "auc": m.meta["metrics"]["test_auc"],
                 "algorithm": m.meta["chosen_algorithm"], "rows": m.meta["rows"]}
                for m in self.models.values()
            ],
        }

    def catalogue(self) -> list[dict]:
        return [m.meta for m in self.models.values()]


_engine: RiskEngine | None = None


def engine() -> RiskEngine:
    global _engine
    if _engine is None:
        _engine = RiskEngine()
    return _engine
