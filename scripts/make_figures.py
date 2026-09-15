"""Figures for the capstone report, drawn from the trained artefacts."""
from __future__ import annotations
import json
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
ART, OUT = ROOT / "artifacts", ROOT / "docs" / "figures"
OUT.mkdir(parents=True, exist_ok=True)
plt.rcParams.update({"font.size": 12.5, "font.family": "DejaVu Sans",
                     "axes.spines.top": False, "axes.spines.right": False})
TEAL, SLATE = "#0d6e6e", "#2b4552"
KEYS = ["diabetes", "heart", "kidney", "liver", "breast"]
META = {k: json.loads((ART / f"{k}_meta.json").read_text()) for k in KEYS}


def roc_curves():
    fig, ax = plt.subplots(figsize=(6.5, 5))
    colours = plt.cm.viridis(np.linspace(0.15, 0.85, len(KEYS)))
    for (k, m), c in zip(META.items(), colours):
        ax.plot(m["roc"]["fpr"], m["roc"]["tpr"], lw=2, color=c,
                label=f"{m['label']} (AUC {m['metrics']['test_auc']:.3f})")
    ax.plot([0, 1], [0, 1], ls="--", lw=1, color="#8ba1ac", label="Chance")
    ax.set_xlabel("False positive rate"); ax.set_ylabel("True positive rate")
    ax.set_title("ROC curves on the held-out test split", pad=12)
    ax.legend(loc="lower right", frameon=False, fontsize=8.5)
    fig.tight_layout(); fig.savefig(OUT / "fig_roc_curves.png", dpi=200); plt.close(fig)


def algorithm_contest():
    algos = ["LogisticRegression", "RandomForest", "XGBoost"]
    fig, ax = plt.subplots(figsize=(8.6, 5.0))
    x = np.arange(len(KEYS)); w = 0.26
    for i, a in enumerate(algos):
        vals = [next((r["cv_auc_mean"] for r in META[k]["leaderboard"] if r["algorithm"] == a), 0)
                for k in KEYS]
        ax.bar(x + (i - 1) * w, vals, w, label=a,
               color=[TEAL, "#5aa9a9", SLATE][i])
    for i, k in enumerate(KEYS):
        best = META[k]["chosen_algorithm"]
        j = algos.index(best)
        v = next(r["cv_auc_mean"] for r in META[k]["leaderboard"] if r["algorithm"] == best)
        ax.text(i + (j - 1) * w, v + 0.012, "*", ha="center", fontsize=17, color="#b3261e")
    ax.set_xticks(x); ax.set_xticklabels([META[k]["label"].replace(" (Malignancy)", "") for k in KEYS],
                                         rotation=12, ha="right", fontsize=11)
    ax.set_ylabel("Mean 5-fold CV ROC-AUC"); ax.set_ylim(0.6, 1.04)
    ax.set_title("Three-algorithm contest per disease  (* = selected)", pad=30)
    ax.legend(frameon=False, fontsize=12, ncol=3, loc="lower center",
              bbox_to_anchor=(0.5, 1.0))
    fig.tight_layout(); fig.savefig(OUT / "fig_algorithm_contest.png", dpi=200); plt.close(fig)


def metrics_panel():
    fig, ax = plt.subplots(figsize=(8, 4.4))
    names = ["AUC", "Accuracy", "F1", "Precision", "Recall"]
    x = np.arange(len(KEYS)); w = 0.16
    cols = plt.cm.viridis(np.linspace(0.12, 0.8, len(names)))
    for i, n in enumerate(names):
        key = {"AUC": "test_auc", "Accuracy": "accuracy", "F1": "f1",
               "Precision": "precision", "Recall": "recall"}[n]
        ax.bar(x + (i - 2) * w, [META[k]["metrics"][key] for k in KEYS], w, label=n, color=cols[i])
    ax.set_xticks(x); ax.set_xticklabels([META[k]["label"].replace(" (Malignancy)", "") for k in KEYS],
                                         rotation=12, ha="right", fontsize=11)
    ax.set_ylim(0, 1.08); ax.set_ylabel("Score")
    ax.set_title("Test-split performance of the selected model per disease", pad=12)
    ax.legend(frameon=False, ncol=5, fontsize=8.5, loc="upper center", bbox_to_anchor=(0.5, -0.22))
    fig.tight_layout(); fig.savefig(OUT / "fig_metrics_panel.png", dpi=200, bbox_inches="tight"); plt.close(fig)


def triage_accuracy():
    sm = json.loads((ART / "symptoms_meta.json").read_text())
    ks = sorted(sm["top3_accuracy_by_symptom_count"], key=int)
    vals = [sm["top3_accuracy_by_symptom_count"][k] * 100 for k in ks]
    fig, ax = plt.subplots(figsize=(6.6, 4.4))
    bars = ax.bar([f"{k} symptoms" for k in ks], vals, color=TEAL, width=0.55)
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v + 1.5, f"{v:.1f}%", ha="center", fontsize=13, fontweight="bold")
    ax.set_ylim(0, 105); ax.set_ylabel("Top-3 accuracy (%)")
    ax.set_title(f"Triage recovery from partial information\n{sm['n_conditions']} conditions, "
                 f"{sm['n_symptoms']} symptoms", pad=12, fontsize=13)
    fig.tight_layout(); fig.savefig(OUT / "fig_triage_accuracy.png", dpi=200); plt.close(fig)


def llm_benchmark():
    rows = json.loads((ART / "llm_benchmark.json").read_text())
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(10.4, 4.5))
    names = [r["model"] for r in rows]
    cols = [TEAL if r["score"] == max(x["score"] for x in rows) else "#9db8b8" for r in rows]
    b = a1.barh(names, [r["score"] for r in rows], color=cols, height=0.55)
    for bb, r in zip(b, rows):
        a1.text(r["score"] + 1.5, bb.get_y() + bb.get_height() / 2, f"{r['score']:.0f}",
                va="center", fontsize=13, fontweight="bold")
    a1.set_xlim(0, 115); a1.set_xlabel("Weighted suitability score (/100)")
    a1.set_title("Task-specific benchmark", fontsize=13); a1.invert_yaxis()

    x = np.arange(len(rows)); w = 0.38
    a2.bar(x - w/2, [r["leak_rate"] * 100 for r in rows], w, label="Reasoning leak %", color="#b3261e")
    a2.bar(x + w/2, [r["dose_violations"] for r in rows], w, label="Dose violations", color="#c2560f")
    a2.set_xticks(x); a2.set_xticklabels(names, fontsize=11)
    a2.set_title("Safety failures (lower is better)", fontsize=13)
    a2.legend(frameon=False, fontsize=11)
    fig.tight_layout(); fig.savefig(OUT / "fig_llm_benchmark.png", dpi=200); plt.close(fig)


def triage_sources():
    sm = json.loads((ART / "symptoms_meta.json").read_text())
    src = sm["conditions_by_source"]
    fig, ax = plt.subplots(figsize=(5.6, 4.4))
    labels = list(src); vals = [src[k] for k in labels]
    ax.pie(vals, labels=[f"{k}\n({v})" for k, v in zip(labels, vals)], autopct="%1.0f%%",
           colors=[TEAL, "#5aa9a9", "#a8cfcf"], startangle=110,
           wedgeprops={"edgecolor": "white", "linewidth": 2}, textprops={"fontsize": 12.5})
    fig.tight_layout(); fig.savefig(OUT / "fig_triage_sources.png", dpi=200); plt.close(fig)


def architecture(wide: bool = False):
    """The request path, drawn rather than described.

    Two aspects: a squarer one for the report (shown at 160mm, so the type
    must survive a 0.66x reduction) and a wide one that fills a 16:9 slide.
    """
    fig, ax = plt.subplots(figsize=(14.5, 5.4) if wide else (10.0, 6.4))
    ax.set_xlim(0, 10); ax.set_ylim(0, 10); ax.axis("off")

    def box(x, y, w, h, text, fc="#ffffff", ec=SLATE, fs=12, bold=False, tc=SLATE):
        r = plt.Rectangle((x, y), w, h, facecolor=fc, edgecolor=ec, lw=1.3,
                          zorder=2, joinstyle="round")
        ax.add_patch(r)
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs,
                color=tc, fontweight="bold" if bold else "normal", zorder=3, linespacing=1.5)

    def arrow(x1, y1, x2, y2, label=None):
        ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                    arrowprops=dict(arrowstyle="-|>", color=SLATE, lw=1.3, shrinkA=2, shrinkB=2))
        if label:
            ax.text((x1 + x2) / 2, (y1 + y2) / 2 + 0.18, label, ha="center",
                    fontsize=10, color="#5c7683", style="italic")

    box(0.2, 8.5, 2.6, 1.0, "Patient utterance\n(plain English)", fc="#e6f4f3", ec=TEAL, bold=True, tc=TEAL)
    box(3.6, 8.5, 3.0, 1.0, "Deterministic entity extractor\nregex + range validation", fc="#ffffff")
    box(7.4, 8.5, 2.4, 1.0, "Symptom triage\n244 conditions", fc="#ffffff")
    arrow(2.8, 9.0, 3.6, 9.0)
    arrow(5.1, 8.5, 5.1, 7.6)
    arrow(6.6, 9.0, 7.4, 9.0)

    box(3.1, 6.6, 4.0, 1.0, "Canonical clinical record\n58 fields, observed / imputed tagged",
        fc="#f2fafa", ec=TEAL, bold=True, tc=TEAL)

    models = [("Diabetes\nXGBoost", 0.2), ("Heart\nLogReg", 2.16), ("Kidney\nLogReg", 4.12),
              ("Liver\nRandomForest", 6.08), ("Breast\nLogReg", 8.04)]
    for label, x in models:
        box(x, 4.7, 1.76, 1.0, label, fc="#ffffff", fs=11)
        arrow(5.1, 6.6, x + 0.88, 5.7)

    box(1.6, 3.0, 6.8, 0.95, "Probability  +  SHAP attribution  +  provenance",
        fc="#f2fafa", ec=TEAL, bold=True, tc=TEAL)
    for _, x in models:
        arrow(x + 0.88, 4.7, 5.0, 3.95)

    box(0.6, 1.05, 4.0, 1.55, "Local language model\ngemma3:4b over loopback\n(observed features only)",
        fc="#ffffff", ec=TEAL, fs=11)
    box(5.4, 1.05, 4.0, 1.55, "Deterministic prescribing\nIndian OTC formulary\n(chosen in code)",
        fc="#ffffff", ec=TEAL, fs=11)
    arrow(3.6, 3.0, 2.6, 2.6)
    arrow(6.4, 3.0, 7.4, 2.6)

    box(2.6, 0.0, 4.8, 0.85, "Patient-facing note and prescription",
        fc="#e6f4f3", ec=TEAL, bold=True, tc=TEAL)
    arrow(2.6, 1.05, 4.2, 0.85)
    arrow(7.4, 1.05, 5.8, 0.85)

    ax.text(5.0, 9.85, "IHACDP request path", ha="center", fontsize=15, fontweight="bold", color=DARK if False else SLATE)
    fig.tight_layout(); name = "fig_architecture_wide" if wide else "fig_architecture"
    fig.savefig(OUT / f"{name}.png", dpi=200, bbox_inches="tight")
    plt.close(fig)



def shap_ckd():
    """A real SHAP attribution for one CKD record, split by provenance.

    The slide claims imputed features score but never reach the narrator, so
    the figure has to show both groups and mark which ones are withheld.
    """
    import sys
    sys.path.insert(0, str(ROOT))
    from backend.services.predictor import engine

    record = {                      # what a patient actually volunteered
        "age": 58, "diastolic_bp": 90, "serum_creatinine": 3.2, "blood_urea": 96,
        "haemoglobin": 9.4, "urine_albumin": 3, "specific_gravity": 1.010,
        "hypertension_dx": True, "diabetes_dx": True, "appetite": "poor",
    }
    out = engine().models["kidney"].predict(record)
    contribs = out["contributions"][:8][::-1]
    if not contribs:
        print("  ! no SHAP contributions for the CKD example"); return

    labels = [c["label"] for c in contribs]
    vals = [c["shap"] for c in contribs]
    observed = [c["observed"] for c in contribs]

    fig, ax = plt.subplots(figsize=(8.0, 4.6))
    y = np.arange(len(vals))
    for i, (v, obs) in enumerate(zip(vals, observed)):
        ax.barh(y[i], v, color=(TEAL if v > 0 else SLATE) if obs else "#c9d6d6",
                edgecolor="none" if obs else "#9fb3b3",
                hatch=None if obs else "///", height=0.66)
    ax.set_yticks(y)
    ax.set_yticklabels([f"{l}" + ("" if o else "  (imputed)")
                        for l, o in zip(labels, observed)], fontsize=12)
    ax.axvline(0, color="#8fa3a3", lw=0.8)
    ax.set_xlabel("SHAP contribution to the CKD risk score")
    ax.set_title(f"Chronic Kidney Disease  ·  modelled risk {out['probability']:.2f}",
                 fontsize=13.5, pad=10)
    ax.text(0.0, -0.22, "Shaded bars are imputed: they move the score but never reach the narrator.",
            transform=ax.transAxes, fontsize=11, color=SLATE)
    fig.tight_layout(); fig.savefig(OUT / "fig_shap_ckd.png", dpi=200); plt.close(fig)


if __name__ == "__main__":
    architecture(wide=True); print("  architecture_wide")
    for fn in (architecture, roc_curves, algorithm_contest, metrics_panel, triage_accuracy, llm_benchmark, triage_sources, shap_ckd):
        fn(); print(f"  {fn.__name__}")
    print(f"\n{len(list(OUT.glob('*.png')))} figures -> docs/figures/")