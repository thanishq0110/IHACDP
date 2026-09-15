# IHACDP — Intelligent Healthcare Analytics & Clinical Decision Support Platform

An offline clinical decision support system. A patient describes symptoms in plain language;
five validated machine-learning models score disease risk; a locally-run language model explains
the result using **only** the evidence the patient actually provided.

---

## Design thesis

> The language model does not diagnose. It narrates what the statistical models found.

An LLM asked to diagnose from lab values produces fluent, confident prose with **no measurable
error rate**. A gradient-boosted classifier trained on 75,000 labelled records produces a
probability with a **published AUC**. IHACDP uses the second for the decision and the first for
the explanation — and keeps a hard wall between them.

```
Patient utterance
      │
      ├─▶ Regex entity extractor ──▶ canonical clinical record (58 fields)
      │                                    │
      │                                    ├─▶ diabetes mapper ─▶ XGBoost           ─┐
      │                                    ├─▶ heart    mapper ─▶ LogisticRegression │
      │                                    ├─▶ kidney   mapper ─▶ LogisticRegression ├─▶ probability
      │                                    ├─▶ liver    mapper ─▶ RandomForest       │   + SHAP
      │                                    └─▶ breast   mapper ─▶ LogisticRegression ┘   + provenance
      │                                                                                   │
      └─▶ local LLM ◀── OBSERVED features + SHAP drivers + AUC ◀────────────────────────┘
                │
                └─▶ streamed clinical note · drug classes only · never a dose
```

### Three safety mechanisms

| Mechanism | What it prevents |
|---|---|
| **Observed/imputed tagging** | Missing inputs are median-imputed so models can still run, but every feature carries its provenance and the LLM receives only *observed* ones. It cannot narrate a finding that was really a placeholder. |
| **Deterministic extraction** | Lab values are lifted by range-validated pattern matching, never paraphrased by a neural network before reaching a classifier. |
| **Class-only prescribing** | Management output names drug *classes*. Dose, frequency and brand are structurally withheld and flagged for a licensed prescriber. |

---

## Everyday complaints

Chronic risk models are the wrong tool for a head cold or a sprained ankle. A
separate triage layer covers **69 conditions** across
**165 symptoms**, from two sources:

| Source | Conditions | What it covers |
|---|---:|---|
| Public symptom/condition matrix | 41 | infectious and chronic disease |
| Curated clinical patterns | 28 | injuries, pains, ENT, eye, dental, mental health |

The public matrix is built around infectious and chronic disease and contains
nothing for what people actually present with. Lumbar pain mapped to *Cervical
spondylosis* — a neck condition — because no entry for mechanical back pain
existed. The curated set closes that: ankle sprain, suspected fracture, muscle
strain, tendonitis, concussion, minor burn, mechanical low back pain, sciatica,
frozen shoulder, tension/cluster/sinus headache, otitis media, tonsillitis,
conjunctivitis, toothache, influenza, food poisoning, constipation, dehydration,
heat exhaustion, anxiety, insomnia, menstrual cramps. Each condition records its
provenance in `artifacts/symptoms_meta.json`.

### Routing

A complaint the symptom vocabulary recognises goes to the symptom track, which
asks only about symptoms in ordinary words and never for a measured value.
Chronic questioning engages only on a chronic signal.

### Ranking

By IDF-weighted overlap with what was actually reported, not by a fitted
classifier. A naive-Bayes fit treats every unmentioned symptom as confirmed
absent, which collapses onto whichever condition has the fewest symptoms — it
ranked a head cold as AIDS at 46%.

Three guards on top:

- **Defining features.** Conditions overlapping only on generic symptoms need a
  distinguishing one present. A twisted ankle was proposing *Minor burn*,
  because both swell.
- **Reserved uncertainty.** Probability holds mass back for what has not been
  asked, so a complaint mapping cleanly onto one condition cannot read as 100%
  certain off two symptoms.
- **Evidence-scaled shortlist.** One symptom yields one candidate, not three.

### On the metrics

The public source repeats every pattern ~120 times; trained as shipped it scores
100% and means nothing. After de-duplication and merging, cross-validated
accuracy is 0.916. The figure worth quoting is partial-information
recovery — how often the right condition lands in the top three when only a few
symptoms are volunteered:

| Symptoms volunteered | Top-3 accuracy |
|---:|---:|
| 2 | 50.6% |
| 3 | 61.1% |
| 4 | 66.6% |

## Model performance

Selected by mean ROC-AUC over 5-fold stratified CV; figures from the held-out 20% test split.

| Disease | Dataset | Records | Winner | Test AUC | Accuracy | F1 |
|---|---|---:|---|---:|---:|---:|
| Type 2 Diabetes | UCI #891 CDC BRFSS | 75,346 | XGBoost | 0.827 | 74.9% | 0.749 |
| Coronary Heart Disease | UCI #45 Cleveland | 303 | LogisticRegression | 0.950 | 86.9% | 0.867 |
| Chronic Kidney Disease | UCI #336 | 400 | LogisticRegression | 1.000 | 97.5% | 0.980 |
| Chronic Liver Disease | UCI #225 ILPD | 583 | RandomForest | 0.782 | 70.1% | 0.790 |
| Breast Cancer | UCI #17 WDBC | 569 | LogisticRegression | 0.995 | 97.4% | 0.964 |

**Two honest caveats to raise before a reviewer does:**

1. **CKD AUC 1.000 is a property of the dataset, not a claim of perfection.** The UCI CKD set is
   near-linearly separable on specific gravity, albumin and haemoglobin. It is a demonstration of
   the pipeline, not evidence of clinical performance.
2. **ILPD (liver) at 0.782 is genuinely hard** — small, imbalanced and noisy. It is reported
   unmassaged rather than tuned until it looked good.

---

## LLM selection

Candidate local models were scored on the behaviours this application needs — not general trivia.
See `scripts/bench_llm.py`; results in `artifacts/llm_benchmark.json`.

| Criterion | Weight | Why it matters here |
|---|---:|---|
| Reasoning-leak rate | 30 | Hybrid-thinking models emit chain-of-thought as patient-facing text |
| Conciseness (≤4 sentences) | 15 | A real clinician does not deliver paragraphs mid-history |
| One question per turn | 10 | Multi-question turns read as a form, not a consultation |
| Red-flag escalation | 10 | Must interrupt history and direct to emergency care |
| Report section adherence | 15 | Downstream rendering depends on the six headings |
| Evidence faithfulness | 10 | Numbers in the note must trace to observed data |
| Zero dose violations | 10 | Prescribing safety boundary |

### Results

Three candidates were benchmarked on an Apple M2 / 8 GB. Each ran a four-turn history, a red-flag
escalation probe and a full grounded report.

| Model | Score | Leak | Concise | 1-Question | Red-flag | Sections | Dose violations | Faithfulness | Latency | Tok/s |
|---|---:|---:|---:|---:|---|---:|---:|---:|---:|---:|
| **gemma3:4b** ✅ | **100.0** | 0% | 100% | 100% | PASS | 6/6 | **0** | 100% | 4.2 s | 11.4 |
| llama3.2:3b | 90.0 | 0% | 100% | 100% | PASS | 6/6 | 3 | 100% | 2.1 s | 17.1 |
| qwen3:4b | 42.5 | **75%** | 0% | 0% | PASS | 6/6 | 6 | 100% | 10.9 s | 24.0 |

**Selected: `gemma3:4b`.**

Two findings worth reporting:

1. **The fastest model is not the safest.** `llama3.2:3b` was ~2× faster but suggested specific
   doses three times — crossing the prescribing-safety boundary. Gemma 3 was the only candidate
   that never did.
2. **Hybrid-reasoning models are unusable as a patient-facing persona without mitigation.**
   `qwen3:4b` emitted its chain-of-thought as patient-visible text in 75% of turns, and neither
   Ollama's `think: false` parameter nor the `/no_think` directive suppressed it. A defensive
   `strip_reasoning()` filter remains in `backend/services/llm.py` as a safety net.

```bash
./.venv/bin/python scripts/bench_llm.py
```

---

## Running it

Requires macOS on Apple Silicon, and about 12 GB of free disk on a first install.

```bash
./setup.sh      # once - installs everything
./run.sh        # start it
```

Then open <http://127.0.0.1:8000>.

`setup.sh` checks the machine, creates the Python environment, installs Ollama
and downloads the language model, fetches every dataset, trains all six models
and runs the test suite. It is safe to re-run: anything already in place is left
alone, and it asks only for the disk space the remaining steps need. Nothing in
it requires sudo.

Once setup has finished the application never touches the network again. The
language model runs on the machine, the API keeps no server-side session, and
there is no outbound request path in the code.

| | |
|---|---|
| Different port | `PORT=8001 ./run.sh` |
| Different model | `IHACDP_LLM_MODEL=llama3.2:3b ./run.sh` |
| Retrain from scratch | `rm -rf artifacts && ./setup.sh` |
| Run the tests | `./.venv/bin/python -m pytest tests -q` |

## Layout

```
backend/
  main.py                 FastAPI app, SSE streaming, static hosting
  config.py               runtime config, risk bands
  services/
    clinical.py           58-field canonical record + per-disease mappers
    extract.py            deterministic clinical entity extraction
    predictor.py          risk engine, SHAP, observed/imputed provenance
    llm.py                local Ollama transport, reasoning stripper
    consultation.py       physician persona, history taking, grounded reporting
frontend/                 landing · about · consultation · model performance
scripts/
  fetch_data.py           dataset ingestion with source fallbacks
  train.py                3-algorithm bake-off per disease
  bench_llm.py            LLM selection benchmark
artifacts/                trained models, metrics, ROC curves, benchmark
```

## API

| Endpoint | Purpose |
|---|---|
| `GET /api/health` | service + local model status |
| `GET /api/diseases` | catalogue, metrics, leaderboards, ROC curves |
| `GET /api/fields` | canonical clinical field dictionary |
| `POST /api/chat` | SSE — extraction, streamed reply, readiness |
| `POST /api/assess` | deterministic risk assessment |
| `POST /api/report` | SSE — assessment + streamed clinical note |

## Consultation flow

The history is taken in two stages rather than as a questionnaire:

1. **Understand the complaint.** On the opening turns the model is barred from asking about labs,
   vitals or lifestyle. It asks open questions about the presenting problem only.
2. **Targeted enquiry.** The complaint is matched against symptom cues to rank lines of enquiry
   (cardiac / renal / hepatic / metabolic). The server then names *exactly one* field per turn, drawn
   from the relevant disease's ask-order, skipping anything already on record or already asked.

A prior question is detected by scanning earlier assistant turns, so nothing is ever asked twice.

Interactive reference at `/api/docs`.

## Privacy

Stateless by design. The transcript and clinical record live only in the browser tab. The server
persists nothing, the LLM is reached over loopback, and there is no outbound request path in the
application.

