## 3.1 Introduction

This chapter sets out the methodology by which IHACDP (Intelligent Healthcare Analytics and Clinical Decision Support Platform) was designed, trained and assembled. The platform accepts a description of symptoms written in ordinary English, assesses it through a symptom-triage layer and five supervised risk models, and then uses a locally-run language model to explain the outcome to the patient. The methodological commitment that shapes every decision described below is the separation of judgement from narration: the statistical components decide, the language model only explains, and a hard architectural wall prevents the latter from performing the former. Each section that follows describes one stage of that pipeline — data selection, preprocessing, entity extraction, model selection, triage scoring, explanation, consultation flow and prescribing — together with the design alternatives that were tested and rejected. Where a result is weaker than it might appear, or stronger than it should be believed, the limitation is stated rather than smoothed over.

## 3.2 System Architecture

IHACDP is a stateless local web application. A FastAPI application served by uvicorn exposes the consultation endpoint over Server-Sent Events, so that generated text streams to the browser token by token; the frontend is hand-written HTML, CSS and JavaScript with no build step and no content-delivery-network dependency; Ollama serves the gemma3:4b language model over the loopback interface. No component contacts a remote host at any point during a consultation. The transcript and the accumulated clinical record live only in the browser tab, and no server-side session or database records them, so a closed tab is the end of the data.

The request path is deterministic up to the final narration stage:

```
  Patient utterance (plain English)
              |
              v
  +-------------------------------------------+
  | Deterministic regex entity extractor      |   NOT the LLM
  | range-validated, unit-aware                |
  +-------------------------------------------+
              |
              v
  +-------------------------------------------+
  | Canonical clinical record (58 fields)     |
  | every field tagged OBSERVED or IMPUTED    |
  +-------------------------------------------+
              |
      +-------+--------------------------+
      |                                  |
      v                                  v
  +--------------------+     +--------------------------+
  | Symptom triage     |     | Per-disease mappers      |
  | 244 conditions     |     | record -> model features |
  | IDF-weighted       |     +--------------------------+
  | overlap scoring    |                 |
  +--------------------+                 v
      |                        +--------------------------+
      |                        | 5 scikit-learn pipelines |
      |                        | diabetes, CHD, CKD,      |
      |                        | liver, breast cancer     |
      |                        +--------------------------+
      |                                  |
      |                                  v
      |                        +--------------------------+
      |                        | probability + SHAP       |
      |                        | attribution + provenance |
      |                        +--------------------------+
      |                                  |
      +-------------+--------------------+
                    v
      +-------------------------------------+
      | Evidence filter: OBSERVED only      |
      +-------------------------------------+
                    |
                    v
      +-------------------------------------+
      | gemma3:4b via Ollama (loopback)     |  narrates, never decides
      +-------------------------------------+
                    |
                    v
      +-------------------------------------+
      | Deterministic prescribing module    |  code, not model
      +-------------------------------------+
                    |
                    v
          SSE stream -> browser tab
```

The evidence filter at the penultimate stage is the mechanism that enforces the design thesis. Because every field of the clinical record carries a provenance tag, the assembler can withhold imputed values entirely, so the model is never in a position to describe a finding that was in reality a training-set median filling a blank.

## 3.3 Datasets

Five public clinical datasets from the UCI Machine Learning Repository underpin the risk models, and three independent sources were merged to build the symptom-triage knowledge base.

| Source | Provenance | Rows / Patterns | Features | Role |
|---|---|---|---|---|
| Type 2 Diabetes | UCI #891, CDC BRFSS survey | 75,346 | 21 | Risk model |
| Coronary Heart Disease | UCI #45, Cleveland | 303 | 13 | Risk model |
| Chronic Kidney Disease | UCI #336 | 400 | 24 | Risk model |
| Chronic Liver Disease | UCI #225, ILPD | 583 | 10 | Risk model |
| Breast Cancer (Malignancy) | UCI #17, WDBC | 569 | 30 | Risk model |
| Public symptom/condition matrix | Open dataset, 4,920 raw rows de-duplicated to 304 unique patterns | 41 conditions | 389-symptom shared vocabulary | Triage |
| SymCat | Conditional probabilities P(symptom \| condition) over 801 conditions | 174 conditions adopted | 389-symptom shared vocabulary | Triage |
| Curated clinical patterns | Authored for this project | 29 conditions | 389-symptom shared vocabulary | Triage |

The disparity in scale is deliberate and consequential. The diabetes corpus is a large population survey, while the Cleveland, CKD, ILPD and WDBC sets are small clinical cohorts of a few hundred records each; the reported performance of the latter four must therefore be read as an estimate from a held-out split of a few dozen patients, not as a population-level claim.

The merged triage base covers 244 conditions across a vocabulary of 389 symptoms, expressed as 516 unique symptom patterns. The public matrix required de-duplication before it was usable at all: it repeats every pattern roughly 120 times, so a classifier trained on it as distributed scores a meaningless 100% because the test split is a verbatim copy of the training split. SymCat supplies far broader coverage, but adopting all 801 of its conditions was tested and rejected — top-3 recovery fell from 61% to 18%, because a common head cold then had to out-rank several hundred rare diagnoses competing for the same handful of symptoms. The 29 curated patterns cover presentations that no public dataset carries in usable form: ankle sprain, suspected fracture, muscle strain, tendonitis, concussion, minor burn, mechanical low back pain, sciatica, frozen shoulder, tension, cluster and sinus headache, otitis media, tonsillitis, conjunctivitis, toothache, influenza, food poisoning, constipation, dehydration, heat exhaustion, anxiety, insomnia, menstrual cramps, and a localiser for appendicitis.

## 3.4 Data Preprocessing

Each of the five datasets is handled by a scikit-learn pipeline in which every transformation is fitted on the training split alone and applied unchanged to the test split, so that no test-set statistic can leak into a fitted parameter. Numerical fields are median-imputed, because the clinical variables involved — blood pressure, serum creatinine, bilirubin, cell-nucleus geometry — are skewed and a mean imputer would be dragged by the tail. Categorical fields are mode-imputed. Continuous features are standardised, which matters most for the three datasets whose selected model is logistic regression, where unscaled features would make the coefficient magnitudes incomparable and slow convergence. Genuinely ordered categories, such as the banded general-health and age variables in the BRFSS diabetes data, are ordinal-encoded so that their ordering is preserved rather than destroyed by one-hot expansion.

The step that distinguishes this pipeline from a conventional one is provenance tagging. At the point where a patient's clinical record is assembled, every one of the 58 canonical fields is marked either observed — the patient actually stated it, or it was extracted from what they stated — or imputed, meaning the pipeline supplied a fill value so that the model could be scored at all. Both classes of value pass to the model, because a scikit-learn pipeline requires a complete feature vector; only the observed subset is released to the language model. Without this separation the narration layer would be free to write a sentence such as "your cholesterol is elevated" about a figure the patient never gave, which is precisely the failure mode the platform is built to prevent.

## 3.5 Deterministic Clinical Entity Extraction

Numbers are extracted from patient utterances by a regular-expression entity extractor with explicit range validation, not by the language model. This is a deliberate rejection of the obvious design. A language model asked to parse "my sugar was one forty this morning" will usually return 140, but its failure mode is silent and unbounded: it may return a plausible number that was never said, normalise units without saying so, or hallucinate a second measurement to fill a field it believes should be populated. A regular expression has the opposite failure profile — it either matches or it does not, and when it does not the field simply remains unobserved, which is a safe state because the provenance tag will then keep it out of the narration.

Each extracted quantity is checked against a physiologically plausible range for its field before it is admitted to the clinical record. A systolic pressure of 1,200 or an age of 400 is discarded rather than propagated, and unit variants are normalised in code. The extractor is therefore auditable line by line, reproducible across runs, and testable by unit test — three properties a generative parser cannot offer.

## 3.6 Risk Model Selection

No algorithm was assumed in advance. For each of the five diseases, three candidates — logistic regression, random forest and XGBoost — were trained and compared under 5-fold stratified cross-validation on the training portion, with stratification preserving the class balance in every fold. Selection was by mean cross-validated ROC-AUC, chosen over accuracy because several of the datasets are imbalanced and a majority-class predictor can post a respectable accuracy while being clinically useless. The winning configuration was then refitted on the full training split and evaluated once on a held-out 20% test split, which was untouched during selection.

The contest produced a genuinely mixed outcome rather than a single dominant algorithm: XGBoost won on the large diabetes survey, random forest on the small and noisy liver data, and logistic regression on the coronary, kidney and breast-cancer sets. This is consistent with the sample sizes involved — the gradient-boosted model has the data to exploit interactions only in the 75,346-row corpus, whereas on a 303-row cohort a regularised linear model generalises better.

Two results require honest qualification. The chronic kidney disease model attains a test AUC of 1.0000, which is a property of the UCI CKD dataset rather than evidence of clinical perfection: the data are near-linearly separable on specific gravity, albumin and haemoglobin, and any competent classifier will separate them. It should be read as a sanity check that the pipeline works, not as a performance claim. Conversely, the liver model reaches only 0.7821 on ILPD, which is a small, imbalanced and noisy dataset that is genuinely hard; the figure is reported unmassaged, without resampling tricks or threshold tuning chosen to flatter it.

## 3.7 Symptom Triage Layer

Ranking of candidate conditions is performed by inverse-document-frequency-weighted overlap between the symptoms the patient has reported and each condition's stored pattern, not by a fitted probabilistic classifier. The IDF weighting means that a symptom shared by dozens of conditions, such as fatigue, contributes little discriminative weight, while a rare and specific symptom contributes a great deal.

A naive-Bayes formulation was implemented first and rejected on evidence. Its defect is structural: it treats every symptom the patient has not mentioned as confirmed absent, whereas in a real consultation an unmentioned symptom is simply unknown. The resulting likelihood collapses onto whichever condition carries the fewest symptoms in its pattern, because that condition accrues the fewest absence penalties. In testing it once ranked an ordinary head cold as AIDS at 46% confidence. Since patients volunteer two to four symptoms and are silent about 385 others, this is not an edge case but the normal operating condition of the system.

Four guards constrain the shipped scorer. Defining-feature gates prevent a condition from ranking without its pathognomonic symptom; a prevalence prior keeps common presentations ahead of rare ones at equal evidence; a reserved uncertainty mass ensures that a thin history cannot yield a confident answer; and output is suppressed entirely when the leading candidates are tied, on the principle that presenting an arbitrary tie-break as a finding is worse than presenting nothing. Measured by the number of symptoms volunteered, top-3 accuracy of the shipped scorer is 78.4% at two symptoms, 88.8% at three, and 90.4% at four.

## 3.8 Explainability with SHAP

Every risk score is accompanied by SHAP attributions computed over the model's feature vector, giving the signed contribution of each feature to that individual prediction. The attributions serve two purposes. Internally they are a debugging instrument, exposing cases where a model leans on a feature for the wrong reason. Externally they are the raw material of the explanation: the narration is constructed from the highest-magnitude contributions, so that the patient is told which of their own reported findings moved the estimate and in which direction.

Critically, the attribution list is filtered by provenance before it reaches the language model. A feature that carried a median fill value may well receive a non-zero SHAP value, but it is withheld, because an explanation built on an imputed value would be an explanation of the training set rather than of the patient.

## 3.9 Consultation Design

History-taking proceeds in two stages. In the first, the system's only objective is to understand the complaint, and no field directive is injected into the model's context until the patient has described something in their own words; this prevents the interview from opening with an interrogation about laboratory values the patient has no reason to expect. In the second stage the system moves to targeted enquiry, and the prompt assembler names exactly one clinical field per turn. The one-question-per-turn rule was a scored criterion in model selection precisely because a model that asks four questions at once produces an unusable transcript.

Questioning is also track-routed. An everyday complaint never routes to laboratory questions, and the chronic-disease line of questioning engages only once a chronic signal appears in the record. IHACDP closes the history on its own judgement of sufficiency and generates the assessment without the patient pressing a button, so that the consultation ends the way a clinical encounter ends rather than the way a web form ends.

## 3.10 Deterministic Prescribing

Medicines are selected and dosed in code, from a fixed Indian over-the-counter formulary comprising Dolo 650, Brufen 400, Sinarest, Strepsils, Cetzine, Digene, Electral, Otrivin, Volini, Betadine and Isabgol. The language model is never permitted to name a drug or a dose; it may only describe a decision the formulary module has already made. This is an architectural constraint rather than a prompt instruction, and it was made non-negotiable by the benchmarking result described in the following chapter, in which a candidate model volunteered specific doses three times unprompted.

The module enforces explicit safety rules: never two paracetamol-containing products in one recommendation, never two agents of the same drug class, and dosing only for over-the-counter medicines. Conditions that require assessment rather than self-medication — appendicitis, meningitis, sepsis, stroke, suspected fracture and concussion — return urgent-care advice and no medicine at all. The implementation comprises 5,186 lines of code covered by 71 passing regression tests, and the whole system runs fully offline on an Apple Silicon Mac with 8 GB of RAM, installed by a single `./setup.sh` invocation.