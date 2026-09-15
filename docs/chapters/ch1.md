## 1.1 Background

The first point of contact between a patient and the health system is rarely a clinician. It is a search engine, a messaging application, a neighbourhood pharmacist, or a household member with a remembered remedy. A person who wakes with abdominal pain, a persistent cough or an unusual degree of fatigue must decide, entirely unaided, whether the sensible response is rest, a visit to a local clinic, or an immediate trip to a hospital. That decision is made without a stethoscope, without a blood pressure cuff, without a laboratory report and, in the overwhelming majority of cases, without any professional input at all. It is, nevertheless, a triage decision, and the quality of the information available at that moment materially affects the outcome.

Computational support for this moment has developed along two largely separate lines. The first is the tradition of statistical risk modelling, in which supervised classifiers are fitted to curated clinical cohorts and produce calibrated probabilities of a defined outcome. These models are measurable: their discrimination, calibration and error profiles can be reported on held-out data and interrogated by a reviewer. Their weakness is expressive. A pipeline that returns a probability of 0.71 for type 2 diabetes tells a lay reader almost nothing about why that number was produced, and post-hoc attribution methods such as SHAP, while mathematically informative, yield feature-importance vectors rather than explanations a patient can act upon.

The second line is the recent generation of conversational systems built on large language models. These systems converse fluently, ask follow-up questions, restate a complaint in accessible language and produce an answer that reads with considerable authority. Their weakness is epistemic. When a language model is asked to name a condition, there is no held-out test split behind its answer, no reported sensitivity or specificity, and no way to distinguish a well-grounded inference from a confidently generated one. Worse, a language model presented with an incomplete clinical picture will frequently supply the missing parts itself, describing findings that the patient never reported.

IHACDP — the Intelligent Healthcare Analytics and Clinical Decision Support Platform — is built on the position that these two traditions should be combined without being blended. The statistical models decide; the language model narrates; and a strict architectural boundary separates the two. The system runs entirely offline on commodity hardware, so that no clinical narrative leaves the machine on which it was produced.

## 1.2 Motivation

The motivation for the project is partly architectural and partly contextual.

The architectural motivation arises directly from the failure mode described above. In an interactive consultation, a patient volunteers a handful of details and leaves the remainder unknown. A machine learning pipeline handles this by imputation: a missing serum creatinine becomes the training-set median, a missing body mass index becomes a cohort average. This is statistically defensible and clinically invisible — until a language model is handed the completed feature vector and, in good faith, informs the patient that their creatinine is elevated. The patient never gave a creatinine value. The number was manufactured by the pipeline. A narrative generated from imputed data is not merely imprecise; it fabricates clinical findings, and it does so in fluent, trustworthy prose. Preventing this specific failure is the design thesis of the entire system: every feature carries a provenance tag recording whether it was observed or imputed, and only observed features are ever exposed to the language model.

The contextual motivation is the state of Indian primary care. The physician-to-population ratio in India leaves large parts of the country, particularly outside metropolitan centres, with limited access to a qualified general practitioner for an unscheduled complaint. Consultation loads compress the available time per patient. The predictable consequence is widespread self-medication: a patient interprets their own symptoms, selects a medicine at a pharmacy counter, and doses it by guesswork or by analogy with a previous illness. The risks are familiar — duplicated active ingredients across branded products, non-steroidal anti-inflammatory drugs taken on empty stomachs, antibiotics taken for viral illness, and serious presentations such as appendicitis or concussion treated as ordinary discomfort until they escalate.

Any tool meant to help at this moment must therefore accept a hard constraint: the patient has no measuring equipment and no laboratory report. A system that requires fasting glucose, serum albumin or a fine-needle aspirate before it will say anything useful has excluded exactly the user it was built for. IHACDP is designed to be useful from plain-English symptom description alone, to invite laboratory values only where the patient has them, and to be explicit about how much of its assessment rests on what was actually supplied.

## 1.3 Problem Statement

The problem addressed by this project can be stated precisely. Existing patient-facing symptom assessment tools fall into two categories, each with a disqualifying defect for unsupervised home use.

Language-model-driven assistants diagnose fluently but possess no measurable error rate. They cannot be audited against a test split, they hallucinate findings to fill gaps in an incomplete history, and their confidence is uncorrelated with their correctness. Classifier-driven tools, by contrast, are measurable but inarticulate. They emit probabilities and feature weights that a lay user cannot interpret, they typically require structured numeric input that a patient at home does not possess, and they offer no conversational mechanism for eliciting the history in the first place.

The problem is therefore to construct a system that preserves the measurability of supervised classifiers and the communicative capability of a language model while making it structurally impossible for the language model to perform the diagnostic function or to describe evidence the patient did not provide. Subsidiary to this are three further requirements: the system must elicit a usable clinical history through natural conversation rather than a form; it must produce medicine recommendations through deterministic, auditable code rather than generative text; and it must operate wholly offline, since clinical narratives are among the most sensitive categories of personal data.

## 1.4 Objectives

### 1.4.1 Primary Objectives

The primary objectives concern the core decision architecture. The first is to train and validate five supervised risk models — for type 2 diabetes, coronary heart disease, chronic kidney disease, chronic liver disease and breast cancer malignancy — with model families selected by five-fold cross-validated ROC-AUC from logistic regression, random forest and XGBoost, and with all reported figures drawn from a held-out twenty per cent test split. The second is to construct a symptom triage layer covering 244 conditions across 389 symptoms that ranks candidate conditions by IDF-weighted overlap with the symptoms the patient actually volunteered, rather than by a fitted classifier whose absence assumptions are invalid in a conversational setting. The third is the provenance wall: a canonical clinical record in which every field is tagged observed or imputed, with only observed fields reaching the language model. The fourth is the selection of a local language model by measured task behaviour — reasoning leakage, conciseness, one-question-per-turn discipline, red-flag escalation, section adherence, evidence faithfulness and dose-violation count — rather than by general benchmark performance. The fifth is a deterministic prescribing module that selects and doses over-the-counter medicines in code.

### 1.4.2 Secondary Objectives

The secondary objectives concern usability, safety and reproducibility: a two-stage consultation flow that understands the complaint before interrogating specific fields and that names exactly one field per turn; separated everyday and chronic questioning tracks so that a sprained ankle never triggers laboratory enquiry; stateless operation in which the transcript and clinical record exist only in the browser tab; a dependency-light deployment with no build step and no content delivery network, installable by a single setup script and runnable offline on an eight-gigabyte Apple Silicon machine; and a regression suite sufficient to protect the safety-critical paths.

| No. | Objective | Type | Verification |
|-----|-----------|------|--------------|
| O1 | Train and validate five disease risk models with CV-selected families | Primary | Held-out 20% test AUC, accuracy, F1, precision, recall, Brier |
| O2 | Build a 244-condition, 389-symptom triage layer ranked by IDF-weighted overlap | Primary | Top-3 accuracy by count of volunteered symptoms |
| O3 | Enforce an observed/imputed provenance wall between models and narration | Primary | Field-level provenance tagging; evidence-faithfulness scoring |
| O4 | Select a local language model on task behaviour, not trivia | Primary | Weighted benchmark over seven behavioural criteria |
| O5 | Generate medicine recommendations deterministically in code | Primary | Formulary rules: no duplicate class, no duplicate paracetamol, OTC only |
| O6 | Elicit history through a two-stage, one-field-per-turn conversation | Secondary | Consultation-flow regression tests |
| O7 | Separate everyday and chronic questioning tracks | Secondary | Track-routing tests |
| O8 | Operate statelessly and wholly offline | Secondary | Loopback-only model serving; no server-side transcript store |
| O9 | Provide single-script installation on commodity hardware | Secondary | `./setup.sh` on an 8 GB Apple Silicon Mac |
| O10 | Maintain a regression suite over safety-critical paths | Secondary | 71 passing regression tests |

## 1.5 Scope of the Project

The scope of IHACDP is deliberately bounded. The system is a decision *support* platform intended for patient-facing preliminary assessment and safety-netting. It is not a diagnostic device, it is not a substitute for clinical examination, and it does not claim regulatory clearance of any kind. Its assessments are probabilistic statements derived from public research datasets and are presented as such.

Two limitations are stated plainly here rather than deferred. The chronic kidney disease model attains a test ROC-AUC of 1.0000. This figure is a property of the UCI CKD dataset, which is near-linearly separable on specific gravity, albumin and haemoglobin; it is emphatically not a claim of clinical perfection, and it would not survive contact with an unselected population. Conversely, the chronic liver disease model attains 0.7821 on the Indian Liver Patient Dataset, which is small, class-imbalanced and noisy. That figure is reported unmassaged, because a genuinely difficult problem reported honestly is more informative than a difficult problem disguised.

| In scope | Out of scope |
|----------|--------------|
| Conversational symptom intake in plain English | Voice, image or wearable-sensor input |
| Triage across 244 conditions from 389 symptoms | Exhaustive coverage of rare and tropical disease |
| Risk assessment for five specified chronic conditions | Any condition outside those five models |
| Narration of observed evidence by a local language model | Diagnosis, staging or prognosis by the language model |
| Deterministic dosing from an Indian OTC formulary | Prescription-only medicines and paediatric dosing |
| Urgent-care referral for red-flag presentations | Emergency dispatch or clinician hand-off integration |
| Fully offline, stateless, single-machine operation | Cloud hosting, multi-user accounts, longitudinal records |
| Interpretation of laboratory values the patient already holds | Ordering, acquiring or validating laboratory tests |
| Adult, self-reporting users | Paediatric, obstetric and intensive-care populations |

Within these boundaries the system is complete and self-contained: 5,186 lines of code, 71 passing regression tests, and an installation path that requires no network connection after setup.