## Slide 1: IHACDP — Intelligent Healthcare Analytics and Clinical Decision Support Platform

- Offline clinical decision support with statistical risk models and a local language model
- Shiva Charan Ambati — 23BRS1203
- Atla Abhinav Karthik Reddy — 23BCE1085
- Kotha Rithvik Reddy — 23BCE1582
- B.Tech, School of Computer Science and Engineering, VIT Chennai
- Guide: Dr. Vivekanandan M (53646)
- Capstone Project Review

## Slide 2: Agenda

- Problem, motivation, and the Indian primary-care context
- Literature review: symptom checkers, ML risk prediction, explainability
- Research gap, objectives, and system architecture
- Datasets, methodology, and the observed/imputed wall
- Risk model selection, language model benchmark, SHAP explainability
- Results, limitations, future scope, conclusion

## Slide 3: Problem Definition and Domain Understanding

- Patients describe symptoms in plain English, not clinical vocabulary
- Clinical decision support must map free text to structured risk assessment
- Language models are fluent but unreliable as diagnostic engines
- Statistical models are reliable but opaque to a layperson
- Core problem: combine both without letting the generator decide outcomes
- Constraint: must run offline, with no patient data leaving the device

## Slide 4: Motivation: the Indian Primary-Care Context

- Primary care is the first and often only point of contact
- Consultation time is short; symptom histories are taken rapidly
- Connectivity and cloud access are unreliable outside urban centres
- Sending health data to remote APIs raises privacy and cost concerns
- Self-medication from an OTC counter is common and unguided
- An offline, on-device assistant fits this setting directly

## Slide 5: Literature Review: Symptom Checkers and Their Limitations

- Public symptom checkers rank conditions from a checklist of symptoms
- Coverage is skewed to conditions that public datasets happen to carry
- Injuries and musculoskeletal complaints are poorly represented
- Probabilistic fits misread silence: unmentioned symptoms treated as absent
- Naive Bayes on our data ranked a head cold as AIDS at 46%
- Outputs rarely separate what the patient said from what was assumed

## Slide 6: Literature Review: ML Risk Prediction and Explainability

- Supervised models on UCI clinical datasets are a well-established baseline
- LogisticRegression, RandomForest and XGBoost dominate tabular clinical risk work
- ROC-AUC under cross-validation is the standard selection criterion
- SHAP attributes a prediction to individual feature contributions
- Explainability literature rarely addresses who the explanation is written for
- Gap between a SHAP plot and a sentence a patient understands

## Slide 7: Research Gap

- Symptom checkers rank conditions but do not quantify chronic disease risk
- Risk models quantify but do not read plain-English complaints
- LLM health assistants generate diagnoses without statistical grounding
- Imputed feature values silently enter explanations as if observed
- No offline system unites triage, risk scoring, SHAP and narration
- IHACDP targets exactly that junction

## Slide 8: Objectives and Scope

- Accept plain-English symptom descriptions and extract entities deterministically
- Triage across 244 conditions using an interpretable ranking, not a fitted classifier
- Score five chronic disease risks with cross-validated supervised models
- Explain every prediction with SHAP over observed features only
- Narrate results with a local language model that never diagnoses
- Run fully offline on commodity hardware; no cloud, no build step

## Slide 9: System Architecture (the Request Path)

- Plain-English input enters a FastAPI endpoint; the system is stateless
- Deterministic regex entity extraction with range validation parses the text
- Features are tagged observed or imputed; only observed ones cross the wall
- Triage layer ranks conditions by IDF-weighted symptom overlap
- Risk models score; SHAP attributes each prediction to observed features
- Ollama serves gemma3:4b over loopback to narrate, not decide

## Slide 10: Datasets Used

| Source | Dataset | Size | Role |
|---|---|---|---|
| UCI #891 | CDC BRFSS | 75,346 rows | Type 2 Diabetes risk |
| UCI #45 | Cleveland | 303 rows | Coronary Heart Disease risk |
| UCI #336 | CKD | 400 rows | Chronic Kidney Disease risk |
| UCI #225 | ILPD | 583 rows | Chronic Liver Disease risk |
| UCI #17 | WDBC | 569 rows | Breast Cancer risk |
| Public matrix | Symptom matrix | 41 conditions | Triage patterns |
| SymCat | SymCat | 174 conditions | Triage patterns |
| Curated | Clinical patterns | 29 patterns | Injuries, musculoskeletal |

## Slide 11: Methodology: Deterministic Extraction and the Observed/Imputed Wall

- Regex entity extraction with range validation; no model parses the input
- Every extracted feature carries a tag: observed or imputed
- Imputed values fill the model's input vector but never reach the narrator
- The language model sees only evidence the patient actually gave
- Prevents fabricated explanations built on statistical placeholders
- Determinism makes extraction auditable and reproducible across runs

## Slide 12: Methodology: Triage Layer and Why IDF Overlap Beat Naive Bayes

- 244 conditions, 389 symptoms, 516 patterns merged from three sources
- Public symptom matrix 41, SymCat 174, plus 29 curated clinical patterns
- Curated patterns cover injuries and musculoskeletal complaints no public dataset carries
- Ranking uses IDF-weighted symptom overlap, not a fitted classifier
- Naive Bayes treats unmentioned symptoms as confirmed absent
- That fit once ranked a head cold as AIDS at 46%

## Slide 13: Risk Model Selection

| Condition | Winning model | ROC-AUC | Accuracy |
|---|---|---|---|
| Type 2 Diabetes | XGBoost | 0.8268 | 0.7490 |
| Coronary Heart Disease | LogisticRegression | 0.9502 | 0.8689 |
| Chronic Kidney Disease | LogisticRegression | 1.0000 | 0.9750 |
| Chronic Liver Disease | RandomForest | 0.7821 | 0.7009 |
| Breast Cancer | LogisticRegression | 0.9954 | 0.9737 |

- Selection by 5-fold cross-validated ROC-AUC across three candidate families
- Reported figures come from a held-out 20% test split

## Slide 14: Language Model Benchmark

| Model | Score | Reasoning leak | Dose violations | Latency |
|---|---|---|---|---|
| gemma3:4b | 100/100 | 0% | 0 | 4.17s |
| llama3.2:3b | 90.0 | — | 3 | 2.07s |
| qwen3:4b | 42.5 | 75% | — | — |

- Task-specific benchmark on our own prompts, not general trivia
- gemma3:4b: 100% concise, 100% one-question, red-flag PASS, 6/6 sections
- qwen3:4b emitted chain-of-thought as patient-visible text
- Neither think:false nor /no_think suppressed that leak; gemma3:4b selected

## Slide 15: Explainability with SHAP (Worked CKD Example)

- CKD model is LogisticRegression, AUC 1.0000, accuracy 0.9750 on held-out data
- SHAP attributes the risk score to individual input features
- Only features tagged observed are eligible to appear in the explanation
- Imputed values contribute to the score but are withheld from narration
- gemma3:4b converts the surviving attributions into plain-language reasoning
- The number comes from the model; the sentence comes from the narrator

## Slide 16: Deterministic Prescribing and Urgent-Care Escalation

- Prescribing is deterministic from a curated Indian OTC formulary
- Formulary: Dolo 650, Brufen 400, Sinarest, Strepsils, Cetzine, Digene, Electral, Otrivin, Volini
- Hard rule: no duplicate active ingredients across a suggestion set
- Hard rule: no dose emitted on prescription-only drugs
- Appendicitis, meningitis, sepsis, stroke, suspected fracture, concussion bypass prescribing
- Those conditions return urgent-care advice instead of any medication

## Slide 17: Results: Triage Accuracy and Graded System Validation

| Symptoms given | Top-3 triage accuracy |
|---|---|
| 2 symptoms | 78.4% |
| 3 symptoms | 88.8% |
| 4 symptoms | 90.4% |

- Accuracy improves as the patient supplies more symptoms, as expected
- Five risk models validated on held-out 20% splits, AUC 0.7821 to 1.0000
- 5,186 lines of code covered by 71 regression tests
- Full pipeline runs offline on an 8 GB Apple Silicon Mac

## Slide 18: Limitations, Stated Honestly

- CKD AUC 1.0000 reflects near-linear separability of that dataset, not clinical perfection
- ILPD at 0.7821 is genuinely hard; reported unmassaged rather than tuned upward
- Cleveland (303), CKD (400), ILPD (583), WDBC (569) are small datasets
- Triage coverage is bounded at 244 conditions and 389 symptoms
- Top-3 accuracy from 2 symptoms is 78.4%; sparse input degrades ranking
- System is stateless: no longitudinal history, no follow-up tracking

## Slide 19: Future Scope

- Extend triage coverage beyond 244 conditions and 389 symptoms
- Validate risk models on Indian cohort data rather than UCI splits
- Improve chronic liver disease performance above the 0.7821 baseline
- Add multilingual input for Indian languages at the extraction layer
- Evaluate newer small local models against the same task-specific benchmark
- Clinical usability study with primary-care practitioners

## Slide 20: Conclusion and Thank You

- Statistical models decide; the language model only narrates the decision
- Observed/imputed wall keeps explanations grounded in patient-given evidence
- Five risk models, AUC 0.7821 to 1.0000, selected by cross-validated ROC-AUC
- Triage reaches 90.4% top-3 accuracy from four symptoms
- 5,186 lines, 71 regression tests, fully offline on 8 GB hardware
- Thank you — questions welcome