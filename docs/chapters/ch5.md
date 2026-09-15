## 5.1 Introduction

This chapter reports the empirical results obtained from IHACDP and discusses what they do and do not establish. The evaluation is organised around the four components that together produce a consultation: the five supervised risk models, the symptom triage layer, the local language model, and the explainability and safety machinery that binds them. Each component is measured on the criterion appropriate to it. The risk models are measured on discrimination and calibration against held-out clinical data. The triage layer is measured on top-3 recovery of the correct condition as a function of how many symptoms the patient volunteers. The language model is measured on task behaviour rather than on general-knowledge benchmarks, because the role it occupies in this system is narrow and behavioural. The system as a whole is measured by graded end-to-end consultations that escalate from a trivial complaint to a surgical emergency.

Two points of method apply throughout. First, every figure reported here comes from the artefact as it is actually shipped, not from a favourable intermediate configuration; where a shipped design scores lower than a discarded alternative, the shipped figure is the one reported and the discrepancy is explained. Second, two results in this chapter are flattering in a way that would mislead if presented without qualification, and both are addressed directly in Section 5.8 rather than left for an examiner to discover.

## 5.2 Evaluation Metrics

Six metrics are used to characterise the supervised models, and it is worth stating precisely what each measures before the numbers are presented.

**Accuracy** is the proportion of test cases classified correctly at the default decision threshold of 0.5. It is the most intuitive metric and the least informative on medical data, because a dataset in which 90% of patients are healthy yields 90% accuracy from a model that predicts "healthy" unconditionally.

**Precision** is the proportion of positive predictions that are genuinely positive, and answers the question: when the system raises a concern, how often is that concern real? Low precision produces unnecessary alarm and unnecessary referral.

**Recall** (sensitivity) is the proportion of genuinely positive cases the model identifies, and answers the complementary question: of the patients who have the condition, how many does the system catch? In a screening context recall carries greater weight than precision, because the cost of a missed disease exceeds the cost of a false alarm that a clinician subsequently dismisses.

**F1 score** is the harmonic mean of precision and recall, summarising the trade-off between them in a single number and penalising models that achieve one at the expense of the other.

**ROC-AUC** is the area under the receiver operating characteristic curve. It equals the probability that a randomly chosen positive case receives a higher score than a randomly chosen negative case, and is therefore threshold-independent: it measures the quality of the ranking the model produces rather than the quality of one arbitrary cut applied to that ranking.

**Brier score** is the mean squared error between predicted probability and observed outcome, and is a measure of calibration rather than discrimination. Lower is better. It matters here because IHACDP presents probabilities to the patient as risk bands; a model that ranks correctly but is systematically over-confident would produce technically defensible orderings attached to misleading numbers.

ROC-AUC was adopted as the model-selection criterion for three reasons. It is insensitive to class imbalance, which is present to differing degrees in all five datasets. It is insensitive to the decision threshold, which is a deployment choice that should not be entangled with the choice of algorithm. And it evaluates the ranking itself, which is what the downstream banding and explanation machinery actually consumes. Accuracy, precision, recall, F1 and Brier score are reported alongside AUC as diagnostic context, but they did not decide which algorithm was retained.

## 5.3 Risk Model Results

Each of the five disease models was fitted with three candidate algorithms — logistic regression, random forest and XGBoost — and the winner selected by mean ROC-AUC under 5-fold stratified cross-validation on the training portion. All figures in Table 5.1 are then computed on a held-out 20% test split that took no part in either fitting or selection.

**Table 5.1: Held-out test performance of the five supervised risk models**

| Disease | Dataset | Rows | Features | Selected model | AUC | Accuracy | F1 | Precision | Recall | Brier |
|---|---|---|---|---|---|---|---|---|---|---|
| Type 2 Diabetes | UCI #891 CDC BRFSS | 75,346 | 21 | XGBoost | 0.8268 | 0.7490 | 0.7491 | 0.7053 | 0.7987 | 0.1693 |
| Coronary Heart Disease | UCI #45 Cleveland | 303 | 13 | LogisticRegression | 0.9502 | 0.8689 | 0.8667 | 0.8125 | 0.9286 | 0.0964 |
| Chronic Kidney Disease | UCI #336 | 400 | 24 | LogisticRegression | 1.0000 | 0.9750 | 0.9796 | 1.0000 | 0.9600 | 0.0117 |
| Chronic Liver Disease | UCI #225 ILPD | 583 | 10 | RandomForest | 0.7821 | 0.7009 | 0.7904 | 0.7857 | 0.7952 | 0.1704 |
| Breast Cancer (Malignancy) | UCI #17 WDBC | 569 | 30 | LogisticRegression | 0.9954 | 0.9737 | 0.9639 | 0.9756 | 0.9524 | 0.0222 |

The per-disease leaderboard is more instructive than any single row of the table, because the algorithm that won differs across the five problems in a way that tracks the structure of the data rather than any general superiority of one method. Logistic regression won three of the five contests, and in each case the dataset is small and the signal close to linear: Cleveland at 303 rows and 13 features, CKD at 400 rows and 24 features, and WDBC at 569 rows and 30 morphometric measurements. On data of this size a gradient-boosted ensemble has ample capacity to memorise the training partition, and cross-validation duly penalised it; the regularised linear model generalised better and, as a secondary benefit, produces coefficients that are directly interpretable.

XGBoost won only where the data justified it. The BRFSS diabetes set is two orders of magnitude larger at 75,346 rows, and its predictors — body mass index, general health, blood pressure, physical activity, age band — interact in ways a linear decision surface cannot capture. There, the additional capacity of boosting was supported by sufficient data and it took the selection.

Random forest won the liver problem, and it is important not to read that as a success. ILPD is the weakest of the five datasets: 583 rows, only 10 features, a marked class imbalance toward positive cases, and known label noise. The random forest won the internal contest at 0.7821 AUC because bagging tolerates noisy small-sample data better than either alternative, not because it found strong structure. Its Brier score of 0.1704 is the worst in the table alongside diabetes, confirming that the probabilities it emits are the least trustworthy of the five.

A final observation concerns the relationship between discrimination and calibration. The two best-discriminating models are also the best-calibrated — CKD at 0.0117 and breast cancer at 0.0222 — while the two weakest discriminators carry Brier scores an order of magnitude worse. This is expected but consequential: it means that where IHACDP is least certain, its numeric output is also least precise, and the risk banding presented to the patient must be read accordingly.

## 5.4 Symptom Triage Results

The triage layer covers 244 conditions across 389 symptoms, assembled from 516 unique presentation patterns drawn from three sources: 41 conditions from a public symptom/condition matrix, 174 from SymCat, and 29 clinical patterns curated specifically for this project. Ranking is performed by IDF-weighted overlap between the symptoms the patient reported and each condition's pattern, with four guards applied on top: defining-feature gates, a prevalence prior, a reserved uncertainty mass, and suppression of the suggestion entirely when the field is too closely tied to separate.

**Table 5.2: Top-3 accuracy of the shipped triage scorer by number of symptoms volunteered**

| Symptoms volunteered | Top-3 accuracy |
|---|---|
| 2 | 78.4% |
| 3 | 88.8% |
| 4 | 90.4% |

The steep improvement between two and three symptoms, and the flattening thereafter, is the expected behaviour of an overlap-based scorer: the third symptom usually resolves the ambiguity that two symptoms leave open, while the fourth adds comparatively little because the field has already narrowed. The practical consequence is visible in the deployed behaviour: where a single generic symptom leaves the candidate field effectively tied, no shortlist is offered at all, and the consultation continues until either the symptom picture separates or the patient indicates they have nothing further to add.

It must be stated clearly which scorer these figures describe. They describe the IDF-weighted overlap ranker that is actually shipped, not a fitted probabilistic classifier — and the fitted alternatives score better on paper while being useless in deployment. Training a classifier on the public symptom matrix as distributed returns approximately 100% accuracy, but that figure is an artefact: the source repeats every pattern roughly 120 times, inflating 4,920 rows into what are only 304 unique presentations, so the test split is very largely a copy of the training split. De-duplication was therefore performed before any figure was computed. A naive Bayes fit over the de-duplicated corpus fails for a different and more dangerous reason. Because a patient mentions only what troubles them, naive Bayes treats every unmentioned symptom as confirmed absent, which drives the score toward whichever condition has the fewest symptoms in its pattern. In testing this behaviour ranked a common head cold as AIDS at 46% confidence. The IDF-weighted ranker was adopted precisely because it treats silence as absence of evidence rather than as evidence of absence, and its lower headline number is the honest one.

Corpus composition was also decided on measured grounds rather than on coverage for its own sake. SymCat supplies 801 conditions, but adopting all of them collapsed top-3 recovery from 61% to 18%, because a head cold then had to out-rank several hundred rare diagnoses competing on partially overlapping symptoms. The corpus was restricted to 190 everyday presentations plus 13 conditions that are dangerous to miss, which preserves ranking quality without discarding the rare conditions that carry real consequence. The 29 curated patterns fill a gap no public dataset covers: musculoskeletal and everyday injury presentations such as ankle sprain, suspected fracture, muscle strain, tendonitis, concussion, minor burn, mechanical low back pain and sciatica, along with common complaints like tension and sinus headache, otitis media, tonsillitis, conjunctivitis, toothache, influenza, food poisoning, dehydration, heat exhaustion, anxiety, insomnia and menstrual cramps, plus an appendicitis localiser.

## 5.5 Language Model Benchmark Results

Three locally-servable models were benchmarked by `scripts/bench_llm.py` on task behaviour rather than general knowledge, since the model's role in IHACDP is confined to narrating a decision that has already been made elsewhere. Seven weighted criteria were used: absence of reasoning leakage (30), conciseness (15), report-section adherence (15), one question per turn (10), red-flag escalation (10), evidence faithfulness (10), and zero dose violations (10).

**Table 5.3: Language model benchmark results**

| Model | Score | Reasoning leak | Concise | 1 question/turn | Red flag | Sections | Dose violations | Faithfulness | Latency | Throughput |
|---|---|---|---|---|---|---|---|---|---|---|
| gemma3:4b | 100.0 | 0% | 100% | 100% | PASS | 6/6 | 0 | 100% | 4.17 s | 11.4 tok/s |
| llama3.2:3b | 90.0 | 0% | 100% | 100% | PASS | 6/6 | 3 | 100% | 2.07 s | 17.1 tok/s |
| qwen3:4b | 42.5 | 75% | 0% | 0% | PASS | 6/6 | 6 | 100% | 10.86 s | 24.0 tok/s |

gemma3:4b was selected on a perfect score and is the model shipped. Two findings from this benchmark are worth recording.

The first is that the fastest model was not the safest. llama3.2:3b responded in half the latency of the selected model and produced faster tokens, and it matched gemma3 on leakage, conciseness, turn discipline and faithfulness. It lost the contest on a single criterion: it volunteered specific medicine doses on three occasions. In a system whose prescribing layer is deterministic by design, a model that improvises dosage defeats the safety property the architecture exists to guarantee, and no latency advantage compensates for that.

The second is that hybrid-reasoning models are unusable as a patient-facing persona in their current form. qwen3:4b emitted chain-of-thought as patient-visible text in 75% of turns, and neither Ollama's `think:false` parameter nor an inline `/no_think` directive suppressed it. Its 0% scores on conciseness and one-question-per-turn follow directly from the same cause. Notably, qwen3 had the highest raw throughput of the three and still the worst latency, because it spent that throughput generating reasoning the patient should never have seen.

## 5.6 Explainability Results

Every risk assessment produced by IHACDP is accompanied by SHAP attributions computed over the fitted pipeline, and every feature carries a provenance tag marking it as observed or imputed. Only observed features are passed to the language model, which is what prevents the narration from describing a finding that was in reality a training-set median filling a blank field.

A representative chronic kidney disease case illustrates the mechanism. A patient record whose observed values included markedly elevated protein in the urine, reduced urine concentration and low haemoglobin returned a probability of 100%, banded as Very High. The SHAP attribution ranked the three drivers in that order: urinary protein contributed the largest positive push toward the positive class, urine concentration second, haemoglobin third. The narration that reached the patient named exactly those three findings and nothing else, because those were the three the patient had actually supplied. The remaining features of the 24-feature CKD vector were imputed, carried their imputed tag through the pipeline, and were withheld from the model's context entirely — so no sentence in the explanation could attribute significance to a value the patient never reported.

This worked example also demonstrates why the explanation layer must never be generated by the language model from the probability alone. The number 100% is not self-explanatory, and a model asked to justify it would confabulate plausible clinical reasoning. Here the justification is read out of the SHAP vector, which is a property of the fitted model, and the language model's only latitude is in phrasing.

## 5.7 System Validation

End-to-end validation was performed through graded consultations of increasing clinical severity, run against the shipped system with no manual intervention, alongside a regression suite of 71 tests covering the extractor, the mappers, the pipelines, the triage scorer and the prescribing rules; all 71 pass.

At the lowest grade, a plain tension headache was described conversationally. The system asked targeted single-field questions, closed the history on its own without the patient pressing a button, produced an assessment, and issued a deterministic over-the-counter recommendation from the Indian formulary — a single paracetamol product at a coded dose, with the no-duplicate-drug-class and no-two-paracetamol rules enforced in code rather than by the model. Intermediate grades exercised the track split: everyday complaints never routed the conversation into laboratory questioning, and the chronic-questioning track engaged only once a chronic signal appeared in the record.

At the highest grade, a presentation consistent with appendicitis was entered — periumbilical pain migrating to the right lower quadrant with associated features. The system did not offer an analgesic. Appendicitis is one of six conditions (with meningitis, sepsis, stroke, suspected fracture and concussion) for which the prescribing layer is configured to return urgent-care advice instead of medicine, and the case correctly returned that advice. This is the single most important behavioural result in the chapter: the failure mode that matters in an offline triage tool is not an inaccurate probability but a plausible painkiller recommendation that masks a surgical abdomen, and the deterministic prescribing layer is what forecloses it. Because the medicine decision is made in code and not by the language model, this behaviour is a property of the system rather than of a prompt, and cannot regress through a change of model weights.

## 5.8 Discussion

Two results in Table 5.1 require honest qualification.

The chronic kidney disease model returns an AUC of exactly 1.0000. This is a property of the UCI CKD dataset, not a claim of clinical perfection. That dataset is near-linearly separable on three variables — specific gravity, albumin and haemoglobin — so a regularised linear model can partition it cleanly. A perfect AUC on 400 curated records with unambiguous laboratory markers says that the problem as posed in the dataset is easy; it says nothing about performance on an unselected clinical population with early-stage or atypical disease. The figure is reported because it is what the experiment produced, and it is qualified here because reporting it unqualified would be misleading.

The liver model at 0.7821 AUC is the weakest result in the table, and it is reported unmassaged. ILPD is small at 583 rows, imbalanced, carries only 10 features, and is known to contain noisy labels. No resampling, threshold tuning or feature engineering was applied to raise the headline figure, because doing so would produce a better number without producing a better model. A discrimination of 0.78 is genuinely useful as one input among several and genuinely insufficient as a standalone determination, and the system's risk banding and narration are worded accordingly.

A third limitation is architectural rather than statistical. The breast cancer model reports the second-highest AUC in the table at 0.9954, but it is not reachable through conversation. Its 30 features are fine-needle-aspirate cytology morphometry — cell nucleus radius, texture, concavity, fractal dimension and their standard errors and worst-case values — which are measured from a stained slide under a microscope. No patient can report them, and the deterministic entity extractor has nothing to extract from an utterance. The model is therefore retained as a validated component operating on structured input, but the conversational front end cannot route to it. This is a useful demonstration of the boundary of the whole approach: a natural-language front end can only reach the subset of clinical models whose features are things a patient can observe and describe, and expanding coverage in the direction of cytology, imaging or histopathology would require an input modality this system does not have.

Taken together, the results support the design thesis stated at the outset. The statistical layer supplies the decision and the calibrated probability; the SHAP layer supplies the justification from the fitted model rather than from generated text; the provenance tagging ensures the narration cannot exceed the evidence the patient gave; and the deterministic prescribing layer ensures that the most consequential output — whether to recommend a medicine or urgent care — never passes through the language model at all. The language model scored 100 on the benchmark precisely because the benchmark measured how faithfully it stayed within that narrow role.