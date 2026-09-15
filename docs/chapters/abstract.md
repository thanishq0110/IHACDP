Primary healthcare in India is constrained by a severe shortage of physicians, and a large
proportion of the population self-medicates for everyday complaints without any structured
assessment. Digital symptom checkers have been proposed to fill this gap, but the current
generation divides into two unsatisfactory classes. Systems built on large language models
produce fluent and confident clinical text with no measurable error rate and a documented
tendency to fabricate findings, while systems built on supervised classifiers can predict a
risk but cannot explain the reasoning behind it in a form a patient understands.

This project presents IHACDP, an Intelligent Healthcare Analytics and Clinical Decision Support
Platform that resolves this tension by separating the two roles entirely. Statistical models
decide; the language model narrates. A deterministic, range-validated entity extractor lifts
vitals and laboratory values out of ordinary sentences, so no numeric value is paraphrased by a
neural network before it reaches a classifier. Five supervised risk models, each selected from a
contest between Logistic Regression, Random Forest and XGBoost by mean five-fold cross-validated
ROC-AUC, assess diabetes, coronary heart disease, chronic kidney disease, chronic liver disease
and breast malignancy, achieving test-split AUCs of 0.827, 0.950, 1.000, 0.782 and 0.995
respectively. A separate symptom triage layer covers 244 everyday conditions across 389
symptoms, merged from a public symptom matrix, the SymCat knowledge base and a set of curated
clinical patterns covering injuries and musculoskeletal complaints that no public dataset
carries. Ranking is by inverse-document-frequency weighted overlap rather than by a fitted
classifier, because a naive Bayes fit treats every unmentioned symptom as confirmed absent and
collapses onto whichever condition has the fewest symptoms. The shipped scorer recovers the
correct condition within its top three in 78.4 per cent of cases from two volunteered symptoms,
rising to 88.8 per cent from three.

Explainability is provided by SHAP attribution, and every feature carries a provenance tag
recording whether it was observed or imputed. Only observed features are passed to the language
model, which therefore cannot describe a finding that was in reality a training-set median
filling a blank. The language model itself was selected by a task-specific benchmark measuring
reasoning leakage, conciseness, red-flag escalation, evidence faithfulness and prescribing
safety rather than general knowledge; Gemma 3 4B scored 100 out of 100 and was the only
candidate that never suggested a medication dose unprompted. Prescribing is likewise
deterministic, drawing from an Indian over-the-counter formulary with explicit guards against
duplicate active ingredients, while conditions requiring assessment return urgent-care advice in
place of medication.

The entire system runs offline on consumer hardware with eight gigabytes of memory. No clinical
text leaves the device, there are no API keys and no third-party endpoints, and the
implementation comprises 5,186 lines of code validated by 71 regression tests. The work
demonstrates that an auditable division of labour between statistical inference and natural
language generation yields a decision support tool that is both explainable and safe enough to
place in front of a lay user.
