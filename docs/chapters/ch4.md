## 4.1 Introduction

This chapter describes how the design presented in Chapter 3 was realised as working software. IHACDP is implemented as a single Python service that exposes a streaming HTTP API to a hand-written browser front end, with every component — the entity extractor, the five supervised risk models, the symptom-triage scorer, the deterministic prescribing engine and the language model itself — executing on the user's own machine. The implementation totals 5,186 lines of code and is covered by 71 regression tests, all passing.

Two implementation decisions shape the whole system and are worth restating before the detail. First, the language model does not diagnose; it narrates. All probabilistic judgement is produced by fitted scikit-learn and XGBoost pipelines, and the model receives only their output together with the features the patient actually supplied. Second, every field in the clinical record is tagged as observed or imputed at the point of construction, and only observed fields cross into the language model's prompt. Both decisions had to be enforced in code rather than in prose instructions, and Sections 4.5 and 4.4 respectively describe the module boundary and the benchmark that verify them.

## 4.2 Tools and Technologies

| Layer | Technology | Role in IHACDP |
|---|---|---|
| Language | Python 3 | All backend logic, training scripts and tests |
| Web framework | FastAPI | Routing, request validation, JSON contracts |
| Server | uvicorn | ASGI server; serves the API and the static front end |
| Streaming transport | Server-Sent Events (SSE) | Token-by-token delivery of narration to the browser |
| Classical ML | scikit-learn | Pipelines, preprocessing, LogisticRegression, RandomForest, cross-validation |
| Gradient boosting | XGBoost | Selected model for the Type 2 Diabetes classifier |
| Explanation | SHAP | Per-prediction feature attribution for each risk model |
| Data handling | pandas | Dataset loading, cleaning and feature assembly |
| Language model runtime | Ollama (loopback only) | Serves gemma3:4b locally with no network egress |
| Selected language model | gemma3:4b | Narration, consultation turns and report prose |
| Front end | Hand-written HTML, CSS, JavaScript | Chat interface and report view; no build step, no CDN |
| Testing | Regression test suite (71 tests) | Guards extraction ranges, triage ranking and prescribing rules |

No JavaScript bundler, package registry fetch at runtime, or content-delivery network is used. The front end is plain static files because any build toolchain would have introduced a network dependency into a system whose central claim is offline operation.

## 4.3 System Requirements

| Requirement | Minimum | Recommended |
|---|---|---|
| Machine | Apple Silicon Mac | Apple Silicon Mac |
| Memory | 8 GB RAM | 16 GB RAM |
| Free disk space | ~12 GB | ~12 GB or more |
| Operating system | macOS with Python 3 available | Same |
| Network | Required once, for `./setup.sh` to fetch dependencies and the model | Same; not required thereafter |
| Browser | Any modern browser with SSE support | Same |

The 8 GB figure is a tested floor rather than an estimate: gemma3:4b was chosen partly because it runs within that envelope alongside the Python process and the five loaded pipelines. The disk requirement is dominated by the Ollama model weights and the Python scientific stack. After setup completes, the system requires no network connection at all, and the absence of egress is a property of the deployment rather than a configuration option.

## 4.4 Language Model Selection and Benchmark

The narration model was not chosen by reputation or parameter count. A dedicated harness, `scripts/bench_llm.py`, was written to score candidate models on the behaviours that actually matter for a patient-facing clinical assistant, on the grounds that standard knowledge benchmarks say nothing about whether a model will leak its own reasoning into a patient's screen or volunteer a drug dose.

Seven criteria were defined and weighted. Reasoning-leak avoidance carried the largest weight, 30, because a model that prints its deliberation as patient-visible text destroys the consultation persona outright. Conciseness and report-section adherence carried 15 each; one-question-per-turn discipline, red-flag escalation, evidence faithfulness and absence of dose suggestions carried 10 each. Evidence faithfulness tests whether the model confines itself to the observed features supplied to it, and the dose criterion is strictly binary in spirit: prescribing is deterministic in IHACDP, so any dose emitted by the model is a boundary violation regardless of whether the dose happens to be correct.

| Model | Score | Reasoning leak | Concise | One question/turn | Red flag | Sections | Dose violations | Faithfulness | Latency | Throughput |
|---|---|---|---|---|---|---|---|---|---|---|
| **gemma3:4b** | **100.0** | 0% | 100% | 100% | PASS | 6/6 | 0 | 100% | 4.17 s | 11.4 tok/s |
| llama3.2:3b | 90.0 | 0% | 100% | 100% | PASS | 6/6 | 3 | 100% | 2.07 s | 17.1 tok/s |
| qwen3:4b | 42.5 | 75% | 0% | 0% | PASS | 6/6 | 6 | 100% | 10.86 s | 24.0 tok/s |

Two findings emerged that would not have been visible from published benchmarks. The first is that the fastest model was not the safest: llama3.2:3b generated tokens at 17.1 tok/s against gemma3:4b's 11.4 and answered in half the time, yet suggested specific medicine doses on three occasions, encroaching on a decision the system reserves for code. The second concerns hybrid-reasoning models. qwen3:4b emitted chain-of-thought as patient-visible text in 75% of turns, and neither Ollama's `think:false` option nor the `/no_think` directive suppressed it. Its high raw throughput was therefore irrelevant, and the same behaviour also destroyed its conciseness and one-question-per-turn scores. gemma3:4b was selected on a perfect weighted score of 100.0, accepting the slower latency as the price of behaviour that never crossed the boundaries the architecture depends on.

## 4.5 Module Design

The backend is partitioned so that each responsibility has exactly one owner, and so that the wall between statistical inference and language generation corresponds to an actual module boundary rather than to a convention.

| Module | Responsibility |
|---|---|
| `config` | Central configuration: model paths, Ollama endpoint, thresholds and feature constants used across modules |
| `main` | FastAPI application; HTTP routes, request validation and SSE streaming of narration to the browser |
| `clinical` | Definition and construction of the canonical clinical record of 58 fields, including the observed/imputed provenance tag carried by every field |
| `extract` | Deterministic regex entity extraction from patient utterances, with range validation on every parsed value; performed without the language model |
| `predictor` | Per-disease feature mappers, loading and execution of the five sklearn pipelines, probability output and SHAP attribution |
| `llm` | The sole interface to Ollama; prompt assembly from observed features only, streaming, and enforcement of the narration-not-diagnosis contract |
| `consultation` | Two-stage dialogue control: understanding the complaint before any field directive is injected, then targeted enquiry naming exactly one field per turn, and closing the history automatically |
| `symptoms` | The triage layer over 244 conditions and 389 symptoms; IDF-weighted overlap ranking with defining-feature gates, prevalence prior, reserved uncertainty mass and tie suppression |
| `prescribing` | Deterministic selection and dosing from the Indian over-the-counter formulary, with class and duplicate-ingredient guards and urgent-care substitution for conditions requiring assessment |
| `vocab` | Shared clinical vocabulary: symptom synonyms, canonical field names and the lexical resources used by extraction and triage |

The division has a defensive purpose. Because `extract` and `predictor` never call `llm`, and because `llm` is given a filtered view of the clinical record, there is no code path by which a generated sentence can influence a probability, and no path by which an imputed value can appear in narration. Similarly, `prescribing` is reached only after triage and is never given the model's text, so the formulary rules — no two paracetamol products, no two drugs of the same class, dosing only for over-the-counter medicines, and urgent-care advice in place of medicine for appendicitis, meningitis, sepsis, stroke, suspected fracture and concussion — hold unconditionally.

## 4.6 Deployment and Reproducibility

Deployment is handled by a single script, `./setup.sh`, which installs the Python dependencies, pulls gemma3:4b through Ollama and prepares the trained pipelines so that the system can be brought up with no further manual configuration. This was a deliberate requirement: a project whose contribution is a set of boundaries is only credible if an examiner can reproduce those boundaries on their own machine without assembling an environment by hand.

Once setup completes, IHACDP runs fully offline on an Apple Silicon Mac with 8 GB of RAM. Ollama is reached over loopback only, the front end loads no remote asset, and no telemetry or remote inference call exists anywhere in the codebase. The service is also stateless by design: the transcript and the clinical record live only in the browser tab, so no patient narrative is written to disk by the server and closing the tab ends the record's existence.

Reproducibility of behaviour, as opposed to reproducibility of installation, rests on the 71 regression tests. These cover the range validation applied to extracted values, the ranking behaviour of the triage scorer including its guard conditions, and the prescribing rules that must never be violated. They are the mechanism by which the guarantees claimed in this chapter remain true as the code changes, and all 71 pass on the submitted build.