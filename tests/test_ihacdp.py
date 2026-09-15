"""IHACDP regression tests. Run: ./.venv/bin/python -m pytest tests -q"""
import re
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pytest
from fastapi.testclient import TestClient

from backend.services.extract import extract, extract_in_context, question_topic
from backend.services import clinical, consultation
from backend.services import symptoms as sx
from backend.services import prescribing
from backend.services.predictor import engine
from backend.services.llm import strip_reasoning
from backend.main import app

client = TestClient(app)


# ---------------- extraction ----------------
def test_extracts_vitals_from_natural_speech():
    r = extract("I'm 58, male. My BP was 148/94 and I weigh 92 kg at 172 cm.")
    assert r["age"] == 58 and r["sex"] == "male"
    assert r["systolic_bp"] == 148 and r["diastolic_bp"] == 94
    assert r["weight_kg"] == 92 and r["height_cm"] == 172


def test_extracts_lab_panel():
    r = extract("Creatinine 2.4, urea 64, haemoglobin 10.1, SGPT 88, SGOT 96, bilirubin 2.8")
    assert r["serum_creatinine"] == 2.4 and r["blood_urea"] == 64
    assert r["haemoglobin"] == 10.1 and r["alt_sgpt"] == 88
    assert r["ast_sgot"] == 96 and r["total_bilirubin"] == 2.8


def test_negation_is_respected():
    assert extract("I don't smoke and I never drink alcohol.")["smoker"] is False
    assert extract("I smoke ten a day.")["smoker"] is True


def test_out_of_range_values_rejected():
    """A creatinine of 900 is a transcription artefact, not a lab value."""
    assert "serum_creatinine" not in extract("creatinine 900")


def test_bare_answers_resolve_against_the_question():
    """A patient replying "Yes" or "3.8" is answering the previous question."""
    assert extract_in_context("Do you smoke?", "Yes") == {"smoker": True}
    assert extract_in_context("Do you exercise regularly?", "No, not really") == {"physical_activity": False}
    assert extract_in_context("How old are you?", "58") == {"age": 58.0}
    assert extract_in_context("What did your creatinine come back as?", "3.8") == {"serum_creatinine": 3.8}
    assert extract_in_context("Are you male or female?", "Female") == {"sex": "female"}


def test_context_extraction_respects_ranges_and_unknown_topics():
    assert extract_in_context("What is your creatinine?", "900") == {}   # implausible
    assert extract_in_context("How are you today?", "Yes") == {}          # no topic


def test_question_topic_detection():
    assert question_topic("Have your ankles been swelling?") == "pedal_edema"
    assert question_topic("What was your cholesterol?") == "total_cholesterol"
    assert question_topic("Nice weather isn't it?") is None


# ---------------- clinical derivation ----------------
def test_bmi_and_flags_derived():
    d = clinical.derive({"height_cm": 172, "weight_kg": 92, "systolic_bp": 148, "diastolic_bp": 94})
    assert d["bmi"] == pytest.approx(31.1, abs=0.2)
    assert d["hypertension_dx"] == 1


def test_brfss_age_bucketing():
    m = clinical.map_diabetes({"age": 58})
    assert m["Age"] == 8          # 55-59 bucket


def test_coverage_gate():
    assert clinical.coverage({}, "heart") == 0.0
    rich = {"age": 58, "sex": "male", "chest_pain_type": 1, "systolic_bp": 148,
            "total_cholesterol": 256, "max_heart_rate": 122, "exercise_angina": True}
    assert clinical.coverage(rich, "heart") == 1.0


# ---------------- prescribing ----------------
def test_conditions_needing_assessment_are_not_medicated():
    """Suspected appendicitis was prescribed paracetamol, which masks the very
    sign that decides whether someone goes to theatre."""
    e = sx.engine()
    found = e.extract("severe pain in my lower right tummy with nausea and fever")
    conditions = e.predict(found)
    assert conditions and conditions[0]["condition"] == "Appendicitis", conditions
    assert conditions[0]["urgent"] is True
    assert prescribing.needs_assessment(conditions) is True
    assert prescribing.build(found, conditions) == []


def test_cannot_miss_conditions_are_not_handicapped_by_the_prior():
    """Appendicitis ranked below food poisoning on a textbook presentation."""
    from backend.services.symptoms import prior_for
    for c in ("Appendicitis", "Meningitis", "Sepsis", "Stroke"):
        assert prior_for(c) >= 1.0, f"{c} is penalised at {prior_for(c)}"


def test_never_prescribes_two_paracetamol_products():
    """It issued Calpol and Sinarest together - both contain paracetamol."""
    for symptoms in ({"high_fever": True, "runny_nose": True, "continuous_sneezing": True,
                      "throat_irritation": True, "headache": True},
                     {"headache": True, "mild_fever": True, "muscle_pain": True}):
        items = prescribing.build(symptoms)
        assert sum(1 for m in items if m["paracetamol"]) <= 1, [m["brand"] for m in items]


def test_never_prescribes_two_of_the_same_class():
    items = prescribing.build({"runny_nose": True, "continuous_sneezing": True,
                               "congestion": True, "itching": True, "high_fever": True})
    seen = set()
    for m in items:
        cls = prescribing.CLASSES.get(m["key"], set())
        assert not (cls & seen), f"{m['brand']} repeats {cls & seen}"
        seen |= cls


def test_treats_what_was_reported():
    ors = [m["key"] for m in prescribing.build({"diarrhoea": True, "vomiting": True})]
    assert "ors" in ors, ors
    # no painkiller when no pain or fever was mentioned
    assert "paracetamol" not in ors and "ibuprofen" not in ors, ors
    assert "paracetamol" in [m["key"] for m in prescribing.build({"headache": True})]


def test_prescription_carries_dose_duration_and_caution():
    block = prescribing.render(prescribing.build({"headache": True}))
    for field in ("Dose:", "Duration:", "Note:"):
        assert field in block, block
    assert "(" in block and "mg" in block, "brand and strength expected"


def test_nothing_is_prescribed_without_symptoms():
    assert prescribing.build({}) == []
    assert prescribing.render([]) == ""


def test_the_note_never_shows_internals_or_prescribes_in_self_care():
    note = ("## What this might mean\nNo chronic risk model was assessed.\n"
            "## What would help\n* Rest.\n* Use Strepsils to soothe your throat.\n"
            "## When to get help straight away\n* Trouble breathing.")
    out = consultation.scrub_note(note)
    assert "risk model" not in out.lower()
    assert "strepsils" not in out.split("## When to get help")[0].lower()
    assert "Rest." in out


# ---------------- everyday-complaint triage ----------------
def test_common_cold_is_recognised_as_a_cold():
    e = sx.engine()
    found = e.extract("I have a bad fever and a runny nose, keep sneezing")
    assert found.get("high_fever") and found.get("runny_nose") and found.get("continuous_sneezing")
    # "bad fever" must not also register the milder variant via the word "fever"
    assert "mild_fever" not in found
    top = e.predict(found)
    assert top[0]["condition"] == "Common Cold"


def test_unreported_symptoms_are_unknown_not_absent():
    """Scoring silence as denial collapses onto whichever condition has fewest
    symptoms, which once ranked a head cold as AIDS."""
    e = sx.engine()
    top = e.predict(e.extract("sore throat, cough and I feel really tired"))
    names = [c["condition"] for c in top]
    # a sore throat with cough and fatigue fits a cold or flu; either leading is
    # right, but nothing alarming may appear and nothing may be flagged urgent
    assert {"Common Cold", "Influenza"} & set(names), names
    assert all(not c["urgent"] for c in top), names
    assert "AIDS" not in names and "Tuberculosis" not in names, names


def test_urgent_flag_needs_a_strong_match():
    e = sx.engine()
    for c in e.predict(e.extract("I have a bad fever and a runny nose, keep sneezing")):
        if c["urgent"]:
            assert c["probability"] >= 0.45 and len(c["matched_symptoms"]) >= 3


def test_everyday_phrasings_are_understood():
    """People write 'backpain', give weight as '65k' and height in feet."""
    from backend.services.extract import extract
    e = sx.engine()
    assert "back_pain" in e.extract("i have backpain")
    assert "back_pain" in e.extract("back-pain since Monday")
    got = extract("i have backpain , i am 21 and 65k , 5.9 Hight")
    assert got["age"] == 21 and got["weight_kg"] == 65
    assert 174 <= got["height_cm"] <= 176          # 5 ft 9 in
    assert extract("I am 5'9 tall")["height_cm"] == pytest.approx(175.3, abs=0.5)


def test_one_generic_symptom_yields_no_shortlist():
    """'headache' left the top five within 0.1% of each other. Naming a winner
    there invents precision, so nothing is offered until something separates."""
    e = sx.engine()
    assert e.predict(e.extract("I have a headache all day")) == []
    assert e.predict(e.extract("my back hurts")) == []


def test_two_close_leaders_well_ahead_of_the_field_still_count():
    """Sprain versus fracture is a real finding, not a dead heat."""
    e = sx.engine()
    names = [c["condition"] for c in e.predict(e.extract(
        "I twisted my ankle, it is swollen and bruised and I can't move it"))]
    assert {"Ankle sprain", "Suspected fracture"} & set(names), names


def test_shortlist_is_capped_to_the_evidence_given():
    e = sx.engine()
    many = e.predict(e.extract("bad fever runny nose sneezing chills sore throat"))
    assert len(many) == 3 and many[0]["confidence"] == "reasonable"
    assert all(c["probability"] < 0.9 for c in many), "nothing should read as certain"


def test_everyday_complaints_and_injuries_are_covered():
    """The public dataset has no entry for a pulled muscle or a sprained ankle."""
    e = sx.engine()
    cases = {
        "I twisted my ankle playing football, it is swollen and bruised": {"Ankle sprain", "Suspected fracture"},
        "tight band around my head and my neck is stiff": {"Tension headache"},
        "my ear hurts and my hearing is muffled": {"Otitis media", "Acute otitis media",
                                                   "Otitis externa (swimmer's ear)", "Tinnitus of unknown cause"},
        "lower back pain after lifting something heavy, hurts when I move":
            {"Mechanical low back pain", "Lumbago", "Muscle strain", "Sprain or strain"},
        "back pain shooting down my leg with pins and needles": {"Sciatica"},
        "toothache and my gum is swollen": {"Toothache", "Tooth abscess", "Broken tooth"},
        "my eye is red and sticky with discharge": {"Conjunctivitis"},
        "I can't sleep and I feel anxious": {"Anxiety", "Insomnia", "Primary insomnia",
                                             "Depression", "Panic disorder"},
    }
    for text, expected in cases.items():
        names = {c["condition"] for c in e.predict(e.extract(text))}
        assert names & expected, f"{text!r} -> {names}, expected one of {expected}"


def test_appendicitis_is_reachable_from_how_people_describe_it():
    """A cannot-miss condition. 'lower right tummy' previously extracted nothing
    abdominal at all, so it could never be reached."""
    e = sx.engine()
    for text in ["severe pain in my lower right tummy with nausea and fever",
                 "bad pain in the lower right side of my belly and I feel sick",
                 "pain that started near my belly button and moved to the lower right"]:
        found = e.extract(text)
        assert found.get("right_lower_abdominal_pain"), f"{text!r} -> {list(found)}"
        names = [c["condition"] for c in e.predict(found)]
        assert "Appendicitis" in names, f"{text!r} -> {names}"


def test_abdominal_pain_survives_ordinary_word_order():
    e = sx.engine()
    for text in ["pain in my tummy", "my stomach hurts", "my belly hurts", "sore tummy"]:
        found = e.extract(text)
        assert found.get("stomach_pain") or found.get("abdominal_pain"), f"{text!r} -> {list(found)}"


def test_a_condition_needs_a_defining_feature_not_just_generic_overlap():
    """A twisted ankle was ranking Minor burn, because both swell."""
    e = sx.engine()
    names = {c["condition"] for c in e.predict(e.extract(
        "I twisted my ankle playing football, swollen and bruised"))}
    assert "Minor burn" not in names, names
    assert names & {"Ankle sprain", "Suspected fracture"}, names
    # a real burn still reaches the shortlist
    burn = {c["condition"] for c in e.predict(e.extract(
        "I spilled hot water and my arm is red and blistered"))}
    assert "Minor burn" in burn, burn


def test_a_lone_clean_match_is_not_reported_as_certainty():
    e = sx.engine()
    top = e.predict(e.extract("toothache and my gum is swollen"))
    assert top[0]["condition"] == "Toothache"
    assert top[0]["probability"] < 0.75, "two symptoms cannot justify near-certainty"
    assert all(c["probability"] >= 0.03 for c in top), "near-zero candidates are noise"


def test_curated_patterns_are_merged_and_attributed():
    import json
    from backend.config import ARTIFACTS
    meta = json.loads((ARTIFACTS / "symptoms_meta.json").read_text())
    assert meta["n_conditions"] >= 65
    by_source = meta["conditions_by_source"]
    assert by_source.get("curated clinical pattern", 0) >= 20
    assert by_source.get("public dataset", 0) >= 40


def test_questions_narrow_the_conditions_still_in_play():
    """A question must relate to a contender, or it reads as random - this once
    asked about chest pain to narrow down back pain. It is asked against the
    candidates, which exist even when the shortlist is too tied to show."""
    e = sx.engine()
    for text in ["I have back pain", "I have a bad fever", "my ear hurts"]:
        found = e.extract(text)
        contenders = e.candidates(found, limit=5)
        nxt = e.next_symptom(found, set())
        assert nxt, text
        assert any(e.matrix.loc[c, nxt] == 1 for c in contenders), \
            f"{text!r}: asks about {nxt!r}, unrelated to {contenders}"


def test_common_things_outrank_rare_ones_on_thin_evidence():
    """A bare fever ranked as AIDS: rare conditions have short symptom lists,
    so a single generic match gave them high coverage."""
    e = sx.engine()
    names = [c["condition"] for c in e.predict(e.extract("I have a bad fever"))]
    assert names, "a fever should suggest something"
    assert "AIDS" not in names and "Tuberculosis" not in names, names


def test_answering_no_to_a_symptom_does_not_end_the_history():
    """'no' answers the question asked. It was ending the consultation and
    producing a report while a question was still on screen."""
    rec = {"sx_back_pain": True}
    h = [{"role": "user", "content": "i have backpain"},
         {"role": "assistant", "content": "Do you also have any pain in your neck?"},
         {"role": "user", "content": "no"}]
    assert consultation.readiness(rec, h)["should_assess"] is False


def test_an_explicit_sign_off_is_recognised_before_the_reply_is_written():
    """Deciding this after generation let the model ask something the report
    then talked over."""
    rec = {"sx_back_pain": True}
    h = [{"role": "user", "content": "i have backpain"},
         {"role": "assistant", "content": "Anything else alongside it?"},
         {"role": "user", "content": "No, nothing else"}]
    assert consultation.is_closing(rec, h) is True
    assert "STAGE - CLOSING" in consultation.build_messages(h, rec)[0]["content"]


def test_pain_is_never_greeted_as_good_news():
    """It replied 'Okay, that's good to know' to someone reporting back pain."""
    f = consultation.soften_acknowledgement
    for text in ["Okay, that's good to know. Can you tell me where you feel it most?",
                 "That's good to know. How long has it been going on?",
                 "Great! Do you have any other symptoms?",
                 "Perfect, and how old are you?"]:
        out = f(text)
        assert not re.match(r"^\s*(okay|ok|alright)?\s*,?\s*(that'?s\s+)?"
                            r"(good|great|perfect|excellent|wonderful|nice)\b", out, re.I), out
        assert out.strip() and out.rstrip()[-1] in ".?!", out
    # a sympathetic opener must survive untouched
    kept = "I understand that must be uncomfortable. Where is the pain?"
    assert f(kept) == kept


def test_the_closing_turn_never_leaves_a_question_hanging():
    """A 4B model does not reliably obey 'do not end with a question'."""
    for text in ["Okay. Could you please tell me your age and sex?",
                 "Alright. How long has it been going on?",
                 "Could you tell me more?"]:
        out = consultation.finalise_closing(text)
        assert not out.rstrip().endswith("?"), out
        assert out.strip(), out


def test_saying_nothing_else_ends_the_history():
    rec = {"sx_back_pain": True}
    h = [{"role": "user", "content": "I have back pain"},
         {"role": "assistant", "content": "Could you describe the pain a little more?"},
         {"role": "user", "content": "No, nothing else"}]
    assert consultation.readiness(rec, h)["should_assess"] is True


def test_questions_target_the_conditions_still_in_play():
    e = sx.engine()
    found = e.extract("bad fever and a runny nose")
    nxt = e.next_symptom(found, set())
    assert nxt and nxt not in found


def test_a_cardiac_presentation_routes_to_the_risk_models():
    """Exertional chest pain went to the symptom track and was asked about a
    cough, because 'crushing pain in my chest' matched no clinical pattern."""
    from backend.services.extract import extract
    for text in ["I get a crushing pain in my chest when I climb stairs",
                 "my chest feels tight when I walk uphill"]:
        got = extract(text)
        assert got.get("chest_pain_type") == 1, f"{text!r} -> {got}"
        rec, _ = consultation.ingest({}, text)
        assert consultation.track(rec, [{"role": "user", "content": text}]) == "chronic"


def test_lab_defined_complaints_ask_for_the_report_early():
    """Queued behind every demographic question, the turn budget ran out and no
    model was ever assessed."""
    rec = {"sx_pedal_edema": True, "age": 47, "sex": "female"}
    h = [{"role": "user", "content": "my ankles keep swelling and my urine is foamy"}]
    field, order = consultation.next_question(rec, h)
    assert order[0] == "kidney", order
    assert field == consultation.LABS_GATE_FIELD, field


def test_a_chronic_history_gets_more_turns_than_a_cold():
    assert consultation.CHRONIC_MAX_TURNS > consultation.MAX_PATIENT_TURNS
    rich = {"sx_thirst": True}
    h = [{"role": "user", "content": "I am always thirsty and passing urine a lot"},
         {"role": "assistant", "content": "Where do you feel it?"},
         {"role": "user", "content": "No, nothing else"}]
    # a sign-off two turns into a chronic history must not end it
    assert consultation.readiness(rich, h)["should_assess"] is False


def test_everyday_complaints_never_route_to_lab_questions():
    text = "I have a bad fever and a runny nose, keep sneezing"
    rec, _ = consultation.ingest({}, text)
    h = [{"role": "user", "content": text}]
    assert consultation.track(rec, h) == "symptoms"
    sys = consultation.build_messages(h, rec)[0]["content"]
    # the instruction issued must be the symptom one, never a clinical-field one
    assert "ASK ABOUT EXACTLY THIS" not in sys
    assert "in plain everyday words" in sys
    directive = sys.split("ASK EXACTLY THIS AND NOTHING ELSE", 1)[-1].lower()
    for banned in ("creatinine", "cholesterol", "bilirubin", "blood pressure reading"):
        assert banned not in directive


def test_chronic_presentation_still_routes_to_the_risk_models():
    text = "crushing chest pain when I climb stairs, my cholesterol was 256"
    rec, _ = consultation.ingest({}, text)
    assert consultation.track(rec, [{"role": "user", "content": text}]) == "chronic"


def test_a_symptom_only_consultation_can_close():
    """It will never have lab coverage, so it must not be gated on it."""
    rec = {f"sx_{s}": True for s in ("high_fever", "runny_nose", "continuous_sneezing", "chills")}
    h = [{"role": "user", "content": "bad fever runny nose sneezing"},
         {"role": "assistant", "content": "Any chills?"}, {"role": "user", "content": "yes"}]
    r = consultation.readiness(rec, h)
    assert r["track"] == "symptoms" and r["can_assess"] and r["should_assess"]


def test_sex_is_not_invented_from_a_contraction():
    from backend.services.extract import extract
    assert "sex" not in extract("Yes, I'm really tired.")
    assert extract("I am male, 58")["sex"] == "male"


# ---------------- self-closing consultation ----------------
RICH = {"age": 58, "sex": "male", "chest_pain_type": 1, "exercise_angina": True,
        "smoker": True, "hypertension_dx": True, "high_cholesterol_dx": True,
        "systolic_bp": 148, "total_cholesterol": 256, "fasting_glucose": 141}

COMPLAINT = "crushing chest pain when I climb stairs"


def _turns(n, q="And what about that?"):
    """A transcript that opens with a real complaint, as every history must."""
    h = [{"role": "user", "content": COMPLAINT}, {"role": "assistant", "content": q}]
    for i in range(n - 1):
        h += [{"role": "user", "content": f"answer {i}"}, {"role": "assistant", "content": q}]
    return h


def test_closes_when_nothing_useful_is_left_to_ask():
    h = [{"role": "user", "content": COMPLAINT},
         {"role": "assistant", "content": "Have you had any blood tests done recently?"},
         {"role": "user", "content": "No I haven't."}]
    assert consultation.next_question(RICH, h)[0] is None
    assert consultation.is_closing(RICH, h) is True


def test_does_not_close_while_the_record_is_thin():
    h = [{"role": "user", "content": "I have a headache"}]
    assert consultation.is_closing({"age": 40}, h) is False


def test_closes_on_turn_cap_once_a_model_is_assessable():
    assert consultation.is_closing(RICH, _turns(consultation.MAX_PATIENT_TURNS)) is True


def test_always_terminates_at_the_hard_ceiling():
    """A patient must never be questioned indefinitely, even with no coverage."""
    assert consultation.is_closing({}, _turns(consultation.HARD_MAX_TURNS)) is True


def test_patient_declining_a_last_word_closes_the_history():
    h = [{"role": "assistant", "content": "Is there anything else bothering you?"},
         {"role": "user", "content": "No, that's all."}]
    assert consultation.readiness(RICH, h)["should_assess"] is True


def test_patient_adding_more_keeps_the_history_open():
    thin = {"age": 58, "sex": "male", "chest_pain_type": 1}
    h = [{"role": "user", "content": "chest pain"},
         {"role": "assistant", "content": "Is there anything else bothering you?"},
         {"role": "user", "content": "Yes, my ankles have been swelling too."}]
    assert consultation.readiness(thin, h)["should_assess"] is False


def test_closing_prompt_contains_no_contradictory_ask_directive():
    msgs = consultation.build_messages(_turns(consultation.MAX_PATIENT_TURNS), RICH)
    sys = msgs[0]["content"]
    assert "STAGE - CLOSING" in sys
    assert "ASK ABOUT EXACTLY THIS" not in sys


def test_unaskable_fields_are_excluded_from_coverage():
    """Stress-test findings are imputed, so they must not gate readiness."""
    assert "max_heart_rate" not in clinical.KEY_FIELDS["heart"]
    assert "max_heart_rate" not in consultation.ASK_ORDER["heart"]


def test_persona_is_an_online_assistant_not_a_clinic():
    """It was greeting people with 'please do come in and have a seat'."""
    sys = consultation.PERSONA.lower()
    # it must not describe itself as working in a physical place...
    assert "online health assistant" in sys
    assert "clinic" not in sys.split("this is an online chat")[0], "still places itself in a clinic"
    # ...and must explicitly forbid the in-person phrasings it was using
    assert "come in" in sys and "have a seat" in sys
    assert "cannot see, touch or examine" in sys
    opening = consultation.build_messages([{"role": "user", "content": "hello"}], {})[0]["content"]
    assert "must not assume they are unwell" in opening.lower()


def test_never_names_a_symptom_before_the_patient_does():
    """The worst failure this system can make: inventing a complaint."""
    h = [{"role": "user", "content": "hello"}]
    assert consultation.has_complaint(h) is False
    sys = consultation.build_messages(h, {})[0]["content"]
    assert "STAGE - OPENING" in sys
    assert "ASK ABOUT EXACTLY THIS" not in sys
    assert "ASK EXACTLY THIS" not in sys


def test_lab_values_are_gated_behind_having_a_report():
    """A person at home has no lab numbers - check before asking for any."""
    self_reported = {"age": 58, "sex": "male", "chest_pain_type": 1, "exercise_angina": True,
                     "smoker": True, "hypertension_dx": True, "high_cholesterol_dx": True,
                     "systolic_bp": 148}
    h = [{"role": "user", "content": COMPLAINT}]
    assert consultation.next_question(self_reported, h)[0] == consultation.LABS_GATE_FIELD

    declined = h + [{"role": "assistant", "content": "Have you had any blood tests recently?"},
                    {"role": "user", "content": "No, I haven't."}]
    assert consultation.labs_status(declined) is False
    assert consultation.next_question(self_reported, declined)[0] is None
    assert consultation.is_closing(self_reported, declined) is True


def test_history_always_moves_on_when_a_question_goes_unanswered():
    """No field may be re-asked forever just because its phrasing is unrecognised."""
    h = [{"role": "user", "content": COMPLAINT}]
    targets = []
    for _ in range(8):
        targets.append(consultation.next_question({}, h)[0])
        h += [{"role": "assistant", "content": "Tell me more."},
              {"role": "user", "content": "I'm not sure, sorry."}]
    assert len(set(targets)) >= 3, f"stuck on {set(targets)}"


def test_model_column_names_are_translated_for_humans():
    assert clinical.human("trestbps") == "blood pressure"
    assert clinical.human("GenHlth") == "how you rate your health"
    assert clinical.human("exang") == "chest pain on exertion"
    assert "=" not in clinical.human("Sgpt")


# ---------------- risk engine ----------------
def test_engine_loads_all_five_models():
    assert set(engine().models) == {"diabetes", "heart", "kidney", "liver", "breast"}


def test_low_coverage_is_refused_not_guessed():
    out = engine().assess({"age": 40})
    statuses = {r["disease"]: r["status"] for r in out["results"]}
    assert statuses["kidney"] == "insufficient_data"
    assert all(r["probability"] is None for r in out["results"] if r["status"] != "ok")


def test_probabilities_and_provenance():
    rec = {"age": 47, "sex": "female", "serum_creatinine": 3.8, "blood_urea": 92,
           "haemoglobin": 9.2, "specific_gravity": 1.010, "urine_albumin": 4,
           "hypertension_dx": True}
    out = engine().assess(rec)
    ck = next(r for r in out["results"] if r["disease"] == "kidney")
    assert ck["status"] == "ok"
    assert 0.0 <= ck["probability"] <= 1.0
    assert ck["risk_band"] in ("Low", "Moderate", "High", "Very High")
    assert all("observed" in c for c in ck["contributions"])
    assert any(c["observed"] for c in ck["contributions"])


def test_imputed_features_are_flagged():
    """Heart model run without ca/thal must mark them imputed, never observed."""
    rec = {"age": 58, "sex": "male", "chest_pain_type": 1, "systolic_bp": 148,
           "total_cholesterol": 256, "max_heart_rate": 122, "exercise_angina": True}
    hr = next(r for r in engine().assess(rec)["results"] if r["disease"] == "heart")
    by_name = {c["feature"]: c["observed"] for c in hr["contributions"]}
    for absent in ("ca", "thal"):
        if absent in by_name:
            assert by_name[absent] is False


# ---------------- reasoning stripper ----------------
def test_strip_reasoning_removes_leaked_chain_of_thought():
    leaked = ("The patient said they have chest pain.\n\n"
              "Wait, red flags are critical here.\n\n"
              "I'm sorry to hear that. Does the pain ease when you rest?")
    out = strip_reasoning(leaked)
    assert out.startswith("I'm sorry")
    assert "Wait," not in out


def test_wrapping_quotes_removed_but_inner_quotes_kept():
    assert strip_reasoning('"Does the pain ease when you rest?"') == "Does the pain ease when you rest?"
    assert strip_reasoning("\u201cSmart quoted reply.\u201d") == "Smart quoted reply."
    # a quote used mid-sentence must survive
    assert strip_reasoning('He said "ouch" and left. Does that help?').startswith("He said")
    # ambiguous multi-quote text is left alone rather than mangled
    assert strip_reasoning('"He said "ouch" and left."').startswith('"')


def test_strip_reasoning_never_returns_empty():
    assert strip_reasoning("The patient said something.").strip()


# ---------------- API ----------------
def test_health_endpoint():
    d = client.get("/api/health").json()
    assert d["app"] == "IHACDP" and d["offline"] is True
    assert len(d["models_loaded"]) == 5


def test_diseases_endpoint_exposes_leaderboards():
    ds = client.get("/api/diseases").json()
    assert len(ds) == 5
    for d in ds:
        assert 0.5 <= d["metrics"]["test_auc"] <= 1.0
        assert len(d["leaderboard"]) == 3
        assert d["chosen_algorithm"] in {l["algorithm"] for l in d["leaderboard"]}


def test_assess_endpoint_rejects_empty_record():
    assert client.post("/api/assess", json={"record": {}}).status_code == 400


def test_assess_endpoint_returns_ranked_results():
    r = client.post("/api/assess", json={"record": {
        "age": 58, "sex": "male", "systolic_bp": 148, "total_cholesterol": 256,
        "chest_pain_type": 1, "max_heart_rate": 122, "exercise_angina": True}})
    assert r.status_code == 200
    probs = [x["probability"] for x in r.json()["results"] if x["probability"] is not None]
    assert probs == sorted(probs, reverse=True)


def test_pages_served():
    for p in ("/", "/about", "/consult", "/models"):
        assert client.get(p).status_code == 200
