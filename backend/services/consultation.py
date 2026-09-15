"""Consultation orchestration: physician persona, history taking, grounded reporting."""
from __future__ import annotations
import re
from typing import Any, AsyncIterator

from backend.services import clinical, llm, prescribing, symptoms as sx
from backend.services.extract import extract, extract_in_context, question_topic
from backend.services.predictor import engine

DOCTOR = "IHACDP"

PERSONA = f"""You are IHACDP, an online health assistant. The patient is typing to you from wherever
they are - a phone, a laptop - and you have never met them.
If you ever name yourself you are simply "IHACDP", but you do not announce your name
in replies. Never begin a sentence with "IHACDP" or prefix a question with it.

HOW YOU SPEAK
- Warm, calm, unhurried. Talk like a kind person, never like a form.
- Exactly ONE question per reply. Never present a list.
- Briefly acknowledge what they just told you before moving on.
- Match the acknowledgement to what they actually said. Someone describing pain or
  illness has told you something unwelcome: never call it good, great, perfect,
  excellent, wonderful or nice, and never say "that's good to know". Use "thank you",
  "I see", "I understand", or a short word of sympathy instead.
- Keep replies to 2-3 short sentences. No bullet points, no headings.
- Everyday words only ("blood sugar", not "serum glucose"; "kidney test", not "creatinine"
  unless they raised it first).
- If they sound worried, reassure them before anything else.
- If they do not know an answer, say that is completely fine and move on. Never press,
  never repeat a question they could not answer.

THIS IS AN ONLINE CHAT, NOT A CLINIC
- Never invite them in, offer them a seat, or refer to a room, waiting area or appointment.
- Never say "come in", "have a seat", "let me take a look", "step this way" or anything
  that implies you are in the same place as them.
- You cannot see, touch or examine them. You can ask them to look at or press something
  themselves, but never describe yourself doing it.
- Do not assume they are unwell before they have said so. A greeting is just a greeting.

WHAT AN ORDINARY PERSON ACTUALLY KNOWS
- They know their age, sex, rough height and weight, whether they smoke or drink, how
  active they are, how they have been feeling, and what a doctor has told them before.
- They do NOT have medical equipment at home. Most will not know their cholesterol,
  blood sugar or kidney figures unless they are holding a test report.
- Only ask for a measured value after they have confirmed they have results with them.

WHAT YOU MUST NOT DO
- Never state a diagnosis during history taking.
- Never invent or assume a test result the patient has not given you.
- Never recommend a specific drug or dose in conversation.
- If the patient reports a red flag (crushing chest pain now, breathlessness at rest,
  one-sided weakness, slurred speech, fainting, coughing blood), stop the history and tell
  them plainly to seek emergency care immediately.

ORDER OF ENQUIRY (skip anything already known)
1. Main complaint and how long it has been going on
2. Associated symptoms and red flags
3. Age, sex, height, weight
4. Smoking, alcohol, exercise
5. Known conditions: blood pressure, diabetes, cholesterol, heart, stroke, kidney, liver
6. Any recent blood or urine test results they can read out

STYLE ILLUSTRATION (tone and length only - these are NOT things this patient said,
and you must never refer to their content):
  Two weeks of that sounds wearing. Can you tell me where you feel it most?
  Thanks, that helps. Has anything made it noticeably worse?

Write your reply as plain speech. Never wrap it in quotation marks.

NEVER invent a symptom the patient has not mentioned. Only ever refer to what is in
this conversation or on the record below.
"""



# Presenting-complaint cues -> which disease line of enquiry to pursue first.
FOCUS_CUES: dict[str, list[str]] = {
    "heart": ["chest pain", "chest tightness", "chest discomfort", "crushing", "angina", "palpitation",
              "breathless", "short of breath", "exertion", "climb stairs", "climbing stairs",
              "left arm", "jaw pain", "heart", "sweating with pain"],
    "kidney": ["swelling", "ankle", "ankles", "oedema", "edema", "puffy", "urine", "urinating", "urination",
               "foamy", "frothy", "flank", "back pain", "creatinine", "dialysis", "kidney", "passing water"],
    "liver": ["jaundice", "yellow", "yellowing", "whites of my eyes", "right rib", "upper abdomen",
              "abdominal pain", "nausea", "vomiting", "alcohol", "drink", "dark urine", "itching",
              "liver", "bloating", "belly"],
    "diabetes": ["thirst", "thirsty", "drinking a lot", "passing urine often", "weight loss", "blurred vision",
                 "sugar", "glucose", "tingling", "numbness", "slow healing", "wound", "fatigue", "tired",
                 "diabetes", "hungry"],
}

# Order in which to seek each disease's fields once that line of enquiry is active.
ASK_ORDER: dict[str, list[str]] = {
    "heart": ["chest_pain_type", "exercise_angina", "age", "sex", "smoker",
              "hypertension_dx", "high_cholesterol_dx", "systolic_bp",
              "total_cholesterol", "fasting_glucose"],
    "kidney": ["pedal_edema", "age", "sex", "hypertension_dx", "diabetes_dx", "appetite",
               "anaemia", "serum_creatinine", "blood_urea", "haemoglobin", "urine_albumin",
               "specific_gravity"],
    "liver": ["heavy_alcohol", "age", "sex", "appetite", "total_bilirubin",
              "direct_bilirubin", "alt_sgpt", "ast_sgot", "albumin"],
    "diabetes": ["bmi", "fasting_glucose", "hypertension_dx", "high_cholesterol_dx",
                 "general_health", "physical_activity", "difficulty_walking"],
}


def focus(transcript: list[dict]) -> list[str]:
    """Rank lines of enquiry by what the patient actually complained about."""
    said = " ".join(m["content"].lower() for m in transcript if m["role"] == "user")
    scores = {d: sum(1 for c in cues if c in said) for d, cues in FOCUS_CUES.items()}
    ranked = [d for d, n in sorted(scores.items(), key=lambda kv: -kv[1]) if n > 0]
    return ranked or ["heart", "diabetes", "kidney", "liver"]


SX = "sx_"


def symptom_state(rec: dict) -> dict[str, bool]:
    return {k[len(SX):]: bool(v) for k, v in rec.items() if k.startswith(SX)}


def track(rec: dict, transcript: list[dict]) -> str:
    """Which line of enquiry this consultation belongs to.

    Everyday complaints (fever, colds, aches) must never be routed into the
    chronic-disease questioning, which asks for lab values no one has at home.
    """
    said = " ".join(m["content"].lower() for m in transcript if m["role"] == "user")
    chronic_hits = sum(1 for cues in FOCUS_CUES.values() for c in cues if c in said)
    reported = [v for v in symptom_state(rec).values() if v]
    # an explicit chronic signal: known diagnoses or actual measured values
    hard_chronic = any(rec.get(k) is not None for k in
                       ("serum_creatinine", "total_bilirubin", "alt_sgpt", "total_cholesterol",
                        "fasting_glucose", "hypertension_dx", "diabetes_dx", "high_cholesterol_dx",
                        "exercise_angina", "chest_pain_type"))
    if hard_chronic and chronic_hits >= 1:
        return "chronic"
    # A textbook chronic presentation should not wait for a lab value to be
    # offered: exertional chest pain routed to the symptom track and was asked
    # about a cough.
    if chronic_hits >= 2 or rec.get("chest_pain_type") or rec.get("exercise_angina"):
        return "chronic"
    if reported:
        return "symptoms"
    return "chronic" if chronic_hits else "symptoms"


def has_complaint(transcript: list[dict]) -> bool:
    """Has the patient actually described a symptom yet?

    Until they have, no line of enquiry may be assumed. Naming a symptom the
    patient never mentioned is the worst failure this system can make.
    """
    said = " ".join(m["content"].lower() for m in transcript if m["role"] == "user")
    if not said.strip():
        return False
    if any(c in said for cues in FOCUS_CUES.values() for c in cues):
        return True
    try:
        return any(sx.engine().extract(said).values())
    except Exception:
        return False


# Values a person can answer from memory, versus values that only exist on a
# printed lab report. We never ask for the second kind without checking first.
REPORT_ONLY = {
    "total_cholesterol", "fasting_glucose", "random_glucose", "serum_creatinine",
    "blood_urea", "sodium", "potassium", "haemoglobin", "packed_cell_volume",
    "wbc_count", "rbc_count", "specific_gravity", "urine_albumin", "urine_sugar",
    "total_bilirubin", "direct_bilirubin", "alkaline_phosphatase", "alt_sgpt",
    "ast_sgot", "total_protein", "albumin", "ag_ratio", "rbc_urine", "pus_cells",
}

LABS_GATE = re.compile(
    r"(blood test|lab report|laboratory report|test result|blood work|blood report|"
    r"results to hand|report to hand|recent tests|had any tests)", re.I)


def labs_status(transcript: list[dict]) -> bool | None:
    """True if the patient has a report, False if they declined, None if unasked."""
    msgs = list(transcript)
    for i, m in enumerate(msgs):
        if m["role"] == "assistant" and LABS_GATE.search(m["content"]):
            for nxt in msgs[i + 1:]:
                if nxt["role"] == "user":
                    return not bool(DECLINE.match(nxt["content"].strip()))
    return None


def _asked_already(field: str, transcript: list[dict]) -> bool:
    """True if any earlier clinician turn already covered this field.

    Matching is by question topic rather than keyword overlap, so a question
    phrased any way at all still counts and the history keeps moving.
    """
    for m in transcript:
        if m["role"] == "assistant" and question_topic(m["content"]) == field:
            return True
    return False


LABS_GATE_FIELD = "__labs__"
MAX_ATTEMPTS_PER_FIELD = 2


def _stall(rec: dict, transcript: list[dict]) -> int:
    """How many pending fields to skip because the history has stopped moving.

    Topic detection cannot recognise every phrasing, so a field with no matching
    cue would otherwise be re-asked forever. Every couple of unproductive turns
    we abandon the current field and move on, exactly as a clinician would.
    """
    asked = sum(1 for m in transcript if m["role"] == "assistant")
    captured = sum(1 for v in rec.values() if v is not None)
    return max(0, asked - captured - 1) // MAX_ATTEMPTS_PER_FIELD


def next_question(rec: dict, transcript: list[dict]) -> tuple[str | None, list[str]]:
    """The single most valuable unasked field, plus the active lines of enquiry.

    Anything a person can answer from memory is sought first. Lab values are
    only pursued after confirming the patient actually has a report in front
    of them, and are abandoned entirely if they do not.
    """
    order = focus(transcript)
    have_labs = labs_status(transcript)
    # These are diagnosed on bloods. Asking for the report only after every
    # self-reported field is exhausted meant the turn budget ran out first and
    # no model was ever assessed.
    LAB_DEFINED = {"kidney", "liver"}
    lab_led = order and order[0] in LAB_DEFINED

    skip = _stall(rec, transcript)

    def pending(pool):
        seen = 0
        for disease in order:
            for f in ASK_ORDER.get(disease, []):
                if f in pool and rec.get(f) is None and not _asked_already(f, transcript):
                    if seen < skip:
                        seen += 1
                        continue
                    return f
        return None

    self_reported = pending({f for d in order for f in ASK_ORDER.get(d, [])} - REPORT_ONLY)
    known = sum(1 for d in order for f in ASK_ORDER.get(d, [])
                if f not in REPORT_ONLY and rec.get(f) is not None)
    if lab_led and have_labs is None and known >= 2:
        return LABS_GATE_FIELD, order        # ask for the report now
    if self_reported:
        return self_reported, order

    lab_pending = pending(REPORT_ONLY)
    if lab_pending:
        if have_labs is None:
            return LABS_GATE_FIELD, order      # check before asking for numbers
        if have_labs:
            return lab_pending, order
    return None, order


def _known_block(rec: dict) -> str:
    have = {k: v for k, v in rec.items() if v is not None and not k.startswith("wdbc_")}
    if not have:
        return "Nothing recorded yet - open the consultation."
    return ", ".join(f"{k}={v}" for k, v in list(have.items())[:40])


# One message can easily name three symptoms, and closing on it skips the
# refining questions entirely. Ending early is handled by the patient saying
# they have nothing to add, not by a low count.
MIN_SYMPTOMS_TO_CLOSE = 4


def is_closing(rec: dict, transcript: list[dict]) -> bool:
    """Single source of truth: has this consultation gathered what it needs?

    Both the prompt and the readiness signal derive from this, so the model is
    never told to close and to keep asking in the same breath.
    """
    turns = sum(1 for m in transcript if m["role"] == "user")
    if turns >= HARD_MAX_TURNS:
        return True          # never interrogate a patient indefinitely

    # A sign-off has to be recognised HERE, before the reply is written. Deciding
    # it afterwards let the model ask a question that the report then talked over.
    users = [m["content"] for m in transcript if m["role"] == "user"]
    if users and EXPLICIT_DONE.match(users[-1].strip()):
        # On a chronic presentation "nothing else" answers "any other symptoms?"
        # - it does not mean stop asking their age. Closing on it after two turns
        # gave a diabetes presentation a diagnosis of heat exhaustion.
        if track(rec, transcript) == "chronic":
            if clinical_ready(rec) or turns >= CHRONIC_MAX_TURNS:
                return True
        elif any(symptom_state(rec).values()):
            return True

    if track(rec, transcript) == "symptoms":
        confirmed = symptom_state(rec)
        positives = sum(1 for v in confirmed.values() if v)
        if positives >= MIN_SYMPTOMS_TO_CLOSE:
            return True
        try:
            if _symptom_block(rec, transcript) is None and positives:
                return True   # nothing left that would separate the candidates
        except Exception:
            pass
        return turns >= MAX_PATIENT_TURNS and positives > 0

    field, _order = next_question(rec, transcript)
    if field is None:
        return True
    cap = CHRONIC_MAX_TURNS if track(rec, transcript) == "chronic" else MAX_PATIENT_TURNS
    return turns >= cap and clinical_ready(rec)


def _symptom_block(rec: dict, transcript: list[dict]) -> str | None:
    """Ask about the symptom that best separates the conditions still in play."""
    eng = sx.engine()
    confirmed = symptom_state(rec)
    asked = set()
    for m in transcript:
        if m["role"] == "assistant":
            asked |= set(eng.extract(m["content"]))
    nxt = eng.next_symptom(confirmed, asked)
    if not nxt:
        return None
    return ("ASK EXACTLY THIS AND NOTHING ELSE, in plain everyday words: whether they also have "
            f"{eng.label(nxt)}. Do not ask about blood tests, blood pressure numbers or lifestyle.")


def _needed_block(rec: dict, transcript: list[dict]) -> str:
    field, order = next_question(rec, transcript)
    if field == LABS_GATE_FIELD:
        return ("ASK EXACTLY THIS AND NOTHING ELSE: whether they have had any blood tests done "
                "recently and have the results with them. Do not name any individual test or value. "
                "Make clear it is completely fine if they do not.")
    ask = clinical.FIELDS.get(field, {}).get("ask", field)
    unit = clinical.FIELDS.get(field, {}).get("unit", "")
    hint = f" (expressed in {unit})" if unit and "=" not in unit and len(unit) < 10 else ""
    return (f"Lines of enquiry, most relevant first: {', '.join(order)}.\n"
            f"ASK ABOUT EXACTLY THIS AND NOTHING ELSE: {ask}{hint}.")


def build_messages(history: list[dict], rec: dict) -> list[dict]:
    turns = sum(1 for m in history if m["role"] == "user")
    if not has_complaint(history):
        stage = ("STAGE - OPENING. The patient has NOT yet told you what is wrong. You know nothing "
                 "about their symptoms and you must not assume they are unwell. Greet them warmly "
                 "and ask what they would like help with. You must NOT name, guess at or refer to "
                 "any symptom, condition or body part - not chest pain, not anything - and must not "
                 "say you are sorry they are unwell. Ask nothing about tests, numbers or lifestyle. "
                 "Two short sentences, written as an online message.")
        return [{"role": "system", "content": PERSONA + f"\n\n{stage}"}] + history[-14:]
    if track(rec, history) == "symptoms" and not is_closing(rec, history):
        block = _symptom_block(rec, history)
        if block:
            stage = ("STAGE - TARGETED ENQUIRY. You are working out an everyday complaint. Ask only "
                     "about symptoms, one per reply, in ordinary words. Never ask for a measured "
                     "value, a lab result or a blood pressure reading.")
            sysmsg = (PERSONA + f"\n\n{stage}"
                      + f"\n\nALREADY ON RECORD (never ask again): {_known_block(rec)}"
                      + f"\n\n{block}")
            return [{"role": "system", "content": sysmsg}] + history[-14:]
    if is_closing(rec, history):
        stage = ("STAGE - CLOSING. You have everything you need. Do NOT seek any further detail and do "
                 "NOT end with a question. Warmly tell the patient you have enough to go on and that you "
                 "are reviewing their results now. Two sentences.")
        sys = PERSONA + f"\n\n{stage}\n\nALREADY ON RECORD: {_known_block(rec)}"
        return [{"role": "system", "content": sys}] + history[-14:]
    if turns <= 1:
        stage = ("STAGE - UNDERSTAND THE COMPLAINT FIRST. You do not yet know what is wrong. Do not ask "
                 "about blood tests, blood pressure or lifestyle yet. Acknowledge what they said and ask "
                 "one open question that clarifies the main complaint itself - where it is, what it feels "
                 "like, what brings it on, or how it has changed.")
    else:
        stage = ("STAGE - TARGETED ENQUIRY. You now understand the complaint. Ask only for information that "
                 "matters for THIS presentation, one item per reply, and never revisit anything already asked.")
    sys = (PERSONA
           + f"\n\n{stage}"
           + f"\n\nALREADY ON RECORD (never ask for these again): {_known_block(rec)}"
           + f"\n\n{_needed_block(rec, history)}")
    sys += ("\n\nOUTPUT FORMAT - CRITICAL: Reply with ONLY the words you say to the patient. "
            "Never write your reasoning, never write analysis, never write headings or numbered plans, "
            "never explain what you are about to do. Start directly with what you say aloud. "
            "Maximum 3 sentences.")
    return [{"role": "system", "content": sys}] + history[-14:]


# The clinician turn that closes a history, and the patient replies that accept it.
ANYTHING_ELSE = re.compile(
    r"(anything else|anything more|anything further|something else|else you'?d like|"
    r"else to add|else bothering|else troubling|else worrying|other concerns|"
    r"other symptoms?|any other|alongside (that|the|this)|as well as|"
    r"before i review|before we finish)", re.I)
DECLINE = re.compile(
    r"^\W*(no|nope|nah|nothing|none|that'?s (it|all|everything)|that is (it|all)|"
    r"nothing else|nothing more|i'?m done|i am done|all good|no thanks|no that'?s all|"
    r"not really|i think that'?s|that should be)\b", re.I)

# "no" answers whatever was just asked. Only an unmistakable sign-off ends the
# consultation on its own - otherwise declining one symptom closed the history
# and produced a report while a question was still on screen.
EXPLICIT_DONE = re.compile(
    r"^\W*(no[,.\s]+)?(nothing (else|more|further)|that'?s (it|all|everything)|"
    r"that is (it|all|everything)|i'?m done|i am done|no thanks|all good|"
    r"nope[,.\s]+nothing)\b", re.I)


def readiness(rec: dict, history: list[dict] | None = None) -> dict[str, Any]:
    history = history or []
    cov = {d: round(clinical.coverage(rec, d), 2) for d in ("diabetes", "heart", "kidney", "liver")}
    ready = [d for d, c in cov.items() if c >= 0.45]
    positives = sum(1 for v in symptom_state(rec).values() if v)
    on_symptoms = track(rec, history) == "symptoms"
    # Assessable on symptoms alone, whichever track this is. Tying this to the
    # chronic track meant a patient describing thirst and fatigue - who never
    # reaches lab coverage - could never close, and the consultation looped.
    can = bool(ready) or positives >= 1
    field, _order = next_question(rec, history) if history else (None, [])
    return {
        "coverage": cov, "ready": ready if ready else (["common conditions"] if can else []),
        "can_assess": can,
        "track": "symptoms" if on_symptoms else "chronic",
        "symptoms_reported": positives,
        "exhausted": can and field is None,
        "should_assess": can and _should_assess(rec, history, field),
    }


MAX_PATIENT_TURNS = 8
CHRONIC_MAX_TURNS = 12          # demographics, history and bloods take longer
HARD_MAX_TURNS = 16


def clinical_ready(rec: dict) -> bool:
    return any(clinical.coverage(rec, d) >= 0.45 for d in ("diabetes", "heart", "kidney", "liver"))


def _should_assess(rec: dict, history: list[dict], next_field: str | None) -> bool:
    """IHACDP decides for itself that the history is complete.

    Triggers: nothing clinically useful is left to ask, the consultation has run
    its natural length, or it offered the patient a last word and they declined.
    """
    if not history:
        return False
    if is_closing(rec, history):
        return True
    asst = [m["content"] for m in history if m["role"] == "assistant"]
    users = [m["content"] for m in history if m["role"] == "user"]
    if not asst or not users:
        return False
    # Sign-offs are handled by is_closing, which is track-aware. A second copy
    # of that rule here ignored the track and closed chronic consultations after
    # two turns with no model assessed.
    return bool(ANYTHING_ELSE.search(asst[-1]) and DECLINE.match(users[-1].strip()))


def ingest(rec: dict, text: str, last_question: str = "") -> tuple[dict, dict]:
    """Merge newly extracted facts into the record. Returns (record, newly_found).

    `last_question` is the clinician turn this text answers; a bare "Yes" or
    "3.8" only has meaning relative to what was asked.
    """
    found = extract(text)
    if last_question:
        for k, v in extract_in_context(last_question, text).items():
            found.setdefault(k, v)
    try:
        eng = sx.engine()
        for name, present in eng.extract(text).items():
            found.setdefault(SX + name, present)
        # a bare yes/no answering a symptom question
        # only treat a bare yes/no as answering the question asked; if the reply
        # names a symptom of its own, that is what the patient is telling us
        volunteered = eng.extract(text)
        if last_question and not volunteered:
            asked_sx = eng.extract(last_question)
            if len(asked_sx) == 1:
                only = next(iter(asked_sx))
                t = text.strip()
                if DECLINE.match(t):
                    found.setdefault(SX + only, False)
                elif re.match(r"^\W*(yes|yeah|yep|a bit|a little|sometimes|slightly)\b", t, re.I):
                    found.setdefault(SX + only, True)
    except Exception:
        pass
    merged = dict(rec)
    new = {}
    for k, v in found.items():
        if merged.get(k) is None:
            merged[k] = v
            new[k] = v
    return clinical.derive(merged), new


_CLOSING_LINE = "Let me look at what you've told me now."

# "That's good to know" in reply to someone's pain. A small model reaches for
# these fillers regardless of instruction, so the common ones are rewritten.
_TONE_DEAF = re.compile(
    r"^\s*(?:(?:okay|ok|alright|right|well)\s*[,.]?\s*)?"
    r"(?:that'?s\s+|that\s+is\s+)?"
    r"(good to know|good|great|perfect|excellent|wonderful|lovely|nice|fantastic|brilliant)"
    r"\s*[,.!]?\s*", re.I)


def soften_acknowledgement(text: str) -> str:
    """Strip a cheerful opener from a reply to someone reporting symptoms."""
    m = _TONE_DEAF.match(text or "")
    if not m:
        return text
    rest = text[m.end():].lstrip()
    if not rest:
        return "Thank you for telling me."
    return "Thank you. " + rest[0].upper() + rest[1:] if rest[0].islower() else "Thank you. " + rest


_INSTRUCTION_ECHO = re.compile(
    r"^\s*(number them\b|always give the brand|for a medicine that needs a prescription|"
    r"two or three medicines|one medicine per entry|each on four lines|"
    r"never prescribe the same drug|every medicine must treat|"
    r"write only the note|lay each medicine out)", re.I)


_MODEL_TALK = re.compile(
    r"(chronic risk model|risk model|statistical model|not assessed|was assessed|"
    r"insufficient data|no model)", re.I)


def scrub_note(text: str, rec: dict | None = None) -> str:
    """Remove what the patient should never see.

    Two things the model says despite instruction: that a chronic risk model
    was not assessed, and the names of medicines in the self-care section -
    sometimes medicines that are not even on the prescription.
    """
    brands = [m["brand"].split()[0] for m in prescribing.FORMULARY.values()]
    brand_re = re.compile(r"\b(" + "|".join(re.escape(b) for b in brands) + r")\b", re.I)

    out, in_help = [], False
    for ln in text.splitlines():
        if ln.startswith("## "):
            in_help = ln.strip().lower().startswith("## what would help")
        if _MODEL_TALK.search(ln) and not ln.startswith("## "):
            continue
        if in_help and brand_re.search(ln):
            continue            # self-care advice must not prescribe
        out.append(ln)
    return re.sub(r"\n{3,}", "\n\n", "\n".join(out)).strip()


def strip_instruction_echo(text: str) -> str:
    """Remove any of the formatting rules the model copied into the note."""
    kept = [ln for ln in text.splitlines() if not _INSTRUCTION_ECHO.match(ln)]
    out = "\n".join(kept)
    return re.sub(r"\n{3,}", "\n\n", out).strip()


def finalise_closing(text: str) -> str:
    """Guarantee the closing turn leaves no question hanging.

    A 4B model does not reliably obey "do not end with a question", and when it
    slips the report talks straight over an unanswered one. Enforced here rather
    than hoped for in the prompt.
    """
    sentences = [x for x in re.split(r"(?<=[.!?])\s+", text.strip()) if x.strip()]
    kept = [x for x in sentences if not x.rstrip().endswith("?")]
    out = " ".join(kept).strip()
    if not out:
        out = "Thank you, that gives me what I need."
    if not re.search(r"\b(look|review|go(ing)? (over|through)|results|see what)\b", out, re.I):
        out = f"{out} {_CLOSING_LINE}".strip()
    return out


async def reply_stream(history: list[dict], rec: dict) -> AsyncIterator[str]:
    async for tok in llm.chat_stream(build_messages(history, rec), temperature=0.65, max_tokens=260):
        yield tok


# ---------------- grounded clinical report -----------------------------------
# Brands people actually buy in an Indian pharmacy, each pinned to its generic.
# Supplied explicitly so the model cannot invent a brand or, far worse, pair a
# real brand with the wrong drug.
INDIAN_FORMULARY = """AVAILABLE OVER THE COUNTER IN INDIA (brand = generic):
  Pain and fever   Dolo 650 = paracetamol 650mg · Crocin 500 = paracetamol 500mg
                   Calpol 500 = paracetamol 500mg
  Pain and swelling Brufen 400 = ibuprofen 400mg · Combiflam = ibuprofen 400mg +
                   paracetamol 325mg
  Muscle and joint Volini gel = diclofenac topical · Moov cream = topical analgesic
                   Zandu Balm / Amrutanjan = topical menthol balm
  Acidity          Digene = antacid chewable · Gelusil = antacid · Eno = antacid sachet
  Allergy, itching Cetzine 10 = cetirizine 10mg · Allegra 120 = fexofenadine 120mg
                   Avil 25 = pheniramine 25mg
  Blocked nose     Otrivin nasal drops = xylometazoline · Sinarest = paracetamol +
                   chlorpheniramine + phenylephrine
  Cough, throat    Strepsils lozenges · Honitus syrup · Vicks VapoRub (topical)
  Loose motions    Electral / ORS-L = oral rehydration salts
  Constipation     Isabgol (psyllium husk) · Duphalac = lactulose syrup
  Cuts, grazes     Betadine = povidone-iodine antiseptic
  Vitamins         Zincovit · Becosules = B-complex
NEEDS A DOCTOR'S PRESCRIPTION IN INDIA (name it, never dose it):
  all antibiotics (Augmentin, Azithral, Taxim-O), Pan-D / Omez (PPI),
  Domstal (domperidone), Stugeron (cinnarizine), Zerodol / Aceclofenac,
  steroids, tramadol, and anything for blood pressure, diabetes or cholesterol."""

REPORT_RULES = """You are writing directly to the patient, in plain English.

TONE
- Calm, kind and steady. Never alarming. Never dramatic.
- Speak to them as "you". Short sentences. No medical jargon.
- If you must use a medical word, explain it in the same breath.
- Keep the whole note under about 220 words. Short beats thorough here.
- Use short bullets, not paragraphs. A worried person should be able to read
  this in under a minute.

ABSOLUTE RULES
- Use ONLY the observed values listed. If something is not listed it was NOT
  measured, so do not mention it at all. Never invent a number.
- Write every finding in the plain words given to you. NEVER print a database
  field name such as GenHlth, trestbps, exang, DiffWalk, HighBP or sg, and never
  write "field=value". Say "your blood pressure was 148", not "trestbps=148".
- Give the risk as a plain percentage and say clearly it is an estimate from a
  statistical model, not a diagnosis.
- Do NOT write a prescription or name any medicine, brand or dose. The
  prescription is produced separately and inserted for you. In "What would help"
  stick to self-care - rest, fluids, ice, positioning - and never mention tablets.
- For anything that legally needs a prescription - antibiotics, steroids, strong
  painkillers, anything for blood pressure, diabetes or cholesterol - name the
  medicine and write "prescription required" instead of a dose. You must not put
  a dose on a medicine the patient cannot buy themselves.
- Always give the adult dose, and add the maximum in 24 hours for any painkiller.
- Add one short caution per medicine where it matters (take with food, avoid if
  asthmatic, avoid with other paracetamol products, do not exceed the stated days).

WRITE EXACTLY THESE FOUR SECTIONS AS MARKDOWN (## headings), nothing else:
## What you told me
Two or three short bullets of what they described.
## What this might mean
If likely common conditions are listed, lead with those: name the condition in
plain words, give the match as a percentage, and say which of their symptoms
point to it. Then cover any chronic risk model that was assessed, with its
percentage. Say plainly that these are estimates from a symptom match or a
statistical model, not a diagnosis. If something is flagged as needing urgent
care, say so first, clearly and calmly.
## What would help
Short, practical bullets that fit THIS complaint. Self-care first, then any test
worth asking for.
## When to get help straight away
Three short bullets of red flags for THIS complaint and THIS body part. Derive
them from what the patient actually described - do not reuse a stock list. An
ankle injury warrants signs of a break or lost circulation; a headache warrants
sudden severe onset or neck stiffness; a cough warrants breathing difficulty.
Never list a red flag for a body part the patient did not mention.

No preamble, no sign-off, no extra sections.

Write ONLY the note itself. Never repeat, quote or summarise these instructions -
no sentences about numbering, about brackets, or about what to do for a
prescription medicine. The patient must see medicines, never the rules."""


PRESCRIPTION_HEADING = "## Prescription"


def prescription_block(rec: dict) -> str:
    """The prescription, chosen and dosed in code rather than by the model."""
    conditions = symptom_assessment(rec)
    if prescribing.needs_assessment(conditions):
        return prescribing.URGENT_ADVICE
    return prescribing.render(prescribing.build(symptom_state(rec), conditions))


def splice_prescription(note: str, rec: dict) -> str:
    """Insert the prescription ahead of the safety-netting section."""
    block = prescription_block(rec)
    if not block:
        return note
    section = f"{PRESCRIPTION_HEADING}\n{block}\n"
    marker = "## When to get help"
    i = note.find(marker)
    if i == -1:
        return f"{note.rstrip()}\n\n{section}"
    return f"{note[:i].rstrip()}\n\n{section}\n{note[i:]}"


def symptom_assessment(rec: dict) -> list[dict]:
    try:
        return sx.engine().predict(symptom_state(rec), top=3)
    except Exception:
        return []


def _evidence(rec: dict, assessment: dict) -> str:
    obs = {k: v for k, v in rec.items() if v is not None and not k.startswith("wdbc_")}
    lines = ["OBSERVED PATIENT DATA:"]
    for k, v in obs.items():
        unit = clinical.FIELDS.get(k, {}).get("unit", "")
        lines.append(f"  - {clinical.human(k)}: {v}{(' ' + unit) if unit and '=' not in unit else ''}")

    confirmed = symptom_state(rec)
    if confirmed:
        eng = sx.engine()
        yes = [eng.label(k) for k, v in confirmed.items() if v]
        no = [eng.label(k) for k, v in confirmed.items() if not v]
        if yes:
            lines.append("\nSYMPTOMS REPORTED: " + ", ".join(yes))
        if no:
            lines.append("SYMPTOMS RULED OUT: " + ", ".join(no))

    sxp = symptom_assessment(rec)
    if sxp:
        conf = sxp[0].get("confidence", "low")
        lines.append(f"\nLIKELY COMMON CONDITIONS (symptom match, not a diagnosis; "
                     f"confidence in this shortlist: {conf}):")
        for c in sxp:
            flag = "  [needs urgent care]" if c["urgent"] else ""
            lines.append(f"  - {c['condition']}: {c['probability']:.0%} match"
                         f" via {', '.join(c['matched_symptoms'][:5])}{flag}")
        if conf == "low":
            lines.append("  NOTE: very few symptoms were given, so say plainly that this is a "
                         "limited picture and more detail would narrow it down.")

    assessed = [r for r in assessment["results"] if r["status"] == "ok"]
    if assessed:
        lines.append("\nCHRONIC RISK MODEL OUTPUT:")
    else:
        lines.append("\nNO chronic risk model was assessed. Say NOTHING about chronic risk models, "
                     "statistical models, or anything being unavailable or not assessed. The patient "
                     "must not see any reference to them. Write only about the conditions above.")
    for r in assessed:
        lines.append(f"  - {r['label']}: probability={r['probability']:.1%} band={r['risk_band']} "
                     f"model={r['model']} AUC={r['auc']}")
        drivers = [c for c in r["contributions"] if c["observed"]][:5]
        for c in drivers:
            lines.append(f"       driver: {clinical.human(c['feature'])} = {c['value']} "
                         f"({c['direction']} risk)")
        if r["missing"]:
            lines.append(f"       not measured: {', '.join(clinical.human(m) for m in r['missing'][:5])}")
    return "\n".join(lines)


async def report_stream(rec: dict, assessment: dict, transcript: list[dict]) -> AsyncIterator[str]:
    convo = "\n".join(f"{m['role']}: {m['content']}" for m in transcript[-12:] if m["role"] in ("user", "assistant"))
    msgs = [
        {"role": "system", "content": REPORT_RULES + "\n\n" + INDIAN_FORMULARY},
        {"role": "user", "content": f"{_evidence(rec, assessment)}\n\nCONSULTATION TRANSCRIPT:\n{convo}\n\nWrite the assessment note."},
    ]
    async for tok in llm.chat_stream(msgs, temperature=0.3, max_tokens=1400):
        yield tok
