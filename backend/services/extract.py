"""Deterministic-first clinical entity extraction.

Regex handles numeric labs/vitals (fast, exact, auditable). The LLM is only
consulted for qualitative history the patterns cannot capture. This keeps a 4B
model off the critical path for anything numeric.
"""
from __future__ import annotations
import re
from typing import Any

# field -> (aliases, plausible range)
LABS: dict[str, tuple[list[str], tuple[float, float]]] = {
    "age":                  (["age", "aged", "years old", "year old", "yo", "y/o"], (1, 120)),
    "weight_kg":            (["weight", "weigh", "wt", "kilos", "kilograms"], (20, 300)),
    "height_cm":            (["height", "tall", "ht"], (100, 230)),
    "bmi":                  (["bmi", "body mass index"], (10, 70)),
    "total_cholesterol":    (["cholesterol", "chol", "tc"], (80, 500)),
    "fasting_glucose":      (["fasting glucose", "fasting blood sugar", "fbs", "fasting sugar"], (40, 500)),
    "random_glucose":       (["random glucose", "random blood sugar", "rbs", "blood sugar", "glucose", "sugar"], (40, 600)),
    "max_heart_rate":       (["max heart rate", "maximum heart rate", "peak heart rate", "thalach"], (60, 220)),
    "st_depression":        (["st depression", "oldpeak"], (0, 8)),
    "serum_creatinine":     (["creatinine", "s.creatinine", "scr"], (0.2, 20)),
    "blood_urea":           (["urea", "bun", "blood urea"], (5, 250)),
    "sodium":               (["sodium", "na"], (100, 165)),
    "potassium":            (["potassium", "k+"], (2, 9)),
    "haemoglobin":          (["haemoglobin", "hemoglobin", "hgb", "hb"], (3, 20)),
    "packed_cell_volume":   (["packed cell volume", "pcv", "hematocrit", "haematocrit"], (10, 60)),
    "wbc_count":            (["white cell", "wbc", "leucocyte", "leukocyte"], (1000, 30000)),
    "rbc_count":            (["red cell count", "rbc count"], (2, 8)),
    "specific_gravity":     (["specific gravity", "sg"], (1.0, 1.04)),
    "urine_albumin":        (["urine albumin", "albuminuria"], (0, 5)),
    "total_bilirubin":      (["total bilirubin", "t.bilirubin", "tb", "bilirubin"], (0.1, 60)),
    "direct_bilirubin":     (["direct bilirubin", "d.bilirubin", "db", "conjugated bilirubin"], (0, 30)),
    "alkaline_phosphatase": (["alkaline phosphatase", "alkphos", "alp"], (20, 2500)),
    "alt_sgpt":             (["sgpt", "alt"], (1, 3000)),
    "ast_sgot":             (["sgot", "ast"], (1, 3000)),
    "total_protein":        (["total protein", "tp"], (2, 12)),
    "albumin":              (["serum albumin", "albumin", "alb"], (1, 6)),
    "ag_ratio":             (["a/g ratio", "ag ratio", "albumin globulin"], (0.1, 3)),
}

BOOLS: dict[str, tuple[list[str], list[str]]] = {
    "smoker":              (["smoke", "smoker", "smoking", "cigarette", "tobacco"],
                            ["non-smoker", "never smoked", "don't smoke", "do not smoke", "quit smoking", "no smoking"]),
    "heavy_alcohol":       (["drink heavily", "heavy drinker", "alcoholic", "drink daily", "heavy alcohol"],
                            ["don't drink", "do not drink", "teetotal", "no alcohol", "never drink"]),
    "physical_activity":   (["exercise", "workout", "work out", "gym", "jog", "physically active", "walk daily"],
                            ["sedentary", "no exercise", "don't exercise", "do not exercise", "inactive"]),
    "difficulty_walking":  (["difficulty walking", "trouble walking", "hard to walk", "climbing stairs", "breathless walking"], []),
    "prior_stroke":        (["had a stroke", "previous stroke", "stroke history"], ["no stroke", "never had a stroke"]),
    "prior_heart_disease": (["heart attack", "myocardial infarction", "angina", "bypass", "stent", "heart disease"],
                            ["no heart", "never had a heart"]),
    "hypertension_dx":     (["high blood pressure", "hypertension", "hypertensive", "bp tablets", "bp medication"],
                            ["no hypertension", "normal blood pressure"]),
    "diabetes_dx":         (["diabetic", "diabetes", "metformin", "insulin"], ["no diabetes", "not diabetic"]),
    "high_cholesterol_dx": (["high cholesterol", "hyperlipidemia", "statin", "dyslipidemia"], ["normal cholesterol"]),
    "exercise_angina":     (["chest pain when i walk", "chest pain on exertion", "pain when climbing",
                             "chest tightness when walking", "angina on exertion"], []),
    "pedal_edema":         (["ankle swelling", "swollen ankles", "swelling in my feet", "leg swelling", "pedal edema"], []),
    "anaemia":             (["anemia", "anaemia", "anemic", "anaemic"], []),
    "pus_cell_clumps":     (["pus cell clumps"], []),
    "bacteria":            (["bacteriuria", "bacteria in urine"], []),
}

NEG = re.compile(r"\b(no|not|never|denies|without|negative for|free of|n/?o)\b[^.,;]{0,28}$", re.I)
_BP = re.compile(r"\b(\d{2,3})\s*(?:/|over)\s*(\d{2,3})\b")
# 5'9  ·  5 ft 9  ·  5 foot 9 in  ·  5 feet 9 inches
_FEET = re.compile(r"\b([3-7])\s*(?:'|’|ft|foot|feet)\s*(\d{1,2})?\s*(?:\"|”|in|inch|inches)?", re.I)
# a bare 5.9 or 5 9 is only a height when height is what is being discussed
_FEET_BARE = re.compile(r"\b([3-7])[.\s](\d{1,2})\b")


def feet_to_cm(feet: float, inches: float = 0.0) -> float | None:
    cm = feet * 30.48 + inches * 2.54
    return round(cm, 1) if 100 <= cm <= 230 else None


def parse_height(text: str, assume_feet: bool = False) -> float | None:
    """Height from any of the ways people give it, in cm."""
    m = re.search(r"\b(\d{2,3}(?:\.\d)?)\s*(?:cm|centimet\w*)\b", text, re.I)
    if m and 100 <= float(m.group(1)) <= 230:
        return float(m.group(1))
    m = _FEET.search(text)
    if m:
        cm = feet_to_cm(float(m.group(1)), float(m.group(2) or 0))
        if cm:
            return cm
    if assume_feet:
        m = _FEET_BARE.search(text)
        if m and int(m.group(2)) <= 11:
            return feet_to_cm(float(m.group(1)), float(m.group(2)))
    return None
_BP_LABEL = re.compile(r"\b(?:bp|blood pressure)\b\D{0,12}(\d{2,3})\s*(?:/|over)\s*(\d{2,3})", re.I)


def _num_near(text: str, aliases: list[str], rng: tuple[float, float]) -> float | None:
    lo, hi = rng
    for a in sorted(aliases, key=len, reverse=True):
        pat = rf"{re.escape(a)}\s*(?:is|was|of|at|:|=|-|reads?|came back|level)?\s*([0-9]+(?:\.[0-9]+)?)"
        for m in re.finditer(pat, text, re.I):
            v = float(m.group(1))
            if lo <= v <= hi:
                return v
        pat2 = rf"([0-9]+(?:\.[0-9]+)?)\s*(?:mg/dl|g/dl|iu/l|u/l|mmhg|kg|cm|bpm|meq/l|%)?\s*(?:of\s+)?{re.escape(a)}\b"
        for m in re.finditer(pat2, text, re.I):
            v = float(m.group(1))
            if lo <= v <= hi:
                return v
    return None


def extract(text: str) -> dict[str, Any]:
    t = " " + text.lower().replace("’", "'") + " "
    out: dict[str, Any] = {}

    m = _BP_LABEL.search(t) or _BP.search(t)
    if m:
        s, d = int(m.group(1)), int(m.group(2))
        if 70 <= s <= 260 and 40 <= d <= 160:
            out["systolic_bp"], out["diastolic_bp"] = s, d

    for field, (aliases, rng) in LABS.items():
        v = _num_near(t, aliases, rng)
        if v is not None:
            out[field] = v

    # unit-anchored fallbacks: "172 cm", "92 kg" carry no keyword
    if "height_cm" not in out:
        wants_height = bool(re.search(r"\b(height|tall|hight|heigth)\b", t))
        h = parse_height(t, assume_feet=wants_height)
        if h:
            out["height_cm"] = h
    if "weight_kg" not in out:
        # "65kg", "65 kilos" and the common shorthand "65k"
        m = re.search(r"\b(\d{2,3}(?:\.\d)?)\s*(?:kgs?|kilo\w*|k)\b", t)
        if m and 20 <= float(m.group(1)) <= 300:
            out["weight_kg"] = float(m.group(1))

    if "age" not in out:
        m = re.search(r"\b(?:i'?m|i am|im)\s+(\d{1,3})\b", t)
        if m and 1 <= int(m.group(1)) <= 120:
            out["age"] = float(m.group(1))

    # single letters are deliberately excluded: "m" matches inside "I'm"
    if re.search(r"(?<!\w)(male|man|boy|gentleman)(?!\w)", t) and not re.search(r"(?<!\w)(female|woman)(?!\w)", t):
        out["sex"] = "male"
    elif re.search(r"(?<!\w)(female|woman|girl|lady)(?!\w)", t):
        out["sex"] = "female"

    for field, (pos, neg) in BOOLS.items():
        if any(n in t for n in neg):
            out[field] = False
            continue
        for p in pos:
            i = t.find(p)
            if i >= 0:
                out[field] = not bool(NEG.search(t[max(0, i - 34):i]))
                break

    # the descriptor and the word "chest" can fall either side of each other:
    # "crushing pain in my chest", "chest feels tight", "pressure in my chest"
    _crush = r"(crushing|pressure|tight\w*|squeez\w*|heavy|heaviness|band|vice|elephant)"
    if (re.search(rf"{_crush}[^.]{{0,24}}chest", t) or re.search(rf"chest[^.]{{0,24}}{_crush}", t)
            or "typical angina" in t):
        out["chest_pain_type"] = 1
    elif re.search(r"(sharp|stabbing|burning)[^.]{0,24}chest", t) or re.search(r"chest[^.]{0,24}(sharp|stabbing)", t):
        out["chest_pain_type"] = 2
    elif re.search(r"chest[^.]{0,20}(ache|discomfort|pain)", t) or re.search(r"pain[^.]{0,20}chest", t):
        out["chest_pain_type"] = 3
    elif "no chest pain" in t or "without chest pain" in t:
        out["chest_pain_type"] = 4

    # exertional angina, however it is phrased
    if re.search(r"(chest|it)[^.]{0,40}(when|on)[^.]{0,20}"
                 r"(climb\w*|walk\w*|stairs|uphill|exert\w*|running|exercis\w*)", t) and \
            re.search(r"chest|angina", t):
        out.setdefault("exercise_angina", True)

    if re.search(r"\b(feel (terrible|awful|bad)|very unwell|poor health)\b", t):
        out["general_health"] = 5
    elif re.search(r"\b(not great|unwell|tired all the time|fatigued)\b", t):
        out["general_health"] = 4
    elif re.search(r"\b(fairly good|okay|ok|alright|average)\b", t):
        out["general_health"] = 3
    elif re.search(r"\b(good health|pretty good|quite well)\b", t):
        out["general_health"] = 2
    elif re.search(r"\b(excellent|very healthy|great health)\b", t):
        out["general_health"] = 1

    if re.search(r"\bpoor appetite|no appetite|not eating|lost my appetite\b", t):
        out["appetite"] = "poor"
    elif re.search(r"\bappetite is (good|fine|normal)\b", t):
        out["appetite"] = "good"
    return out


# ---------------------------------------------------------------------------
# Context-aware extraction.
#
# A patient answering "Yes" or "3.8" says nothing a keyword matcher can latch
# onto - the subject lives in the question that was just asked. These helpers
# resolve a bare answer against the topic of the preceding question.
# ---------------------------------------------------------------------------

# Phrasings that identify what a question is ASKING about, beyond the value
# aliases above (a question says "how old are you", never "age 58").
QUESTION_EXTRA: dict[str, list[str]] = {
    "age": ["how old"],
    "sex": ["male or female", "your sex", "your gender", "gender", "female", "man or a woman",
            "a male or", "male or"],
    "weight_kg": ["how much do you weigh", "how much you weigh", "your weight"],
    "height_cm": ["how tall"],
    "smoker": ["do you smoke", "have you smoked", "ever smoked", "smoking"],
    "heavy_alcohol": ["do you drink", "how much do you drink", "alcohol"],
    "physical_activity": ["do you exercise", "physically active", "any exercise", "how active"],
    "difficulty_walking": ["difficulty walking", "trouble walking", "climbing stairs"],
    "systolic_bp": ["blood pressure", "systolic", "systolic blood pressure", "top number"],
    "diastolic_bp": ["diastolic", "diastolic blood pressure", "bottom number", "lower number"],
    "physical_health_days": ["physical health"],
    "mental_health_days": ["mental health"],
    "total_cholesterol": ["cholesterol"],
    "fasting_glucose": ["fasting", "blood sugar", "sugar level"],
    "max_heart_rate": ["heart rate"],
    "exercise_angina": ["pain on exertion", "pain when you walk", "chest pain when"],
    "pedal_edema": ["ankles", "ankle swelling", "swelling in your"],
    "anaemia": ["anaemia", "anemia", "anaemic", "anemic"],
    "appetite": ["your appetite", "appetite been", "eating"],
    "hypertension_dx": ["high blood pressure", "hypertension", "bp tablets"],
    "diabetes_dx": ["diabetes", "diabetic"],
    "high_cholesterol_dx": ["high cholesterol", "statin"],
    "prior_stroke": ["a stroke"],
    "prior_heart_disease": ["heart attack", "angina before", "heart trouble"],
    "serum_creatinine": ["creatinine"],
    "blood_urea": ["urea"],
    "haemoglobin": ["haemoglobin", "hemoglobin"],
    "urine_albumin": ["urine albumin"],
    "specific_gravity": ["specific gravity"],
    "total_bilirubin": ["bilirubin"],
    "alt_sgpt": ["sgpt", "alt"],
    "ast_sgot": ["sgot", "ast"],
    "alkaline_phosphatase": ["alkaline phosphatase"],
    "albumin": ["albumin"],
    "total_protein": ["total protein"],
    "general_health": ["your health overall", "overall health", "general health"],
}

# Plausible ranges for fields that are NOT in LABS (blood pressure is matched by a
# dedicated "148/94" regex, and the ordinal scales never appear as free numbers).
CONTEXT_RANGES: dict[str, tuple[float, float]] = {
    "systolic_bp": (70, 260),
    "diastolic_bp": (40, 160),
    "general_health": (1, 5),
    "chest_pain_type": (1, 4),
    "resting_ecg": (0, 2),
    "st_slope": (1, 3),
    "vessels_colored": (0, 3),
    "thalassemia": (3, 7),
    "urine_sugar": (0, 5),
    "mental_health_days": (0, 30),
    "physical_health_days": (0, 30),
}


def _range_for(field: str) -> tuple[float, float] | None:
    if field in LABS:
        return LABS[field][1]
    return CONTEXT_RANGES.get(field)


_YES = re.compile(r"^\W*(yes|yeah|yep|yup|aye|correct|that'?s right|i do|i have|i am|sure)\b", re.I)
_NO = re.compile(r"^\W*(no|nope|nah|never|not really|none|i don'?t|i do not|i haven'?t|i'?m not)\b", re.I)


def _cue_hit(cue: str, q: str) -> bool:
    """Whole-word match, so a short alias like 'alt' never fires inside 'health'."""
    return re.search(rf"(?<!\w){re.escape(cue)}(?!\w)", q) is not None


def question_topic(question: str) -> str | None:
    """Which canonical field is this question about? Longest whole-word cue wins."""
    q = " " + question.lower() + " "
    best, best_len = None, 0
    for source in (QUESTION_EXTRA,
                   {f: al for f, (al, _r) in LABS.items()},
                   {f: pos for f, (pos, _n) in BOOLS.items()}):
        for field, cues in source.items():
            for c in cues:
                if len(c) > best_len and _cue_hit(c, q):
                    best, best_len = field, len(c)
    return best


def extract_in_context(question: str, answer: str) -> dict[str, Any]:
    """Resolve a bare answer ('Yes', '3.8') against the question that prompted it."""
    field = question_topic(question or "")
    if not field:
        return {}
    a = answer.strip()
    kind = LABS.get(field) and "number" or ("bool" if field in BOOLS else None)
    if field in ("sex", "appetite", "rbc_urine", "pus_cells"):
        kind = "enum"

    # yes / no answers resolve booleans
    if _NO.match(a):
        if field in BOOLS or kind == "bool":
            return {field: False}
    elif _YES.match(a):
        if field in BOOLS or kind == "bool":
            return {field: True}

    if field in ("systolic_bp", "diastolic_bp"):
        m = _BP.search(a)
        if m:
            sv, dv = int(m.group(1)), int(m.group(2))
            if 70 <= sv <= 260 and 40 <= dv <= 160:
                return {"systolic_bp": sv, "diastolic_bp": dv}

    if field == "height_cm":
        h = parse_height(a, assume_feet=True)
        if h:
            return {"height_cm": h}

    # a bare number resolves the numeric field that was asked about
    rng = _range_for(field)
    if rng:
        lo, hi = rng
        for m in re.finditer(r"\b(\d+(?:\.\d+)?)\b", a):
            v = float(m.group(1))
            if lo <= v <= hi:
                return {field: v}

    if field == "sex":
        if re.search(r"\b(male|man|m)\b", a, re.I) and not re.search(r"\bfemale|woman\b", a, re.I):
            return {"sex": "male"}
        if re.search(r"\b(female|woman|f)\b", a, re.I):
            return {"sex": "female"}
    if field == "appetite":
        if re.search(r"\b(poor|bad|no|lost|not good)\b", a, re.I):
            return {"appetite": "poor"}
        if re.search(r"\b(good|fine|normal|ok)\b", a, re.I):
            return {"appetite": "good"}
    return {}
