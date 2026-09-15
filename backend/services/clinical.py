"""Canonical clinical record -> per-disease feature vectors.

One patient conversation populates ONE canonical record; each disease model
receives only the subset of fields it was trained on. Absent fields fall back to
the training-set median inside the sklearn pipeline's imputer.
"""
from __future__ import annotations
from typing import Any

Rec = dict[str, Any]

# ---- canonical fields the consultation agent may collect -------------------
FIELDS: dict[str, dict] = {
    "age":                  {"unit": "years", "kind": "number", "ask": "age"},
    "sex":                  {"unit": "", "kind": "enum", "options": ["male", "female"], "ask": "sex"},
    "height_cm":            {"unit": "", "kind": "number", "ask": "height, in feet and inches or centimetres"},
    "weight_kg":            {"unit": "kg", "kind": "number", "ask": "weight"},
    "bmi":                  {"unit": "kg/m2", "kind": "number", "ask": "BMI"},
    "systolic_bp":          {"unit": "mmHg", "kind": "number", "ask": "systolic blood pressure"},
    "diastolic_bp":         {"unit": "mmHg", "kind": "number", "ask": "diastolic blood pressure"},
    "total_cholesterol":    {"unit": "mg/dL", "kind": "number", "ask": "total cholesterol"},
    "fasting_glucose":      {"unit": "mg/dL", "kind": "number", "ask": "fasting blood glucose"},
    "random_glucose":       {"unit": "mg/dL", "kind": "number", "ask": "random blood glucose"},
    "smoker":               {"unit": "", "kind": "bool", "ask": "smoking history"},
    "heavy_alcohol":        {"unit": "", "kind": "bool", "ask": "heavy alcohol use"},
    "physical_activity":    {"unit": "", "kind": "bool", "ask": "regular physical activity"},
    "general_health":       {"unit": "1=excellent..5=poor", "kind": "number", "ask": "overall self-rated health"},
    "mental_health_days":   {"unit": "days/30", "kind": "number", "ask": "poor mental-health days"},
    "physical_health_days": {"unit": "days/30", "kind": "number", "ask": "poor physical-health days"},
    "difficulty_walking":   {"unit": "", "kind": "bool", "ask": "difficulty walking or climbing stairs"},
    "hypertension_dx":      {"unit": "", "kind": "bool", "ask": "diagnosed high blood pressure"},
    "high_cholesterol_dx":  {"unit": "", "kind": "bool", "ask": "diagnosed high cholesterol"},
    "diabetes_dx":          {"unit": "", "kind": "bool", "ask": "known diabetes"},
    "prior_stroke":         {"unit": "", "kind": "bool", "ask": "previous stroke"},
    "prior_heart_disease":  {"unit": "", "kind": "bool", "ask": "previous heart attack or angina"},
    "chest_pain_type":      {"unit": "1=typical angina,2=atypical,3=non-anginal,4=asymptomatic",
                             "kind": "number", "ask": "chest pain character"},
    "resting_ecg":          {"unit": "0=normal,1=ST-T abnormality,2=LV hypertrophy", "kind": "number", "ask": "resting ECG"},
    "max_heart_rate":       {"unit": "bpm", "kind": "number", "ask": "maximum heart rate achieved"},
    "exercise_angina":      {"unit": "", "kind": "bool", "ask": "chest pain on exertion"},
    "st_depression":        {"unit": "mm", "kind": "number", "ask": "exercise ST depression"},
    "st_slope":             {"unit": "1=upsloping,2=flat,3=downsloping", "kind": "number", "ask": "ST segment slope"},
    "vessels_colored":      {"unit": "0-3", "kind": "number", "ask": "major vessels on fluoroscopy"},
    "thalassemia":          {"unit": "3=normal,6=fixed defect,7=reversible defect", "kind": "number", "ask": "thallium scan"},
    "serum_creatinine":     {"unit": "mg/dL", "kind": "number", "ask": "serum creatinine"},
    "blood_urea":           {"unit": "mg/dL", "kind": "number", "ask": "blood urea"},
    "sodium":               {"unit": "mEq/L", "kind": "number", "ask": "serum sodium"},
    "potassium":            {"unit": "mEq/L", "kind": "number", "ask": "serum potassium"},
    "haemoglobin":          {"unit": "g/dL", "kind": "number", "ask": "haemoglobin"},
    "packed_cell_volume":   {"unit": "%", "kind": "number", "ask": "packed cell volume"},
    "wbc_count":            {"unit": "cells/cmm", "kind": "number", "ask": "white cell count"},
    "rbc_count":            {"unit": "millions/cmm", "kind": "number", "ask": "red cell count"},
    "specific_gravity":     {"unit": "1.005-1.025", "kind": "number", "ask": "urine specific gravity"},
    "urine_albumin":        {"unit": "0-5", "kind": "number", "ask": "urine albumin"},
    "urine_sugar":          {"unit": "0-5", "kind": "number", "ask": "urine sugar"},
    "rbc_urine":            {"unit": "normal/abnormal", "kind": "enum", "options": ["normal", "abnormal"], "ask": "urine RBCs"},
    "pus_cells":            {"unit": "normal/abnormal", "kind": "enum", "options": ["normal", "abnormal"], "ask": "urine pus cells"},
    "pus_cell_clumps":      {"unit": "", "kind": "bool", "ask": "pus cell clumps"},
    "bacteria":             {"unit": "", "kind": "bool", "ask": "bacteriuria"},
    "appetite":             {"unit": "good/poor", "kind": "enum", "options": ["good", "poor"], "ask": "appetite"},
    "pedal_edema":          {"unit": "", "kind": "bool", "ask": "ankle swelling"},
    "anaemia":              {"unit": "", "kind": "bool", "ask": "known anaemia"},
    "total_bilirubin":      {"unit": "mg/dL", "kind": "number", "ask": "total bilirubin"},
    "direct_bilirubin":     {"unit": "mg/dL", "kind": "number", "ask": "direct bilirubin"},
    "alkaline_phosphatase": {"unit": "IU/L", "kind": "number", "ask": "alkaline phosphatase"},
    "alt_sgpt":             {"unit": "IU/L", "kind": "number", "ask": "ALT (SGPT)"},
    "ast_sgot":             {"unit": "IU/L", "kind": "number", "ask": "AST (SGOT)"},
    "total_protein":        {"unit": "g/dL", "kind": "number", "ask": "total protein"},
    "albumin":              {"unit": "g/dL", "kind": "number", "ask": "serum albumin"},
    "ag_ratio":             {"unit": "ratio", "kind": "number", "ask": "albumin/globulin ratio"},
}

BRFSS_AGE_BUCKETS = [(24, 1), (29, 2), (34, 3), (39, 4), (44, 5), (49, 6), (54, 7),
                     (59, 8), (64, 9), (69, 10), (74, 11), (79, 12), (200, 13)]


def _b(v, default=None):
    """Coerce to 1/0; return default when unknown."""
    if v is None:
        return default
    if isinstance(v, bool):
        return int(v)
    if isinstance(v, (int, float)):
        return int(bool(v))
    s = str(v).strip().lower()
    if s in ("yes", "true", "y", "1", "present", "abnormal", "poor"):
        return 1
    if s in ("no", "false", "n", "0", "notpresent", "normal", "good"):
        return 0
    return default


def _n(v, default=None):
    try:
        return float(v)
    except (TypeError, ValueError):
        return default


def derive(rec: Rec) -> Rec:
    """Fill values computable from others (BMI, hypertension flags, glucose)."""
    r = dict(rec)
    if r.get("bmi") is None:
        h, w = _n(r.get("height_cm")), _n(r.get("weight_kg"))
        if h and w and h > 50:
            r["bmi"] = round(w / (h / 100.0) ** 2, 1)
    if r.get("hypertension_dx") is None:
        s, d = _n(r.get("systolic_bp")), _n(r.get("diastolic_bp"))
        if s or d:
            r["hypertension_dx"] = int((s or 0) >= 140 or (d or 0) >= 90)
    if r.get("high_cholesterol_dx") is None and _n(r.get("total_cholesterol")):
        r["high_cholesterol_dx"] = int(_n(r["total_cholesterol"]) >= 240)
    if r.get("random_glucose") is None and _n(r.get("fasting_glucose")):
        r["random_glucose"] = _n(r["fasting_glucose"])
    if r.get("diabetes_dx") is None and _n(r.get("fasting_glucose")):
        r["diabetes_dx"] = int(_n(r["fasting_glucose"]) >= 126)
    return r


def _brfss_age(age):
    a = _n(age)
    if a is None:
        return None
    for cut, code in BRFSS_AGE_BUCKETS:
        if a <= cut:
            return code
    return 13


# ---- per-disease mappers ---------------------------------------------------
def map_diabetes(r: Rec) -> Rec:
    return {
        "HighBP": _b(r.get("hypertension_dx")), "HighChol": _b(r.get("high_cholesterol_dx")),
        "CholCheck": 1, "BMI": _n(r.get("bmi")), "Smoker": _b(r.get("smoker")),
        "Stroke": _b(r.get("prior_stroke")), "HeartDiseaseorAttack": _b(r.get("prior_heart_disease")),
        "PhysActivity": _b(r.get("physical_activity")), "Fruits": None, "Veggies": None,
        "HvyAlcoholConsump": _b(r.get("heavy_alcohol")), "AnyHealthcare": 1, "NoDocbcCost": None,
        "GenHlth": _n(r.get("general_health")), "MentHlth": _n(r.get("mental_health_days")),
        "PhysHlth": _n(r.get("physical_health_days")), "DiffWalk": _b(r.get("difficulty_walking")),
        "Sex": 1 if str(r.get("sex", "")).lower().startswith("m") else (0 if r.get("sex") else None),
        "Age": _brfss_age(r.get("age")), "Education": None, "Income": None,
    }


def map_heart(r: Rec) -> Rec:
    return {
        "age": _n(r.get("age")),
        "sex": 1 if str(r.get("sex", "")).lower().startswith("m") else (0 if r.get("sex") else None),
        "cp": _n(r.get("chest_pain_type")), "trestbps": _n(r.get("systolic_bp")),
        "chol": _n(r.get("total_cholesterol")),
        "fbs": None if _n(r.get("fasting_glucose")) is None else int(_n(r["fasting_glucose"]) > 120),
        "restecg": _n(r.get("resting_ecg")), "thalach": _n(r.get("max_heart_rate")),
        "exang": _b(r.get("exercise_angina")), "oldpeak": _n(r.get("st_depression")),
        "slope": _n(r.get("st_slope")), "ca": _n(r.get("vessels_colored")), "thal": _n(r.get("thalassemia")),
    }


def _yn(v):
    b = _b(v)
    return None if b is None else ("yes" if b else "no")


def map_kidney(r: Rec) -> Rec:
    return {
        "age": _n(r.get("age")), "bp": _n(r.get("diastolic_bp")), "sg": _n(r.get("specific_gravity")),
        "al": _n(r.get("urine_albumin")), "su": _n(r.get("urine_sugar")), "bgr": _n(r.get("random_glucose")),
        "bu": _n(r.get("blood_urea")), "sc": _n(r.get("serum_creatinine")), "sod": _n(r.get("sodium")),
        "pot": _n(r.get("potassium")), "hemo": _n(r.get("haemoglobin")), "pcv": _n(r.get("packed_cell_volume")),
        "wbcc": _n(r.get("wbc_count")), "rbcc": _n(r.get("rbc_count")),
        "rbc": r.get("rbc_urine"), "pc": r.get("pus_cells"),
        "pcc": "present" if _b(r.get("pus_cell_clumps")) else ("notpresent" if r.get("pus_cell_clumps") is not None else None),
        "ba": "present" if _b(r.get("bacteria")) else ("notpresent" if r.get("bacteria") is not None else None),
        "htn": _yn(r.get("hypertension_dx")), "dm": _yn(r.get("diabetes_dx")), "cad": _yn(r.get("prior_heart_disease")),
        "appet": r.get("appetite"), "pe": _yn(r.get("pedal_edema")), "ane": _yn(r.get("anaemia")),
    }


def map_liver(r: Rec) -> Rec:
    return {
        "Age": _n(r.get("age")),
        "Gender": "Male" if str(r.get("sex", "")).lower().startswith("m") else ("Female" if r.get("sex") else None),
        "TB": _n(r.get("total_bilirubin")), "DB": _n(r.get("direct_bilirubin")),
        "Alkphos": _n(r.get("alkaline_phosphatase")), "Sgpt": _n(r.get("alt_sgpt")),
        "Sgot": _n(r.get("ast_sgot")), "TP": _n(r.get("total_protein")),
        "ALB": _n(r.get("albumin")), "A/G Ratio": _n(r.get("ag_ratio")),
    }


def map_breast(r: Rec) -> Rec:
    """WDBC uses FNA cytology morphometry - not obtainable by history taking.
    Values arrive from the cytology panel in the UI, prefixed 'wdbc_'."""
    return {k[5:]: _n(v) for k, v in r.items() if k.startswith("wdbc_")}


MAPPERS = {"diabetes": map_diabetes, "heart": map_heart, "kidney": map_kidney,
           "liver": map_liver, "breast": map_breast}

# fields that carry most of the signal - used for the coverage gate
KEY_FIELDS = {
    "diabetes": ["bmi", "hypertension_dx", "high_cholesterol_dx", "age", "general_health", "difficulty_walking", "physical_activity"],
    # max_heart_rate / ST slope / vessel counts are stress-test findings a patient
    # cannot report, so they are imputed and excluded from the coverage gate.
    "heart": ["age", "sex", "chest_pain_type", "systolic_bp", "total_cholesterol", "exercise_angina"],
    "kidney": ["serum_creatinine", "blood_urea", "haemoglobin", "specific_gravity", "urine_albumin", "hypertension_dx"],
    "liver": ["total_bilirubin", "direct_bilirubin", "alt_sgpt", "ast_sgot", "albumin", "age"],
    "breast": [],
}


def coverage(rec: Rec, disease: str) -> float:
    keys = KEY_FIELDS.get(disease, [])
    if not keys:
        return 1.0 if any(k.startswith("wdbc_") for k in rec) else 0.0
    present = sum(1 for k in keys if rec.get(k) is not None)
    return present / len(keys)


def missing_key_fields(rec: Rec, disease: str) -> list[str]:
    return [k for k in KEY_FIELDS.get(disease, []) if rec.get(k) is None]


# Dataset column names are not English. Everything shown to a patient, or given
# to the language model to describe, is translated through this table first.
HUMAN: dict[str, str] = {
    # diabetes (BRFSS)
    "HighBP": "high blood pressure", "HighChol": "high cholesterol", "CholCheck": "cholesterol check",
    "BMI": "body mass index", "Smoker": "smoking", "Stroke": "previous stroke",
    "HeartDiseaseorAttack": "previous heart trouble", "PhysActivity": "physical activity",
    "Fruits": "fruit intake", "Veggies": "vegetable intake", "HvyAlcoholConsump": "heavy drinking",
    "AnyHealthcare": "access to healthcare", "NoDocbcCost": "cost barrier to care",
    "GenHlth": "how you rate your health", "MentHlth": "poor mental-health days",
    "PhysHlth": "poor physical-health days", "DiffWalk": "difficulty walking",
    "Sex": "sex", "Age": "age", "Education": "education", "Income": "income",
    # heart (Cleveland)
    "age": "age", "sex": "sex", "cp": "type of chest pain", "trestbps": "blood pressure",
    "chol": "cholesterol", "fbs": "fasting blood sugar", "restecg": "resting ECG",
    "thalach": "maximum heart rate", "exang": "chest pain on exertion",
    "oldpeak": "ST depression on exercise", "slope": "ST segment slope",
    "ca": "number of narrowed vessels", "thal": "thallium scan result",
    # kidney
    "bp": "blood pressure", "sg": "urine concentration", "al": "protein in urine",
    "su": "sugar in urine", "bgr": "blood sugar", "bu": "urea, a kidney test",
    "sc": "creatinine, a kidney test", "sod": "sodium", "pot": "potassium",
    "hemo": "haemoglobin", "pcv": "packed cell volume", "wbcc": "white cell count",
    "rbcc": "red cell count", "rbc": "red cells in urine", "pc": "pus cells in urine",
    "pcc": "pus cell clumps", "ba": "bacteria in urine", "htn": "high blood pressure",
    "dm": "diabetes", "cad": "heart disease", "appet": "appetite",
    "pe": "ankle swelling", "ane": "anaemia",
    # liver (ILPD)
    "TB": "total bilirubin", "DB": "direct bilirubin", "Alkphos": "alkaline phosphatase",
    "Sgpt": "ALT, a liver enzyme", "Sgot": "AST, a liver enzyme",
    "TP": "total protein", "ALB": "albumin", "A/G Ratio": "albumin to globulin ratio",
    "Gender": "sex",
}


def human(feature: str) -> str:
    """Plain-English name for a model feature."""
    if feature in HUMAN:
        return HUMAN[feature]
    if feature in FIELDS:
        return FIELDS[feature].get("ask", feature)
    return feature.replace("_", " ")
