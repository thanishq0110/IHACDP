"""Deterministic prescribing.

The language model narrates; it does not choose drugs. Asked to, a 4B model
paired Calpol with Sinarest (both paracetamol), prescribed a decongestant to
"soothe your stomach", and dropped the dosing lines entirely. Selection and
dosing therefore happen here, in reviewable code, and the rendered block is
handed to the model as fixed text.

Adult doses for medicines sold over the counter in India. Anything needing a
prescription is named without a dose.
"""
from __future__ import annotations
from typing import Any

# Each entry: what it is, the dose, how long, one caution, and whether it
# contains paracetamol - so two paracetamol products can never both be issued.
FORMULARY: dict[str, dict[str, Any]] = {
    "paracetamol": {
        "brand": "Dolo 650", "generic": "Paracetamol 650mg", "for": "pain and fever",
        "dose": "1 tablet three times a day, after food", "duration": "up to 5 days",
        "note": "Do not take more than 4 tablets in 24 hours.", "paracetamol": True,
    },
    "ibuprofen": {
        "brand": "Brufen 400", "generic": "Ibuprofen 400mg", "for": "pain, swelling and inflammation",
        "dose": "1 tablet three times a day, after food", "duration": "up to 5 days",
        "note": "Take with food. Avoid if you have asthma, stomach ulcers or kidney problems.",
        "paracetamol": False,
    },
    "topical_analgesic": {
        "brand": "Volini gel", "generic": "Diclofenac topical", "for": "local muscle and joint pain",
        "dose": "Apply a thin layer to the painful area, 3 to 4 times a day",
        "duration": "up to 7 days", "note": "For external use only. Do not apply to broken skin.",
        "paracetamol": False,
    },
    "cold_combination": {
        "brand": "Sinarest", "generic": "Paracetamol + Chlorpheniramine + Phenylephrine",
        "for": "blocked nose, sneezing and fever together",
        "dose": "1 tablet twice a day, after food", "duration": "up to 3 days",
        "note": "Causes drowsiness - do not drive. Contains paracetamol, so do not take another "
                "paracetamol tablet alongside it.", "paracetamol": True,
    },
    "nasal_drops": {
        "brand": "Otrivin nasal drops", "generic": "Xylometazoline 0.1%", "for": "a blocked nose",
        "dose": "1 spray in each nostril, 2 to 3 times a day", "duration": "no more than 5 days",
        "note": "Using it longer than 5 days can make congestion worse.", "paracetamol": False,
    },
    "lozenge": {
        "brand": "Strepsils", "generic": "Amylmetacresol lozenges", "for": "a sore throat",
        "dose": "1 lozenge every 2 to 3 hours as needed", "duration": "up to 5 days",
        "note": "Do not exceed 12 lozenges in 24 hours.", "paracetamol": False,
    },
    "cough_syrup": {
        "brand": "Honitus syrup", "generic": "Herbal cough syrup", "for": "a cough and throat irritation",
        "dose": "1 teaspoon three times a day", "duration": "up to 5 days",
        "note": "See a doctor if the cough lasts more than 2 weeks.", "paracetamol": False,
    },
    "antihistamine": {
        "brand": "Cetzine 10", "generic": "Cetirizine 10mg", "for": "itching, rash and allergy",
        "dose": "1 tablet at night", "duration": "up to 5 days",
        "note": "May cause drowsiness.", "paracetamol": False,
    },
    "antacid": {
        "brand": "Digene", "generic": "Antacid chewable", "for": "acidity and indigestion",
        "dose": "1 to 2 tablets after meals and at bedtime", "duration": "up to 5 days",
        "note": "Leave 2 hours between this and any other medicine.", "paracetamol": False,
    },
    "ors": {
        "brand": "Electral", "generic": "Oral rehydration salts", "for": "replacing fluids lost through "
                "loose motions or vomiting",
        "dose": "1 sachet in 1 litre of clean water, sipped through the day",
        "duration": "while symptoms last",
        "note": "This is the most important treatment for loose motions. Seek care if you cannot keep "
                "fluids down.", "paracetamol": False,
    },
    "laxative": {
        "brand": "Isabgol", "generic": "Psyllium husk", "for": "constipation",
        "dose": "1 to 2 teaspoons in a glass of water at night", "duration": "up to 7 days",
        "note": "Drink plenty of water through the day.", "paracetamol": False,
    },
    "antiseptic": {
        "brand": "Betadine", "generic": "Povidone-iodine 5%", "for": "cleaning a cut or graze",
        "dose": "Apply to the cleaned area twice a day", "duration": "until healed",
        "note": "For external use only.", "paracetamol": False,
    },
}

# Which medicines suit which reported symptoms. Order is priority order.
BY_SYMPTOM: list[tuple[str, set[str]]] = [
    ("ors", {"diarrhoea", "vomiting"}),
    ("antacid", {"acidity", "indigestion", "stomach_pain", "abdominal_pain"}),
    ("laxative", {"constipation", "hard_stools", "straining_to_pass_stool"}),
    ("antiseptic", {"skin_redness_local", "blistering"}),
    ("antihistamine", {"itching", "skin_rash", "continuous_sneezing", "watering_from_eyes"}),
    ("nasal_drops", {"congestion", "sinus_pressure", "runny_nose"}),
    ("lozenge", {"throat_irritation", "difficulty_swallowing"}),
    ("cough_syrup", {"cough", "phlegm"}),
    ("cold_combination", {"runny_nose", "continuous_sneezing", "congestion"}),
    ("topical_analgesic", {"muscle_pain", "back_pain", "joint_pain", "knee_pain", "neck_pain",
                           "swelling_at_injury", "movement_stiffness"}),
    ("ibuprofen", {"swelling_at_injury", "joint_pain", "muscle_pain", "back_pain", "recent_injury",
                   "swelling_joints", "tooth_pain", "gum_swelling"}),
    ("paracetamol", {"headache", "high_fever", "mild_fever", "severe_pain", "ear_pain",
                     "tooth_pain", "muscle_pain", "back_pain", "throat_irritation"}),
]

# A medicine may only be issued if none of its classes is already covered.
CLASSES: dict[str, set[str]] = {
    "paracetamol": {"analgesic"},
    "ibuprofen": {"analgesic", "nsaid"},
    "topical_analgesic": {"topical"},
    "cold_combination": {"analgesic", "antihistamine", "decongestant"},
    "nasal_drops": {"decongestant"},
    "lozenge": {"throat"},
    "cough_syrup": {"cough"},
    "antihistamine": {"antihistamine"},
    "antacid": {"antacid"},
    "ors": {"rehydration"},
    "laxative": {"laxative"},
    "antiseptic": {"antiseptic"},
}

MAX_MEDICINES = 3

# These need assessing, not medicating. A painkiller for suspected appendicitis
# masks the very sign that decides whether someone goes to theatre.
NO_SELF_TREATMENT = {
    "Appendicitis", "Meningitis", "Sepsis", "Stroke", "Heart attack",
    "Subarachnoid hemorrhage", "Acute pancreatitis", "Acute glaucoma", "Glaucoma",
    "Pyelonephritis", "Diverticulitis", "Suspected fracture", "Concussion",
    "Paralysis (brain hemorrhage)", "Heat stroke", "Crushing injury",
}

URGENT_ADVICE = (
    "**Do not take any medicine for this yet.**\n"
    "Go to a hospital or see a doctor today so this can be examined properly. "
    "Painkillers can hide the signs a doctor needs to see."
)


def needs_assessment(conditions: list[dict[str, Any]] | None) -> bool:
    """True when the leading explanation should be examined, not self-treated."""
    if not conditions:
        return False
    return str(conditions[0].get("condition", "")) in NO_SELF_TREATMENT


def build(symptoms: dict[str, bool], conditions: list[dict[str, Any]] | None = None
          ) -> list[dict[str, Any]]:
    """Choose medicines for the symptoms actually reported."""
    if needs_assessment(conditions):
        return []
    reported = {s for s, v in symptoms.items() if v}
    if not reported:
        return []

    # rank by how much of the complaint each medicine addresses, then by the
    # listed priority - a fever should not lose its paracetamol to a lozenge
    ranked = sorted(
        ((key, len(triggers & reported), i) for i, (key, triggers) in enumerate(BY_SYMPTOM)
         if triggers & reported),
        key=lambda t: (-t[1], t[2]),
    )

    chosen: list[str] = []
    covered: set[str] = set()
    for key, _overlap, _i in ranked:
        if len(chosen) >= MAX_MEDICINES:
            break
        if key in chosen:
            continue
        if FORMULARY[key]["paracetamol"] and any(FORMULARY[c]["paracetamol"] for c in chosen):
            continue
        cls = CLASSES.get(key, set())
        if cls & covered:
            continue            # that job is already done by something chosen
        chosen.append(key)
        covered |= cls
    return [dict(FORMULARY[k], key=k) for k in chosen]


def render(items: list[dict[str, Any]]) -> str:
    """The prescription block, exactly as the patient should see it."""
    if not items:
        return ""
    lines = []
    for i, m in enumerate(items, 1):
        lines.append(f"**{i}. {m['brand']} ({m['generic']})** - for {m['for']}")
        lines.append(f"Dose: {m['dose']}")
        lines.append(f"Duration: {m['duration']}")
        lines.append(f"Note: {m['note']}")
        lines.append("")
    return "\n".join(lines).strip()
