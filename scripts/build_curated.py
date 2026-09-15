"""Curated symptom patterns for everyday complaints the public datasets miss.

Public symptom->condition matrices are built around infectious and chronic
disease. They contain almost nothing for the things people actually walk in
with: a pulled muscle, a tension headache, an ear infection, a sprained ankle.
These patterns are encoded from standard clinical presentations so the triage
layer can recognise them, and each carries its own provenance.

Written as data, not scraped, so every row is reviewable.
"""
from __future__ import annotations
import csv
from pathlib import Path

OUT = Path(__file__).resolve().parents[1] / "data" / "curated" / "curated_conditions.csv"

# condition -> symptoms that typically present with it
CURATED: dict[str, list[str]] = {
    # ---- headache ----------------------------------------------------------
    "Tension headache": [
        "headache", "band_like_head_pressure", "neck_pain", "movement_stiffness",
        "fatigue", "tenderness_to_touch", "stress",
    ],
    "Cluster headache": [
        "headache", "one_sided_head_pain", "severe_pain", "watering_from_eyes",
        "redness_of_eyes", "runny_nose", "restlessness",
    ],
    "Sinusitis": [
        "sinus_pressure", "headache", "congestion", "runny_nose", "phlegm",
        "facial_pain", "mild_fever", "loss_of_smell", "cough",
    ],
    "Dehydration headache": [
        "headache", "thirst", "dry_mouth", "dark_urine", "dizziness", "fatigue",
    ],

    # ---- back, neck and limb pain ------------------------------------------
    "Mechanical low back pain": [
        "back_pain", "recent_injury", "pain_on_movement", "movement_stiffness",
        "tenderness_to_touch", "muscle_pain",
    ],
    "Sciatica": [
        "back_pain", "pain_radiating_to_leg", "numbness_tingling", "weakness_in_limbs",
        "pain_on_movement", "one_sided_pain",
    ],
    "Muscle strain": [
        "muscle_pain", "recent_injury", "pain_on_movement", "tenderness_to_touch",
        "swelling_at_injury", "movement_stiffness",
    ],
    "Tendonitis": [
        "joint_pain", "pain_on_movement", "tenderness_to_touch", "swelling_at_injury",
        "movement_stiffness", "repetitive_activity",
    ],
    "Frozen shoulder": [
        "joint_pain", "limited_movement", "movement_stiffness", "pain_on_movement",
        "night_pain",
    ],

    # ---- injuries ----------------------------------------------------------
    "Ankle sprain": [
        "recent_injury", "joint_pain", "swelling_at_injury", "bruising",
        "limited_movement", "pain_on_movement", "tenderness_to_touch",
    ],
    "Suspected fracture": [
        "recent_injury", "severe_pain", "swelling_at_injury", "bruising",
        "limited_movement", "deformity", "tenderness_to_touch",
    ],
    "Minor burn": [
        "skin_redness_local", "blistering", "severe_pain", "recent_injury", "swelling_at_injury",
    ],
    "Concussion": [
        "recent_injury", "headache", "dizziness", "nausea", "blurred_and_distorted_vision",
        "confusion", "light_sensitivity", "lethargy",
    ],

    # ---- ear, nose, throat, eye, dental ------------------------------------
    "Otitis media": [
        "ear_pain", "hearing_muffled", "mild_fever", "ear_discharge", "irritability", "headache",
    ],
    "Tonsillitis": [
        "throat_irritation", "difficulty_swallowing", "high_fever", "swelled_lymph_nodes",
        "headache", "malaise",
    ],
    "Pharyngitis": [
        "throat_irritation", "difficulty_swallowing", "mild_fever", "cough", "malaise",
    ],
    "Conjunctivitis": [
        "redness_of_eyes", "eye_discharge", "eye_gritty", "watering_from_eyes", "itching",
    ],
    "Toothache": [
        "tooth_pain", "gum_swelling", "pain_on_movement", "tenderness_to_touch", "headache",
    ],

    # ---- gastrointestinal --------------------------------------------------
    # Appendicitis also arrives from SymCat; the two patterns are unioned on
    # merge. This row exists to add the right-lower-quadrant localiser, which no
    # source carries and which is the single most useful thing a patient says.
    "Appendicitis": [
        "right_lower_abdominal_pain", "abdominal_pain", "stomach_pain", "nausea", "vomiting",
        "loss_of_appetite", "mild_fever", "severe_pain", "pain_on_movement",
    ],
    "Influenza": [
        "high_fever", "chills", "muscle_pain", "headache", "fatigue", "cough",
        "throat_irritation", "malaise", "sweating",
    ],
    "Food poisoning": [
        "vomiting", "diarrhoea", "stomach_pain", "nausea", "mild_fever", "cramps", "malaise",
    ],
    "Acute gastritis": [
        "stomach_pain", "nausea", "indigestion", "acidity", "loss_of_appetite", "vomiting",
    ],
    "Constipation": [
        "constipation", "hard_stools", "straining_to_pass_stool", "abdominal_pain", "indigestion",
    ],

    # ---- general -----------------------------------------------------------
    "Dehydration": [
        "thirst", "dry_mouth", "dark_urine", "dizziness", "fatigue", "headache", "lethargy",
    ],
    "Heat exhaustion": [
        "sweating", "dizziness", "nausea", "headache", "fatigue", "cramps", "thirst",
    ],
    "Insect bite reaction": [
        "skin_redness_local", "itching", "swelling_at_injury", "skin_rash", "tenderness_to_touch",
    ],
    "Anxiety": [
        "anxiety", "restlessness", "fast_heart_rate", "breathlessness", "sweating",
        "difficulty_sleeping", "irritability",
    ],
    "Insomnia": [
        "difficulty_sleeping", "fatigue", "irritability", "lethargy", "depression",
    ],
    "Menstrual cramps": [
        "lower_abdominal_cramping", "cramps", "back_pain", "nausea", "headache", "fatigue",
    ],
}

SOURCE = "Curated from standard clinical presentations"


def main() -> None:
    symptoms = sorted({s for v in CURATED.values() for s in v})
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with OUT.open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(symptoms + ["condition"])
        for cond, syms in CURATED.items():
            w.writerow([1 if s in syms else 0 for s in symptoms] + [cond])
    print(f"wrote {OUT.relative_to(OUT.parents[2])}")
    print(f"  {len(CURATED)} conditions · {len(symptoms)} symptoms")
    print(f"  mean symptoms per condition: {sum(len(v) for v in CURATED.values())/len(CURATED):.1f}")
    print(f"  source: {SOURCE}")


if __name__ == "__main__":
    main()
