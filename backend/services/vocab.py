"""One symptom vocabulary across three sources.

SymCat, the public matrix and the curated patterns each name symptoms
differently. Unaliased names survive as their own columns; these aliases fold
the overlapping ones together so a patient saying "tummy pain" can reach a
SymCat condition filed under sharp_abdominal_pain.
"""
from __future__ import annotations

ALIASES: dict[str, str] = {
    # abdomen
    "sharp_abdominal_pain": "stomach_pain",
    "burning_abdominal_pain": "stomach_pain",
    "lower_abdominal_pain": "abdominal_pain",
    "upper_abdominal_pain": "abdominal_pain",
    "pain_of_the_anus": "abdominal_pain",
    "stomach_bloating": "indigestion",
    "heartburn": "acidity",
    "regurgitation": "acidity",
    "diarrhea": "diarrhoea",
    "constipation": "constipation",
    # general
    "fever": "mild_fever",
    "chills": "chills",
    "ache_all_over": "muscle_pain",
    "shortness_of_breath": "breathlessness",
    "difficulty_breathing": "breathlessness",
    "sharp_chest_pain": "chest_pain",
    "burning_chest_pain": "chest_pain",
    "weakness": "fatigue",
    "decreased_appetite": "loss_of_appetite",
    "weight_loss": "weight_loss",
    "excessive_sweating": "sweating",
    "fainting": "dizziness",
    "vertigo": "dizziness",
    # head, eyes, ears, nose, throat
    "sore_throat": "throat_irritation",
    "difficulty_in_swallowing": "difficulty_swallowing",
    "painful_swallowing": "difficulty_swallowing",
    "nasal_congestion": "congestion",
    "sinus_congestion": "sinus_pressure",
    "plugged_feeling_in_ear": "hearing_muffled",
    "diminished_hearing": "hearing_muffled",
    "ringing_in_ear": "hearing_muffled",
    "pain_in_eye": "eye_pain",
    "diminished_vision": "blurred_and_distorted_vision",
    "spots_or_clouds_in_vision": "blurred_and_distorted_vision",
    "eye_redness": "redness_of_eyes",
    "itchiness_of_eye": "eye_gritty",
    "lacrimation": "watering_from_eyes",
    "toothache": "tooth_pain",
    "mouth_pain": "tooth_pain",
    "gum_pain": "gum_swelling",
    # musculoskeletal and injury
    "problems_with_movement": "limited_movement",
    "joint_stiffness_or_tightness": "movement_stiffness",
    "joint_pain": "joint_pain",
    "leg_pain": "muscle_pain",
    "arm_pain": "muscle_pain",
    "shoulder_pain": "joint_pain",
    "knee_pain": "knee_pain",
    "ankle_pain": "joint_pain",
    "wrist_pain": "joint_pain",
    "hip_pain": "hip_joint_pain",
    "low_back_pain": "back_pain",
    "neck_pain": "neck_pain",
    "muscle_pain": "muscle_pain",
    "muscle_stiffness_or_tightness": "movement_stiffness",
    "swollen_or_red_tonsils": "throat_irritation",
    "skin_swelling": "swelling_at_injury",
    "joint_swelling": "swelling_joints",
    "bones_are_painful": "severe_pain",
    "loss_of_sensation": "numbness_tingling",
    "paresthesia": "numbness_tingling",
    "weakness_of_the_arm_or_leg": "weakness_in_limbs",
    "back_cramps_or_spasms": "cramps",
    "muscle_cramps_contractures_or_spasms": "cramps",
    # skin
    "itching_of_skin": "itching",
    "abnormal_appearing_skin": "skin_rash",
    "skin_lesion": "skin_rash",
    "skin_irritation": "skin_redness_local",
    "skin_dryness_peeling_scaliness_or_roughness": "skin_rash",
    "bleeding_or_discharge_from_nipple": "skin_rash",
    "skin_on_leg_or_foot_looks_infected": "skin_redness_local",
    # mind
    "anxiety_and_nervousness": "anxiety",
    "depressive_or_psychotic_symptoms": "depression",
    "depression": "depression",
    "insomnia": "difficulty_sleeping",
    "sleepiness": "lethargy",
    "restlessness": "restlessness",
    "excessive_anger": "irritability",
    "temper_problems": "irritability",
    "low_self_esteem": "depression",
    # urinary
    "painful_urination": "burning_micturition",
    "frequent_urination": "burning_micturition",
    "involuntary_urination": "burning_micturition",
    "retention_of_urine": "burning_micturition",
    "blood_in_urine": "dark_urine",
}


def canonical(symptom: str) -> str:
    """Fold a source-specific symptom name onto the shared vocabulary."""
    s = symptom.strip().lower()
    return ALIASES.get(s, s)
