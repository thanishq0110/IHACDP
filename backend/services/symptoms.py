"""Everyday-complaint triage: fever, colds, headaches, aches and similar.

Separate from the chronic-disease models on purpose. Someone with a head cold
must never be asked for their creatinine, and the questions asked here are
chosen to separate the conditions still in play rather than to fill a form.
"""
from __future__ import annotations
import json, re
from pathlib import Path
from typing import Any

import numpy as np, pandas as pd, joblib

from backend.config import ARTIFACTS

# Everyday words people actually use, mapped onto the dataset's symptom columns.
SYNONYMS: dict[str, list[str]] = {
    "high_fever": ["high fever", "bad fever", "burning up", "high temperature", "fever of"],
    "mild_fever": ["mild fever", "slight fever", "low fever", "feverish", "temperature", "fever"],
    "headache": ["headache", "head ache", "head hurts", "migraine", "head is pounding"],
    "cough": ["cough", "coughing", "chesty cough", "dry cough"],
    "continuous_sneezing": ["sneezing", "sneeze", "keep sneezing"],
    "runny_nose": ["runny nose", "running nose", "nose is running", "snotty"],
    "congestion": ["blocked nose", "stuffy nose", "congested", "congestion", "bunged up"],
    "throat_irritation": ["sore throat", "scratchy throat", "throat hurts", "throat irritation"],
    "phlegm": ["phlegm", "mucus", "flem"],
    "chills": ["chills", "shivering", "shivery", "cold shivers"],
    "fatigue": ["tired", "tiredness", "exhausted", "fatigue", "no energy", "worn out"],
    "malaise": ["unwell", "run down", "rough", "malaise", "off colour", "off color"],
    "vomiting": ["vomiting", "throwing up", "being sick", "puking", "vomit"],
    "nausea": ["nausea", "nauseous", "feel sick", "queasy"],
    "diarrhoea": ["diarrhoea", "diarrhea", "loose motions", "runny stomach", "the runs"],
    "stomach_pain": ["stomach pain", "tummy pain", "belly pain", "stomach ache", "tummy ache",
                     "pain in my stomach", "pain in my tummy", "pain in my belly",
                     "my stomach hurts", "my tummy hurts", "my belly hurts",
                     "stomach is hurting", "tummy is hurting", "sore stomach", "sore tummy"],
    "abdominal_pain": ["abdominal pain", "pain in my abdomen", "gut pain", "abdomen hurts",
                       "pain in my gut", "cramping in my abdomen"],
    # The classic appendicitis localiser. Kept as its own symptom because it is
    # far more specific than generic abdominal pain, and so carries real weight.
    "right_lower_abdominal_pain": [
        "lower right tummy", "lower right abdomen", "lower right side", "lower right belly",
        "right lower tummy", "right lower abdomen", "right lower quadrant", "right iliac fossa",
        "bottom right of my tummy", "bottom right of my stomach", "right side of my lower tummy",
        "pain on my lower right", "lower right hand side of my tummy",
        # the textbook migration: periumbilical pain that settles right-iliac
        "moved to the lower right", "moved to my lower right", "moved down to the right",
        "started near my belly button and moved", "around my belly button then moved",
    ],
    "back_pain": ["back pain", "backache", "back ache", "my back hurts", "sore back", "lower back"],
    "neck_pain": ["neck pain", "stiff neck", "neck hurts", "sore neck",
                  "pain in my neck", "pain in your neck", "my neck is sore"],
    "joint_pain": ["joint pain", "joints hurt", "aching joints", "sore joints"],
    "knee_pain": ["knee pain", "knee hurts", "sore knee", "knees hurt"],
    "hip_joint_pain": ["hip pain", "hip hurts"],
    "muscle_pain": ["muscle pain", "muscle ache", "body ache", "body aches", "aching all over",
                    "leg pain", "legs hurt", "sore legs", "aching legs", "legs ache", "my legs ache",
                    "arms ache", "aches and pains"],
    "cramps": ["cramps", "cramping"],
    "swelling_joints": ["swollen joints", "joint swelling"],
    "movement_stiffness": ["stiffness", "stiff", "hard to move"],
    "dizziness": ["dizzy", "dizziness", "lightheaded", "light headed", "giddy"],
    "loss_of_appetite": ["no appetite", "lost my appetite", "not eating", "off my food"],
    "weight_loss": ["losing weight", "weight loss", "lost weight"],
    "breathlessness": ["breathless", "short of breath", "cant breathe", "can't breathe",
                       "shortness of breath", "wheezing"],
    "chest_pain": ["chest pain", "pain in my chest", "chest hurts", "chest tightness"],
    "burning_micturition": ["burning when i pee", "burning urination", "stings when i pee", "burning pee"],
    "bladder_discomfort": ["bladder discomfort", "bladder pain"],
    "itching": ["itching", "itchy", "scratching"],
    "skin_rash": ["rash", "skin rash", "spots on my skin", "red patches"],
    "sweating": ["sweating", "sweaty", "night sweats"],
    "watering_from_eyes": ["watery eyes", "eyes watering", "eyes streaming"],
    "redness_of_eyes": ["red eyes", "bloodshot", "red eye", "eye is red", "eyes are red",
                        "eye has gone red", "eye looks red"],
    "sinus_pressure": ["sinus pressure", "sinus pain", "face pressure"],
    "constipation": ["constipation", "constipated", "cant go", "can't go"],
    "acidity": ["acidity", "heartburn", "acid reflux", "burning in my chest after eating"],
    "indigestion": ["indigestion", "bloated", "bloating"],
    "yellowish_skin": ["yellow skin", "skin has gone yellow", "jaundice"],
    "yellowing_of_eyes": ["yellow eyes", "eyes have gone yellow", "whites of my eyes are yellow"],
    "dark_urine": ["dark urine", "dark pee", "urine is dark"],
    "irritability": ["irritable", "short tempered", "snappy"],
    "depression": ["depressed", "low mood", "depression"],
    "anxiety": ["anxious", "anxiety", "panicky"],
    "blurred_and_distorted_vision": ["blurred vision", "blurry vision", "vision is blurred"],
    "stiff_neck": ["stiff neck"],
    "weakness_in_limbs": ["weak arms", "weak legs", "limb weakness", "weakness in my limbs"],
    "fast_heart_rate": ["heart racing", "palpitations", "fast heartbeat", "racing heart"],
    "swelled_lymph_nodes": ["swollen glands", "swollen lymph nodes", "glands are up"],
    "mucoid_sputum": ["coughing up mucus", "mucoid sputum"],
    "restlessness": ["restless", "cant settle", "can't settle"],
    "lethargy": ["lethargic", "sluggish", "no get up and go"],

    # ---- injury and musculoskeletal ---------------------------------------
    "recent_injury": ["i fell", "had a fall", "twisted it", "twisted my", "hurt it",
                      "injured", "injury", "banged it", "knocked it", "after lifting",
                      "lifting something heavy", "playing football", "at the gym", "sprained"],
    "swelling_at_injury": ["swollen", "swelling", "puffed up", "it has swelled"],
    "bruising": ["bruise", "bruised", "bruising", "black and blue", "discoloured"],
    "pain_on_movement": ["hurts when i move", "painful to move", "hurts to move",
                         "pain when i bend", "worse when i move"],
    "tenderness_to_touch": ["tender", "sore to touch", "hurts to touch", "tender to touch"],
    "limited_movement": ["cant move it", "can't move it", "limited movement",
                         "cant lift it", "can't lift it", "restricted movement"],
    "numbness_tingling": ["numb", "numbness", "tingling", "pins and needles"],
    "pain_radiating_to_leg": ["down my leg", "shoots down my leg", "radiates down",
                              "pain goes down my leg", "shooting down"],
    "severe_pain": ["severe pain", "really bad pain", "excruciating", "unbearable pain",
                    "agony", "worst pain"],
    "one_sided_pain": ["one side", "on one side", "only the left", "only the right"],
    "deformity": ["looks deformed", "out of shape", "bent out", "looks wrong", "misshapen"],
    "night_pain": ["worse at night", "pain at night", "wakes me at night"],
    "repetitive_activity": ["repetitive", "same movement", "over and over", "typing all day"],

    # ---- head --------------------------------------------------------------
    "band_like_head_pressure": ["tight band", "band around my head", "pressure around my head",
                                "tight across my forehead", "like a vice"],
    "one_sided_head_pain": ["one side of my head", "half my head", "behind one eye"],
    "facial_pain": ["face pain", "pain in my face", "cheek pain", "face hurts",
                    "pain around my cheeks"],
    "light_sensitivity": ["light hurts", "sensitive to light", "photophobia", "bright light hurts"],
    "loss_of_smell": ["cant smell", "can't smell", "lost my sense of smell", "no sense of smell"],
    "confusion": ["confused", "muddled", "foggy", "cant think straight", "can't think straight"],

    # ---- ear, eye, mouth, throat ------------------------------------------
    "ear_pain": ["earache", "ear ache", "ear hurts", "ear pain", "pain in my ear"],
    "ear_discharge": ["ear discharge", "fluid from my ear", "runny ear"],
    "hearing_muffled": ["muffled hearing", "cant hear properly", "can't hear properly",
                        "hearing is muffled", "ear feels blocked"],
    "eye_discharge": ["sticky eye", "discharge from my eye", "gunk in my eye", "crusty eye",
                      "eye discharge", "weeping eye", "pus in my eye", "sticky with discharge",
                      "sticky and weeping"],
    "eye_gritty": ["gritty", "feels like sand", "grit in my eye"],
    "tooth_pain": ["toothache", "tooth ache", "tooth hurts", "tooth pain"],
    "gum_swelling": ["swollen gum", "gum is swollen", "swollen gums"],
    "difficulty_swallowing": ["hard to swallow", "painful to swallow", "cant swallow",
                              "can't swallow", "hurts to swallow"],

    # ---- general -----------------------------------------------------------
    "thirst": ["thirsty", "very thirsty", "really thirsty", "cant stop drinking"],
    "dry_mouth": ["dry mouth", "mouth is dry", "mouth feels dry"],
    "difficulty_sleeping": ["cant sleep", "can't sleep", "trouble sleeping", "insomnia",
                            "not sleeping", "awake all night"],
    "stress": ["stressed", "under stress", "stressful"],
    "blistering": ["blister", "blisters", "blistered"],
    "skin_redness_local": ["red skin", "skin is red", "gone red", "red patch"],
    "hard_stools": ["hard stools", "hard poo", "dry stools"],
    "straining_to_pass_stool": ["straining", "struggling to go", "pushing to go"],
    "lower_abdominal_cramping": ["period pain", "period cramps", "lower tummy cramps",
                                 "cramping in my lower tummy", "menstrual cramps"],
}

# Some conditions share only generic symptoms (swelling, recent injury, pain)
# with unrelated complaints. Without a defining feature present they are not
# candidates at all - a twisted ankle is not a burn because both swell.
GATES: dict[str, set[str]] = {
    "Minor burn": {"blistering", "skin_redness_local"},
    "Insect bite reaction": {"itching", "skin_rash", "skin_redness_local"},
    "Concussion": {"headache", "confusion", "dizziness", "blurred_and_distorted_vision"},
    "Conjunctivitis": {"redness_of_eyes", "eye_discharge", "eye_gritty"},
    "Toothache": {"tooth_pain", "gum_swelling"},
    "Otitis media": {"ear_pain", "ear_discharge", "hearing_muffled"},
    "Sciatica": {"pain_radiating_to_leg", "numbness_tingling"},
    "Menstrual cramps": {"lower_abdominal_cramping"},
    "Frozen shoulder": {"limited_movement", "night_pain"},
    "Heat exhaustion": {"sweating", "thirst", "cramps"},
    "Dehydration": {"thirst", "dry_mouth", "dark_urine"},
    "Dehydration headache": {"thirst", "dry_mouth", "dark_urine"},
    "Suspected fracture": {"deformity", "severe_pain", "limited_movement", "recent_injury"},
}

# Common things are common. Without a prevalence prior, a rare condition with a
# short symptom list wins on a single generic symptom - a bare fever was ranking
# as AIDS. These tiers are coarse on purpose; the point is ordering, not epidemiology.
COMMON = {
    "Common Cold", "Influenza", "Allergy", "Tension headache", "Migraine", "Sinusitis",
    "Acute sinusitis", "Pharyngitis", "Tonsillitis", "Acute otitis media", "Otitis media",
    "Otitis externa (swimmer's ear)", "Conjunctivitis", "Gastroenteritis", "Food poisoning",
    "Acute gastritis", "Constipation", "Chronic constipation", "Urinary tract infection",
    "Sprain or strain", "Ankle sprain", "Muscle strain", "Mechanical low back pain", "Lumbago",
    "Sciatica", "Tendonitis", "Tendinitis", "Toothache", "Tooth abscess", "Acne",
    "Anxiety", "Insomnia", "Primary insomnia", "Menstrual cramps", "Dehydration",
    "Heat exhaustion", "Insect bite reaction", "Contact dermatitis", "Eczema",
    "Fungal infection", "Asthma", "Bronchitis", "Acute bronchitis", "Hay fever",
    "Osteoarthritis", "Osteoarthristis", "Arthritis", "GERD", "Gastroesophageal reflux disease",
    "Minor burn", "Burn", "Concussion", "Suspected fracture", "Frozen shoulder",
    "Cervical spondylosis", "Chronic back pain", "Vertigo", "Dizziness",
}
# Uncommon, and missing them is survivable. Conditions that must not be missed
# are deliberately absent: a prevalence prior is a reasonable tie-breaker, but
# it had appendicitis ranking below food poisoning on a textbook presentation.
RARE = {
    "AIDS", "Tuberculosis", "Malaria", "Dengue", "Typhoid",
    "Chicken pox", "Alcoholic hepatitis", "Hepatitis B", "Hepatitis C", "Hepatitis D",
    "Hepatitis E", "hepatitis A", "Chronic cholestasis", "Psoriasis", "Impetigo",
    "Heat stroke", "Crushing injury",
}
PRIOR_COMMON, PRIOR_DEFAULT, PRIOR_RARE = 1.35, 1.0, 0.55


def prior_for(condition: str) -> float:
    if condition in COMMON:
        return PRIOR_COMMON
    if condition in RARE:
        return PRIOR_RARE
    return PRIOR_DEFAULT


_URGENT_DEFAULT = {"Heart attack", "Paralysis (brain hemorrhage)", "Pneumonia", "Tuberculosis",
                   "Malaria", "Dengue", "Typhoid", "AIDS"}


class SymptomEngine:
    def __init__(self):
        bundle = joblib.load(ARTIFACTS / "symptoms_model.joblib")
        self.model = bundle["model"]
        self.symptoms: list[str] = bundle["symptoms"]
        self.meta = json.loads((ARTIFACTS / "symptoms_meta.json").read_text())
        self.matrix = pd.read_csv(ARTIFACTS / "symptoms_matrix.csv", index_col=0)
        self.matrix = self.matrix.reindex(columns=self.symptoms, fill_value=0)
        # One source of truth: anything that must be assessed rather than
        # self-treated is also flagged urgent. The trained metadata predates the
        # cannot-miss list, so appendicitis was never flagged at all.
        from backend.services.prescribing import NO_SELF_TREATMENT
        self.urgent = set(self.meta.get("urgent") or _URGENT_DEFAULT) | NO_SELF_TREATMENT
        self._patterns = self._build_patterns()

    def _build_patterns(self) -> list[tuple[re.Pattern, str]]:
        pats: list[tuple[re.Pattern, str]] = []
        for s in self.symptoms:
            phrases = list(SYNONYMS.get(s, []))
            plain = s.replace("_", " ").strip()
            if plain not in phrases:
                phrases.append(plain)
            for ph in phrases:
                if len(ph) < 4:
                    continue
                # people write "backpain" and "back-pain" as often as "back pain"
                body = r"[\s\-]?".join(re.escape(w) for w in ph.split())
                pats.append((re.compile(rf"(?<!\w){body}(?!\w)", re.I), s))
        # longest phrase first so "high fever" wins over "fever"
        pats.sort(key=lambda t: -len(t[0].pattern))
        return pats

    def label(self, symptom: str) -> str:
        return symptom.replace("_", " ").replace("  ", " ").strip()

    def extract(self, text: str) -> dict[str, bool]:
        """Symptoms named in free text, with negation handling.

        Matched text is consumed so a longer phrase wins outright: "bad fever"
        registers high_fever and cannot also register mild_fever via "fever".
        """
        found: dict[str, bool] = {}
        t = " " + text.lower() + " "
        for pat, sym in self._patterns:
            m = pat.search(t)
            if not m or sym in found:
                continue
            window = t[max(0, m.start() - 28):m.start()]
            negated = re.search(r"\b(no|not|never|without|don'?t have|haven'?t|denies)\b[^.,;]{0,24}$",
                                window, re.I)
            found[sym] = not bool(negated)
            t = t[:m.start()] + " " * (m.end() - m.start()) + t[m.end():]
        return found

    def vector(self, confirmed: dict[str, bool]) -> pd.DataFrame:
        row = {s: int(bool(confirmed.get(s, False))) for s in self.symptoms}
        return pd.DataFrame([row], columns=self.symptoms)

    def _score(self, confirmed: dict[str, bool]) -> pd.Series:
        """Weighted overlap with what was reported. The single source of ranking,
        so the questions asked narrow the same set the patient is shown."""
        yes = [s for s, v in confirmed.items() if v and s in self.matrix.columns]
        no = [s for s, v in confirmed.items() if not v and s in self.matrix.columns]
        if not yes:
            return pd.Series(0.0, index=self.matrix.index)
        idf = self._idf()
        total = float(idf[yes].sum()) or 1.0
        score = self.matrix[yes].mul(idf[yes], axis=1).sum(axis=1) / total
        if no:
            score = score - 0.4 * self.matrix[no].mul(idf[no], axis=1).sum(axis=1) / float(idf[no].sum() or 1.0)
        coverage = self.matrix[yes].sum(axis=1) / self.matrix.sum(axis=1).clip(lower=1)
        score = (score + 0.15 * coverage).clip(lower=0.0)
        score = score * pd.Series({c: prior_for(str(c)) for c in score.index})
        # a condition lacking its defining feature is not in the running at all
        for cond, gate in GATES.items():
            if cond in score.index and not (gate & set(yes)):
                score[cond] = 0.0
        return score

    def candidates(self, confirmed: dict[str, bool], limit: int = 8) -> list[str]:
        """Conditions still consistent with what has been confirmed and denied."""
        return list(self._score(confirmed).sort_values(ascending=False).head(limit).index)

    def next_symptom(self, confirmed: dict[str, bool], asked: set[str]) -> str | None:
        """The unasked symptom worth asking about next.

        Pure information gain picks technically excellent but bizarre questions
        ("any skin rash?" for back pain). A clinician confirms the leading
        explanation first, so symptoms belonging to the front-runner are
        preferred, and only then the best general discriminator.
        """
        cands = self.candidates(confirmed, limit=6)
        if len(cands) <= 1:
            return None
        sub = self.matrix.loc[cands]
        leader = sub.loc[cands[0]]

        # Fall back only to symptoms belonging to a serious contender. Scanning
        # the whole vocabulary produces technically-optimal but absurd questions
        # (asking about chest pain to narrow down back pain).
        contenders = sub.loc[cands[:3]]
        relevant = {s for s in self.symptoms if contenders[s].max() == 1}

        best_lead, best_lead_gap = None, 1.1
        best_any, best_any_gap = None, 1.1
        for s in self.symptoms:
            if s in confirmed or s in asked or s not in relevant:
                continue
            share = float(sub[s].mean())
            if share in (0.0, 1.0):
                continue
            gap = abs(share - 0.5)
            if leader[s] == 1 and gap < best_lead_gap:
                best_lead, best_lead_gap = s, gap
            if gap < best_any_gap:
                best_any, best_any_gap = s, gap
        return best_lead or best_any

    def _idf(self) -> pd.Series:
        """A symptom shared by many conditions tells us little; a rare one a lot."""
        n = len(self.matrix)
        freq = self.matrix.sum(axis=0).clip(lower=1)
        return np.log(n / freq) + 1.0

    def predict(self, confirmed: dict[str, bool], top: int = 3) -> list[dict[str, Any]]:
        """Rank conditions by weighted overlap with the symptoms actually reported.

        A symptom the patient has not mentioned is unknown, not absent. Treating
        silence as denial (as a naive-Bayes fit over this matrix does) collapses
        onto whichever condition has the fewest symptoms, which is how a head
        cold ends up ranked as something far more serious.
        """
        yes = [s for s, v in confirmed.items() if v and s in self.matrix.columns]
        no = [s for s, v in confirmed.items() if not v and s in self.matrix.columns]
        if not yes:
            return []
        # One or two symptoms cannot separate 41 conditions. Offering a long
        # list there reads as noise and can alarm someone over an incidental
        # overlap, so the shortlist is capped to the evidence actually given.
        top = min(top, 1 if len(yes) < 2 else (2 if len(yes) < 3 else 3))
        score = self._score(confirmed)
        # normalise over a fixed field of five, never over the shortened list,
        # or a single displayed candidate would always read as 100%
        field = score.sort_values(ascending=False).head(5)
        # A single common symptom leaves the whole field in a dead heat -
        # "headache" separated the top five by 0.1%, and naming a winner there
        # invents precision. Two close leaders well ahead of the rest is a real
        # finding though (sprain vs fracture), so the comparison is against the
        # field, not against the runner-up.
        if len(field) >= 4 and float(field.iloc[0]) > 0:
            spread = (float(field.iloc[0]) - float(field.iloc[3])) / float(field.iloc[0])
            if spread < 0.05:
                return []
        # Reserve mass for everything we have not asked about. Without it, a
        # complaint that maps cleanly onto one condition reads as 100% certain
        # off two symptoms, which it plainly is not.
        unexplained = 0.75 if len(yes) < 3 else (0.4 if len(yes) < 5 else 0.2)
        denom = (float(field.sum()) + unexplained) or 1.0
        out = []
        for cond, raw in field.head(top).items():
            if float(raw) / denom < 0.03:
                continue        # a near-zero match is noise, not a candidate

            cond = str(cond)
            matched = [s for s in yes if self.matrix.loc[cond, s] == 1]
            share = float(raw) / denom
            # a serious condition sharing one or two generic symptoms is not a
            # reason to alarm someone; require a genuinely strong match
            # A cannot-miss condition leading the shortlist is warning enough.
            # Requiring a strong match as well meant appendicitis, which rarely
            # scores high against 244 conditions, was never flagged at all.
            leads = len(out) == 0
            flag_urgent = cond in self.urgent and (
                (share >= 0.45 and len(matched) >= 3) or (leads and len(matched) >= 3))
            out.append({
                "condition": cond,
                "probability": round(share, 4),
                "confidence": "low" if len(yes) < 3 else ("moderate" if len(yes) < 5 else "reasonable"),
                "urgent": flag_urgent,
                "matched_symptoms": [self.label(s) for s in matched],
                "typical_symptoms": [self.label(s) for s in self.matrix.columns[self.matrix.loc[cond] == 1]][:10],
            })
        return out


_engine: SymptomEngine | None = None


def engine() -> SymptomEngine:
    global _engine
    if _engine is None:
        _engine = SymptomEngine()
    return _engine
