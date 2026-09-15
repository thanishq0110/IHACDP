"""Select the SymCat conditions a self-triage tool should actually know.

All 801 is the wrong set: a head cold should not have to out-rank 800 rare
diagnoses, and adopting the lot dropped top-3 recovery from 61% to 18%. This
keeps everyday presentations, plus a short list of conditions that are not
everyday but are dangerous to miss.
"""
from __future__ import annotations
import re
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "data" / "raw" / "symcat.csv"
OUT = ROOT / "data" / "raw" / "symcat_keep.csv"

EVERYDAY = re.compile(r"""(
 sprain|strain|fracture|dislocat|concussion|burn|contusion|open wound|laceration|
 injury|whiplash|tendin|bursitis|sciatica|lumbago|back pain|neck pain|
 osteoarthritis|arthritis|gout|carpal tunnel|adhesive capsulitis|
 headache|migraine|
 otitis|sinusitis|tonsill|pharyng|laryngitis|common cold|influenza|croup|bronchitis|
 pneumonia|asthma|allerg|hay fever|
 conjunctivitis|stye|chalazion|corneal abrasion|
 tooth|dental|gingivitis|
 gastroenteritis|gastritis|reflux|peptic ulcer|constipation|hemorrhoid|haemorrhoid|
 diarrhea|food|indigest|irritable bowel|
 urinary tract|cystitis|kidney stone|
 anxiety|depress|insomnia|panic|
 menstrua|dysmenorrhea|vaginitis|
 dermatitis|eczema|psoriasis|acne|urticaria|hives|cellulitis|abscess|boil|
 fungal|ringworm|scabies|lice|wart|
 anemia|anaemia|dehydrat|heat exhaust|heat stroke|fainting|
 diabet|hypertens|thyroid|obesity|
 chickenpox|measles|mumps|shingles|herpes|
 fever|vertigo|tinnitus|
 sore throat|cough|nosebleed|insect bite|sunburn
)""", re.I | re.X)

# Not everyday, but missing them would be the failure that matters.
CANNOT_MISS = re.compile(r"""(
 appendicitis|meningitis|glaucoma|pulmonary embolism|deep vein thrombosis|
 heart attack|myocardial|stroke|transient ischemic|sepsis|
 ectopic pregnancy|testicular torsion|anaphylaxis|
 pyelonephritis|cholecystitis|pancreatitis|diverticulitis|
 bowel obstruction|gastrointestinal hemorrhage|subarachnoid
)""", re.I | re.X)


def main() -> None:
    e = pd.read_csv(SRC)
    conds = sorted(e["condition"].unique())
    keep, why = [], {}
    for c in conds:
        if EVERYDAY.search(c):
            keep.append(c); why[c] = "everyday"
        elif CANNOT_MISS.search(c):
            keep.append(c); why[c] = "cannot-miss"
    pd.DataFrame({"condition": keep, "reason": [why[c] for c in keep]}).to_csv(OUT, index=False)
    n_ev = sum(1 for c in keep if why[c] == "everyday")
    print(f"[filter] {len(conds)} SymCat conditions -> {len(keep)} kept "
          f"({n_ev} everyday, {len(keep)-n_ev} cannot-miss), {len(conds)-len(keep)} dropped")
    print("  cannot-miss kept:", ", ".join(c for c in keep if why[c] == "cannot-miss")[:300])


if __name__ == "__main__":
    main()
