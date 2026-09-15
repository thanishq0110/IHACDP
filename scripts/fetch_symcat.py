"""SymCat: 801 conditions with P(symptom | condition).

The file nests seven levels deep as selection2_..._selection1_* column families.
Reading only the first level silently yields 50 conditions instead of 801, so
every level ending in _symptoms_disorder is unioned.

Emits a long-format edge list: condition, symptom, probability.
"""
from __future__ import annotations
import io, urllib.request
from pathlib import Path
import pandas as pd

URL = ("https://raw.githubusercontent.com/teliov/symcat-to-synthea/"
       "master/symcat/symcat-801-diseases.csv")
OUT = Path(__file__).resolve().parents[1] / "data" / "raw" / "symcat.csv"


def normalise(name: str) -> str:
    """SymCat symptom names -> the snake_case vocabulary used elsewhere."""
    return (name.strip().lower()
            .replace("'", "").replace("(", "").replace(")", "")
            .replace("/", " ").replace("-", " ")
            .replace(",", " ").split() and
            "_".join(name.strip().lower().replace("'", "").replace("(", "")
                     .replace(")", "").replace("/", " ").replace("-", " ")
                     .replace(",", " ").split()))


def fetch() -> pd.DataFrame:
    req = urllib.request.Request(URL, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=180) as r:
        raw = r.read()
    print(f"    downloaded {len(raw)/1e6:.1f} MB")
    return pd.read_csv(io.BytesIO(raw), low_memory=False)


def parse(df: pd.DataFrame) -> pd.DataFrame:
    prefixes = sorted({c.split("_symptoms")[0] for c in df.columns
                       if c.endswith("_symptoms_disorder")})
    print(f"    {len(prefixes)} nesting levels")
    rows: dict[tuple[str, str], float | None] = {}
    for p in prefixes:
        d, s, pr = f"{p}_symptoms_disorder", f"{p}_symptoms_name", f"{p}_symptoms_probability"
        if not {d, s}.issubset(df.columns):
            continue
        cols = [d, s] + ([pr] if pr in df.columns else [])
        sub = df[cols].dropna(subset=[d, s])
        for t in sub.itertuples(index=False):
            cond = str(t[0]).strip()
            sym = normalise(str(t[1]))
            prob = float(t[2]) if len(cols) > 2 and pd.notna(t[2]) else None
            if not cond or not sym:
                continue
            prev = rows.get((cond, sym))
            rows[(cond, sym)] = prob if prev is None else max(prev, prob or 0)
    out = pd.DataFrame(
        [{"condition": c, "symptom": s, "probability": p} for (c, s), p in rows.items()]
    )
    return out.sort_values(["condition", "symptom"]).reset_index(drop=True)


if __name__ == "__main__":
    print("[symcat] fetching ...")
    edges = parse(fetch())
    OUT.parent.mkdir(parents=True, exist_ok=True)
    edges.to_csv(OUT, index=False)
    print(f"  -> {OUT.name}  edges={len(edges):,}  "
          f"conditions={edges['condition'].nunique()}  symptoms={edges['symptom'].nunique()}")
    if edges["probability"].notna().any():
        print(f"     probability present on {edges['probability'].notna().mean():.0%} of edges")
