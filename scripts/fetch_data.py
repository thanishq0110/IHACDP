"""IHACDP - dataset ingestion. Pulls 5 disease datasets, normalises to CSV in data/raw/."""
from __future__ import annotations
import io, sys, warnings, urllib.request
from pathlib import Path
import pandas as pd

warnings.filterwarnings("ignore")
RAW = Path(__file__).resolve().parents[1] / "data" / "raw"
RAW.mkdir(parents=True, exist_ok=True)


def _uci(uid: int) -> pd.DataFrame:
    from ucimlrepo import fetch_ucirepo
    ds = fetch_ucirepo(id=uid)
    X, y = ds.data.features, ds.data.targets
    df = pd.concat([X, y], axis=1)
    df.columns = [str(c).strip() for c in df.columns]
    return df


def _http_csv(urls: list[str]) -> pd.DataFrame | None:
    for u in urls:
        try:
            req = urllib.request.Request(u, headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(req, timeout=45) as r:
                raw = r.read()
            df = pd.read_csv(io.BytesIO(raw))
            if len(df) > 50:
                print(f"    source ok: {u.split('/')[-1]}")
                return df
        except Exception as e:
            print(f"    miss ({type(e).__name__}): {u[:70]}")
    return None


def diabetes() -> pd.DataFrame:
    df = _uci(891)
    df = df.rename(columns={"Diabetes_binary": "target"})
    if "target" not in df.columns:
        df = df.rename(columns={df.columns[-1]: "target"})
    if len(df) > 80000:
        parts = [g.sample(min(len(g), 40000), random_state=42) for _, g in df.groupby("target")]
        df = pd.concat(parts, axis=0)
    return df.reset_index(drop=True)


def heart() -> pd.DataFrame:
    df = _uci(45)
    df["target"] = (pd.to_numeric(df["num"], errors="coerce").fillna(0) > 0).astype(int)
    return df.drop(columns=["num"])


def kidney() -> pd.DataFrame:
    df = _uci(336)
    tcol = [c for c in df.columns if c.lower() in ("class", "classification")][0]
    t = df[tcol].astype(str).str.strip().str.lower()
    df["target"] = (t == "ckd").astype(int)
    df = df.drop(columns=[tcol])
    return df.replace({"?": None, "\t?": None})


def liver() -> pd.DataFrame:
    df = _uci(225)
    tcol = [c for c in df.columns if "select" in c.lower()][-1]
    df["target"] = (pd.to_numeric(df[tcol], errors="coerce") == 1).astype(int)
    return df.drop(columns=[tcol])


def stroke() -> pd.DataFrame | None:
    df = _http_csv([
        "https://raw.githubusercontent.com/AlexTeboul/healthcare-dataset-stroke-data/main/healthcare-dataset-stroke-data.csv",
        "https://raw.githubusercontent.com/Chiranjeevi-Thanikonda/Stroke-Prediction/main/healthcare-dataset-stroke-data.csv",
        "https://raw.githubusercontent.com/plotly/datasets/master/healthcare-dataset-stroke-data.csv",
        "https://raw.githubusercontent.com/sagarnildass/stroke_prediction/master/healthcare-dataset-stroke-data.csv",
    ])
    if df is None:
        return None
    df.columns = [c.strip() for c in df.columns]
    df = df.rename(columns={"stroke": "target"})
    return df.drop(columns=[c for c in df.columns if c.lower() == "id"], errors="ignore")


def symptoms() -> pd.DataFrame | None:
    """Symptom -> condition patterns for common, everyday complaints.

    The published file repeats each pattern ~120 times; duplicates are dropped
    here so nothing downstream reports an inflated accuracy.
    """
    df = _http_csv([
        "https://raw.githubusercontent.com/itachi9604/healthcare-chatbot/master/Data/Training.csv",
        "https://raw.githubusercontent.com/anujdutt9/Disease-Prediction-from-Symptoms/master/dataset/training_data.csv",
    ])
    if df is None:
        return None
    df = df.loc[:, ~df.columns.str.startswith("Unnamed")]
    tcol = "prognosis" if "prognosis" in df.columns else df.columns[-1]
    df = df.rename(columns={tcol: "condition"})
    df["condition"] = df["condition"].astype(str).str.strip()
    before = len(df)
    df = df.drop_duplicates().reset_index(drop=True)
    print(f"    de-duplicated {before} -> {len(df)} unique symptom patterns")
    return df


def breast() -> pd.DataFrame:
    """Guaranteed UCI fallback if the stroke mirror is unreachable."""
    df = _uci(17)
    tcol = [c for c in df.columns if c.lower() in ("diagnosis", "class")][0]
    df["target"] = (df[tcol].astype(str).str.strip().str.upper() == "M").astype(int)
    return df.drop(columns=[tcol])


LOADERS = {"diabetes": diabetes, "heart": heart, "kidney": kidney, "liver": liver,
           "breast": breast, "symptoms": symptoms}

if __name__ == "__main__":
    ok, failed = [], []
    for name, fn in LOADERS.items():
        print(f"[fetch] {name} ...")
        try:
            df = fn()
            if df is None:
                raise RuntimeError("no reachable source")
            df.to_csv(RAW / f"{name}.csv", index=False)
            if "target" in df.columns:
                print(f"  -> {name}.csv  rows={len(df):,}  cols={df.shape[1]}  pos_rate={df['target'].mean():.1%}")
            else:
                print(f"  -> {name}.csv  rows={len(df):,}  cols={df.shape[1]}  classes={df['condition'].nunique()}")
            ok.append(name)
        except Exception as e:
            print(f"  !! {name} FAILED: {type(e).__name__}: {e}")
            failed.append(name)

    print(f"\nSUMMARY ok={ok} failed={failed}")
    sys.exit(0 if len(ok) >= 4 else 1)
