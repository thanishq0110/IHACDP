"""IHACDP runtime configuration. Fully local - no external network calls at inference."""
from pathlib import Path
import os

ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS = ROOT / "artifacts"
FRONTEND = ROOT / "frontend"

OLLAMA_HOST = os.getenv("IHACDP_OLLAMA_HOST", "http://127.0.0.1:11434")
LLM_MODEL = os.getenv("IHACDP_LLM_MODEL", "gemma3:4b")  # selected by scripts/bench_llm.py
LLM_TIMEOUT = float(os.getenv("IHACDP_LLM_TIMEOUT", "180"))
LLM_NUM_CTX = int(os.getenv("IHACDP_LLM_NUM_CTX", "4096"))

APP_NAME = "IHACDP"
APP_FULL = "Intelligent Healthcare Analytics & Clinical Decision Support Platform"
VERSION = "1.0.0"

RISK_BANDS = [(0.00, 0.20, "Low"), (0.20, 0.50, "Moderate"), (0.50, 0.75, "High"), (0.75, 1.01, "Very High")]
MIN_COVERAGE = 0.45


def band(p: float) -> str:
    for lo, hi, name in RISK_BANDS:
        if lo <= p < hi:
            return name
    return "Unknown"
