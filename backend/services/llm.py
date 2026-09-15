"""Local LLM transport. Talks only to a loopback Ollama daemon - no egress."""
from __future__ import annotations
import json, re
from typing import AsyncIterator
import httpx

from backend.config import OLLAMA_HOST, LLM_MODEL, LLM_TIMEOUT, LLM_NUM_CTX

_THINK = re.compile(r"<think>.*?</think>\s*", re.S | re.I)


def strip_think(t: str) -> str:
    return _THINK.sub("", t).replace("<think>", "").replace("</think>", "").strip()


async def health() -> dict:
    try:
        async with httpx.AsyncClient(timeout=5) as c:
            r = await c.get(f"{OLLAMA_HOST}/api/tags")
            tags = [m["name"] for m in r.json().get("models", [])]
        return {"online": True, "model": LLM_MODEL, "available": tags,
                "model_ready": any(t.split(":")[0] == LLM_MODEL.split(":")[0] for t in tags)}
    except Exception as e:
        return {"online": False, "model": LLM_MODEL, "available": [], "model_ready": False,
                "error": f"{type(e).__name__}"}


async def chat_stream(messages: list[dict], temperature: float = 0.6,
                      max_tokens: int = 700) -> AsyncIterator[str]:
    payload = {
        "model": LLM_MODEL, "messages": messages, "stream": True, "think": False,
        "options": {"temperature": temperature, "num_predict": max_tokens,
                    "num_ctx": LLM_NUM_CTX, "top_p": 0.9, "repeat_penalty": 1.05},
    }
    buf, in_think = "", False
    async with httpx.AsyncClient(timeout=LLM_TIMEOUT) as c:
        async with c.stream("POST", f"{OLLAMA_HOST}/api/chat", json=payload) as r:
            r.raise_for_status()
            async for line in r.aiter_lines():
                if not line.strip():
                    continue
                try:
                    chunk = json.loads(line)
                except json.JSONDecodeError:
                    continue
                tok = chunk.get("message", {}).get("content", "")
                if not tok:
                    if chunk.get("done"):
                        break
                    continue
                buf += tok
                # suppress any <think> block the model emits despite think=False
                if "<think>" in buf and not in_think:
                    in_think = True
                    buf = buf.split("<think>", 1)[0]
                if in_think:
                    if "</think>" in buf + tok:
                        in_think = False
                        buf = ""
                    continue
                yield tok


async def complete(messages: list[dict], temperature: float = 0.2,
                   max_tokens: int = 600, as_json: bool = False) -> str:
    payload = {
        "model": LLM_MODEL, "messages": messages, "stream": False, "think": False,
        "options": {"temperature": temperature, "num_predict": max_tokens, "num_ctx": LLM_NUM_CTX},
    }
    if as_json:
        payload["format"] = "json"
    async with httpx.AsyncClient(timeout=LLM_TIMEOUT) as c:
        r = await c.post(f"{OLLAMA_HOST}/api/chat", json=payload)
        r.raise_for_status()
        return strip_think(r.json().get("message", {}).get("content", ""))


async def complete_json(messages: list[dict], fallback: dict | None = None) -> dict:
    try:
        txt = await complete(messages, as_json=True)
        m = re.search(r"\{.*\}", txt, re.S)
        return json.loads(m.group(0)) if m else (fallback or {})
    except Exception:
        return fallback or {}


_LEAK = re.compile(
    r"^\s*(we are given|we are in a scenario|the patient (said|mentioned|says)|the user (said|is|also|mentioned)|"
    r"first,? i must|i must (respond|acknowledge|not|check|ask)|response structure|let me think|"
    r"per the order|wait[,.]|okay,? so|so,? the|step \d|my task|according to the instruction|"
    r"i should ask|note:|important:|analysis:|reasoning:|thinking:|but note|however,? (note|the key))",
    re.I)
_PLAN_LINE = re.compile(r"^\s*\d+[\.\)]\s+(acknowledge|then ask|ask |check |respond|asses)", re.I)


def strip_reasoning(text: str) -> str:
    """Defensive: remove meta-reasoning a hybrid-thinking model leaks as plain content.

    Kept deliberately conservative - if stripping would remove everything, fall back
    to the trailing paragraph, which is where such models place the real reply.
    """
    t = strip_think(text)
    paras = [p.strip() for p in re.split(r"\n\s*\n", t) if p.strip()]
    kept = [p for p in paras if not _LEAK.match(p) and not _PLAN_LINE.match(p)]
    kept = [p for p in kept if not (p.startswith(("1.", "2.", "3.")) and len(p) < 90)]
    if kept:
        return _unquote("\n\n".join(kept).strip())
    for p in reversed(paras):
        if not _LEAK.match(p):
            return _unquote(p.strip())
    return _unquote(paras[-1].strip()) if paras else t.strip()


def _unquote(t: str) -> str:
    """Drop wrapping quotes when the model quotes its whole reply back."""
    for a, b in (('"', '"'), ("\u201c", "\u201d"), ("'", "'")):
        if len(t) <= 2 or not (t.startswith(a) and t.endswith(b)):
            continue
        # symmetric delimiters must appear exactly twice; asymmetric once each
        n = t.count(a) if a == b else min(t.count(a), t.count(b))
        if (a == b and n == 2) or (a != b and n == 1):
            return t[1:-1].strip()
    return t
