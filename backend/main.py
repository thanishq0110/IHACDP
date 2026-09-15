"""IHACDP API. Stateless by design: the client holds the transcript and record,
so no patient data is ever persisted server-side."""
from __future__ import annotations
import json
from typing import Any

from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse, FileResponse, JSONResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from backend.config import APP_NAME, APP_FULL, VERSION, FRONTEND, LLM_MODEL
from backend.services import consultation, llm, clinical
from backend.services.predictor import engine

@asynccontextmanager
async def lifespan(_app: FastAPI):
    engine()
    print(f"[{APP_NAME}] ready - model={LLM_MODEL}")
    yield


app = FastAPI(title=APP_FULL, version=VERSION, docs_url="/api/docs",
              openapi_url="/api/openapi.json", lifespan=lifespan)


class Msg(BaseModel):
    role: str
    content: str


class ChatIn(BaseModel):
    messages: list[Msg] = Field(default_factory=list)
    record: dict[str, Any] = Field(default_factory=dict)


class AssessIn(BaseModel):
    record: dict[str, Any] = Field(default_factory=dict)


class ReportIn(BaseModel):
    record: dict[str, Any] = Field(default_factory=dict)
    messages: list[Msg] = Field(default_factory=list)


def sse(event: dict) -> str:
    return f"data: {json.dumps(event, default=str)}\n\n"


@app.get("/api/health")
async def health():
    h = await llm.health()
    return {"app": APP_NAME, "version": VERSION, "llm": h,
            "models_loaded": list(engine().models.keys()), "offline": True}


@app.get("/api/diseases")
async def diseases():
    out = []
    for m in engine().catalogue():
        out.append({k: m[k] for k in ("key", "label", "desc", "source", "rows", "n_features",
                                      "prevalence", "chosen_algorithm", "metrics", "leaderboard",
                                      "roc", "confusion")})
    return out


@app.get("/api/fields")
async def fields():
    return {"fields": clinical.FIELDS, "key_fields": clinical.KEY_FIELDS}


@app.post("/api/chat")
async def chat(body: ChatIn):
    history = [m.model_dump() for m in body.messages]
    last_user = next((m["content"] for m in reversed(history) if m["role"] == "user"), "")
    prior = [m for m in history[:-1] if m["role"] == "assistant"] if history else []
    last_question = prior[-1]["content"] if prior else ""
    record, new_fields = consultation.ingest(body.record, last_user, last_question)

    async def gen():
        yield sse({"type": "extracted", "fields": new_fields})
        acc = []
        try:
            async for tok in consultation.reply_stream(history, record):
                acc.append(tok)
                yield sse({"type": "token", "t": tok})
        except Exception as e:
            yield sse({"type": "error", "message": f"Local model unavailable ({type(e).__name__}). "
                                                   f"Ensure Ollama is running."})
            return
        clean = consultation.soften_acknowledgement(llm.strip_reasoning("".join(acc)))
        full = history + [{"role": "assistant", "content": clean}]
        ready = consultation.readiness(record, full)
        if ready.get("should_assess"):
            clean = consultation.finalise_closing(clean)
            full[-1]["content"] = clean
        yield sse({"type": "final", "text": clean, "record": record, "readiness": ready})
        yield sse({"type": "done"})

    return StreamingResponse(gen(), media_type="text/event-stream",
                             headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})


@app.post("/api/assess")
async def assess(body: AssessIn):
    if not body.record:
        raise HTTPException(400, "empty record")
    return engine().assess(body.record)


@app.post("/api/report")
async def report(body: ReportIn):
    rec = clinical.derive(body.record)
    assessment = engine().assess(rec)
    transcript = [m.model_dump() for m in body.messages]

    assessment["symptom_conditions"] = consultation.symptom_assessment(rec)

    async def gen():
        yield sse({"type": "assessment", "data": assessment})
        acc = []
        try:
            async for tok in consultation.report_stream(rec, assessment, transcript):
                acc.append(tok)
                yield sse({"type": "token", "t": tok})
        except Exception as e:
            yield sse({"type": "error", "message": f"Local model unavailable ({type(e).__name__})."})
            return
        note = consultation.strip_instruction_echo(llm.strip_reasoning("".join(acc)))
        note = consultation.scrub_note(note, rec)
        yield sse({"type": "final", "text": consultation.splice_prescription(note, rec)})
        yield sse({"type": "done"})

    return StreamingResponse(gen(), media_type="text/event-stream",
                             headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})


# ---- static frontend --------------------------------------------------------
for sub in ("css", "js", "assets"):
    d = FRONTEND / sub
    if d.exists():
        app.mount(f"/{sub}", StaticFiles(directory=str(d)), name=sub)


def _page(name: str):
    p = FRONTEND / name
    if not p.exists():
        return JSONResponse({"detail": f"{name} missing"}, status_code=404)
    return FileResponse(p)


@app.exception_handler(404)
async def not_found(request, exc):
    """API callers get JSON; browsers get the branded page."""
    if request.url.path.startswith("/api/"):
        return JSONResponse({"detail": "Not found"}, status_code=404)
    return FileResponse(FRONTEND / "error.html", status_code=404)


@app.exception_handler(500)
async def server_error(request, exc):
    if request.url.path.startswith("/api/"):
        return JSONResponse({"detail": "Internal error"}, status_code=500)
    return RedirectResponse("/error?code=500", status_code=302)


@app.get("/error")
async def error_page():
    return _page("error.html")


@app.get("/")
async def index():
    return _page("index.html")


@app.get("/about")
async def about():
    return _page("about.html")


@app.get("/consult")
async def consult():
    return _page("consult.html")


@app.get("/models")
async def models_page():
    return _page("models.html")
