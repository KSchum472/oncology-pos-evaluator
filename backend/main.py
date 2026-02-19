"""
Oncology PoS Evaluator – Production Backend
Stack: FastAPI · sentence-transformers · numpy · Anthropic SDK

Architecture:
  POST /api/chat       → RAG retrieval + Claude API → streaming response
  POST /api/ingest     → Add new knowledge base entry + embed
  GET  /api/kb         → List all knowledge base entries
  GET  /api/kb/search  → Semantic search
  POST /api/feedback   → User confirms/corrects PoS scores (learning loop)
"""

import os, json, time, hashlib
from pathlib import Path
from typing import Optional
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import numpy as np
import anthropic
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded

# ── Optional: sentence-transformers for local embeddings
# If not installed, falls back to keyword search
try:
    from sentence_transformers import SentenceTransformer
    EMBED_MODEL = SentenceTransformer("all-MiniLM-L6-v2")   # 80MB, fast, good quality
    HAS_EMBEDDINGS = True
except ImportError:
    HAS_EMBEDDINGS = False
    print("⚠ sentence-transformers not found. Using keyword fallback.")


# ════════════════════════════════════════════════════════════════════
# VECTOR STORE  (file-backed JSON + numpy arrays)
# In production: swap for Pinecone / Weaviate / pgvector
# ════════════════════════════════════════════════════════════════════
KB_PATH = Path("knowledge_base.json")
EMBED_PATH = Path("embeddings.npy")
META_PATH = Path("embed_meta.json")   # maps embedding row → doc id


def load_kb() -> list[dict]:
    if KB_PATH.exists():
        return json.loads(KB_PATH.read_text())
    return []


def save_kb(docs: list[dict]):
    KB_PATH.write_text(json.dumps(docs, ensure_ascii=False, indent=2))


def embed_text(text: str) -> Optional[np.ndarray]:
    if HAS_EMBEDDINGS:
        return EMBED_MODEL.encode(text, normalize_embeddings=True)
    return None


def cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b))   # already normalized


def rebuild_index(docs: list[dict]):
    """Rebuild full embedding matrix from KB."""
    if not HAS_EMBEDDINGS or not docs:
        return
    texts = [f"{d['title']} {d['content']}" for d in docs]
    matrix = EMBED_MODEL.encode(texts, normalize_embeddings=True, show_progress_bar=False)
    np.save(EMBED_PATH, matrix)
    META_PATH.write_text(json.dumps([d["id"] for d in docs]))


def semantic_search(query: str, top_k: int = 5) -> list[dict]:
    docs = load_kb()
    if not docs:
        return []

    if HAS_EMBEDDINGS and EMBED_PATH.exists() and META_PATH.exists():
        # True vector similarity search
        q_vec = embed_text(query)
        matrix = np.load(EMBED_PATH)
        meta   = json.loads(META_PATH.read_text())
        sims   = [cosine_sim(q_vec, matrix[i]) for i in range(len(meta))]
        ranked = sorted(zip(sims, meta), reverse=True)[:top_k]
        id_set = {doc_id for sim, doc_id in ranked if sim > 0.25}
        return [d for d in docs if d["id"] in id_set]
    else:
        # Keyword fallback
        q = query.lower()
        scored = []
        for d in docs:
            score = sum(3 for tag in d.get("tags", []) if tag.replace("_", " ") in q)
            score += sum(1 for w in d["content"].lower().split() if len(w) > 4 and w in q)
            if score > 0:
                scored.append((score, d))
        return [d for _, d in sorted(scored, reverse=True)[:top_k]]


# ════════════════════════════════════════════════════════════════════
# SYSTEM PROMPT
# ════════════════════════════════════════════════════════════════════
BASE_SYSTEM = """Du bist ein hochspezialisierter Oncology PoS (Probability of Success) Algorithmus
mit automatischer RAG-Wissensdatenbank.

## PFLICHT-CHECK: Modalitätsspezifische Biomarker-Anforderung
- TKI / Signal-mAb   → Treibermutation zwingend (Überexpression allein = ROT -40%)
- ADC                → Überexpression ausreichend; DXd-Bystander = kein IHC-Cutoff (+15%)
- RLT                → PET + Internalisierung + Tumor/Organ-Ratio (ohne PET = -25%)
- BiTE / BiAb        → Beide Targets klinisch einzeln validiert
- SERD               → ESR1/acquired resistance + Companion Diagnostic
- mRNA-Vakzine       → TMB/Neoantigen-Last + CPI-Kombination
- CPI                → PD-L1 / TMB / MSI je nach Indikation

## 7 NEGATIVFALL-TYPEN
1. Falscher Biomarker (Überexpression ≠ Treiber für mAb/TKI): -40 bis -60%
2. Subgruppen-Falle ohne molekulare Validierung: -20 bis -35%
3. Pleiotropes Target: -35 bis -50%
4. Payload-Toxizität (Rova-T): -40 bis -55%
5. Klassen-Versagen (TIGIT-Muster): -30 bis -50%
6. Falsches Krankheitsmodell: -25 bis -35%
7. Endpunkt-Design-Fehler: -15 bis -25%

## PoS-Gewichtung: PTRS 40% | Klinisch 25% | OS-Proxy 20% | Kommerziell 15%

## RAG-KONTEXT
Im Nutzer-Prompt findest du automatisch abgerufene Dokumente aus der Wissensdatenbank.
Nutze diese für Benchmarks, Biomarker-Validierung und Klassen-Warnungen.

Führe den Nutzer durch 7 Schritte. Erstelle am Ende den vollständigen PoS-Report.
Antworte immer auf Deutsch.
"""


def build_system(rag_docs: list[dict]) -> str:
    if not rag_docs:
        return BASE_SYSTEM
    ctx = "\n\n## ABGERUFENE WISSENSDATENBANK\n"
    for d in rag_docs:
        ctx += f"### [{d['type'].upper()}] {d['title']} ({d['date']})\n{d['content']}\n\n"
    return BASE_SYSTEM + ctx


# ════════════════════════════════════════════════════════════════════
# MODELS
# ════════════════════════════════════════════════════════════════════
class ChatRequest(BaseModel):
    messages: list[dict]            # [{role, content}]
    session_id: Optional[str] = None


class IngestRequest(BaseModel):
    title: str
    content: str
    type: str                       # approval | phase3_failure | pipeline | negative_case
    date: str                       # YYYY-MM-DD
    source: str
    tags: list[str]


class FeedbackRequest(BaseModel):
    session_id: str
    candidate: str
    pos_predicted: int
    pos_actual: Optional[int] = None   # filled later when outcome known
    flags_confirmed: list[str] = []
    flags_rejected: list[str] = []
    notes: Optional[str] = None


# ════════════════════════════════════════════════════════════════════
# FEEDBACK STORE  (enables future fine-tuning / prompt calibration)
# ════════════════════════════════════════════════════════════════════
FEEDBACK_PATH = Path("feedback_log.jsonl")


def log_feedback(fb: FeedbackRequest):
    entry = fb.dict()
    entry["timestamp"] = time.time()
    with open(FEEDBACK_PATH, "a") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")


# ════════════════════════════════════════════════════════════════════
# APP
# ════════════════════════════════════════════════════════════════════
app = FastAPI(title="Oncology PoS Evaluator API", version="2.0.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"],
                   allow_methods=["*"], allow_headers=["*"])

# Rate limiting
limiter = Limiter(key_func=get_remote_address)
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])


@app.get("/api/health")
def health():
    return {"status": "ok", "kb_size": len(load_kb()),
            "embeddings": HAS_EMBEDDINGS, "model": "claude-sonnet-4-5-20250929"}


@app.get("/api/kb")
def list_kb():
    return {"documents": load_kb(), "total": len(load_kb())}


@app.get("/api/kb/search")
def search_kb(q: str, top_k: int = 5):
    results = semantic_search(q, top_k)
    return {"query": q, "results": results, "count": len(results)}


@app.post("/api/kb/ingest")
def ingest(req: IngestRequest):
    docs = load_kb()
    doc_id = hashlib.md5(
        f"{req.title}{req.date}".encode()).hexdigest()[:8]
    doc = {"id": f"kb_{doc_id}", **req.dict()}

    # Avoid duplicates
    if any(d["id"] == doc["id"] for d in docs):
        raise HTTPException(400, "Document already exists")

    docs.append(doc)
    save_kb(docs)
    rebuild_index(docs)   # Re-embed entire KB (fast for <10k docs)
    return {"status": "ingested", "id": doc["id"],
            "kb_size": len(docs), "embeddings_rebuilt": HAS_EMBEDDINGS}


@app.post("/api/kb/ingest/bulk")
def bulk_ingest(items: list[IngestRequest]):
    docs = load_kb()
    added = 0
    for req in items:
        doc_id = hashlib.md5(f"{req.title}{req.date}".encode()).hexdigest()[:8]
        doc = {"id": f"kb_{doc_id}", **req.dict()}
        if not any(d["id"] == doc["id"] for d in docs):
            docs.append(doc)
            added += 1
    save_kb(docs)
    rebuild_index(docs)
    return {"added": added, "total": len(docs)}


@app.post("/api/chat")
@limiter.limit("10/day")  # Max 10 Bewertungen pro Tag pro IP
def chat(req: ChatRequest, request: Request):
    """Streaming RAG + Claude response."""
    # Build search query from last 3 messages
    query_context = " ".join(
        m["content"] for m in req.messages[-3:] if m["role"] == "user"
    )
    rag_docs = semantic_search(query_context, top_k=4)
    system = build_system(rag_docs)

    def generate():
        # Send RAG metadata first
        yield f"data: {json.dumps({'type': 'rag_docs', 'docs': [{'id':d['id'],'title':d['title'],'type':d['type']} for d in rag_docs]})}\n\n"

        # Stream Claude response
        with client.messages.stream(
            model="claude-sonnet-4-5-20250929",
            max_tokens=1800,
            system=system,
            messages=req.messages,
        ) as stream:
            for text in stream.text_stream:
                yield f"data: {json.dumps({'type': 'text', 'delta': text})}\n\n"

        yield f"data: {json.dumps({'type': 'done'})}\n\n"

    return StreamingResponse(generate(), media_type="text/event-stream")


@app.post("/api/feedback")
def feedback(req: FeedbackRequest):
    log_feedback(req)
    return {"status": "logged", "message": "Feedback gespeichert. Wird für nächste Algorithmus-Kalibrierung verwendet."}


@app.get("/api/feedback/stats")
def feedback_stats():
    if not FEEDBACK_PATH.exists():
        return {"total": 0, "entries": []}
    entries = [json.loads(l) for l in FEEDBACK_PATH.read_text().splitlines() if l.strip()]
    return {"total": len(entries), "recent": entries[-5:]}


@app.post("/api/calibrate")
def calibrate(tier: int = 1, apply: bool = False):
    """
    Run self-learning calibration.
    
    tier=1: Prompt adjustment (simple)
    tier=2: Bayesian parameter update (statistical)
    tier=3: Fine-tuning dataset prep (advanced)
    """
    import subprocess
    cmd = ["python", "self_learning.py", f"--tier={tier}"]
    if apply:
        cmd.append("--apply")
    else:
        cmd.append("--analyze")
    
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    return {
        "tier": tier,
        "applied": apply,
        "stdout": result.stdout,
        "stderr": result.stderr,
        "success": result.returncode == 0,
    }
