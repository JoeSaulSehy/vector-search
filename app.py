import os, json, numpy as np, faiss, torch, html
from flask import Flask, request, jsonify, make_response
from transformers import AutoTokenizer, AutoModel

# ---------- Config ----------
API_KEY = os.getenv("SB_API_KEY", "")  # set this in Railway
MODEL_NAME = os.getenv("MODEL_NAME", "ahmedrachid/FinancialBERT")
INDEX_PATH = os.getenv("FAISS_INDEX", "data/faiss_ip.index")
META_PATH  = os.getenv("FAISS_META",  "data/metadata.json")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ---------- Load model & index on startup ----------
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModel.from_pretrained(MODEL_NAME).to(DEVICE)
model.eval()

index = faiss.read_index(INDEX_PATH)
with open(META_PATH, "r", encoding="utf-8") as f:
    id2meta = json.load(f)

@torch.no_grad()
def embed_texts(texts, batch_size=16):
    vecs = []
    max_pos = getattr(model.config, "max_position_embeddings", 512)
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i+batch_size]
        enc = tokenizer(batch, padding=True, truncation=False, return_tensors="pt")
        if enc["input_ids"].shape[1] > max_pos:
            enc = tokenizer(batch, padding=True, truncation=True, max_length=max_pos, return_tensors="pt")
        enc = {k: v.to(DEVICE) for k, v in enc.items()}
        out = model(**enc).last_hidden_state
        mask = enc["attention_mask"].unsqueeze(-1)
        sent_emb = ((out * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-9)).cpu().numpy().astype("float32")
        vecs.append(sent_emb)
    X = np.vstack(vecs).astype("float32")
    faiss.normalize_L2(X)
    return X

def run_search(query, top_k=5):
    q_emb = embed_texts([query], batch_size=1)
    scores, ids = index.search(q_emb, top_k)
    results = []
    for score, idx in zip(scores[0], ids[0]):
        meta = id2meta[int(idx)]
        results.append({
            "source": meta["source"],
            "chunk_idx": meta["chunk_idx"],
            "char_start": meta["char_start"],
            "char_end": meta["char_end"],
            "score": float(score),
            "text": meta["text"],
        })
    return results

def render_html(results, q):
    items = []
    for r in results:
        items.append(
            "<li>"
            f"<strong>{html.escape(r['source'])}</strong> · score {r['score']:.3f}"
            f"<div style='white-space:pre-wrap;margin-top:4px'>{html.escape(r['text'])}</div>"
            "</li>"
        )
    body = "".join(items) if items else "<li>No results</li>"
    return f"<h3>Results for: {html.escape(q)}</h3><ul>{body}</ul>"

app = Flask(__name__)

def authorized(req):  # simple API key check
    key = req.headers.get("X-API-Key", "")
    return (API_KEY and key == API_KEY)

@app.route("/search", methods=["GET"])
def search_route():
    if not authorized(request):
        return make_response("Unauthorized", 401)
    q = (request.args.get("q") or "").strip()
    if not q:
        return make_response("Missing query (?q=...)", 400)
    try:
        k = int(request.args.get("k", 5))
    except ValueError:
        k = 5
    results = run_search(q, top_k=k)
    if (request.args.get("format") or "html").lower() == "json":
        return jsonify({"query": q, "results": results})
    resp = make_response(render_html(results, q), 200)
    resp.headers["Content-Type"] = "text/html; charset=utf-8"
    return resp

@app.route("/health")
def health():
    return "ok", 200

if __name__ == "__main__":
    port = int(os.getenv("PORT", 8000))
    app.run(host="0.0.0.0", port=port)
