"""
Flask API for the ICD-10 coding assistant.

Endpoints:
  GET  /health              -> liveness check
  POST /predict              -> {"note": "...", "top_k": 5} -> ICD-10 code suggestions

Model selection is controlled by the MODEL_BACKEND env var:
  MODEL_BACKEND=tfidf     (default, works everywhere, no downloads)
  MODEL_BACKEND=semantic  (pretrained transformer embeddings, needs HF access)
"""
import os
import time
from pathlib import Path
from flask import Flask, request, jsonify
from flask_cors import CORS

from model_baseline import TfidfIcdCoder, ARTIFACT_PATH as TFIDF_ARTIFACT_PATH

app = Flask(__name__)
CORS(app)  # allow the demo webpage (different origin) to call this API

MODEL_BACKEND = os.environ.get("MODEL_BACKEND", "tfidf")
_coder = None


def get_coder():
    global _coder
    if _coder is not None:
        return _coder

    if MODEL_BACKEND == "semantic":
        from model_semantic import SemanticIcdCoder, ARTIFACT_PATH as SEM_PATH

        if Path(SEM_PATH).exists():
            _coder = SemanticIcdCoder.load(SEM_PATH)
        else:
            _coder = SemanticIcdCoder()
            _coder.save()
    else:
        if Path(TFIDF_ARTIFACT_PATH).exists():
            _coder = TfidfIcdCoder.load(TFIDF_ARTIFACT_PATH)
        else:
            _coder = TfidfIcdCoder()
            _coder.save()
    return _coder


@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok", "model_backend": MODEL_BACKEND})


@app.route("/predict", methods=["POST"])
def predict():
    payload = request.get_json(force=True, silent=True) or {}
    note = payload.get("note", "").strip()
    top_k = int(payload.get("top_k", 5))

    if not note:
        return jsonify({"error": "Field 'note' is required and cannot be empty."}), 400
    if not (1 <= top_k <= 20):
        return jsonify({"error": "'top_k' must be between 1 and 20."}), 400

    coder = get_coder()
    request_start = time.time()
    results, model_latency_ms = coder.predict(note, top_k=top_k)
    total_latency_ms = round((time.time() - request_start) * 1000, 2)

    return jsonify(
        {
            "model_backend": MODEL_BACKEND,
            "note_preview": note[:120],
            "suggested_codes": results,
            "model_latency_ms": model_latency_ms,
            "total_latency_ms": total_latency_ms,
        }
    )


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 7860))  # 7860 = HF Spaces default port
    app.run(host="0.0.0.0", port=port)
