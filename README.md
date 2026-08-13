# ICD-10 Auto-Coding Assistant

An AI-powered assistant that takes a clinician's free-text note and suggests
the most likely ICD-10-CM diagnosis codes, to speed up medical coding and
reduce claim-denial risk.

## Project structure

```
icd10-coder/
├── data/
│   └── icd10_codes.csv         # 71,704 official ICD-10-CM codes + descriptions
├── app/
│   ├── model_baseline.py       # TF-IDF + cosine similarity (no downloads needed)
│   ├── model_semantic.py       # Pretrained biomedical embeddings (transfer learning)
│   └── api.py                  # Flask API (/health, /predict)
├── benchmark.py                 # Compares baseline vs semantic on labeled test notes
├── requirements.txt
├── Dockerfile
└── README.md
```

## How it works

1. Every official ICD-10-CM code has a text description (e.g. `J18.9` →
   "Pneumonia, unspecified organism").
2. We vectorize all 71,704 descriptions once, ahead of time.
3. At inference time, we vectorize the clinician's note the same way and
   return the top-k closest ICD-10 codes by cosine similarity.
4. Two interchangeable vectorization strategies are implemented, so we can
   **benchmark** a simple approach against a pretrained-model approach:
   - `tfidf` — classic bag-of-words TF-IDF (fast, zero downloads, weak on synonyms)
   - `semantic` — pretrained biomedical sentence embeddings via
     `sentence-transformers` (transfer learning, understands meaning/synonyms,
     e.g. matches "community acquired pneumonia" text to the pneumonia code
     even without exact word overlap)

## Local setup

```bash
pip install -r requirements.txt

# Baseline (works everywhere, no internet needed after install)
cd app && python model_baseline.py      # builds + saves the TF-IDF index
python api.py                            # starts Flask app on :7860

# Semantic model (needs huggingface.co access — run on your machine or Colab)
pip install sentence-transformers torch
cd app && python model_semantic.py      # downloads pretrained model, builds index
MODEL_BACKEND=semantic python api.py
```

Test it:
```bash
curl -X POST http://127.0.0.1:7860/predict \
  -H "Content-Type: application/json" \
  -d '{"note": "45 year old with poorly controlled type 2 diabetes", "top_k": 5}'
```

## Benchmarking

```bash
python benchmark.py --backend tfidf      # runs anywhere
python benchmark.py --backend semantic   # needs HF access
python benchmark.py --backend both       # side-by-side comparison
```

**Results so far (TF-IDF baseline, 5-note labeled test set):**

| Metric | Value |
|---|---|
| Top-1 accuracy | 0% |
| Top-5 accuracy | 60% |
| Avg inference latency | ~25 ms |
| Index build time | ~1.1 s |

Interpretation for your deck: the baseline is fast and needs zero downloads,
but it matches on literal word overlap, not clinical meaning — it often
surfaces a *symptom* code (e.g. "shortness of breath") instead of the
underlying *diagnosis* code (e.g. pneumonia). This is the concrete
justification for the semantic/pretrained-model upgrade — run the same
benchmark with `--backend semantic` on a machine with Hugging Face access
and drop the resulting table in next to this one.

## Docker

```bash
docker build -t icd10-coder .
docker run -p 7860:7860 icd10-coder
```

The Dockerfile bakes in the TF-IDF backend by default (zero-cost, no
external calls at runtime). To containerize the semantic backend instead,
uncomment the `sentence-transformers`/`torch` lines in `requirements.txt`
and set `MODEL_BACKEND=semantic` — note this makes the image much larger
and requires HF access at container build/run time.

## Free deployment options (no cost)

1. **Hugging Face Spaces (recommended)** — free tier, natively supports
   Docker Spaces, and runs *inside* the HF ecosystem so the semantic model
   downloads with no restrictions. Push this repo, add a `README.md` header
   with `sdk: docker`, and it deploys automatically.
2. **Render.com free web service** — connect this repo, set the Docker
   runtime, free tier sleeps after inactivity but works for a demo.
3. **Railway.app free trial tier** — similar to Render, Docker-native.
4. **Local + ngrok** — for a live class demo without deploying anywhere
   permanent: `python app/api.py` locally, then `ngrok http 7860`.

## Requirements checklist (mapped to course deliverables)

- [x] Pretrained / open-source model usage (no training from scratch)
- [x] Transfer learning component (pretrained sentence-embedding model)
- [x] Model benchmarking (TF-IDF vs semantic, accuracy + latency table)
- [x] Notebook → script (model_baseline.py / model_semantic.py as standalone scripts)
- [x] Script → Flask API (app/api.py)
- [x] Docker containerization (Dockerfile)
- [ ] Cloud/mobile hosting — pick one of the free options above and deploy
- [ ] ML Canvas (business + product framing — see Project 1 discussion)
- [ ] Demo (record a short screen capture of the API responding to a note)
