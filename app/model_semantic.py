"""
Semantic ICD-10 coding model: pretrained sentence-embedding model
(transfer learning, no fine-tuning required) + cosine similarity.

This is the "upgrade" model to benchmark against the TF-IDF baseline.
It uses a pretrained model from Hugging Face to embed both the ICD-10
code descriptions and the clinical note into the same semantic vector
space, so it can match on MEANING rather than exact word overlap
(e.g. matching "community acquired pneumonia" to J18.9 even if the
note never says the literal word "pneumonia").

NOTE ON RUNNING THIS FILE:
This requires downloading pretrained weights from huggingface.co.
It will NOT run inside network-restricted sandboxes (e.g. this dev
container), but it WILL run:
  - on your own machine (pip install -r requirements.txt)
  - in Google Colab
  - in a Hugging Face Space (recommended free deployment target,
    since the Space runs natively inside the HF ecosystem)

Suggested pretrained models (general -> biomedical-specialized):
  - "sentence-transformers/all-MiniLM-L6-v2"   (fast, general-purpose, ~80MB)
  - "pritamdeka/S-PubMedBert-MS-MARCO"         (biomedical-tuned embeddings)
  - "emilyalsentzer/Bio_ClinicalBERT"          (clinical-notes-tuned, needs mean pooling)

Start with all-MiniLM-L6-v2 to get something working fast, then swap
in the PubMedBert variant for the "upgrade" benchmark comparison.
"""
import time
import joblib
import numpy as np
import pandas as pd
from pathlib import Path

DATA_PATH = Path(__file__).resolve().parent.parent / "data" / "icd10_codes.csv"
ARTIFACT_PATH = Path(__file__).resolve().parent.parent / "data" / "semantic_artifacts.joblib"

DEFAULT_MODEL_NAME = "pritamdeka/S-PubMedBert-MS-MARCO"


class SemanticIcdCoder:
    def __init__(self, data_path: Path = DATA_PATH, model_name: str = DEFAULT_MODEL_NAME):
        from sentence_transformers import SentenceTransformer  # deferred import

        self.model_name = model_name
        self.df = pd.read_csv(data_path)
        self.df["description"] = self.df["description"].fillna("")

        # device="cpu" forced to avoid NaN/Inf overflow issues seen on Apple
        # Silicon (MPS) backends with some sentence-transformer models.
        self.model = SentenceTransformer(model_name, device="cpu")
        self.embeddings = self.model.encode(
            self.df["description"].tolist(),
            batch_size=64,
            show_progress_bar=True,
            normalize_embeddings=True,
        )

        # A handful of very short/degenerate description rows can produce
        # NaN/Inf embeddings after normalization (division by ~zero norm).
        # Sanitize them to exact zero vectors so they can never win a
        # similarity match, and report how many were affected.
        bad_rows = ~np.isfinite(self.embeddings).all(axis=1)
        n_bad = int(bad_rows.sum())
        if n_bad:
            print(f"[model_semantic] Sanitized {n_bad} degenerate embedding row(s).")
        self.embeddings = np.nan_to_num(self.embeddings, nan=0.0, posinf=0.0, neginf=0.0)
        self.embeddings = self.embeddings.astype(np.float32)

    def save(self, path: Path = ARTIFACT_PATH):
        joblib.dump(
            {"model_name": self.model_name, "embeddings": self.embeddings, "df": self.df},
            path,
        )

    @classmethod
    def load(cls, path: Path = ARTIFACT_PATH):
        from sentence_transformers import SentenceTransformer

        obj = cls.__new__(cls)
        artifacts = joblib.load(path)
        obj.model_name = artifacts["model_name"]
        obj.embeddings = artifacts["embeddings"]
        obj.df = artifacts["df"]
        obj.model = SentenceTransformer(obj.model_name, device="cpu")
        return obj

    def predict(self, note_text: str, top_k: int = 5):
        start = time.time()
        query_emb = self.model.encode([note_text], normalize_embeddings=True)
        query_emb = np.nan_to_num(query_emb, nan=0.0, posinf=0.0, neginf=0.0)
        query_emb = query_emb.astype(np.float32)

        # errstate suppresses benign warnings from some BLAS backends (seen
        # on Apple Accelerate) when multiplying matrices containing exact
        # zero rows -- the resulting values are still mathematically correct.
        with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
            sims = (self.embeddings @ query_emb.T).flatten()
        sims = np.nan_to_num(sims, nan=-1.0, posinf=1.0, neginf=-1.0)

        top_idx = np.argsort(sims)[::-1][:top_k]
        latency_ms = (time.time() - start) * 1000

        results = []
        for idx in top_idx:
            results.append(
                {
                    "code": self.df.iloc[idx]["code"],
                    "description": self.df.iloc[idx]["description"],
                    "category": self.df.iloc[idx]["category"],
                    "confidence": round(float(sims[idx]), 4),
                }
            )
        return results, round(latency_ms, 2)


if __name__ == "__main__":
    print("Building semantic index over", DATA_PATH)
    coder = SemanticIcdCoder()
    coder.save()
    print("Saved artifacts to", ARTIFACT_PATH)

    sample_note = (
        "Patient presents with productive cough, fever of 101.5F, and shortness "
        "of breath for 3 days. Chest x-ray shows right lower lobe infiltrate. "
        "Diagnosis: community acquired pneumonia."
    )
    results, latency = coder.predict(sample_note, top_k=5)
    print(f"\nSample note prediction (latency: {latency} ms):")
    for r in results:
        print(f"  {r['code']}  ({r['confidence']})  {r['description']}")
