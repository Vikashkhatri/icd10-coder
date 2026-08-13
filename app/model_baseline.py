"""
Baseline ICD-10 coding model: TF-IDF + cosine similarity.

Approach:
- Treat each ICD-10 code's official description as a "document"
- Vectorize all code descriptions with TF-IDF
- At inference time, vectorize the clinician's free-text note the same way
- Return the top-k ICD-10 codes whose descriptions are most similar (cosine)

This requires no model downloads / no training data of clinical notes,
which makes it a safe, zero-cost, fully offline baseline to benchmark
against a transformer-based semantic model.
"""
import time
import joblib
import pandas as pd
from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

DATA_PATH = Path(__file__).resolve().parent.parent / "data" / "icd10_codes.csv"
ARTIFACT_PATH = Path(__file__).resolve().parent.parent / "data" / "tfidf_artifacts.joblib"


class TfidfIcdCoder:
    def __init__(self, data_path: Path = DATA_PATH):
        self.df = pd.read_csv(data_path)
        self.df["description"] = self.df["description"].fillna("")
        self.vectorizer = TfidfVectorizer(
            stop_words="english",
            ngram_range=(1, 2),
            max_features=50000,
        )
        self.matrix = self.vectorizer.fit_transform(self.df["description"])

    def save(self, path: Path = ARTIFACT_PATH):
        joblib.dump(
            {"vectorizer": self.vectorizer, "matrix": self.matrix, "df": self.df}, path
        )

    @classmethod
    def load(cls, path: Path = ARTIFACT_PATH):
        obj = cls.__new__(cls)
        artifacts = joblib.load(path)
        obj.vectorizer = artifacts["vectorizer"]
        obj.matrix = artifacts["matrix"]
        obj.df = artifacts["df"]
        return obj

    def predict(self, note_text: str, top_k: int = 5):
        start = time.time()
        query_vec = self.vectorizer.transform([note_text])
        sims = cosine_similarity(query_vec, self.matrix).flatten()
        top_idx = sims.argsort()[::-1][:top_k]
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
    print("Building TF-IDF index over", DATA_PATH)
    coder = TfidfIcdCoder()
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
