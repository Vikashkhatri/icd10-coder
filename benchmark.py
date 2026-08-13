"""
Benchmarking harness: compares the TF-IDF baseline against the
pretrained semantic embedding model on a small labeled test set of
synthetic clinical notes with known "correct" ICD-10 codes.

Metrics reported per model:
  - Top-1 accuracy       (is the correct code the #1 suggestion?)
  - Top-5 accuracy       (is the correct code anywhere in top 5?)
  - Avg inference latency (ms)
  - Cold-start / index build time (s)

Run:
  python benchmark.py --backend tfidf
  python benchmark.py --backend semantic   # requires HF network access
  python benchmark.py --backend both       # requires HF network access
"""
import argparse
import time
import sys
import csv
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "app"))

TEST_SET_PATH = Path(__file__).resolve().parent / "data" / "mtsamples_test_set.csv"


def load_test_set(path: Path = TEST_SET_PATH):
    """Loads a labeled test set of REAL clinical notes (sourced from the
    public-domain MTSamples corpus, https://mtsamples.com, CC0 license),
    hand-labeled with their correct ICD-10-CM diagnosis code."""
    rows = []
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for r in reader:
            rows.append({"note": r["note"], "expected_code": r["expected_code"]})
    return rows


TEST_SET = load_test_set()


def evaluate(coder, name: str):
    top1_hits = 0
    top5_hits = 0
    category_hits = 0  # same 3-char ICD-10 category, even if exact code differs
    latencies = []

    print(f"\n=== {name} ===")
    for case in TEST_SET:
        results, latency_ms = coder.predict(case["note"], top_k=5)
        latencies.append(latency_ms)
        codes = [r["code"] for r in results]
        expected_category = case["expected_code"][:3]

        top1 = codes[0] == case["expected_code"]
        top5 = case["expected_code"] in codes
        cat_match = codes[0][:3] == expected_category
        top1_hits += top1
        top5_hits += top5
        category_hits += cat_match

        status = "✓" if top5 else ("~" if cat_match else "✗")
        print(f"  {status} expected={case['expected_code']:8s} got_top1={codes[0]:8s} top5={codes}")

    n = len(TEST_SET)
    print(f"\n  Top-1 accuracy (exact code):     {top1_hits}/{n} ({100*top1_hits/n:.0f}%)")
    print(f"  Top-5 accuracy (exact code):     {top5_hits}/{n} ({100*top5_hits/n:.0f}%)")
    print(f"  Top-1 category match (~family):  {category_hits}/{n} ({100*category_hits/n:.0f}%)")
    print(f"  Avg latency:                     {sum(latencies)/n:.2f} ms")


def run_tfidf():
    from model_baseline import TfidfIcdCoder

    t0 = time.time()
    coder = TfidfIcdCoder()
    build_time = time.time() - t0
    print(f"[tfidf] index build time: {build_time:.2f}s")
    evaluate(coder, "TF-IDF Baseline")


def run_semantic():
    from model_semantic import SemanticIcdCoder

    t0 = time.time()
    coder = SemanticIcdCoder()
    build_time = time.time() - t0
    print(f"[semantic] index build time: {build_time:.2f}s")
    evaluate(coder, "Semantic (Pretrained PubMedBERT)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend", choices=["tfidf", "semantic", "both"], default="tfidf")
    args = parser.parse_args()

    if args.backend in ("tfidf", "both"):
        run_tfidf()
    if args.backend in ("semantic", "both"):
        run_semantic()
