FROM python:3.11-slim

WORKDIR /code

# System deps kept minimal for a small, fast image
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY data/ ./data/
COPY app/ ./app/
COPY demo.html .

# Pre-build the TF-IDF index at build time so the container starts instantly
RUN cd app && python model_baseline.py

ENV MODEL_BACKEND=tfidf
ENV PORT=7860
EXPOSE 7860

WORKDIR /code/app
CMD ["gunicorn", "--bind", "0.0.0.0:7860", "--workers", "2", "--timeout", "120", "api:app"]