from fastapi import FastAPI
import joblib
import pandas as pd
import os
import csv
from datetime import datetime

from app.schema import JobPostRequest, PredictionResponse
from app.config import MODEL_PATH, THRESHOLD, LOG_FILE

from app.bert_utils import (
    load_bert_model,
    load_tokenizer,
    bert_predict
)

# ------------------------------------------------
# APP INIT
# ------------------------------------------------
app = FastAPI(title="Fake Job Detection API")

# ------------------------------------------------
# LOAD MODELS (ONCE)
# ------------------------------------------------
ml_model = joblib.load(MODEL_PATH)

bert_model = load_bert_model(
    "artifacts/fake_job_bert_full.pt"
)
bert_tokenizer = load_tokenizer(
    "artifacts/bert_tokenizer"
)

# ------------------------------------------------
# ROUTES
# ------------------------------------------------
@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict", response_model=PredictionResponse)
def predict(job: JobPostRequest):

    full_text = (
        job.title + " " +
        job.company_profile + " " +
        job.description + " " +
        job.requirements + " " +
        job.benefits
    )

    # ---------- Classical ML ----------
    text_length = len(full_text)

    input_df = pd.DataFrame([{
        "full_text": full_text,
        "telecommuting": job.telecommuting,
        "has_company_logo": job.has_company_logo,
        "has_questions": job.has_questions,
        "text_length": text_length
    }])

    ml_prob = ml_model.predict_proba(input_df)[0][1]

    # ---------- BERT ----------
    bert_prob = bert_predict(
        bert_model,
        bert_tokenizer,
        full_text
    )

    # ---------- Ensemble ----------
    final_prob = (ml_prob + bert_prob) / 2
    label = "Fake" if final_prob >= THRESHOLD else "Real"

    # ---------- Logging ----------
    log_prediction(
        {
            "full_text": full_text,
            "telecommuting": job.telecommuting,
            "has_company_logo": job.has_company_logo,
            "has_questions": job.has_questions,
            "ml_prob": round(float(ml_prob), 3),
            "bert_prob": round(float(bert_prob), 3),
        },
        label,
        round(float(final_prob), 3)
    )

    return PredictionResponse(
        label=label,
        ml_confidence=round(float(ml_prob), 3),
        bert_confidence=round(float(bert_prob), 3),
        ensemble_confidence=round(float(final_prob), 3)
    )


# ------------------------------------------------
# LOGGING
# ------------------------------------------------
def log_prediction(data: dict, label: str, confidence: float):
    os.makedirs(os.path.dirname(LOG_FILE), exist_ok=True)
    file_exists = os.path.exists(LOG_FILE)

    with open(LOG_FILE, "a", newline="") as f:
        writer = csv.writer(f)

        if not file_exists:
            writer.writerow([
                "timestamp",
                "full_text",
                "telecommuting",
                "has_company_logo",
                "has_questions",
                "ml_prob",
                "bert_prob",
                "final_label",
                "confidence"
            ])

        writer.writerow([
            datetime.utcnow().isoformat(),
            data["full_text"],
            data["telecommuting"],
            data["has_company_logo"],
            data["has_questions"],
            data["ml_prob"],
            data["bert_prob"],
            label,
            confidence
        ])
