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

from fastapi.middleware.cors import CORSMiddleware

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


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

    ml_fake_prob = ml_model.predict_proba(input_df)[0][1]

    # ---------- BERT ----------
    bert_fake_prob = bert_predict(
        bert_model,
        bert_tokenizer,
        full_text
    )

    # ---------- Weighted ensemble (BERT-dominant) ----------
    final_prob = (0.7 * bert_fake_prob) + (0.3 * ml_fake_prob)
    label = "Fake" if final_prob >= THRESHOLD else "Real"

    # ---------- Logging ----------
    log_prediction(
        {
            "job_title": job.title,
            "telecommuting": job.telecommuting,
            "has_company_logo": job.has_company_logo,
            "has_questions": job.has_questions,
            "ml_fake_prob": round(float(ml_fake_prob), 3),
            "bert_fake_prob": round(float(bert_fake_prob), 3),
        },
        label,
        round(float(final_prob), 3)
    )
    import time

    time.sleep(5)  # sleeps for 5 seconds
    return PredictionResponse(
        label=label,
        fake_confidence_score=round(float(final_prob), 3)
    )


# ------------------------------------------------
# LOGGING
# ------------------------------------------------
def log_prediction(data: dict, label: str, fake_confidence_score: float):
    os.makedirs(os.path.dirname(LOG_FILE), exist_ok=True)
    file_exists = os.path.exists(LOG_FILE)

    with open(LOG_FILE, "a", newline="") as f:
        writer = csv.writer(f)

        if not file_exists:
            writer.writerow([
                "timestamp",
                "job_title",
                "telecommuting",
                "has_company_logo",
                "has_questions",
                "ml_fake_prob",
                "bert_fake_prob",
                "final_label",
                "fake_confidence_score"
            ])

        writer.writerow([
            datetime.utcnow().isoformat(),
            data["job_title"],
            data["telecommuting"],
            data["has_company_logo"],
            data["has_questions"],
            data["ml_fake_prob"],
            data["bert_fake_prob"],
            label,
            fake_confidence_score
        ])
