# Fake Job Detection API (ML + BERT)

This project implements a Fake Job Posting Detection system using:

- Classical ML model (TF-IDF + Logistic Regression)
- BERT-based deep learning model (DistilBERT)
- FastAPI for serving predictions
- Docker for deployment

The API performs ensemble prediction by combining both models.

-------------------------------------------------
Project Structure
-------------------------------------------------
app/\
 ├── main.py\
 ├── bert_utils.py\
 ├── schema.py\
 ├── config.py

artifacts/\
 ├── fake_job_detector.pkl\
 ├── fake_job_bert_full.pt\
 └── bert_tokenizer/\
 
      ├── vocab.txt\
      ├── tokenizer.json\
      ├── tokenizer_config.json\
      └── special_tokens_map.json

logs/\
 └── predictions.csv

-------------------------------------------------
Models Used
-------------------------------------------------
1. Classical ML Model
   - TF-IDF text features
   - Metadata features (telecommuting, logo, questions)
   - Logistic Regression classifier

2. BERT Model
   - DistilBERT (distilbert-base-uncased)
   - Fine-tuned on fake job posting dataset
   - Saved as PyTorch weights (.pt)

Final decision is made using ensemble averaging.

-------------------------------------------------
API Endpoints
-------------------------------------------------
GET /health
- Health check endpoint

POST /predict
- Predict whether a job post is Fake or Real

-------------------------------------------------
Sample Request (JSON)
-------------------------------------------------
{
  "title": "Senior Software Engineer",
  "description": "We are looking for an experienced backend engineer...",
  "requirements": "5+ years experience in Python",
  "company_profile": "A well-established fintech company",
  "benefits": "Health insurance, flexible hours",
  "telecommuting": 1,
  "has_company_logo": 1,
  "has_questions": 0
}

-------------------------------------------------
Sample Response
-------------------------------------------------
{
  "label": "Fake",
  "confidence": 0.82
}

-------------------------------------------------
Running Locally (Without Docker)
-------------------------------------------------
pip install -r requirements.txt

uvicorn app.main:app --host 0.0.0.0 --port 8000

-------------------------------------------------
Running with Docker
-------------------------------------------------

1. Build Docker image
docker build -t fake-job-predictor:v1 .

2. Run container in the background
docker run -d \
  -p 8000:8000 \
  -v $(pwd)/logs:/app/logs \
  --name fake-job-api \
  fake-job-predictor:v1


3. Access API
http://localhost:8000/docs

4. Quick checklist (print-worthy)

docker ps

docker ps -a

docker logs <container>

docker run (foreground)

verify artifacts exist

verify dependencies

-------------------------------------------------
Design Decisions
-------------------------------------------------
- Classical ML provides fast inference and interpretability
- BERT captures semantic meaning and contextual patterns
- Ensemble improves robustness and reduces false positives
- Models are loaded once at startup to minimize latency
- Architecture supports future integration with job portals

-------------------------------------------------
Notes
-------------------------------------------------
- CPU inference latency for BERT is ~200-400 ms per request
- System is designed for batch or async scaling
- Dataset source: Kaggle fake job posting dataset

-------------------------------------------------
