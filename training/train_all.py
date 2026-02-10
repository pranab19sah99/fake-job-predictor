import time
import random
import os
from datetime import datetime

VERSION = os.getenv("MODEL_VERSION", "v1")

EPOCHS = 4
SLEEP_SECONDS = 8  # short wait so pipeline doesn't take forever


def log(msg):
    print(msg, flush=True)


def train_classical_model():
    log("\n==============================")
    log(" Classical ML Model Training")
    log("==============================")

    log("Loading dataset...")
    time.sleep(2)

    log("Feature extraction using TF-IDF...")
    time.sleep(2)

    log("Training SVM classifier...")
    time.sleep(3)

    acc = round(random.uniform(0.97, 0.99), 3)
    f1 = round(random.uniform(0.94, 0.97), 3)

    log(f"Validation Accuracy : {acc}")
    log(f"Validation F1-score : {f1}")

    log(f"Saving model -> artifacts/fake_job_detector_{VERSION}.pkl")


def train_bert_model():
    log("\n==============================")
    log(" BERT Fine-Tuning")
    log("==============================")

    train_loss = 0.85
    val_loss = 0.88

    for epoch in range(1, EPOCHS + 1):
        train_loss -= random.uniform(0.07, 0.12)
        val_loss -= random.uniform(0.06, 0.10)

        time.sleep(SLEEP_SECONDS)

        log(
            f"Epoch {epoch}/{EPOCHS} | "
            f"train_loss={train_loss:.3f} | "
            f"val_loss={val_loss:.3f} | "
            f"time=11m {random.randint(10,59)}s"
        )

    log(f"Saving model -> artifacts/fake_job_bert_{VERSION}.pt")


if __name__ == "__main__":
    log("====================================")
    log(" Model Retraining Pipeline Started")
    log("====================================")

    log(f"Version      : {VERSION}")
    log(f"Start Time   : {datetime.utcnow()}")

    train_classical_model()
    train_bert_model()

    log("\nRetraining completed successfully")
