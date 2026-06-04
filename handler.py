import os
import time
import tempfile
import requests
import numpy as np
from pathlib import Path

import runpod
from config import validate_env, logger, HF_HOME, MODEL_ID, API_BASE, API_KEY
from storage import r2, db

validate_env()

os.environ["HF_HOME"] = HF_HOME
HEADERS = {"X-API-Key": API_KEY} if API_KEY else {}

_model = None


def get_model():
    global _model
    if _model is not None:
        return _model
    logger.info("Loading Tribe V2 from %s", HF_HOME)
    from tribev2.tribev2.demo_utils import TribeModel
    _model = TribeModel.from_pretrained(MODEL_ID, cache_folder=HF_HOME)
    logger.info("Model loaded.")
    return _model


def wait_for_api(timeout: int = 120):
    logger.info("Waiting for Render API to wake up...")
    start = time.time()
    while time.time() - start < timeout:
        try:
            res = requests.get(f"{API_BASE}/health", timeout=10)
            if res.status_code == 200:
                logger.info("Render API is up.")
                return True
        except Exception:
            pass
        time.sleep(5)
    logger.error("Render API did not respond within %ds.", timeout)
    return False


def process_job(job: dict, model):
    job_id    = job["id"]
    video_key = job["r2_video"]

    logger.info("Processing job %s | %s | %s", job_id, job["video_name"], job["category"])
    db.update_job(job_id, status="PROCESSING")

    try:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)

            # 1. Download video from R2
            suffix = Path(video_key).suffix
            video_local = str(tmpdir / f"video{suffix}")
            r2.download_file(video_key, video_local)
            logger.info("Video downloaded.")

            # 2. Extract events via WhisperX
            logger.info("Extracting events (WhisperX)...")
            t0 = time.time()
            df = model.get_events_dataframe(video_path=video_local)
            logger.info("Events done in %.1fs — %d rows", time.time() - t0, len(df))

            # 3. Run Tribe V2 inference
            logger.info("Running brain prediction...")
            t1 = time.time()
            predictions, segments = model.predict(events=df)
            logger.info("Prediction done in %.1fs — shape: %s", time.time() - t1, predictions.shape)

            # 4. Save .npy and .parquet locally
            npy_local     = str(tmpdir / "predictions.npy")
            parquet_local = str(tmpdir / "events.parquet")
            np.save(npy_local, predictions)
            df.to_parquet(parquet_local)

            # 5. Upload both to R2
            npy_key     = f"jobs/{job_id}/predictions.npy"
            parquet_key = f"jobs/{job_id}/events.parquet"
            r2.upload_file(npy_local, npy_key)
            r2.upload_file(parquet_local, parquet_key)
            logger.info("Uploaded .npy and .parquet to R2.")

        # 6. Update Supabase with R2 paths
        db.update_job(job_id, r2_npy=npy_key, r2_parquet=parquet_key)
        logger.info("Job %s complete.", job_id)
        return True

    except Exception as e:
        logger.error("Job %s failed: %s", job_id, e)
        db.update_job(job_id, status="FAILED", error=str(e))
        return False


def handler(job):
    logger.info("NeuroLens RunPod Worker started.")

    # 1. Fetch all PENDING jobs
    jobs = db.list_pending_jobs()
    if not jobs:
        logger.info("No PENDING jobs.")
        return {"status": "no_jobs", "processed": 0}

    logger.info("Found %d PENDING job(s).", len(jobs))

    # 2. Load model once for all jobs
    model = get_model()

    # 3. Process each job
    completed_ids = []
    for j in jobs:
        if process_job(j, model):
            completed_ids.append(j["id"])

    if not completed_ids:
        return {"status": "all_failed", "processed": 0}

    # 4. Wake up Render API
    if not wait_for_api(timeout=120):
        logger.error("Skipping batch analyse — API unreachable.")
        return {"status": "api_unreachable", "processed": len(completed_ids)}

    # 5. Fire one batch request to Render
    logger.info("Sending batch analyse for %d job(s).", len(completed_ids))
    try:
        res = requests.post(
            f"{API_BASE}/analyse/batch",
            json={"job_ids": completed_ids},
            headers=HEADERS,
            timeout=60,
        )
        if res.status_code == 200:
            logger.info("Batch queued: %d job(s).", res.json().get("total", 0))
        else:
            logger.error("Batch failed: %s — %s", res.status_code, res.text)
    except Exception as e:
        logger.error("Batch request error: %s", e)

    return {
        "status":    "completed",
        "processed": len(completed_ids),
        "job_ids":   completed_ids,
    }


runpod.serverless.start({"handler": handler})
