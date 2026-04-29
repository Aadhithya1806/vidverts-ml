import os
import time
import tempfile
import requests
import numpy as np
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# ── config ────────────────────────────────────────────────────────────────────
HF_HOME       = os.getenv("HF_HOME", "/workspace/.hf_home")
MODEL_ID      = "facebook/tribev2"
CACHE_FOLDER  = HF_HOME
API_BASE      = os.getenv("API_BASE", "https://vidverts-api.onrender.com")
API_KEY       = os.getenv("API_KEY", "")
HEADERS       = {"X-API-Key": API_KEY} if API_KEY else {}

os.environ["HF_HOME"] = HF_HOME

# ── imports (after HF_HOME is set) ───────────────────────────────────────────
import boto3
from botocore.config import Config
from supabase import create_client

SUPABASE_URL  = os.getenv("SUPABASE_URL")
SUPABASE_KEY  = os.getenv("SUPABASE_ANON_KEY")
R2_ACCESS_KEY = os.getenv("R2_ACCESS_KEY_ID")
R2_SECRET_KEY = os.getenv("R2_SECRET_ACCESS_KEY")
R2_ENDPOINT   = os.getenv("R2_ENDPOINT")
R2_BUCKET     = os.getenv("R2_BUCKET_NAME", "neurolens-storage")


# ── clients ───────────────────────────────────────────────────────────────────
def r2_client():
    return boto3.client(
        "s3",
        endpoint_url=R2_ENDPOINT,
        aws_access_key_id=R2_ACCESS_KEY,
        aws_secret_access_key=R2_SECRET_KEY,
        region_name="auto",
        config=Config(signature_version="s3v4"),
    )

def db_client():
    return create_client(SUPABASE_URL, SUPABASE_KEY)


# ── r2 helpers ────────────────────────────────────────────────────────────────
def r2_download(r2_key: str, local_path: str):
    print(f"  Downloading from R2: {r2_key}")
    r2_client().download_file(R2_BUCKET, r2_key, local_path)

def r2_upload(local_path: str, r2_key: str):
    print(f"  Uploading to R2: {r2_key}")
    r2_client().upload_file(local_path, R2_BUCKET, r2_key)
    return r2_key


# ── model ─────────────────────────────────────────────────────────────────────
_model = None

def get_model():
    global _model
    if _model is not None:
        return _model
    print("Loading Tribe V2 model...")
    from tribev2.tribev2.demo_utils import TribeModel
    _model = TribeModel.from_pretrained(MODEL_ID, cache_folder=CACHE_FOLDER)
    print("Model loaded.")
    return _model


# ── process one job ───────────────────────────────────────────────────────────
def process_job(job: dict):
    job_id    = job["id"]
    video_key = job["r2_video"]

    print(f"\n{'='*60}")
    print(f"Processing job: {job_id}")
    print(f"Video: {job['video_name']}  |  Category: {job['category']}")

    db = db_client()

    # Mark as PROCESSING
    db.table("analysis_jobs").update({"status": "PROCESSING"}).eq("id", job_id).execute()

    try:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)

            # 1. Download video from R2
            suffix = Path(video_key).suffix
            video_local = str(tmpdir / f"video{suffix}")
            r2_download(video_key, video_local)

            # 2. Load model
            model = get_model()

            # 3. Extract events (WhisperX inside)
            print("  Extracting events (WhisperX)...")
            t0 = time.time()
            df = model.get_events_dataframe(video_path=video_local)
            print(f"  Events done in {time.time()-t0:.1f}s — {len(df)} rows")

            # 4. Run inference
            print("  Running brain prediction...")
            t1 = time.time()
            predictions, segments = model.predict(events=df)
            print(f"  Prediction done in {time.time()-t1:.1f}s — shape: {predictions.shape}")

            # 5. Save .npy and .parquet locally
            npy_local     = str(tmpdir / "predictions.npy")
            parquet_local = str(tmpdir / "events.parquet")
            np.save(npy_local, predictions)
            df.to_parquet(parquet_local)

            # 6. Upload to R2
            npy_key     = f"jobs/{job_id}/predictions.npy"
            parquet_key = f"jobs/{job_id}/events.parquet"
            r2_upload(npy_local, npy_key)
            r2_upload(parquet_local, parquet_key)

        # 7. Update job in Supabase with npy + parquet paths
        db.table("analysis_jobs").update({
            "r2_npy":     npy_key,
            "r2_parquet": parquet_key,
        }).eq("id", job_id).execute()

        print(f"  Uploaded .npy and .parquet to R2")

        # 8. Call CPU analyser
        print(f"  Calling analyser...")
        res = requests.post(f"{API_BASE}/analyse/{job_id}", headers=HEADERS, timeout=600)
        if res.status_code == 200:
            print(f"  Analysis complete.")
        else:
            print(f"  Analyser error: {res.text}")
            db.table("analysis_jobs").update({
                "status": "FAILED",
                "error":  f"Analyser error: {res.text}"
            }).eq("id", job_id).execute()

    except Exception as e:
        print(f"  ERROR: {e}")
        db.table("analysis_jobs").update({
            "status": "FAILED",
            "error":  str(e)
        }).eq("id", job_id).execute()


# ── main loop ─────────────────────────────────────────────────────────────────
def main():
    print("NeuroLens — Vast.ai GPU Handler")
    print(f"API: {API_BASE}")
    print(f"HF_HOME: {HF_HOME}")

    db = db_client()

    # Fetch all PENDING jobs
    result = db.table("analysis_jobs").select("*").eq("status", "PENDING").order("created_at").execute()
    jobs = result.data

    if not jobs:
        print("No PENDING jobs found. Exiting.")
        return

    print(f"\nFound {len(jobs)} PENDING job(s)")

    for job in jobs:
        process_job(job)

    print(f"\nAll jobs processed.")


if __name__ == "__main__":
    main()
