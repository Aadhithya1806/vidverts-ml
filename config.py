import os
import logging
from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

HF_HOME       = os.getenv("HF_HOME")
MODEL_ID      = os.getenv("MODEL_ID")
API_BASE      = os.getenv("API_BASE")
API_KEY       = os.getenv("API_KEY")

R2_ACCESS_KEY_ID     = os.getenv("R2_ACCESS_KEY_ID")
R2_SECRET_ACCESS_KEY = os.getenv("R2_SECRET_ACCESS_KEY")
R2_ENDPOINT          = os.getenv("R2_ENDPOINT")
R2_BUCKET_NAME       = os.getenv("R2_BUCKET_NAME")

SUPABASE_URL      = os.getenv("SUPABASE_URL")
SUPABASE_ANON_KEY = os.getenv("SUPABASE_ANON_KEY")


def validate_env():
    required = {
        "HF_HOME":              HF_HOME,
        "MODEL_ID":             MODEL_ID,
        "API_BASE":             API_BASE,
        "API_KEY":              API_KEY,
        "R2_ACCESS_KEY_ID":     R2_ACCESS_KEY_ID,
        "R2_SECRET_ACCESS_KEY": R2_SECRET_ACCESS_KEY,
        "R2_ENDPOINT":          R2_ENDPOINT,
        "R2_BUCKET_NAME":       R2_BUCKET_NAME,
        "SUPABASE_URL":         SUPABASE_URL,
        "SUPABASE_ANON_KEY":    SUPABASE_ANON_KEY,
    }
    missing = [k for k, v in required.items() if not v]
    if missing:
        logger.error("Missing env vars: %s", ", ".join(missing))
        raise SystemExit(1)
    logger.info("ENV OK — all variables loaded.")
