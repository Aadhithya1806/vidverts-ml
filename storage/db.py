from datetime import datetime, timezone
from supabase import create_client
from config import SUPABASE_URL, SUPABASE_ANON_KEY


def _client():
    return create_client(SUPABASE_URL, SUPABASE_ANON_KEY)


def create_job(video_name: str, category: str, r2_video: str) -> dict:
    data = (
        _client()
        .table("analysis_jobs")
        .insert({
            "video_name": video_name,
            "category":   category,
            "r2_video":   r2_video,
            "status":     "PENDING",
        })
        .execute()
    )
    return data.data[0]


def update_job(job_id: str, **kwargs) -> dict:
    data = (
        _client()
        .table("analysis_jobs")
        .update(kwargs)
        .eq("id", job_id)
        .execute()
    )
    return data.data[0]


def get_job(job_id: str) -> dict | None:
    data = (
        _client()
        .table("analysis_jobs")
        .select("*")
        .eq("id", job_id)
        .execute()
    )
    return data.data[0] if data.data else None


def list_jobs(limit: int = 50) -> list:
    data = (
        _client()
        .table("analysis_jobs")
        .select("*")
        .order("created_at", desc=True)
        .limit(limit)
        .execute()
    )
    return data.data
