import boto3
from botocore.config import Config
from config import R2_ACCESS_KEY_ID, R2_SECRET_ACCESS_KEY, R2_ENDPOINT, R2_BUCKET_NAME


def _client():
    return boto3.client(
        "s3",
        endpoint_url=R2_ENDPOINT,
        aws_access_key_id=R2_ACCESS_KEY_ID,
        aws_secret_access_key=R2_SECRET_ACCESS_KEY,
        region_name="auto",
        config=Config(signature_version="s3v4"),
    )


def upload_file(local_path: str, r2_key: str) -> str:
    """Upload a local file to R2. Returns the r2_key."""
    _client().upload_file(local_path, R2_BUCKET_NAME, r2_key)
    return r2_key


def download_file(r2_key: str, local_path: str):
    """Download an R2 object to a local path."""
    _client().download_file(R2_BUCKET_NAME, r2_key, local_path)


def presigned_url(r2_key: str, expires_in: int = 3600) -> str:
    """Generate a pre-signed GET URL valid for expires_in seconds."""
    return _client().generate_presigned_url(
        "get_object",
        Params={"Bucket": R2_BUCKET_NAME, "Key": r2_key},
        ExpiresIn=expires_in,
    )


def delete_file(r2_key: str):
    _client().delete_object(Bucket=R2_BUCKET_NAME, Key=r2_key)
