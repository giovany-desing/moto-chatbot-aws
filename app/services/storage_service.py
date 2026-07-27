"""
Servicio S3 — subida y descarga de PDFs de manuales.
"""
import boto3
from botocore.exceptions import ClientError

from app.core.config import settings
from app.core.logging import get_logger

logger = get_logger(__name__)

_s3 = boto3.client("s3", region_name=settings.AWS_REGION)

MANUALES_PREFIX = "manuales/"
PROCESADOS_PREFIX = "procesados/"
CONVERSACIONES_PREFIX = "conversaciones/"


def upload_pdf(filename: str, content: bytes) -> str:
    key = f"{MANUALES_PREFIX}{filename}"
    _s3.put_object(
        Bucket=settings.S3_BUCKET_NAME,
        Key=key,
        Body=content,
        ContentType="application/pdf",
    )
    logger.info(f"PDF subido a S3: {key}")
    return key


def download_pdf(filename: str) -> bytes:
    key = f"{MANUALES_PREFIX}{filename}"
    try:
        response = _s3.get_object(Bucket=settings.S3_BUCKET_NAME, Key=key)
        return response["Body"].read()
    except ClientError as exc:
        logger.error(f"Error descargando {key}: {exc}")
        raise


def object_exists(filename: str) -> bool:
    key = f"{MANUALES_PREFIX}{filename}"
    try:
        _s3.head_object(Bucket=settings.S3_BUCKET_NAME, Key=key)
        return True
    except ClientError:
        return False
