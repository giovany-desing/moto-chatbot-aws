"""
Servicio de embeddings con Amazon Bedrock Titan Text Embeddings V2.

NOTA CRÍTICA: las cuentas AWS nuevas tienen cuota on-demand en 0 para
modelos de embeddings de Bedrock. Si ves ThrottlingException/AccessDenied,
contacta AWS Support para aumentar la cuota, o prueba el prefijo
cross-region "us." delante del model_id.
"""
import json
import time
import boto3
from botocore.exceptions import ClientError

from app.core.config import settings
from app.core.logging import get_logger

logger = get_logger(__name__)

_bedrock = boto3.client("bedrock-runtime", region_name=settings.BEDROCK_REGION)

_MAX_RETRIES = 5
_BASE_DELAY = 1.0


def _invoke_with_retry(body: dict) -> dict:
    last_exc: Exception | None = None
    for attempt in range(_MAX_RETRIES):
        try:
            response = _bedrock.invoke_model(
                modelId=settings.EMBEDDING_MODEL,
                body=json.dumps(body),
                contentType="application/json",
                accept="application/json",
            )
            return json.loads(response["body"].read())
        except ClientError as exc:
            error_code = exc.response.get("Error", {}).get("Code", "")
            last_exc = exc
            if error_code in ("ThrottlingException", "TooManyRequestsException"):
                delay = _BASE_DELAY * (2 ** attempt)
                logger.warning(f"Throttling en Bedrock embeddings, reintento en {delay}s (intento {attempt + 1})")
                time.sleep(delay)
                continue
            raise
    raise last_exc  # type: ignore[misc]


def embed_text(text: str) -> list[float]:
    """Genera el embedding de un solo texto (usado en indexación)."""
    result = _invoke_with_retry({"inputText": text[:8000]})
    return result["embedding"]


def embed_query(question: str) -> list[float]:
    """Genera el embedding de una pregunta (usado en búsqueda)."""
    return embed_text(question)


def embed_batch(texts: list[str]) -> list[list[float]]:
    """
    Genera embeddings para una lista de textos.
    Titan V2 no soporta batch nativo por request — se invoca secuencialmente
    con reintentos individuales para tolerar throttling parcial.
    """
    embeddings = []
    for text in texts:
        embeddings.append(embed_text(text))
    return embeddings
