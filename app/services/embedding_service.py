"""
Servicio de embeddings.

Proveedor conmutable vía settings.EMBEDDING_PROVIDER:
- "local"   -> BGE-M3 (BAAI/bge-m3) vía sentence-transformers, corre en esta
               misma máquina/proceso, sin costo por request.
- "bedrock" -> Amazon Titan Text Embeddings V2 (código original conservado,
               para casos atípicos que se decida volver a usar Bedrock).

NOTA sobre Bedrock (heredada): las cuentas AWS nuevas tienen cuota on-demand
en 0 para modelos de embeddings de Bedrock. Si ves ThrottlingException/
AccessDenied, contacta AWS Support o prueba el prefijo cross-region "us."
delante del model_id.
"""
import json
import time

import boto3
from botocore.exceptions import ClientError

from app.core.config import settings
from app.core.logging import get_logger

logger = get_logger(__name__)

_MAX_RETRIES = 5
_BASE_DELAY = 1.0

_bedrock = boto3.client("bedrock-runtime", region_name=settings.BEDROCK_REGION)

# --- Modelo local (BGE-M3) — carga perezosa (singleton) ---------------------
_local_model = None


def _get_local_model():
    global _local_model
    if _local_model is None:
        from app.core.model_cache import ensure_writable_hf_cache
        ensure_writable_hf_cache()  # debe ir ANTES del import de sentence_transformers

        import torch
        from sentence_transformers import SentenceTransformer

        device = "mps" if torch.backends.mps.is_available() else "cpu"
        # En CPU (Lambda) cargamos en fp16 -- reduce memoria de ~2.2GB a
        # ~1.1GB por modelo. Necesario porque la cuenta de AWS tiene un
        # tope de memoria de Lambda de 3008MB (cuota de cuenta, no el
        # limite tecnico de la plataforma que es 10240MB), y los DOS
        # modelos locales (embeddings + reranker) en fp32 no caben juntos.
        # Perdida de precision verificada como insignificante: similitud
        # coseno de 0.9999994 entre el mismo texto en fp32 vs fp16.
        model_kwargs = {"dtype": torch.float16} if device == "cpu" else {}
        logger.info(f"Cargando modelo de embeddings local '{settings.LOCAL_EMBEDDING_MODEL}' en device={device}{' (fp16)' if model_kwargs else ''}")
        _local_model = SentenceTransformer(settings.LOCAL_EMBEDDING_MODEL, device=device, model_kwargs=model_kwargs)
    return _local_model


def _embed_local_batch(texts: list[str]) -> list[list[float]]:
    model = _get_local_model()
    vectors = model.encode(
        texts,
        normalize_embeddings=True,
        batch_size=16,
        show_progress_bar=False,
    )
    return vectors.tolist()


# --- Bedrock (Titan V2) — conservado tal cual, para casos atípicos ----------
def _bedrock_invoke_with_retry(body: dict) -> dict:
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


def _embed_bedrock_batch(texts: list[str]) -> list[list[float]]:
    # Titan V2 no soporta batch nativo por request: se invoca secuencialmente.
    return [_bedrock_invoke_with_retry({"inputText": t[:8000]})["embedding"] for t in texts]


# --- Interfaz pública (sin cambios para quien la consume) -------------------
def embed_text(text: str) -> list[float]:
    """Genera el embedding de un solo texto (usado en indexación)."""
    return embed_batch([text])[0]


def embed_query(question: str) -> list[float]:
    """Genera el embedding de una pregunta (usado en búsqueda)."""
    return embed_text(question)


def embed_batch(texts: list[str]) -> list[list[float]]:
    """Genera embeddings para una lista de textos, usando el proveedor configurado."""
    if settings.EMBEDDING_PROVIDER == "local":
        return _embed_local_batch(texts)
    elif settings.EMBEDDING_PROVIDER == "bedrock":
        return _embed_bedrock_batch(texts)
    raise ValueError(f"EMBEDDING_PROVIDER desconocido: {settings.EMBEDDING_PROVIDER!r}")
