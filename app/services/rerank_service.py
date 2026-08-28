"""
Reranking -- segundo pase de evaluacion de relevancia sobre los
candidatos que ya trajo el retrieval hibrido (dense + sparse + RRF).

A diferencia de RRF (que solo mira POSICIONES en dos rankings, sin leer
el contenido), el reranker lee la pregunta y cada candidato JUNTOS, en
un solo pase, y les asigna un puntaje de relevancia real.

Proveedor conmutable vía settings.RERANK_PROVIDER:
- "local"   -> BAAI/bge-reranker-v2-m3 (cross-encoder), vía sentence-transformers,
               corre en esta misma máquina/proceso, sin costo por request.
- "bedrock" -> Rerank API de Bedrock (bedrock-agent-runtime), un servicio
               DISTINTO al bedrock-runtime que puede usar embedding_service.py
               (si EMBEDDING_PROVIDER=bedrock) y llm_service.py (si
               LLM_PROVIDER=bedrock). Conservado para casos atípicos.
"""
import boto3
from botocore.exceptions import ClientError

from app.core.config import settings
from app.core.logging import get_logger

logger = get_logger(__name__)

_bedrock_agent = boto3.client("bedrock-agent-runtime", region_name=settings.BEDROCK_REGION)
_RERANK_MODEL_ARN_TEMPLATE = "arn:aws:bedrock:{region}::foundation-model/{model_id}"

_local_reranker = None
_RUTA_HORNEADA = "/var/task/models/bge-reranker-v2-m3"


def _get_local_reranker():
    global _local_reranker
    if _local_reranker is None:
        import os
        import torch
        from sentence_transformers import CrossEncoder

        modelo_a_cargar = _RUTA_HORNEADA if os.path.isdir(_RUTA_HORNEADA) else settings.LOCAL_RERANK_MODEL

        device = "mps" if torch.backends.mps.is_available() else "cpu"
        model_kwargs = {"dtype": torch.float16} if device == "cpu" else {}
        logger.info(f"Cargando reranker '{modelo_a_cargar}' en device={device}{' (fp16)' if model_kwargs else ''}")
        _local_reranker = CrossEncoder(modelo_a_cargar, device=device, model_kwargs=model_kwargs)
    return _local_reranker


def _rerank_local(query: str, candidatos: list[dict], top_k: int) -> list[dict]:
    reranker = _get_local_reranker()
    pares = [[query, c["text"]] for c in candidatos]
    scores = reranker.predict(pares)

    candidatos_con_score = [
        {**c, "relevance": round(float(score), 4)}
        for c, score in zip(candidatos, scores)
    ]
    candidatos_con_score.sort(key=lambda c: c["relevance"], reverse=True)
    return candidatos_con_score[:top_k]


def _rerank_bedrock(query: str, candidatos: list[dict], top_k: int) -> list[dict]:
    model_arn = _RERANK_MODEL_ARN_TEMPLATE.format(
        region=settings.BEDROCK_REGION, model_id=settings.RERANK_MODEL
    )

    sources = [
        {
            "inlineDocumentSource": {"textDocument": {"text": c["text"]}, "type": "TEXT"},
            "type": "INLINE",
        }
        for c in candidatos
    ]

    try:
        response = _bedrock_agent.rerank(
            queries=[{"textQuery": {"text": query}, "type": "TEXT"}],
            sources=sources,
            rerankingConfiguration={
                "type": "BEDROCK_RERANKING_MODEL",
                "bedrockRerankingConfiguration": {
                    "modelConfiguration": {"modelArn": model_arn},
                    "numberOfResults": top_k,
                },
            },
        )
    except ClientError as exc:
        logger.warning(f"Reranking Bedrock falló, se usa el orden previo sin reordenar: {exc}")
        return candidatos[:top_k]

    resultado = []
    for r in response["results"]:
        candidato = dict(candidatos[r["index"]])
        candidato["relevance"] = round(float(r["relevanceScore"]), 4)
        resultado.append(candidato)
    return resultado


def rerank(query: str, candidatos: list[dict], top_k: int) -> list[dict]:
    """
    candidatos: lista de dicts, cada uno con al menos la clave "text".
    Devuelve los top_k candidatos reordenados por relevancia real,
    con "relevance" reemplazado por el score del reranker.
    Si el reranker falla, degrada de forma segura devolviendo los
    primeros top_k del orden que ya traía.
    """
    if not candidatos:
        return []

    try:
        if settings.RERANK_PROVIDER == "local":
            return _rerank_local(query, candidatos, top_k)
        elif settings.RERANK_PROVIDER == "bedrock":
            return _rerank_bedrock(query, candidatos, top_k)
        raise ValueError(f"RERANK_PROVIDER desconocido: {settings.RERANK_PROVIDER!r}")
    except Exception as exc:
        logger.warning(f"Reranking falló, se usa el orden previo sin reordenar: {exc}")
        return candidatos[:top_k]
