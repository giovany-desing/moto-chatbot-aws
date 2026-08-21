"""
Reranking -- segundo pase de evaluacion de relevancia sobre los
candidatos que ya trajo el retrieval hibrido (dense + sparse + RRF).

A diferencia de RRF (que solo mira POSICIONES en dos rankings, sin leer
el contenido), el reranker lee la pregunta y cada candidato JUNTOS, en
un solo pase, y les asigna un puntaje de relevancia real. Usa el Rerank
API de Bedrock (bedrock-agent-runtime), un servicio DISTINTO al
bedrock-runtime que puede usar embedding_service.py (si EMBEDDING_PROVIDER=bedrock)
y llm_service.py (si LLM_PROVIDER=bedrock).
"""
import boto3
from botocore.exceptions import ClientError

from app.core.config import settings
from app.core.logging import get_logger

logger = get_logger(__name__)

_bedrock_agent = boto3.client("bedrock-agent-runtime", region_name=settings.BEDROCK_REGION)

_RERANK_MODEL_ARN_TEMPLATE = "arn:aws:bedrock:{region}::foundation-model/{model_id}"


def rerank(query: str, candidatos: list[dict], top_k: int) -> list[dict]:
    """
    candidatos: lista de dicts, cada uno con al menos la clave "text".
    Devuelve los top_k candidatos reordenados por relevancia real,
    con "relevance" reemplazado por el score del reranker.
    Si el reranker falla (ej. Bedrock bloqueado), degrada de forma
    segura devolviendo los primeros top_k del orden que ya traia.
    """
    if not candidatos:
        return []

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
        logger.warning(f"Reranking fallo, se usa el orden previo sin reordenar: {exc}")
        return candidatos[:top_k]

    resultado = []
    for r in response["results"]:
        candidato = dict(candidatos[r["index"]])
        candidato["relevance"] = round(float(r["relevanceScore"]), 4)
        resultado.append(candidato)
    return resultado
