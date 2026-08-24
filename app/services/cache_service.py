"""
Servicio de caché -- dos capas:
1. Redis (hash exacto): rapida, pero solo funciona si la pregunta es
   textualmente identica a una ya cacheada.
2. Postgres/pgvector (semantica): si el hash exacto falla, busca por
   similitud de embedding contra preguntas cacheadas recientemente --
   "cada cuanto se cambia el aceite" SI hace match con "cuando toca
   cambiar el aceite", cosa que el hash exacto nunca detectaria.

Nota de diseno: RediSearch (busqueda vectorial nativa de Redis) NO esta
disponible porque ElastiCache estandar no soporta modulos de Redis (eso
requiere Redis Enterprise/Redis Stack) -- por eso la capa semantica vive
en Postgres, no en Redis.

Tambien incluye memoria conversacional (Redis).
"""
from __future__ import annotations
import hashlib
import json

import redis

from app.core.config import settings
from app.core.logging import get_logger

logger = get_logger(__name__)

_redis_client: redis.Redis | None = None


def get_client() -> redis.Redis:
    global _redis_client
    if _redis_client is None:
        _redis_client = redis.Redis(
            host=settings.REDIS_HOST,
            port=settings.REDIS_PORT,
            decode_responses=True,
            socket_connect_timeout=5,
            socket_timeout=5,
        )
    return _redis_client


def _cache_key(question: str, filename: str | None) -> str:
    raw = f"{question.strip().lower()}::{filename or ''}"
    digest = hashlib.md5(raw.encode("utf-8")).hexdigest()
    return f"cache:respuesta:{digest}"


def get(question: str, filename: str | None = None) -> dict | None:
    try:
        client = get_client()
        raw = client.get(_cache_key(question, filename))
        return json.loads(raw) if raw else None
    except redis.RedisError as exc:
        logger.warning(f"Redis no disponible para lectura de caché: {exc}")
        return None


def set(question: str, result: dict, filename: str | None = None) -> None:
    try:
        client = get_client()
        client.setex(
            _cache_key(question, filename),
            settings.CACHE_TTL_SECONDS,
            json.dumps(result, default=str),
        )
    except redis.RedisError as exc:
        logger.warning(f"Redis no disponible para escritura de caché: {exc}")


def _memory_key(session_id: str) -> str:
    return f"memoria:sesion:{session_id}"


def get_memory(session_id: str) -> list[dict]:
    if not session_id:
        return []
    try:
        client = get_client()
        raw = client.get(_memory_key(session_id))
        return json.loads(raw) if raw else []
    except redis.RedisError as exc:
        logger.warning(f"Redis no disponible para lectura de memoria: {exc}")
        return []


def save_memory(session_id: str, question: str, answer: str) -> None:
    if not session_id:
        return
    try:
        client = get_client()
        history = get_memory(session_id)
        history.append({"role": "user", "content": question})
        history.append({"role": "assistant", "content": answer})
        history = history[-(settings.MEMORY_MAX_MESSAGES):]
        client.setex(_memory_key(session_id), settings.MEMORY_TTL_SECONDS, json.dumps(history))
    except redis.RedisError as exc:
        logger.warning(f"Redis no disponible para escritura de memoria: {exc}")


# Stopwords y palabras genericas del dominio (verbos/sustantivos que
# aparecen en CASI TODA pregunta de mantenimiento -- "moto", "cambiar",
# "cada cuanto", etc.) -- se excluyen del calculo de solapamiento
# porque no discriminan entre temas distintos (aceite vs filtro de
# aire vs llantas), que es justo el problema que este chequeo evita.
_STOPWORDS_ES = {
    "que", "qué", "cual", "cuál", "cuando", "cuándo", "cada", "cuanto", "cuánto",
    "cuanta", "cuánta", "como", "cómo", "donde", "dónde", "por", "para", "con",
    "sin", "sobre", "entre", "hasta", "desde", "de", "del", "al", "a", "el", "la",
    "los", "las", "un", "una", "unos", "unas", "se", "su", "sus", "es", "son",
    "esta", "está", "están", "debe", "deben", "debo", "hay", "tiene", "tienen",
    "toca", "hace", "hacerle", "hacer", "va", "van", "moto", "motor", "motocicleta",
    "vehiculo", "vehículo", "manual", "taller",
    "cambiar", "cambia", "cambio", "cambios", "reemplazar", "reemplaza", "reemplazo",
}


def _palabras_clave(texto: str) -> set[str]:
    import re
    tokens = re.findall(r"[a-záéíóúñ]+", texto.lower())
    return {t for t in tokens if t not in _STOPWORDS_ES and len(t) > 2}


def _solapamiento_suficiente(pregunta_a: str, pregunta_b: str) -> bool:
    """
    Verifica que dos preguntas compartan al menos una palabra clave REAL
    (no generica del dominio), ademas de la similitud semantica. Sin
    esto, "cambio de aceite" y "cambio de filtro de aire" pueden
    resultar semanticamente muy similares (comparten la plantilla
    "cada cuanto se cambia X de la moto") aunque sean temas distintos --
    confirmado con datos reales: 0.8521 de similitud entre esos dos
    temas, MAS ALTO que la pregunta correcta (0.8253).
    """
    palabras_a = _palabras_clave(pregunta_a)
    palabras_b = _palabras_clave(pregunta_b)
    if not palabras_a or not palabras_b:
        return False
    interseccion = palabras_a & palabras_b
    union = palabras_a | palabras_b
    jaccard = len(interseccion) / len(union)
    return len(interseccion) >= 1 and jaccard >= 0.3


def get_semantic(question: str, query_embedding: list[float], filename: str | None = None) -> dict | None:
    """
    Busca en cache_semantico (Postgres/pgvector) una pregunta previa lo
    suficientemente similar (>= CACHE_SIMILARITY_THRESHOLD) Y con
    solapamiento real de palabras clave (ver _solapamiento_suficiente),
    dentro de la ventana de CACHE_TTL_SECONDS. Se llama SOLO si el hash
    exacto (get()) no encontro nada.
    """
    from app.services.db_service import db_cursor

    embedding_literal = "[" + ",".join(str(x) for x in query_embedding) + "]"

    try:
        with db_cursor() as (conn, cur):
            cur.execute(
                """
                SELECT question_text, result_json, 1 - (question_embedding <=> %s::vector) AS similitud
                FROM cache_semantico
                WHERE filename IS NOT DISTINCT FROM %s
                  AND created_at > NOW() - (%s || ' seconds')::interval
                ORDER BY question_embedding <=> %s::vector
                LIMIT 5
                """,
                [embedding_literal, filename, settings.CACHE_TTL_SECONDS, embedding_literal],
            )
            candidatos = cur.fetchall()
    except Exception as exc:
        logger.warning(f"Cache semantico no disponible para lectura: {exc}")
        return None

    for question_text, result_json, similitud in candidatos:
        if similitud < settings.CACHE_SIMILARITY_THRESHOLD:
            break  # ordenado por similitud descendente -- lo siguiente es aun menos similar
        if _solapamiento_suficiente(question, question_text):
            logger.info(f"Cache semantico HIT (similitud={similitud:.4f}, pregunta cacheada: {question_text!r})")
            return json.loads(result_json)
        logger.info(f"Cache semantico candidato RECHAZADO por bajo solapamiento de palabras clave (similitud={similitud:.4f}, pregunta cacheada: {question_text!r})")

    return None


def set_semantic(question: str, query_embedding: list[float], result: dict, filename: str | None = None) -> None:
    """Guarda la pregunta + su embedding + resultado en cache_semantico, para futuros hits por similitud."""
    from app.services.db_service import db_cursor

    embedding_literal = "[" + ",".join(str(x) for x in query_embedding) + "]"

    try:
        with db_cursor() as (conn, cur):
            cur.execute(
                """
                INSERT INTO cache_semantico (filename, question_text, question_embedding, result_json)
                VALUES (%s, %s, %s::vector, %s)
                """,
                [filename, question, embedding_literal, json.dumps(result, default=str)],
            )
    except Exception as exc:
        logger.warning(f"Cache semantico no disponible para escritura: {exc}")
