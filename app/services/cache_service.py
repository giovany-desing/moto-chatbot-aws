"""
Servicio Redis — caché de respuestas (hash exacto) y memoria conversacional.
"""
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
