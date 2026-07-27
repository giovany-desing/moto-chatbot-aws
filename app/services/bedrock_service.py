"""
Servicio Amazon Nova Lite (Bedrock) — generación de respuestas.

El formato de mensajes de Nova es distinto al de la API de mensajes de
Claude/Anthropic — usa el bloque "content": [{"text": "..."}] dentro
de cada mensaje del array "messages", y "system" como lista separada.
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

_SYSTEM_PROMPT = (
    "Eres un asistente técnico para mecánicos de motocicletas. Respondes en "
    "español, de forma precisa y concreta, basándote únicamente en el "
    "contexto de manuales de taller proporcionado. Cuando cites una "
    "especificación, indica de qué página del manual proviene. Si no "
    "tienes información suficiente en el contexto, dilo claramente en "
    "lugar de inventar datos."
)


def _build_context_block(chunks: list[dict]) -> str:
    if not chunks:
        return "No se encontró contexto relevante en los manuales indexados."
    partes = [
        f"[Manual: {c['filename']} — página {c['page']}]\n{c['text']}"
        for c in chunks
    ]
    return "\n\n---\n\n".join(partes)


def _build_messages(question: str, context: list[dict], memory: list[dict], tool_results: list[dict] | None = None) -> list[dict]:
    messages = []
    for turno in memory[-settings.MEMORY_MAX_MESSAGES:]:
        messages.append({"role": turno["role"], "content": [{"text": turno["content"]}]})

    contexto_texto = _build_context_block(context)
    prompt = f"Contexto de manuales:\n{contexto_texto}\n\nPregunta del mecánico: {question}"

    if tool_results:
        resultados_texto = json.dumps(tool_results, ensure_ascii=False, default=str)
        prompt += f"\n\nResultados de herramientas consultadas: {resultados_texto}"

    messages.append({"role": "user", "content": [{"text": prompt}]})
    return messages


def _invoke_with_retry(payload: dict) -> dict:
    last_exc: Exception | None = None
    for attempt in range(_MAX_RETRIES):
        try:
            response = _bedrock.invoke_model(
                modelId=settings.LLM_MODEL,
                body=json.dumps(payload),
                contentType="application/json",
                accept="application/json",
            )
            return json.loads(response["body"].read())
        except ClientError as exc:
            error_code = exc.response.get("Error", {}).get("Code", "")
            last_exc = exc
            if error_code in ("ThrottlingException", "TooManyRequestsException"):
                delay = _BASE_DELAY * (2 ** attempt)
                logger.warning(f"Throttling en Bedrock Nova, reintento en {delay}s (intento {attempt + 1})")
                time.sleep(delay)
                continue
            raise
    raise last_exc  # type: ignore[misc]


def generate(question: str, context: list[dict], memory: list[dict] | None = None) -> str:
    payload = {
        "schemaVersion": "messages-v1",
        "system": [{"text": _SYSTEM_PROMPT}],
        "messages": _build_messages(question, context, memory or []),
        "inferenceConfig": {"maxTokens": 1024, "temperature": 0.2, "topP": 0.9},
    }
    result = _invoke_with_retry(payload)
    return result["output"]["message"]["content"][0]["text"]


def generate_with_tools(question: str, context: list[dict], tools: list[dict], memory: list[dict] | None = None, system_prompt: str | None = None) -> dict:
    """
    Primera llamada al LLM con definición de herramientas MCP disponibles.
    Devuelve {"text": str, "tool_calls": [...]} — tool_calls vacío si el
    modelo respondió directamente sin usar herramientas.
    system_prompt permite reutilizar esta función con distintos "roles"
    (mecánico interno, atención al cliente, etc.) sin duplicar código.
    """
    payload = {
        "schemaVersion": "messages-v1",
        "system": [{"text": system_prompt or _SYSTEM_PROMPT}],
        "messages": _build_messages(question, context, memory or []),
        "toolConfig": {"tools": tools},
        "inferenceConfig": {"maxTokens": 1024, "temperature": 0.2, "topP": 0.9},
    }
    result = _invoke_with_retry(payload)
    return _parse_tool_response(result)


def generate_with_tool_results(question: str, context: list[dict], tool_results: list[dict], memory: list[dict] | None = None, system_prompt: str | None = None) -> dict:
    payload = {
        "schemaVersion": "messages-v1",
        "system": [{"text": system_prompt or _SYSTEM_PROMPT}],
        "messages": _build_messages(question, context, memory or [], tool_results),
        "inferenceConfig": {"maxTokens": 1024, "temperature": 0.2, "topP": 0.9},
    }
    result = _invoke_with_retry(payload)
    return _parse_tool_response(result)


def generate_with_tools_and_prompt(question: str, context: list[dict], tools: list[dict], memory: list[dict] | None, system_prompt: str) -> dict:
    """Alias explícito de generate_with_tools() para llamadas que siempre pasan un system_prompt propio (ej. sales_service.py)."""
    return generate_with_tools(question, context, tools, memory, system_prompt=system_prompt)


def generate_with_tool_results_and_prompt(question: str, context: list[dict], tool_results: list[dict], memory: list[dict] | None, system_prompt: str) -> dict:
    """Alias explícito de generate_with_tool_results() para llamadas que siempre pasan un system_prompt propio (ej. sales_service.py)."""
    return generate_with_tool_results(question, context, tool_results, memory, system_prompt=system_prompt)


def _parse_tool_response(result: dict) -> dict:
    content_blocks = result.get("output", {}).get("message", {}).get("content", [])
    text_parts = [b["text"] for b in content_blocks if "text" in b]
    tool_calls = [
        {"name": b["toolUse"]["name"], "parameters": b["toolUse"].get("input", {})}
        for b in content_blocks
        if "toolUse" in b
    ]
    return {"text": "\n".join(text_parts), "tool_calls": tool_calls}
