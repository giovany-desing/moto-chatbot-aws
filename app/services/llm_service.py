"""
Servicio de generación de respuestas del LLM.

Proveedor conmutable vía settings.LLM_PROVIDER:
- "groq"    -> Groq (Llama 3.3 70B), protocolo nativo de tool-calling
               (OpenAI-compatible: role="tool" + tool_call_id).
- "bedrock" -> Amazon Nova Lite (código original conservado, para casos
               atípicos que exijan mayor nivel de exigencia intelectual).
"""
import json
import time

import boto3
from botocore.exceptions import ClientError
from groq import Groq
from groq import APIConnectionError, APIStatusError

from app.core.config import settings
from app.core.logging import get_logger

logger = get_logger(__name__)

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

_groq_client: Groq | None = None
_bedrock = boto3.client("bedrock-runtime", region_name=settings.BEDROCK_REGION)


def _get_groq_client() -> Groq:
    global _groq_client
    if _groq_client is None:
        if not settings.GROQ_API_KEY:
            raise RuntimeError("GROQ_API_KEY no está configurada en el entorno (.env)")
        _groq_client = Groq(api_key=settings.GROQ_API_KEY)
    return _groq_client


def _build_context_block(chunks: list[dict]) -> str:
    if not chunks:
        return "No se encontró contexto relevante en los manuales indexados."
    partes = [
        f"[Manual: {c['filename']} — página {c['page']}]\n{c['text']}"
        for c in chunks
    ]
    return "\n\n---\n\n".join(partes)


def _build_user_prompt(question: str, context: list[dict]) -> str:
    contexto_texto = _build_context_block(context)
    return f"Contexto de manuales:\n{contexto_texto}\n\nPregunta del mecánico: {question}"


# =====================================================================
# GROQ (proveedor por defecto) — protocolo nativo de tool-calling
# =====================================================================

def _bedrock_tools_to_openai(tools: list[dict]) -> list[dict]:
    """Convierte el formato toolSpec/inputSchema de Bedrock al formato
    function-calling de OpenAI/Groq, sin tocar los MCP servers."""
    converted = []
    for t in tools:
        spec = t["toolSpec"]
        converted.append({
            "type": "function",
            "function": {
                "name": spec["name"],
                "description": spec["description"],
                "parameters": spec["inputSchema"]["json"],
            },
        })
    return converted


def _groq_invoke_with_retry(**kwargs):
    client = _get_groq_client()
    last_exc: Exception | None = None
    for attempt in range(_MAX_RETRIES):
        try:
            return client.chat.completions.create(**kwargs)
        except (APIStatusError, APIConnectionError) as exc:
            status = getattr(exc, "status_code", None)
            last_exc = exc
            if status == 429 or isinstance(exc, APIConnectionError):
                delay = _BASE_DELAY * (2 ** attempt)
                logger.warning(f"Throttling/conexión en Groq, reintento en {delay}s (intento {attempt + 1})")
                time.sleep(delay)
                continue
            raise
    raise last_exc  # type: ignore[misc]


def _groq_generate(question: str, context: list[dict], memory: list[dict], system_prompt: str | None = None) -> str:
    messages = [{"role": "system", "content": system_prompt or _SYSTEM_PROMPT}]
    for turno in memory[-settings.MEMORY_MAX_MESSAGES:]:
        messages.append({"role": turno["role"], "content": turno["content"]})
    messages.append({"role": "user", "content": _build_user_prompt(question, context)})

    response = _groq_invoke_with_retry(
        model=settings.GROQ_MODEL,
        messages=messages,
        temperature=0.2,
        top_p=0.9,
        max_tokens=1024,
    )
    return response.choices[0].message.content


def _groq_generate_simple(prompt: str, max_tokens: int) -> str:
    response = _groq_invoke_with_retry(
        model=settings.GROQ_MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.0,
        top_p=0.9,
        max_tokens=max_tokens,
    )
    return response.choices[0].message.content


def _groq_run_agentic(question, context, tools, execute_tool, memory=None, system_prompt=None, max_iterations=4) -> dict:
    """
    Loop de tool-calling con protocolo nativo de Groq: cada tool_call
    devuelto por el modelo se ejecuta y su resultado se agrega como un
    mensaje role="tool" con el tool_call_id correspondiente, preservando
    el estado completo de la conversación entre iteraciones (en vez de
    inyectar los resultados como texto plano en el prompt).
    """
    openai_tools = _bedrock_tools_to_openai(tools)

    messages = [{"role": "system", "content": system_prompt or _SYSTEM_PROMPT}]
    for turno in (memory or [])[-settings.MEMORY_MAX_MESSAGES:]:
        messages.append({"role": turno["role"], "content": turno["content"]})
    messages.append({"role": "user", "content": _build_user_prompt(question, context)})

    tools_used: list[str] = []
    iteraciones_restantes = max_iterations

    while iteraciones_restantes > 0:
        iteraciones_restantes -= 1
        response = _groq_invoke_with_retry(
            model=settings.GROQ_MODEL,
            messages=messages,
            tools=openai_tools,
            tool_choice="auto",
            temperature=0.2,
            top_p=0.9,
            max_tokens=1024,
        )
        message = response.choices[0].message

        if not message.tool_calls:
            return {"text": message.content or "", "tools_used": tools_used}

        messages.append({
            "role": "assistant",
            "content": message.content,
            "tool_calls": [
                {
                    "id": tc.id,
                    "type": "function",
                    "function": {"name": tc.function.name, "arguments": tc.function.arguments},
                }
                for tc in message.tool_calls
            ],
        })

        for tc in message.tool_calls:
            nombre_tool = tc.function.name
            try:
                parametros = json.loads(tc.function.arguments)
            except json.JSONDecodeError:
                logger.error(f"Argumentos inválidos del modelo para tool {nombre_tool}: {tc.function.arguments!r}")
                parametros = {}

            logger.info(f"Ejecutando herramienta MCP (Groq nativo): {nombre_tool}")
            result = execute_tool(nombre_tool, parametros)
            tools_used.append(nombre_tool)

            messages.append({
                "role": "tool",
                "tool_call_id": tc.id,
                "content": json.dumps(result, ensure_ascii=False, default=str),
            })

    logger.warning(f"Se alcanzó el límite de {max_iterations} iteraciones de tool-calling sin respuesta final")
    return {
        "text": "No pude completar la consulta tras varias herramientas. Por favor reformula tu pregunta.",
        "tools_used": tools_used,
    }


# =====================================================================
# BEDROCK (Amazon Nova Lite) — conservado para casos atípicos
# =====================================================================

def _bedrock_build_messages(question, context, memory, tool_results=None) -> list[dict]:
    messages = []
    for turno in memory[-settings.MEMORY_MAX_MESSAGES:]:
        messages.append({"role": turno["role"], "content": [{"text": turno["content"]}]})

    prompt = _build_user_prompt(question, context)
    if tool_results:
        resultados_texto = json.dumps(tool_results, ensure_ascii=False, default=str)
        prompt += f"\n\nResultados de herramientas consultadas: {resultados_texto}"

    messages.append({"role": "user", "content": [{"text": prompt}]})
    return messages


def _bedrock_invoke_with_retry(payload: dict) -> dict:
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


def _bedrock_generate(question, context, memory, system_prompt=None) -> str:
    payload = {
        "schemaVersion": "messages-v1",
        "system": [{"text": system_prompt or _SYSTEM_PROMPT}],
        "messages": _bedrock_build_messages(question, context, memory),
        "inferenceConfig": {"maxTokens": 1024, "temperature": 0.2, "topP": 0.9},
    }
    result = _bedrock_invoke_with_retry(payload)
    return result["output"]["message"]["content"][0]["text"]


def _bedrock_generate_simple(prompt: str, max_tokens: int) -> str:
    payload = {
        "schemaVersion": "messages-v1",
        "messages": [{"role": "user", "content": [{"text": prompt}]}],
        "inferenceConfig": {"maxTokens": max_tokens, "temperature": 0.0, "topP": 0.9},
    }
    result = _bedrock_invoke_with_retry(payload)
    return result["output"]["message"]["content"][0]["text"]


def _bedrock_parse_tool_response(result: dict) -> dict:
    content_blocks = result.get("output", {}).get("message", {}).get("content", [])
    text_parts = [b["text"] for b in content_blocks if "text" in b]
    tool_calls = [
        {"name": b["toolUse"]["name"], "parameters": b["toolUse"].get("input", {})}
        for b in content_blocks
        if "toolUse" in b
    ]
    return {"text": "\n".join(text_parts), "tool_calls": tool_calls}


def _bedrock_run_agentic(question, context, tools, execute_tool, memory=None, system_prompt=None, max_iterations=4) -> dict:
    payload = {
        "schemaVersion": "messages-v1",
        "system": [{"text": system_prompt or _SYSTEM_PROMPT}],
        "messages": _bedrock_build_messages(question, context, memory or []),
        "toolConfig": {"tools": tools},
        "inferenceConfig": {"maxTokens": 1024, "temperature": 0.2, "topP": 0.9},
    }
    result = _bedrock_invoke_with_retry(payload)
    parsed = _bedrock_parse_tool_response(result)

    tools_used: list[str] = []
    tool_results: list[dict] = []
    iteraciones_restantes = max_iterations

    while parsed["tool_calls"] and iteraciones_restantes > 0:
        iteraciones_restantes -= 1
        for tool_call in parsed["tool_calls"]:
            logger.info(f"Ejecutando herramienta MCP (Bedrock): {tool_call['name']}")
            result_tool = execute_tool(tool_call["name"], tool_call["parameters"])
            tool_results.append({"tool": tool_call["name"], "result": result_tool})
            tools_used.append(tool_call["name"])

        payload = {
            "schemaVersion": "messages-v1",
            "system": [{"text": system_prompt or _SYSTEM_PROMPT}],
            "messages": _bedrock_build_messages(question, context, memory or [], tool_results),
            "inferenceConfig": {"maxTokens": 1024, "temperature": 0.2, "topP": 0.9},
        }
        result = _bedrock_invoke_with_retry(payload)
        parsed = _bedrock_parse_tool_response(result)

    return {"text": parsed["text"], "tools_used": tools_used}


# =====================================================================
# Interfaz pública — usada por rag_service.py, sales_service.py, context_service.py
# =====================================================================

def generate(question: str, context: list[dict], memory: list[dict] | None = None) -> str:
    """Generación simple con contexto RAG, sin herramientas."""
    if settings.LLM_PROVIDER == "groq":
        return _groq_generate(question, context, memory or [])
    elif settings.LLM_PROVIDER == "bedrock":
        return _bedrock_generate(question, context, memory or [])
    raise ValueError(f"LLM_PROVIDER desconocido: {settings.LLM_PROVIDER!r}")


def generate_simple(prompt: str, max_tokens: int = 200) -> str:
    """Llamada simple sin RAG/memoria/herramientas (ej. Contextual Retrieval)."""
    if settings.LLM_PROVIDER == "groq":
        return _groq_generate_simple(prompt, max_tokens)
    elif settings.LLM_PROVIDER == "bedrock":
        return _bedrock_generate_simple(prompt, max_tokens)
    raise ValueError(f"LLM_PROVIDER desconocido: {settings.LLM_PROVIDER!r}")


def run_agentic(question: str, context: list[dict], tools: list[dict], execute_tool, memory: list[dict] | None = None, system_prompt: str | None = None, max_iterations: int = 4) -> dict:
    """
    Orquesta el loop completo de tool-calling (múltiples iteraciones hasta
    que el modelo responda sin pedir más herramientas o se agote
    max_iterations). Devuelve {"text": str, "tools_used": list[str]}.

    execute_tool: callable(nombre: str, parametros: dict) -> dict — normalmente
    app.mcp.taller_server.execute_tool o app.mcp.ventas_server.execute_tool.
    """
    if settings.LLM_PROVIDER == "groq":
        return _groq_run_agentic(question, context, tools, execute_tool, memory, system_prompt, max_iterations)
    elif settings.LLM_PROVIDER == "bedrock":
        return _bedrock_run_agentic(question, context, tools, execute_tool, memory, system_prompt, max_iterations)
    raise ValueError(f"LLM_PROVIDER desconocido: {settings.LLM_PROVIDER!r}")
