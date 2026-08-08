"""
Orquestador RAG. Flujo:
1. Caché Redis (hit -> respuesta inmediata)
2. Embed de la pregunta (Bedrock Titan)
3. Retrieval HIBRIDO en pgvector (dense + sparse + RRF + reranking)
4. Memoria conversacional (Redis, si hay session_id)
5. Generación con Amazon Nova, opcionalmente con herramientas MCP
6. Guardar en caché y memoria
"""
from app.core.config import settings
from app.core.logging import get_logger
from app.services import cache_service, embedding_service, vector_service, bedrock_service

logger = get_logger(__name__)

_PREVIEW_CHARS = 240


def _fuentes(chunks: list[dict]) -> list[dict]:
    return [
        {
            "filename": c["filename"],
            "page": c["page"],
            "relevance": c["relevance"],
            "text_preview": c["text"][:_PREVIEW_CHARS],
        }
        for c in chunks
    ]


def answer(question: str, filename: str | None = None, session_id: str | None = None) -> dict:
    cached = cache_service.get(question, filename)
    if cached:
        return {**cached, "from_cache": True}

    query_embedding = embedding_service.embed_query(question)
    chunks = vector_service.search_hybrid(query_embedding, question, filename)
    memory = cache_service.get_memory(session_id or "")

    respuesta_texto = bedrock_service.generate(question, chunks, memory)

    result = {
        "answer": respuesta_texto,
        "sources": _fuentes(chunks),
        "from_cache": False,
        "tools_used": [],
    }

    cache_service.set(question, result, filename)
    cache_service.save_memory(session_id or "", question, respuesta_texto)
    return result


def answer_with_tools(question: str, filename: str | None = None, session_id: str | None = None) -> dict:
    """
    Igual que answer(), pero permite que el LLM invoque herramientas MCP
    (inventario, órdenes de servicio, historial de vehículos) antes de
    responder. Ver app/mcp/taller_server.py.
    """
    from app.mcp.taller_server import get_tools_schema, execute_tool

    cached = cache_service.get(question, filename)
    if cached:
        return {**cached, "from_cache": True}

    query_embedding = embedding_service.embed_query(question)
    chunks = vector_service.search_hybrid(query_embedding, question, filename)
    memory = cache_service.get_memory(session_id or "")
    tools = get_tools_schema()

    response = bedrock_service.generate_with_tools(question, chunks, tools, memory)

    tool_results: list[dict] = []
    max_iteraciones = 4
    while response.get("tool_calls") and max_iteraciones > 0:
        max_iteraciones -= 1
        for tool_call in response["tool_calls"]:
            nombre_tool = tool_call["name"]
            logger.info(f"Ejecutando herramienta MCP: {nombre_tool}")
            result = execute_tool(tool_call["name"], tool_call["parameters"])
            tool_results.append({"tool": tool_call["name"], "result": result})

        response = bedrock_service.generate_with_tool_results(question, chunks, tool_results, memory)

    result = {
        "answer": response["text"],
        "sources": _fuentes(chunks),
        "from_cache": False,
        "tools_used": [t["tool"] for t in tool_results],
    }

    cache_service.set(question, result, filename)
    cache_service.save_memory(session_id or "", question, result["answer"])
    return result
