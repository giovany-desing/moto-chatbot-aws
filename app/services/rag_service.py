"""
Orquestador RAG. Flujo:
1. Caché Redis (hit -> respuesta inmediata)
2. Embed de la pregunta (embedding_service, proveedor conmutable: local/bedrock)
3. Retrieval HIBRIDO en pgvector (dense + sparse + RRF + reranking)
4. Memoria conversacional (Redis, si hay session_id)
5. Generación con llm_service (proveedor conmutable: groq/bedrock),
   opcionalmente con herramientas MCP vía tool-calling nativo
6. Guardar en caché y memoria
"""
from app.core.config import settings
from app.core.logging import get_logger
from app.services import cache_service, embedding_service, vector_service, llm_service

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

    respuesta_texto = llm_service.generate(question, chunks, memory)

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

    response = llm_service.run_agentic(question, chunks, tools, execute_tool, memory)

    result = {
        "answer": response["text"],
        "sources": _fuentes(chunks),
        "from_cache": False,
        "tools_used": response["tools_used"],
    }

    cache_service.set(question, result, filename)
    cache_service.save_memory(session_id or "", question, result["answer"])
    return result
