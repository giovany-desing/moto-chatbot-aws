"""
Orquestador RAG. Flujo:
1. Caché EXACTA Redis (hash -> hit inmediato, la mas rapida)
2. Embed de la pregunta (embedding_service, proveedor conmutable: local/bedrock)
3. Caché SEMANTICA Postgres/pgvector (similitud >= CACHE_SIMILARITY_THRESHOLD
   -> hit, sin llamar al LLM aunque la pregunta no sea textualmente identica)
4. Retrieval HIBRIDO en pgvector (dense + sparse + RRF + reranking)
4b. Corrective RAG (grading liviano): si NINGUN chunk supera
    CRAG_RELEVANCE_THRESHOLD, se descartan todos -- evita forzar una
    respuesta con contexto irrelevante/debil (ver
    vector_service.chunks_son_relevantes())
5. Memoria conversacional (Redis, si hay session_id)
6. Generación con llm_service (proveedor conmutable: groq/bedrock),
   opcionalmente con herramientas MCP vía tool-calling nativo
7. Guardar en ambas capas de caché y en memoria
"""
from app.core.config import settings
from app.core.logging import get_logger
from app.core.observability import observe
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


@observe()
def answer(question: str, filename: str | None = None, session_id: str | None = None) -> dict:
    cached = cache_service.get(question, filename)
    if cached:
        return {**cached, "from_cache": True}

    query_embedding = embedding_service.embed_query(question)

    cached_semantico = cache_service.get_semantic(question, query_embedding, filename)
    if cached_semantico:
        return {**cached_semantico, "from_cache": True}

    chunks = vector_service.search_hybrid(query_embedding, question, filename)
    if not vector_service.chunks_son_relevantes(chunks):
        logger.info("Corrective RAG: ningún chunk supera el umbral de relevancia, se trata como sin contexto")
        chunks = []
    memory = cache_service.get_memory(session_id or "")

    respuesta_texto = llm_service.generate(question, chunks, memory)

    result = {
        "answer": respuesta_texto,
        "sources": _fuentes(chunks),
        "from_cache": False,
        "tools_used": [],
    }

    cache_service.set(question, result, filename)
    cache_service.set_semantic(question, query_embedding, result, filename)
    cache_service.save_memory(session_id or "", question, respuesta_texto)
    return result


@observe()
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

    cached_semantico = cache_service.get_semantic(question, query_embedding, filename)
    if cached_semantico:
        return {**cached_semantico, "from_cache": True}

    chunks = vector_service.search_hybrid(query_embedding, question, filename)
    if not vector_service.chunks_son_relevantes(chunks):
        logger.info("Corrective RAG: ningún chunk supera el umbral de relevancia, se trata como sin contexto")
        chunks = []
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
    cache_service.set_semantic(question, query_embedding, result, filename)
    cache_service.save_memory(session_id or "", question, result["answer"])
    return result
