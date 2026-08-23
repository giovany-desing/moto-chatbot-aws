"""
Orquestador del asistente de atención al cliente / ventas ("empleado
digital" comercial). Mismo patrón que rag_service.answer_with_tools(),
pero con un system prompt de tono comercial y las herramientas de
ventas (app/mcp/ventas_server.py) en vez de las de taller.
LLM vía llm_service (proveedor conmutable: groq/bedrock).
"""
from app.core.config import settings
from app.core.logging import get_logger
from app.core.observability import observe
from app.core.guardrails import detect_prompt_injection
from app.services import cache_service, embedding_service, vector_service, llm_service

logger = get_logger(__name__)

_PREVIEW_CHARS = 240

_SALES_SYSTEM_PROMPT = (
    "Eres el asistente virtual de atención al cliente de una empresa de "
    "motocicletas. Hablas en español, de forma cercana, clara y profesional, "
    "las 24 horas del día. Ayudas a los clientes a resolver dudas sobre "
    "motos, financiamiento, mantenimiento y repuestos, y los guías durante "
    "el proceso de compra recomendando productos según sus necesidades. "
    "Cuando menciones una cuota de financiamiento, SIEMPRE aclara que es "
    "una estimación referencial y no una oferta formal. Si el cliente pide "
    "hablar con un asesor humano, o si su solicitud requiere algo que no "
    "puedes resolver tú (negociación de precio, trámite formal de crédito, "
    "agendar una prueba de manejo), regístralo como lead usando la "
    "herramienta correspondiente y comunícale que un asesor se pondrá en "
    "contacto. No inventes precios, tasas ni disponibilidad — usa siempre "
    "las herramientas para consultar datos reales."
)


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
def answer_cliente(question: str, session_id: str | None = None) -> dict:
    from app.mcp.ventas_server import get_tools_schema, execute_tool

    if detect_prompt_injection(question):
        return {
            "answer": "No puedo procesar esa solicitud. Si tienes una pregunta sobre nuestras motos, financiamiento o repuestos, con gusto te ayudo.",
            "sources": [],
            "from_cache": False,
            "tools_used": [],
            "blocked": True,
        }

    cached = cache_service.get(question, "cliente")
    if cached:
        return {**cached, "from_cache": True}

    query_embedding = embedding_service.embed_query(question)

    cached_semantico = cache_service.get_semantic(question, query_embedding, "cliente")
    if cached_semantico:
        return {**cached_semantico, "from_cache": True}

    chunks = vector_service.search_hybrid(query_embedding, question)
    memory = cache_service.get_memory(session_id or "")
    tools = get_tools_schema()

    response = llm_service.run_agentic(
        question, chunks, tools, execute_tool, memory, system_prompt=_SALES_SYSTEM_PROMPT
    )

    result = {
        "answer": response["text"],
        "sources": _fuentes(chunks),
        "from_cache": False,
        "tools_used": response["tools_used"],
    }

    cache_service.set(question, result, "cliente")
    cache_service.set_semantic(question, query_embedding, result, "cliente")
    cache_service.save_memory(session_id or "", question, result["answer"])
    return result
