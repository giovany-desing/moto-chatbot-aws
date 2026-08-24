"""
Contextual Retrieval — genera una frase de contexto por chunk situándolo
dentro del documento completo, siguiendo la técnica documentada por
Anthropic (https://www.anthropic.com/news/contextual-retrieval). Esa
frase se antepone al chunk antes de generar su embedding, mejorando la
calidad de la búsqueda semántica sin cambiar el texto que se le muestra
al usuario final (el chunk original se sigue guardando intacto).
"""
from app.core.config import settings
from app.core.logging import get_logger
from app.services import llm_service

logger = get_logger(__name__)

_CONTEXT_PROMPT_TEMPLATE = """<document>
{document}
</document>
Aqui esta el fragmento que queremos situar dentro del documento completo:
<chunk>
{chunk}
</chunk>
Da un contexto breve y conciso (1-2 frases) que situe este fragmento
dentro del documento completo, con el unico proposito de mejorar la
busqueda por recuperacion de este fragmento. Responde unicamente con
el contexto, nada mas."""


def generate_chunk_context(document_text: str, chunk_text: str) -> str:
    """
    Genera una frase de contexto para un chunk usando Nova Lite.
    document_text se trunca para controlar costo y latencia
    (ver CONTEXT_MAX_DOC_CHARS en la configuracion).
    """
    documento_recortado = document_text[: settings.CONTEXT_MAX_DOC_CHARS]
    prompt = _CONTEXT_PROMPT_TEMPLATE.format(document=documento_recortado, chunk=chunk_text)

    try:
        contexto = llm_service.generate_simple(prompt)
        return contexto.strip()
    except Exception as exc:
        logger.warning(f"No se pudo generar contexto para el chunk, se indexa sin el: {exc}")
        return ""


def contextualize_chunk(document_text: str, chunk_text: str) -> str:
    """
    Devuelve el texto que se debe usar para generar el EMBEDDING
    (contexto + chunk) -- no reemplaza el chunk original que se guarda
    y se le muestra al usuario al citar la fuente.
    """
    if not settings.CONTEXTUAL_RETRIEVAL_ENABLED:
        return chunk_text

    contexto = generate_chunk_context(document_text, chunk_text)
    if not contexto:
        return chunk_text
    return f"{contexto}\n\n{chunk_text}"
