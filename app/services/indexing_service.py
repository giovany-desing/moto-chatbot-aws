"""
Pipeline de indexación: PDF -> texto -> chunks -> embeddings -> pgvector.

Chunking RECURSIVO basado en TOKENS (no caracteres): intenta cortar por
párrafos primero, luego por líneas, luego por oraciones, y solo cae a
corte fijo por caracteres como último recurso -- respetando siempre un
presupuesto de tokens reales (via tiktoken), que es lo que realmente
consume el modelo de embeddings/LLM, no un proxy aproximado como el
conteo de caracteres que usaba la version anterior.
"""
import fitz  # PyMuPDF
import tiktoken

from app.core.config import settings
from app.core.logging import get_logger

logger = get_logger(__name__)

_ENCODER = tiktoken.get_encoding("cl100k_base")

# Separadores probados en orden: parrafo -> linea -> oracion -> palabra.
# "" es el ultimo recurso (corte fijo por caracteres, para texto sin
# ningun separador natural, ej. codigos o texto extraido mal del PDF).
_SEPARADORES = ["\n\n", "\n", ". ", " ", ""]


def extract_text_by_page(pdf_bytes: bytes) -> list[dict]:
    """Devuelve [{"page": int, "text": str}, ...] usando PyMuPDF."""
    paginas = []
    with fitz.open(stream=pdf_bytes, filetype="pdf") as doc:
        for numero, pagina in enumerate(doc, start=1):
            texto = pagina.get_text("text").strip()
            if texto:
                paginas.append({"page": numero, "text": texto})
    return paginas


def _contar_tokens(texto: str) -> int:
    return len(_ENCODER.encode(texto))


def _dividir_recursivo(texto: str, separadores: list[str]) -> list[str]:
    """
    Parte el texto en piezas "atomicas" probando cada separador en orden
    hasta que cada pieza quepa dentro del presupuesto de tokens. Todavia
    NO fusiona piezas pequeñas entre si -- eso lo hace _fusionar_con_solape().
    """
    texto = texto.strip()
    if not texto:
        return []

    if _contar_tokens(texto) <= settings.CHUNK_SIZE_TOKENS:
        return [texto]

    if not separadores:
        return [texto]

    separador, *resto = separadores

    if separador == "":
        # Ultimo recurso: corte fijo por caracteres (aprox. 4 chars/token,
        # razon conservadora para no exceder el presupuesto real).
        chars_por_pieza = max(settings.CHUNK_SIZE_TOKENS * 4, 50)
        return [
            texto[i : i + chars_por_pieza].strip()
            for i in range(0, len(texto), chars_por_pieza)
            if texto[i : i + chars_por_pieza].strip()
        ]

    piezas = [p.strip() for p in texto.split(separador) if p.strip()]
    if len(piezas) <= 1:
        # el separador no partio nada real (ej. no hay saltos de linea) --
        # probar el siguiente separador de la lista
        return _dividir_recursivo(texto, resto)

    resultado = []
    for pieza in piezas:
        if _contar_tokens(pieza) <= settings.CHUNK_SIZE_TOKENS:
            resultado.append(pieza)
        else:
            resultado.extend(_dividir_recursivo(pieza, resto))
    return resultado


def _fusionar_con_solape(piezas: list[str], max_tokens: int, overlap_tokens: int) -> list[str]:
    """
    Fusiona piezas atomicas consecutivas hasta acercarse al presupuesto
    de tokens por chunk, manteniendo un solape real (en tokens) entre
    chunks consecutivos para no perder contexto en el corte.
    """
    if not piezas:
        return []

    chunks = []
    actual: list[str] = []
    tokens_actual = 0

    for pieza in piezas:
        tokens_pieza = _contar_tokens(pieza)

        if actual and tokens_actual + tokens_pieza > max_tokens:
            chunks.append(" ".join(actual))

            solape: list[str] = []
            tokens_solape = 0
            for p in reversed(actual):
                t = _contar_tokens(p)
                if tokens_solape + t > overlap_tokens:
                    break
                solape.insert(0, p)
                tokens_solape += t
            actual = solape
            tokens_actual = tokens_solape

        actual.append(pieza)
        tokens_actual += tokens_pieza

    if actual:
        chunks.append(" ".join(actual))

    return chunks


def chunk_text(paginas: list[dict]) -> list[dict]:
    """
    Chunking recursivo basado en tokens: respeta parrafos/lineas/oraciones
    cuando existen, y solo cae a corte por caracteres para texto sin
    ninguna estructura reconocible. Descarta fragmentos por debajo de
    CHUNK_MIN_SIZE_TOKENS.
    """
    chunks = []

    for pagina in paginas:
        piezas = _dividir_recursivo(pagina["text"], _SEPARADORES)
        fragmentos = _fusionar_con_solape(piezas, settings.CHUNK_SIZE_TOKENS, settings.CHUNK_OVERLAP_TOKENS)

        for fragmento in fragmentos:
            if _contar_tokens(fragmento) >= settings.CHUNK_MIN_SIZE_TOKENS:
                chunks.append({"page": pagina["page"], "text": fragmento})

    return chunks
