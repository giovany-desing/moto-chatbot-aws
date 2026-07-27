"""
Pipeline de indexación: PDF -> texto -> chunks -> embeddings -> pgvector.
"""
import fitz  # PyMuPDF

from app.core.config import settings
from app.core.logging import get_logger

logger = get_logger(__name__)


def extract_text_by_page(pdf_bytes: bytes) -> list[dict]:
    """Devuelve [{"page": int, "text": str}, ...] usando PyMuPDF."""
    paginas = []
    with fitz.open(stream=pdf_bytes, filetype="pdf") as doc:
        for numero, pagina in enumerate(doc, start=1):
            texto = pagina.get_text("text").strip()
            if texto:
                paginas.append({"page": numero, "text": texto})
    return paginas


def chunk_text(paginas: list[dict]) -> list[dict]:
    """
    Chunking por caracteres: CHUNK_SIZE con CHUNK_OVERLAP,
    descartando fragmentos menores a CHUNK_MIN_SIZE.
    """
    chunks = []
    size = settings.CHUNK_SIZE
    overlap = settings.CHUNK_OVERLAP
    min_size = settings.CHUNK_MIN_SIZE

    for pagina in paginas:
        texto = pagina["text"]
        inicio = 0
        while inicio < len(texto):
            fin = inicio + size
            fragmento = texto[inicio:fin].strip()
            if len(fragmento) >= min_size:
                chunks.append({"page": pagina["page"], "text": fragmento})
            if fin >= len(texto):
                break
            inicio = fin - overlap

    return chunks
