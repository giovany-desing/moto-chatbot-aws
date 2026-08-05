"""
IndexingWorker — se dispara por mensajes SQS (ver lambdas/worker_handler.py).

Flujo: descarga PDF de S3 -> extrae texto con PyMuPDF -> genera chunks
(1000 chars, 200 overlap) -> (opcional) contextualiza cada chunk con
Contextual Retrieval -> genera embeddings en batch con Bedrock Titan V2
-> guarda en pgvector -> marca el manual como indexado.

Contextual Retrieval (ver app/services/context_service.py): si está
activo, el embedding de cada chunk se genera a partir del chunk +
una frase de contexto que lo sitúa dentro del documento completo.
El texto ORIGINAL del chunk (sin el contexto agregado) es el que se
guarda y se le muestra al usuario al citar la fuente -- el contexto
solo mejora la calidad de la búsqueda, nunca se ve en la respuesta.
"""
from app.core.config import settings
from app.core.logging import get_logger
from app.services import storage_service, embedding_service, vector_service, context_service
from app.services.indexing_service import extract_text_by_page, chunk_text

logger = get_logger(__name__)


class IndexingWorker:
    def process(self, filename: str) -> dict:
        logger.info(f"Iniciando indexación de {filename}")

        pdf_bytes = storage_service.download_pdf(filename)
        paginas = extract_text_by_page(pdf_bytes)
        logger.info(f"{filename}: extraídas {len(paginas)} páginas con texto")

        fragmentos = chunk_text(paginas)
        logger.info(f"{filename}: generados {len(fragmentos)} chunks")

        if not fragmentos:
            logger.warning(f"{filename}: no se generaron chunks, el PDF puede ser escaneado sin OCR")
            return {"filename": filename, "status": "sin_contenido", "chunks": 0}

        if settings.CONTEXTUAL_RETRIEVAL_ENABLED:
            logger.info(f"{filename}: generando contexto por chunk (Contextual Retrieval activo)")
            documento_completo = "\n".join(p["text"] for p in paginas)
            textos_para_embedding = [
                context_service.contextualize_chunk(documento_completo, c["text"])
                for c in fragmentos
            ]
        else:
            textos_para_embedding = [c["text"] for c in fragmentos]

        embeddings = embedding_service.embed_batch(textos_para_embedding)

        for fragmento, embedding in zip(fragmentos, embeddings):
            fragmento["embedding"] = embedding

        manual_id = vector_service.register_manual(filename)
        total = vector_service.save_chunks(manual_id, filename, fragmentos)

        logger.info(f"{filename}: indexación completa ({total} chunks)")
        return {"filename": filename, "status": "indexado", "chunks": total}
