"""
IndexingWorker — se dispara por mensajes SQS (ver lambdas/worker_handler.py).
"""
from app.core.logging import get_logger
from app.services import storage_service, embedding_service, vector_service
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

        textos = [c["text"] for c in fragmentos]
        embeddings = embedding_service.embed_batch(textos)

        for fragmento, embedding in zip(fragmentos, embeddings):
            fragmento["embedding"] = embedding

        manual_id = vector_service.register_manual(filename)
        total = vector_service.save_chunks(manual_id, filename, fragmentos)

        logger.info(f"{filename}: indexación completa ({total} chunks)")
        return {"filename": filename, "status": "indexado", "chunks": total}
