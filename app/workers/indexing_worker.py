"""
IndexingWorker — se dispara por mensajes SQS (ver lambdas/worker_handler.py).

Flujo: descarga PDF de S3 -> extrae texto con PyMuPDF -> genera chunks
(1000 chars, 200 overlap) -> procesa POR LOTES pequeños: (opcional)
contextualiza cada chunk con Contextual Retrieval -> genera embeddings
en batch -> GUARDA el lote en pgvector inmediatamente -> repite con el
siguiente lote -> marca el manual como indexado al final.

Procesar por lotes (en vez de todo el documento de una sola vez) es
deliberado: si el proceso falla a mitad de camino (rate limit del LLM,
caida de red, timeout de un tunel de acceso a la BD, etc.), los lotes
ya guardados no se pierden -- re-ejecutar el proceso solo repite desde
donde se quedo, no desde cero. Esto tambien evita perder el costo real
de las llamadas al LLM ya hechas cuando algo falla despues.

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

_BATCH_SIZE = 10


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

        documento_completo = "\n".join(p["text"] for p in paginas) if settings.CONTEXTUAL_RETRIEVAL_ENABLED else None

        # Registrar el manual DESDE EL INICIO (no al final) -- si el
        # proceso muere a mitad de camino, el manual ya existe con lo
        # que alcanzó a guardarse, no queda huérfano ni se pierde el id.
        manual_id = vector_service.register_manual(filename)

        total_guardados = 0
        con_contexto = 0
        total_lotes = (len(fragmentos) + _BATCH_SIZE - 1) // _BATCH_SIZE

        for i in range(0, len(fragmentos), _BATCH_SIZE):
            lote = fragmentos[i : i + _BATCH_SIZE]
            numero_lote = (i // _BATCH_SIZE) + 1
            logger.info(f"{filename}: procesando lote {numero_lote}/{total_lotes} ({len(lote)} chunks)")

            if settings.CONTEXTUAL_RETRIEVAL_ENABLED:
                textos_para_embedding = []
                for chunk in lote:
                    contexto = context_service.generate_chunk_context(documento_completo, chunk["text"])
                    if contexto:
                        con_contexto += 1
                        textos_para_embedding.append(f"{contexto}\n\n{chunk['text']}")
                    else:
                        textos_para_embedding.append(chunk["text"])
            else:
                textos_para_embedding = [c["text"] for c in lote]

            embeddings = embedding_service.embed_batch(textos_para_embedding)
            for fragmento, embedding in zip(lote, embeddings):
                fragmento["embedding"] = embedding

            vector_service.save_chunk_batch(manual_id, filename, lote)
            total_guardados += len(lote)
            logger.info(f"{filename}: lote {numero_lote}/{total_lotes} guardado ({total_guardados}/{len(fragmentos)} chunks acumulados)")

        if settings.CONTEXTUAL_RETRIEVAL_ENABLED:
            logger.info(f"{filename}: {con_contexto}/{len(fragmentos)} chunks obtuvieron contexto real (el resto se indexó sin él por límites del proveedor LLM)")

        total_paginas = max((c["page"] for c in fragmentos), default=0)
        vector_service.finalize_manual(manual_id, total_guardados, total_paginas)

        logger.info(f"{filename}: indexación completa ({total_guardados} chunks)")
        return {"filename": filename, "status": "indexado", "chunks": total_guardados, "con_contexto": con_contexto if settings.CONTEXTUAL_RETRIEVAL_ENABLED else None}
