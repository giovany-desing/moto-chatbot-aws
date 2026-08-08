"""
Servicio pgvector -- almacenamiento y busqueda de similitud vectorial,
mas retrieval hibrido (dense + sparse) con Reciprocal Rank Fusion, mas
un segundo pase de reranking real sobre los candidatos fusionados.

- Dense: similitud coseno contra los embeddings guardados (pgvector,
  indice HNSW) -- captura significado, sinonimos, parafraseo.
- Sparse: busqueda de texto completo nativa de PostgreSQL (tsvector +
  indice GIN, funcion ts_rank_cd) -- pondera por frecuencia de termino,
  favorece codigos, referencias, nombres propios y siglas.
- RRF: fusiona dense + sparse por POSICION (no lee contenido).
- Reranking (rerank_service.py): SI lee el contenido -- evalua la
  pregunta y cada candidato fusionado juntos, en un solo pase, y
  reordena con un puntaje de relevancia real antes del corte final.
"""
from app.core.config import settings
from app.core.logging import get_logger
from app.services import rerank_service
from app.services.db_service import db_cursor

logger = get_logger(__name__)

_RRF_K = 60  # constante estandar de Reciprocal Rank Fusion
_CANDIDATE_K = 15  # candidatos por rama antes de fusionar/rerankear


def register_manual(filename: str) -> int:
    with db_cursor() as (conn, cur):
        cur.execute(
            """
            INSERT INTO manuales (filename, indexed)
            VALUES (%s, FALSE)
            ON CONFLICT (filename) DO UPDATE SET updated_at = NOW()
            RETURNING id
            """,
            [filename],
        )
        return cur.fetchone()[0]


def save_chunks(manual_id: int, filename: str, chunks: list[dict]) -> int:
    """
    chunks: [{"page": int, "text": str, "embedding": list[float]}, ...]
    """
    with db_cursor() as (conn, cur):
        for chunk in chunks:
            cur.execute(
                """
                INSERT INTO chunks (manual_id, filename, page, text, embedding)
                VALUES (%s, %s, %s, %s, %s)
                """,
                [manual_id, filename, chunk["page"], chunk["text"], chunk["embedding"]],
            )
        cur.execute(
            """
            UPDATE manuales
            SET chunks = %s, pages = %s, indexed = TRUE, updated_at = NOW()
            WHERE id = %s
            """,
            [len(chunks), max((c["page"] for c in chunks), default=0), manual_id],
        )
    logger.info(f"Guardados {len(chunks)} chunks para {filename}")
    return len(chunks)


def search(query_embedding: list[float], filename: str | None = None, top_k: int | None = None) -> list[dict]:
    """Retrieval DENSE puro (similitud coseno). Se mantiene disponible para comparacion/pruebas."""
    top_k = top_k or settings.TOP_K_RESULTS
    embedding_literal = "[" + ",".join(str(x) for x in query_embedding) + "]"

    with db_cursor() as (conn, cur):
        if filename:
            cur.execute(
                """
                SELECT id, filename, page, text, 1 - (embedding <=> %s::vector) AS relevance
                FROM chunks
                WHERE filename = %s
                ORDER BY embedding <=> %s::vector
                LIMIT %s
                """,
                [embedding_literal, filename, embedding_literal, top_k],
            )
        else:
            cur.execute(
                """
                SELECT id, filename, page, text, 1 - (embedding <=> %s::vector) AS relevance
                FROM chunks
                ORDER BY embedding <=> %s::vector
                LIMIT %s
                """,
                [embedding_literal, embedding_literal, top_k],
            )
        rows = cur.fetchall()

    return [
        {"id": r[0], "filename": r[1], "page": r[2], "text": r[3], "relevance": round(float(r[4]), 4)}
        for r in rows
    ]


def search_sparse(query_text: str, filename: str | None = None, top_k: int | None = None) -> list[dict]:
    """Retrieval SPARSE puro (full-text search nativo de PostgreSQL)."""
    top_k = top_k or settings.TOP_K_RESULTS

    with db_cursor() as (conn, cur):
        if filename:
            cur.execute(
                """
                SELECT id, filename, page, text, ts_rank_cd(text_search, plainto_tsquery('spanish', %s)) AS relevance
                FROM chunks
                WHERE filename = %s AND text_search @@ plainto_tsquery('spanish', %s)
                ORDER BY relevance DESC
                LIMIT %s
                """,
                [query_text, filename, query_text, top_k],
            )
        else:
            cur.execute(
                """
                SELECT id, filename, page, text, ts_rank_cd(text_search, plainto_tsquery('spanish', %s)) AS relevance
                FROM chunks
                WHERE text_search @@ plainto_tsquery('spanish', %s)
                ORDER BY relevance DESC
                LIMIT %s
                """,
                [query_text, query_text, top_k],
            )
        rows = cur.fetchall()

    return [
        {"id": r[0], "filename": r[1], "page": r[2], "text": r[3], "relevance": round(float(r[4]), 4)}
        for r in rows
    ]


def _reciprocal_rank_fusion(result_lists: list[list[dict]], top_k: int) -> list[dict]:
    """
    Combina varias listas ya ordenadas por relevancia en un solo ranking,
    sin necesitar que sus puntajes sean comparables entre si -- solo usa
    la POSICION de cada resultado en cada lista.
    score(chunk) = suma, por cada lista donde aparece, de 1 / (k + rank)
    """
    scores: dict[int, float] = {}
    rows_by_id: dict[int, dict] = {}

    for resultados in result_lists:
        for rank, row in enumerate(resultados, start=1):
            chunk_id = row["id"]
            scores[chunk_id] = scores.get(chunk_id, 0.0) + 1.0 / (_RRF_K + rank)
            rows_by_id.setdefault(chunk_id, row)

    ordenados = sorted(scores.items(), key=lambda item: item[1], reverse=True)
    resultado_final = []
    for chunk_id, score_fusionado in ordenados[:top_k]:
        fila = dict(rows_by_id[chunk_id])
        fila["relevance"] = round(score_fusionado, 4)
        resultado_final.append(fila)
    return resultado_final


def search_hybrid(query_embedding: list[float], query_text: str, filename: str | None = None, top_k: int | None = None) -> list[dict]:
    """
    Retrieval HIBRIDO completo: dense + sparse -> RRF -> reranking.
    Este es el metodo que usan rag_service.py y sales_service.py.
    """
    top_k = top_k or settings.TOP_K_RESULTS

    dense_resultados = search(query_embedding, filename, top_k=_CANDIDATE_K)
    sparse_resultados = search_sparse(query_text, filename, top_k=_CANDIDATE_K)

    if not sparse_resultados:
        candidatos = dense_resultados
    else:
        candidatos = _reciprocal_rank_fusion([dense_resultados, sparse_resultados], _CANDIDATE_K)

    if settings.RERANKING_ENABLED:
        return rerank_service.rerank(query_text, candidatos, top_k)

    return candidatos[:top_k]


def list_manuales() -> list[dict]:
    with db_cursor() as (conn, cur):
        cur.execute(
            """
            SELECT filename, pages, chunks, indexed, created_at
            FROM manuales
            ORDER BY created_at DESC
            """
        )
        rows = cur.fetchall()
    return [
        {
            "filename": r[0],
            "pages": r[1],
            "chunks": r[2],
            "indexed": r[3],
            "created_at": r[4],
        }
        for r in rows
    ]
