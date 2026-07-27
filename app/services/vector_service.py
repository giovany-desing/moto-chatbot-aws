"""
Servicio pgvector — almacenamiento y búsqueda de similitud vectorial.
Usa el operador <=> (distancia coseno) sobre el índice HNSW.
"""
from app.core.config import settings
from app.core.logging import get_logger
from app.services.db_service import db_cursor

logger = get_logger(__name__)


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
    top_k = top_k or settings.TOP_K_RESULTS
    embedding_literal = "[" + ",".join(str(x) for x in query_embedding) + "]"

    with db_cursor() as (conn, cur):
        if filename:
            cur.execute(
                """
                SELECT filename, page, text, 1 - (embedding <=> %s::vector) AS relevance
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
                SELECT filename, page, text, 1 - (embedding <=> %s::vector) AS relevance
                FROM chunks
                ORDER BY embedding <=> %s::vector
                LIMIT %s
                """,
                [embedding_literal, embedding_literal, top_k],
            )
        rows = cur.fetchall()

    return [
        {"filename": r[0], "page": r[1], "text": r[2], "relevance": round(float(r[3]), 4)}
        for r in rows
    ]


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
