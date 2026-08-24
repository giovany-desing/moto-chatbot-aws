"""
Gestión de conexión y migraciones de PostgreSQL/pgvector.

Las migraciones corren en cada cold start con CREATE TABLE IF NOT EXISTS
(funcional pero no ideal para producción — migrar a Alembic más adelante).
"""
import psycopg2
from psycopg2.extensions import connection as PGConnection
from contextlib import contextmanager

from app.core.config import settings
from app.core.logging import get_logger

logger = get_logger(__name__)

_SCHEMA_SQL = f"""
CREATE EXTENSION IF NOT EXISTS vector;
CREATE EXTENSION IF NOT EXISTS pg_trgm;

CREATE TABLE IF NOT EXISTS manuales (
    id         SERIAL PRIMARY KEY,
    filename   TEXT UNIQUE NOT NULL,
    pages      INT DEFAULT 0,
    chunks     INT DEFAULT 0,
    indexed    BOOLEAN DEFAULT FALSE,
    created_at TIMESTAMP DEFAULT NOW(),
    updated_at TIMESTAMP DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS chunks (
    id        SERIAL PRIMARY KEY,
    manual_id INT REFERENCES manuales(id) ON DELETE CASCADE,
    filename  TEXT NOT NULL,
    page      INT NOT NULL,
    text      TEXT NOT NULL,
    parent_text TEXT,
    embedding vector(1024),
    created_at TIMESTAMP DEFAULT NOW()
);

-- Parent-child chunking (version simplificada): "text" es el chunk
-- pequeño que se embebe e indexa para busqueda de precision;
-- "parent_text" es el texto COMPLETO de la pagina de origen, que se
-- expande al construir el contexto para el LLM (mas contexto real sin
-- perder precision en la busqueda). No es deteccion real de secciones
-- (eso requeriria parsear encabezados por tamaño de fuente en el PDF,
-- fuera de alcance de este punto) -- la pagina es el "padre" disponible
-- sin trabajo de parsing adicional.
ALTER TABLE chunks ADD COLUMN IF NOT EXISTS parent_text TEXT;

CREATE INDEX IF NOT EXISTS chunks_embedding_hnsw_idx
ON chunks USING hnsw (embedding vector_cosine_ops)
WITH (m = 16, ef_construction = 64);

-- Retrieval hibrido: columna generada automaticamente para busqueda
-- de texto completo (componente "sparse" -- pesa por frecuencia de
-- termino, favorece codigos/referencias/nombres propios que el
-- embedding puede pasar por alto). Se recalcula sola si "text" cambia.
ALTER TABLE chunks ADD COLUMN IF NOT EXISTS text_search tsvector
    GENERATED ALWAYS AS (to_tsvector('spanish', text)) STORED;

CREATE INDEX IF NOT EXISTS chunks_text_search_gin_idx
ON chunks USING GIN (text_search);

CREATE TABLE IF NOT EXISTS mecanico_perfil (
    id               SERIAL PRIMARY KEY,
    mecanico_id      TEXT UNIQUE NOT NULL,
    taller_id        TEXT,
    motos_frecuentes TEXT[],
    temas_frecuentes TEXT[],
    ultimo_resumen   TEXT,
    total_consultas  INT DEFAULT 0,
    created_at       TIMESTAMP DEFAULT NOW(),
    updated_at       TIMESTAMP DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS feedback (
    id          SERIAL PRIMARY KEY,
    mecanico_id TEXT,
    question    TEXT NOT NULL,
    answer      TEXT NOT NULL,
    rating      INT CHECK (rating BETWEEN 1 AND 5),
    comment     TEXT,
    created_at  TIMESTAMP DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS inventario (
    id              SERIAL PRIMARY KEY,
    nombre          TEXT NOT NULL,
    referencia      TEXT,
    stock           INT DEFAULT 0,
    precio          DECIMAL(10,2),
    ubicacion       TEXT,
    moto_compatible TEXT[],
    stock_minimo    INT DEFAULT 2,
    updated_at      TIMESTAMP DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS ordenes_servicio (
    id          SERIAL PRIMARY KEY,
    cliente     TEXT NOT NULL,
    moto        TEXT NOT NULL,
    placa       TEXT,
    kilometraje INT,
    trabajos    TEXT[],
    repuestos   JSONB,
    estado      TEXT DEFAULT 'abierta',
    mecanico    TEXT,
    valor_total DECIMAL(10,2),
    fecha       TIMESTAMP DEFAULT NOW(),
    completada  TIMESTAMP
);

CREATE TABLE IF NOT EXISTS vehiculos (
    id                  SERIAL PRIMARY KEY,
    placa               TEXT UNIQUE NOT NULL,
    marca               TEXT,
    modelo              TEXT,
    anio                INT,
    cliente             TEXT,
    telefono_cliente    TEXT,
    kilometraje_actual  INT,
    ultimo_servicio     TIMESTAMP,
    proximo_servicio_km INT
);

CREATE TABLE IF NOT EXISTS catalogo_motos (
    id               SERIAL PRIMARY KEY,
    modelo           TEXT NOT NULL,
    marca            TEXT NOT NULL,
    cilindraje       INT,
    precio           DECIMAL(12,2) NOT NULL,
    uso_recomendado  TEXT[],
    descripcion      TEXT,
    disponible       BOOLEAN DEFAULT TRUE,
    imagen_url       TEXT,
    created_at       TIMESTAMP DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS financiamiento_opciones (
    id                  SERIAL PRIMARY KEY,
    entidad             TEXT NOT NULL,
    tasa_interes_mensual DECIMAL(5,4) NOT NULL,
    plazo_max_meses     INT NOT NULL,
    entrada_minima_pct  DECIMAL(5,2) NOT NULL,
    activo              BOOLEAN DEFAULT TRUE
);

CREATE TABLE IF NOT EXISTS cache_semantico (
    id                SERIAL PRIMARY KEY,
    filename          TEXT,
    question_text     TEXT NOT NULL,
    question_embedding vector(1024) NOT NULL,
    result_json       TEXT NOT NULL,
    created_at        TIMESTAMP DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS cache_semantico_embedding_hnsw_idx
ON cache_semantico USING hnsw (question_embedding vector_cosine_ops)
WITH (m = 16, ef_construction = 64);

CREATE TABLE IF NOT EXISTS leads (
    id                   SERIAL PRIMARY KEY,
    nombre_cliente       TEXT,
    telefono             TEXT,
    email                TEXT,
    interes              TEXT,
    motivo_escalamiento  TEXT,
    estado               TEXT DEFAULT 'nuevo',
    session_id           TEXT,
    created_at           TIMESTAMP DEFAULT NOW()
);
"""


def get_connection() -> PGConnection:
    return psycopg2.connect(
        host=settings.DB_HOST,
        port=settings.DB_PORT,
        dbname=settings.DB_NAME,
        user=settings.DB_USER,
        password=settings.DB_PASSWORD,
        connect_timeout=10,
    )


@contextmanager
def db_cursor():
    conn = get_connection()
    try:
        cur = conn.cursor()
        yield conn, cur
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def run_migrations() -> None:
    """Crea extensión pgvector y todas las tablas si no existen. Idempotente."""
    try:
        with db_cursor() as (conn, cur):
            cur.execute(_SCHEMA_SQL)
        logger.info("Migraciones aplicadas correctamente")
    except Exception as exc:
        logger.error(f"Error aplicando migraciones: {exc}")
        raise
