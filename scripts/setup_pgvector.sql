CREATE EXTENSION IF NOT EXISTS vector;

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
    embedding vector(1024),
    created_at TIMESTAMP DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS chunks_embedding_hnsw_idx
ON chunks USING hnsw (embedding vector_cosine_ops)
WITH (m = 16, ef_construction = 64);

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
