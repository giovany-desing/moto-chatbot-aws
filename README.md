
# moto-chatbot-aws

Sistema de IA generativa (RAG + tool-calling) para talleres y concesionarios de motocicletas, desplegado en arquitectura 100% serverless sobre AWS. Expone dos asistentes especializados sobre la misma infraestructura: uno técnico para mecánicos y uno comercial de atención al cliente.

## Tabla de contenidos

- [Arquitectura](#arquitectura)
- [Stack tecnológico](#stack-tecnológico)
- [Cómo funciona (RAG + tool-calling)](#cómo-funciona-rag--tool-calling)
- [Estructura del proyecto](#estructura-del-proyecto)
- [Requisitos previos](#requisitos-previos)
- [Configuración local](#configuración-local)
- [Ejecutar en local](#ejecutar-en-local)
- [Endpoints de la API](#endpoints-de-la-api)
- [Ejemplos de uso](#ejemplos-de-uso)
- [Despliegue en AWS](#despliegue-en-aws)
- [Variables de entorno](#variables-de-entorno)
- [Esquema de base de datos](#esquema-de-base-de-datos)
- [Roadmap / mejoras pendientes](#roadmap--mejoras-pendientes)
- [Notas de seguridad](#notas-de-seguridad)

## Arquitectura


Usuario ──▶ API Gateway ──▶ Lambda API (FastAPI)
│
┌─────────────┼─────────────┐
▼ ▼ ▼
RDS + pgvector ElastiCache Redis Amazon Bedrock
(manuales, datos) (caché + memoria) (Titan Embed V2 + Nova Lite)


Indexación de manuales (flujo asíncrono, desacoplado del chat):


Subida de PDF ──▶ S3 ──▶ SQS ──▶ Lambda Worker ──▶ PyMuPDF (extracción)
│
chunking ──▶ embeddings (Bedrock) ──▶ pgvector


Ambas Lambdas corren dentro de la misma VPC privada que RDS y Redis — no son accesibles públicamente, solo vía el código de la aplicación.

## Stack tecnológico

| Capa | Tecnología | Rol |
|---|---|---|
| Backend | Python 3.11, FastAPI, Pydantic v2 | API y validación de datos |
| Adaptador serverless | Mangum, awslambdaric | Traduce eventos de Lambda/API Gateway a ASGI |
| Cómputo | AWS Lambda (imagen de contenedor), Amazon ECR | Ejecución sin servidor |
| Base de datos | Amazon RDS (PostgreSQL) + pgvector (índice HNSW) | Datos relacionales + búsqueda vectorial |
| Caché / memoria | Amazon ElastiCache (Redis) | Caché de respuestas y memoria conversacional |
| Almacenamiento / colas | Amazon S3, Amazon SQS | PDFs y desacople de indexación |
| IA | Amazon Bedrock — Titan Text Embeddings V2, Amazon Nova Lite | Embeddings y generación de respuestas |
| Procesamiento de PDF | PyMuPDF | Extracción de texto |
| Entrada | Amazon API Gateway (HTTP API) | Punto de entrada público |
| Observabilidad | Amazon CloudWatch Logs | Logs JSON estructurados |

## Cómo funciona (RAG + tool-calling)

1. La pregunta se compara por hash exacto en Redis antes de gastar tokens de IA.
2. Si no hay caché: se genera un embedding de la pregunta (Titan V2) y se buscan los 5 chunks más relevantes en pgvector (similitud coseno).
3. Nova Lite recibe el contexto recuperado junto con un catálogo de herramientas — funciones Python que consultan datos operativos reales (inventario, catálogo, financiamiento, CRM).
4. Si el modelo necesita datos en vivo, invoca una herramienta; el resultado se reinyecta en la conversación (máximo 4 iteraciones) antes de la respuesta final.
5. La respuesta se cachea y se guarda en la memoria de la sesión (`session_id`, TTL 30 min).

Las herramientas de taller (`app/mcp/taller_server.py`) y de ventas (`app/mcp/ventas_server.py`) están completamente aisladas entre sí — cada endpoint solo tiene acceso a su propio catálogo de acciones.

## Estructura del proyecto


app/
├── main.py # entrypoint FastAPI, corre migraciones en cold start
├── api/
│ ├── routes.py # los 5 endpoints
│ └── middleware.py # CORS + logging de requests
├── core/
│ ├── config.py # configuración tipada (Pydantic Settings, desde .env)
│ └── logging.py # logger JSON estructurado para CloudWatch
├── models/
│ └── schemas.py # schemas Pydantic de request/response
├── services/
│ ├── db_service.py # conexión y migraciones de PostgreSQL
│ ├── storage_service.py # subida/descarga de PDFs en S3
│ ├── embedding_service.py # invoca Titan Embeddings V2
│ ├── vector_service.py # búsqueda y almacenamiento en pgvector
│ ├── cache_service.py # caché de respuestas y memoria conversacional (Redis)
│ ├── bedrock_service.py # invoca Nova Lite, arma prompts, parsea tool-calls
│ ├── indexing_service.py # extracción de texto y chunking de PDFs
│ ├── rag_service.py # orquestador del asistente de taller
│ └── sales_service.py # orquestador del asistente de ventas
├── mcp/
│ ├── taller_server.py # herramientas: inventario, órdenes, historial
│ └── ventas_server.py # herramientas: catálogo, financiamiento, leads
└── workers/
└── indexing_worker.py # pipeline completo de indexación de un PDF

lambdas/
├── api_handler.py # handler Lambda de la API (vía Mangum)
└── worker_handler.py # handler Lambda del worker (disparado por SQS)

scripts/
└── setup_pgvector.sql # esquema SQL de referencia


## Requisitos previos

- Python 3.11+ (para desarrollo local; el runtime real corre en Docker)
- Docker Desktop
- AWS CLI v2, configurado con credenciales (`aws configure`)
- Una cuenta de AWS con: RDS PostgreSQL (extensión `pgvector` habilitada), ElastiCache Redis, un bucket S3, una cola SQS, y acceso habilitado a los modelos de Bedrock (`amazon.titan-embed-text-v2:0` y `us.amazon.nova-lite-v1:0`)

## Configuración local

```bash
git clone https://github.com/giovany-desing/moto-chatbot-aws.git
cd moto-chatbot-aws
cp .env.example .env
```

Completa `.env` con las credenciales de tu infraestructura (ver [Variables de entorno](#variables-de-entorno)).

Para desarrollo local sin Docker:

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements-dev.txt
```

## Ejecutar en local

```bash
uvicorn app.main:app --reload --port 8000
```

La API queda disponible en `http://localhost:8000/api/v1`. Nota: como RDS y Redis suelen vivir en una VPC privada, correr en local requiere que tengas acceso de red a ellos (VPN, bastion, o abrirlos temporalmente).

## Endpoints de la API

| Método | Ruta | Descripción |
|---|---|---|
| GET | `/api/v1/health` | Health check |
| GET | `/api/v1/documentos` | Lista manuales indexados |
| POST | `/api/v1/documentos` | Sube un manual PDF (multipart/form-data), dispara indexación asíncrona |
| POST | `/api/v1/chat` | Chat técnico para mecánicos |
| POST | `/api/v1/chat-cliente` | Chat de atención al cliente / ventas |

## Ejemplos de uso

```bash
# Health check
curl -s https://<tu-api-gateway-url>/api/v1/health

# Subir un manual
curl -X POST https://<tu-api-gateway-url>/api/v1/documentos \
  -F "file=@manual.pdf"

# Preguntar (mecánico)
curl -X POST https://<tu-api-gateway-url>/api/v1/chat \
  -H "Content-Type: application/json" \
  -d '{"question": "¿Cada cuántos km se cambia el aceite?", "session_id": "sesion-1"}'

# Preguntar (cliente)
curl -X POST https://<tu-api-gateway-url>/api/v1/chat-cliente \
  -H "Content-Type: application/json" \
  -d '{"question": "Busco una moto económica para domicilios", "session_id": "sesion-2"}'
```

## Despliegue en AWS

```bash
# Build (usar --provenance=false --sbom=false; Lambda no soporta manifiestos de attestation)
docker build --platform linux/amd64 --provenance=false --sbom=false -t moto-chatbot-dev-api:latest .

# Publicar en ECR
aws ecr get-login-password --region us-east-1 | docker login --username AWS --password-stdin <account-id>.dkr.ecr.us-east-1.amazonaws.com
docker tag moto-chatbot-dev-api:latest <account-id>.dkr.ecr.us-east-1.amazonaws.com/moto-chatbot-dev-api:latest
docker push <account-id>.dkr.ecr.us-east-1.amazonaws.com/moto-chatbot-dev-api:latest

# Actualizar la Lambda
aws lambda update-function-code \
  --function-name moto-chatbot-dev-api \
  --image-uri <account-id>.dkr.ecr.us-east-1.amazonaws.com/moto-chatbot-dev-api:latest \
  --region us-east-1
```

El worker (`moto-chatbot-dev-worker`) usa la misma imagen, con el comando de arranque sobreescrito vía `--image-config`:

```bash
aws lambda update-function-configuration \
  --function-name moto-chatbot-dev-worker \
  --image-config '{"Command":["lambdas.worker_handler.handler"]}'
```

## Variables de entorno

| Variable | Descripción |
|---|---|
| `DB_HOST`, `DB_PORT`, `DB_NAME`, `DB_USER`, `DB_PASSWORD` | Conexión a RDS PostgreSQL |
| `REDIS_HOST`, `REDIS_PORT` | Conexión a ElastiCache |
| `S3_BUCKET_NAME` | Bucket donde se guardan los PDFs |
| `SQS_QUEUE_URL` | Cola que dispara la indexación |
| `BEDROCK_REGION`, `LLM_MODEL`, `EMBEDDING_MODEL`, `EMBEDDING_DIMENSIONS` | Configuración de Bedrock |
| `CHUNK_SIZE`, `CHUNK_OVERLAP`, `TOP_K_RESULTS` | Parámetros del pipeline RAG |
| `CACHE_TTL_SECONDS`, `MEMORY_TTL_SECONDS` | TTLs de Redis |

Ver `.env.example` para la lista completa con valores por defecto.

## Esquema de base de datos

10 tablas, creadas automáticamente en cada cold start (`app/services/db_service.py`, idempotente vía `CREATE TABLE IF NOT EXISTS`):

- **RAG**: `manuales`, `chunks` (con columna `vector`, índice HNSW)
- **Taller**: `inventario`, `ordenes_servicio`, `vehiculos`, `mecanico_perfil`
- **Ventas**: `catalogo_motos`, `financiamiento_opciones`, `leads`
- **Otros**: `feedback`

Ver `scripts/setup_pgvector.sql` para el DDL completo de referencia.


