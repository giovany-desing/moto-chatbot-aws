"""
Script de evaluacion RAG (estilo RAGAS) -- corre EN PROCESO contra el
pipeline real (rag_service.py, sales_service.py), sin necesitar una API
desplegada. Cambio respecto a la version original: antes esto llamaba
via HTTP a una API_BASE_URL desplegada en AWS -- pero el pipeline de hoy
(Groq + embeddings/reranking locales) nunca se desplego, asi que
evaluar contra la API vieja hubiera probado el sistema equivocado.

Requiere el mismo acceso a Postgres que el resto del proyecto en local
(tunel SSM + DB_HOST=localhost DB_PORT=15432, ver README).

Metricas (estilo RAGAS):
- Context Precision / Recall: contra expected_pages del dataset
  (verificadas manualmente contra el manual real, no asumidas).
- Faithfulness / Answer Relevance: juzgadas por Amazon Nova Lite via
  Bedrock -- proveedor DISTINTO al que genera las respuestas (Groq),
  deliberado para reducir sesgo de auto-evaluacion. Es un caso atipico
  legitimo para seguir usando Bedrock.

Uso:
    python3 eval/run_evaluation.py                    # solo reporta
    python3 eval/run_evaluation.py --gate              # falla (exit 1)
                                                        # si algun
                                                        # promedio cae
                                                        # bajo el umbral
    python3 eval/run_evaluation.py --gate --threshold 0.75
"""
import argparse
import json
import os
import statistics
import sys

import boto3

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from app.services import rag_service, sales_service  # noqa: E402

BEDROCK_REGION = os.environ.get("BEDROCK_REGION", "us-east-1")
JUDGE_MODEL = os.environ.get("JUDGE_MODEL", "us.amazon.nova-lite-v1:0")
DEFAULT_THRESHOLD = 0.7

_bedrock = boto3.client("bedrock-runtime", region_name=BEDROCK_REGION)


def _judge(prompt: str) -> float | None:
    """Le pide a Nova un puntaje 0.0-1.0 y lo parsea. None si Bedrock falla o no se pudo interpretar."""
    payload = {
        "schemaVersion": "messages-v1",
        "messages": [{"role": "user", "content": [{"text": prompt}]}],
        "inferenceConfig": {"maxTokens": 10, "temperature": 0.0},
    }
    try:
        response = _bedrock.invoke_model(
            modelId=JUDGE_MODEL,
            body=json.dumps(payload),
            contentType="application/json",
            accept="application/json",
        )
    except Exception as exc:
        print(f"    [aviso] el juez de Bedrock fallo: {exc}")
        return None

    result = json.loads(response["body"].read())
    texto = result["output"]["message"]["content"][0]["text"].strip()
    try:
        return max(0.0, min(1.0, float(texto)))
    except ValueError:
        print(f"    [aviso] el juez no devolvio un numero interpretable: '{texto}'")
        return None


def context_precision(retrieved_pages: list[int], expected_pages: list[int]) -> float | None:
    if not expected_pages:
        return None
    if not retrieved_pages:
        return 0.0
    aciertos = len(set(retrieved_pages) & set(expected_pages))
    return round(aciertos / len(retrieved_pages), 3)


def context_recall(retrieved_pages: list[int], expected_pages: list[int]) -> float | None:
    if not expected_pages:
        return None
    aciertos = len(set(retrieved_pages) & set(expected_pages))
    return round(aciertos / len(expected_pages), 3)


def faithfulness(answer: str, context_text: str) -> float | None:
    if not context_text.strip():
        return None
    prompt = (
        f"Contexto:\n{context_text}\n\n"
        f"Respuesta a evaluar:\n{answer}\n\n"
        "¿Cada afirmación de la respuesta está respaldada por el contexto, "
        "sin datos inventados? Responde SOLO con un número entre 0.0 (nada "
        "respaldado) y 1.0 (todo respaldado). Sin texto adicional."
    )
    return _judge(prompt)


def answer_relevance(question: str, answer: str) -> float | None:
    prompt = (
        f"Pregunta:\n{question}\n\n"
        f"Respuesta:\n{answer}\n\n"
        "¿Qué tan bien esta respuesta aborda directamente la pregunta, "
        "sin irse por las ramas ni quedarse corta? Responde SOLO con un "
        "número entre 0.0 (nada relevante) y 1.0 (totalmente relevante). "
        "Sin texto adicional."
    )
    return _judge(prompt)


def evaluar_pregunta(item: dict) -> dict:
    print(f"  -> {item['question']}")

    if item.get("audiencia") == "cliente":
        data = sales_service.answer_cliente(item["question"])
    else:
        data = rag_service.answer_with_tools(item["question"])

    sources = data.get("sources", [])
    retrieved_pages = [s["page"] for s in sources]
    context_text = "\n\n".join(s.get("text_preview", "") for s in sources)

    return {
        "question": item["question"],
        "answer": data.get("answer"),
        "retrieved_pages": retrieved_pages,
        "expected_pages": item.get("expected_pages", []),
        "context_precision": context_precision(retrieved_pages, item.get("expected_pages", [])),
        "context_recall": context_recall(retrieved_pages, item.get("expected_pages", [])),
        "faithfulness": faithfulness(data.get("answer", ""), context_text),
        "answer_relevance": answer_relevance(item["question"], data.get("answer", "")),
        "from_cache": data.get("from_cache"),
    }


def _promedio(resultados: list[dict], clave: str) -> float | None:
    valores = [r[clave] for r in resultados if r[clave] is not None]
    return round(statistics.mean(valores), 3) if valores else None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gate", action="store_true", help="Falla (exit 1) si algun promedio cae bajo el umbral")
    parser.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD)
    args = parser.parse_args()

    with open(os.path.join(os.path.dirname(__file__), "dataset.json"), "r", encoding="utf-8") as f:
        dataset = json.load(f)

    print(f"Evaluando {len(dataset)} preguntas contra el pipeline local (en proceso, sin API desplegada) ...\n")
    resultados = [evaluar_pregunta(item) for item in dataset]

    print("\n=== Resultado detallado ===")
    print(json.dumps(resultados, indent=2, ensure_ascii=False))

    promedios = {
        "context_precision": _promedio(resultados, "context_precision"),
        "context_recall": _promedio(resultados, "context_recall"),
        "faithfulness": _promedio(resultados, "faithfulness"),
        "answer_relevance": _promedio(resultados, "answer_relevance"),
    }

    print("\n=== Promedios ===")
    for nombre, valor in promedios.items():
        print(f"{nombre}: {valor}")

    out_path = os.path.join(os.path.dirname(__file__), "ultimo_resultado.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(resultados, f, indent=2, ensure_ascii=False)
    print(f"\nResultado completo guardado en {out_path}")

    if args.gate:
        fallidas = {k: v for k, v in promedios.items() if v is not None and v < args.threshold}
        if fallidas:
            print(f"\n❌ GATE FALLIDO -- por debajo del umbral {args.threshold}: {fallidas}")
            sys.exit(1)
        print(f"\n✅ GATE APROBADO -- todos los promedios >= {args.threshold}")


if __name__ == "__main__":
    main()
