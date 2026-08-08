"""
Script de evaluacion RAG -- corre LOCALMENTE, fuera de Lambda/AWS Console.

Mide 4 metricas estandar de evaluacion RAG (estilo RAGAS):
- Context Precision: de los chunks recuperados, ¿que fraccion son
  realmente relevantes (su pagina esta en expected_pages)?
- Context Recall: de las paginas relevantes esperadas, ¿que fraccion
  logro traer el retrieval?
- Faithfulness: ¿la respuesta esta respaldada por el contexto real
  (text_preview de las fuentes), o se desvio/invento algo? Evaluado
  por Nova Lite como "juez", via boto3 directo (no pasa por la API).
- Answer Relevance: ¿la respuesta aborda realmente la pregunta?
  Tambien evaluado por Nova Lite como juez.

Requisitos:
- pip install -r requirements.txt
- Credenciales AWS configuradas (aws configure) -- el juez llama a
  Bedrock directamente desde tu Mac, no a traves de la Lambda.
- La API debe estar desplegada y accesible (usa API_BASE_URL).

Uso:
    python3 run_evaluation.py
    API_BASE_URL=https://tu-api/dev python3 run_evaluation.py
"""
import json
import os
import statistics

import boto3
import requests

API_BASE_URL = os.environ.get(
    "API_BASE_URL", "https://fyrr6brbra.execute-api.us-east-1.amazonaws.com/dev"
)
BEDROCK_REGION = os.environ.get("BEDROCK_REGION", "us-east-1")
JUDGE_MODEL = os.environ.get("JUDGE_MODEL", "us.amazon.nova-lite-v1:0")

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
        return None  # no aplica (ej. preguntas de ventas sin manual)
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
    endpoint = "/api/v1/chat-cliente" if item.get("audiencia") == "cliente" else "/api/v1/chat"
    print(f"  -> {item['question']}")

    resp = requests.post(f"{API_BASE_URL}{endpoint}", json={"question": item["question"]}, timeout=60)
    resp.raise_for_status()
    data = resp.json()

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
    with open(os.path.join(os.path.dirname(__file__), "dataset.json"), "r", encoding="utf-8") as f:
        dataset = json.load(f)

    print(f"Evaluando {len(dataset)} preguntas contra {API_BASE_URL} ...\n")
    resultados = [evaluar_pregunta(item) for item in dataset]

    print("\n=== Resultado detallado ===")
    print(json.dumps(resultados, indent=2, ensure_ascii=False))

    print("\n=== Promedios ===")
    print(f"Context Precision: {_promedio(resultados, 'context_precision')}")
    print(f"Context Recall:    {_promedio(resultados, 'context_recall')}")
    print(f"Faithfulness:      {_promedio(resultados, 'faithfulness')}")
    print(f"Answer Relevance:  {_promedio(resultados, 'answer_relevance')}")

    out_path = os.path.join(os.path.dirname(__file__), "ultimo_resultado.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(resultados, f, indent=2, ensure_ascii=False)
    print(f"\nResultado completo guardado en {out_path}")


if __name__ == "__main__":
    main()
