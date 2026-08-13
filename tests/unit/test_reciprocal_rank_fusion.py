"""
Tests de la fusion de rankings (RRF) en app/services/vector_service.py.
Funcion pura -- opera sobre listas de dicts en memoria, sin base de datos.
"""
from app.services.vector_service import _reciprocal_rank_fusion


def test_rrf_prioriza_chunk_que_aparece_en_ambas_listas():
    dense = [
        {"id": 1, "filename": "m.pdf", "page": 1, "text": "a", "relevance": 0.9},
        {"id": 2, "filename": "m.pdf", "page": 2, "text": "b", "relevance": 0.8},
    ]
    sparse = [
        {"id": 2, "filename": "m.pdf", "page": 2, "text": "b", "relevance": 5.0},
        {"id": 3, "filename": "m.pdf", "page": 3, "text": "c", "relevance": 4.0},
    ]
    resultado = _reciprocal_rank_fusion([dense, sparse], top_k=3)
    ids_en_orden = [r["id"] for r in resultado]
    # el chunk 2 aparece en ambas listas -> debe quedar primero
    assert ids_en_orden[0] == 2


def test_rrf_respeta_top_k():
    dense = [{"id": i, "filename": "m.pdf", "page": i, "text": "x", "relevance": 1.0} for i in range(10)]
    resultado = _reciprocal_rank_fusion([dense], top_k=3)
    assert len(resultado) == 3


def test_rrf_lista_vacia_no_falla():
    assert _reciprocal_rank_fusion([[], []], top_k=5) == []
