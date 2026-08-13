"""
Tests del chunking de texto (app/services/indexing_service.py).
Funcion pura -- no requiere mocks de AWS ni base de datos.
"""
from app.services.indexing_service import chunk_text


def test_chunk_text_respeta_tamano_minimo():
    paginas = [{"page": 1, "text": "corto"}]  # menor a CHUNK_MIN_SIZE (100)
    chunks = chunk_text(paginas)
    assert chunks == []


def test_chunk_text_genera_al_menos_un_chunk_para_texto_largo():
    texto_largo = "A" * 500
    paginas = [{"page": 1, "text": texto_largo}]
    chunks = chunk_text(paginas)
    assert len(chunks) >= 1
    assert all(c["page"] == 1 for c in chunks)


def test_chunk_text_preserva_numero_de_pagina():
    paginas = [
        {"page": 1, "text": "B" * 500},
        {"page": 2, "text": "C" * 500},
    ]
    chunks = chunk_text(paginas)
    paginas_en_chunks = {c["page"] for c in chunks}
    assert paginas_en_chunks == {1, 2}


def test_chunk_text_lista_vacia_no_falla():
    assert chunk_text([]) == []
