import pytest

from app.vectorstores.errors import VectorStoreConfigurationError
from app.vectorstores.pgvector_store import (
    _parse_vector_literal,
    _safe_identifier,
    _vector_literal,
)


def test_pgvector_vector_literal_round_trip():
    literal = _vector_literal([0, 1.25, -2])

    assert literal == "[0.0,1.25,-2.0]"
    assert _parse_vector_literal(literal) == [0.0, 1.25, -2.0]


def test_pgvector_rejects_unsafe_table_names():
    with pytest.raises(VectorStoreConfigurationError):
        _safe_identifier("vectors; drop table papers")
