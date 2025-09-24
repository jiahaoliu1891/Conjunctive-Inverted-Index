import pytest

from src.conjunctive_index_binary import ConjunctiveIndexBinary


@pytest.fixture()
def index():
    return ConjunctiveIndexBinary(all_terms=["apple", "banana", "cherry", "date"]) 


def test_encode_terms_basic(index):
    assert index.encode_terms(["apple"]) == 1 << 0
    assert index.encode_terms(["banana"]) == 1 << 1
    assert index.encode_terms(["cherry"]) == 1 << 2
    assert index.encode_terms(["date"]) == 1 << 3
    # Unknown terms are ignored by encode_terms
    assert index.encode_terms(["unknown"]) == 0
    assert index.encode_terms(["apple", "unknown"]) == (1 << 0)


def test_add_and_conjunctive_query_superset(index):
    index.add_document("doc1", ["apple", "banana"])  # 1,2
    index.add_document("doc2", ["banana"])            # 2
    index.add_document("doc3", ["banana", "cherry"]) # 2,3
    index.add_document("doc4", ["apple", "banana", "cherry"])  # 1,2,3

    # Single-term query returns all docs containing that term
    assert index.conjunctive_query(["banana"]) == {"doc1", "doc2", "doc3", "doc4"}

    # Multi-term query requires all terms to be present
    assert index.conjunctive_query(["apple", "banana"]) == {"doc1", "doc4"}

    # Query with terms that no document jointly has
    assert index.conjunctive_query(["cherry", "date"]) == set()


def test_empty_conjunctive_query_returns_all_docs(index):
    index.add_document("doc1", ["apple"]) 
    index.add_document("doc2", ["banana"]) 
    # Adding a document with only unknown terms results in key 0
    index.add_document("docX", ["unknown"]) 

    assert index.conjunctive_query([]) == {"doc1", "doc2", "docX"}


def test_unknown_query_term_raises_key_error(index):
    index.add_document("doc1", ["apple"]) 
    with pytest.raises(KeyError):
        index.conjunctive_query(["unknown"]) 


def test_add_document_ignores_unknown_terms(index):
    index.add_document("doc1", ["apple", "unknown"])  # treated as just {apple}
    index.add_document("doc2", ["banana"]) 

    assert index.conjunctive_query(["apple"]) == {"doc1"}
    assert index.conjunctive_query(["banana"]) == {"doc2"}


def test_duplicate_adds_are_idempotent(index):
    index.add_document("doc1", ["apple", "banana"]) 
    index.add_document("doc1", ["apple", "banana"]) 

    assert index.conjunctive_query(["apple"]) == {"doc1"}
    assert index.conjunctive_query(["banana"]) == {"doc1"}


