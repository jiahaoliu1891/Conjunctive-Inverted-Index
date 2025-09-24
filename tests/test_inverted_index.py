import pytest

from src.inverted_index import InvertedIndex


@pytest.fixture()
def index():
    return InvertedIndex()


def test_add_and_search_single_term(index):
    index.add_document("doc1", ["apple", "banana"]) 
    index.add_document("doc2", ["banana", "cherry"]) 

    assert index.search("banana") == {"doc1", "doc2"}
    assert index.search("apple") == {"doc1"}
    assert index.search("cherry") == {"doc2"}
    assert index.search("date") == set()


def test_conjunctive_search_and_semantics(index):
    index.add_document("doc1", ["apple", "banana"]) 
    index.add_document("doc2", ["banana", "cherry"]) 
    index.add_document("doc3", ["apple", "banana", "cherry"]) 

    assert index.conjunctive_search(["apple", "banana"]) == {"doc1", "doc3"}
    assert index.conjunctive_search(["banana", "cherry"]) == {"doc2", "doc3"}
    assert index.conjunctive_search(["apple", "cherry"]) == {"doc3"}
    assert index.conjunctive_search(["apple", "banana", "cherry"]) == {"doc3"}
    assert index.conjunctive_search(["apple", "date"]) == set()


def test_empty_conjunctive_query_returns_empty(index):
    index.add_document("doc1", ["apple"]) 
    index.add_document("doc2", ["banana"]) 
    assert index.conjunctive_search([]) == set()


def test_duplicate_adds_are_idempotent(index):
    index.add_document("doc1", ["apple", "banana", "banana"]) 
    index.add_document("doc1", ["apple", "banana"]) 

    assert index.search("apple") == {"doc1"}
    assert index.search("banana") == {"doc1"}
    assert index.conjunctive_search(["apple", "banana"]) == {"doc1"}


