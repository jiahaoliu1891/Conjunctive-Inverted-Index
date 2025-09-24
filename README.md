# Conjunctive Inverted Index

This repository contains two minimal, well-tested implementations for multi-term search over documents:

- `src/inverted_index.py`: a classic inverted index with set intersections for AND queries.
- `src/conjunctive_index_binary.py`: a binary-encoded variant that treats each document’s term set as a bitmask and answers conjunctive (AND) queries via bitwise checks.

The goal is to show simple, clear baselines for conjunctive (multi-term) search and highlight trade-offs between straightforward posting-list intersections and compact bitmask-based filtering.

For background and motivation, see the article:
- [From Inverted Index to Conjunctive Inverted Index: Optimizing Multi-Term Queries](https://cake-vinca-a89.notion.site/From-Inverted-Index-to-Conjunctive-Inverted-Index-Optimizing-Multi-Term-Queries-278cf1b2daf080768ac2eb6803d3a39c)


## Installation

Requirements:
- Python 3.8+

Install dev dependency for tests:

```bash
pip install -r requirements.txt
```

## Tests

This repo is test-first. Run the full test suite with:

```bash
pytest -q
```

Relevant tests:
- `tests/test_inverted_index.py`
- `tests/test_conjunctive_index_binary.py`


## Project layout

```
src/
  inverted_index.py              # set-based postings and intersections
  conjunctive_index_binary.py    # bitmask-based conjunctive matching
tests/
  test_inverted_index.py
  test_conjunctive_index_binary.py
benchmark/
  generate_data.py               # placeholder for dataset generation (WIP)
```