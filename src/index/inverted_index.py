from collections import defaultdict
from typing import Dict, Set, List
import json
import os


class InvertedIndex:
    def __init__(self):
        self.index = defaultdict(set)
    
    def add_document(self, doc_id: str, terms: List[str]):
        """Add a document to the index."""
        for term in terms:
            self.index[term].add(doc_id)
    
    def search(self, term: str) -> Set[str]:
        """Search for documents containing a single term."""
        return self.index.get(term, set())
    
    def conjunctive_search(self, terms: List[str]) -> Set[str]:
        """Search for documents containing ALL terms (AND query)."""
        if not terms:
            return set()
        
        # Optimization: start with the smallest posting list
        sorted_terms = sorted(terms, key=lambda t: len(self.index.get(t, [])))
        result = self.index.get(sorted_terms[0], set()).copy()
        
        # Intersect with remaining terms
        for term in sorted_terms[1:]:
            result &= self.index.get(term, set())
            if not result:
                break
        
        return result

    def save(self, output_directory: str) -> None:
        """Persist the inverted index to the given directory.

        Files written:
        - inverted_index.jsonl: one line per term with its posting list
        - inverted_index_meta.json: basic metadata
        """
        os.makedirs(output_directory, exist_ok=True)

        index_path = os.path.join(output_directory, "inverted_index.jsonl")
        meta_path = os.path.join(output_directory, "inverted_index_meta.json")

        with open(index_path, "w", encoding="utf-8") as f:
            for term, doc_ids in self.index.items():
                record = {
                    "term": term,
                    "doc_ids": sorted(doc_ids),
                }
                f.write(json.dumps(record, ensure_ascii=False) + "\n")

        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "num_terms": len(self.index),
                },
                f,
                indent=2,
                ensure_ascii=False,
            )