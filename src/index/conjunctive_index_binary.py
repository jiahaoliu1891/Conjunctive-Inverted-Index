from collections import defaultdict
from typing import List, Set
import json
import os


class ConjunctiveIndexBinary:
    def __init__(self, all_terms: List[str]):
        self.index = defaultdict(set)
        self.term_to_bit = {term: i for i, term in enumerate(all_terms)}
        self.total_terms = len(all_terms)
    
    def encode_terms(self, terms: List[str]) -> int:
        """Encode terms as binary number."""
        binary = 0
        for term in terms:
            if term in self.term_to_bit:
                binary |= (1 << self.term_to_bit[term])
        return binary
    
    def add_document(self, doc_id: str, terms: List[str]):
        """Store document with binary-encoded key."""
        key = self.encode_terms(terms)
        self.index[key].add(doc_id)
    
    def conjunctive_query(self, terms: List[str]) -> Set[str]:
        """Return document ids whose term-set is a superset of the query terms.
        """
        if not terms:
            # Empty conjunction matches all documents
            all_docs: Set[str] = set()
            for doc_ids in self.index.values():
                all_docs.update(doc_ids)
            return all_docs

        # Validate terms exist in the vocabulary to emulate `index.py` behavior
        for term in terms:
            if term not in self.term_to_bit:
                raise KeyError(f"Unknown term: {term}")

        query_bits = self.encode_terms(terms)

        result: Set[str] = set()
        for key_bits, doc_ids in self.index.items():
            if (query_bits & key_bits) == query_bits:
                result.update(doc_ids)
        return result
    
    def save(self, output_directory: str) -> None:
        """Persist the index and metadata to the given directory.

        Files written:
        - conjunctive_index_binary.jsonl: one line per bit-key entry with doc_ids
        - conjunctive_index_meta.json: metadata including term_to_bit mapping
        """
        os.makedirs(output_directory, exist_ok=True)

        index_path = os.path.join(output_directory, "conjunctive_index_binary.jsonl")
        meta_path = os.path.join(output_directory, "conjunctive_index_meta.json")

        # Write index as JSONL for stream-friendly storage
        with open(index_path, "w", encoding="utf-8") as f:
            for key_bits, doc_ids in self.index.items():
                record = {
                    "key_bits": int(key_bits),
                    "doc_ids": sorted(doc_ids),
                }
                f.write(json.dumps(record, ensure_ascii=False) + "\n")

        # Write metadata containing the vocabulary bit assignments
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "total_terms": self.total_terms,
                    "term_to_bit": self.term_to_bit,
                },
                f,
                indent=2,
                ensure_ascii=False,
            )
    