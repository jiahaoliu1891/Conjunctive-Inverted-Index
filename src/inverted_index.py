from collections import defaultdict
from typing import Dict, Set, List


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