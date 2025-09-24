from collections import defaultdict
from typing import *
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt


# ------------------------ #
#       Forward Index      #
# ------------------------ #
def create_forward_index(df: pd.DataFrame, doc_id_column: str, term_columns: List[str]):
    """
    create the forward index: doc_id -> [term_0, term_1, ...]
    TODO: make sure term_columns have distinct values
    """
    if doc_id_column not in df:
        raise Exception(f"doc_id_column: {doc_id_column} not found in df. df columns: {df.columns}")
    
    for term_column in term_columns:
        if term_column not in df:
            raise Exception(f"term_column: {term_column} not found in df. df columns: {df.columns}")
    
    forward_index = defaultdict(set)
    for idx, row in tqdm(df.iterrows(), total=len(df)):
        doc_id = row[doc_id_column]
        for term_column in term_columns:
            forward_index[doc_id].add(row[term_column])
            
    return forward_index


def merge_forward_index(forward_indexes: List[Dict[int, set]]):
    if len(forward_indexes) == 1:
        return forward_indexes
    
    merged_index = defaultdict(set)
    
    for index in forward_indexes:
        for doc_id, terms in index.items():
            if doc_id in merged_index:
                merged_index[doc_id].update(terms)
            else:
                merged_index[doc_id] = terms
    return merged_index


# ------------------------- #
#       Inverted Index      #
# ------------------------- #
def create_inverted_index(forward_index: Dict[str, Set]) -> Dict[str, Set]:
    """
    create inverted index from forward_index
    term_0 -> [doc_id_0, doc_id_1, ...]
    """
    inverted_index = defaultdict(set)
    for doc_id, terms in forward_index.items():
        for term in terms:
            inverted_index[term].add(doc_id)
    return inverted_index


def query_inverted_index(query: List[str], inverted_index: Dict[str, Set]) -> Set:
    if len(query) == 1:
        return inverted_index[query[0]]
    sorted_query = sorted(query, key=lambda x: len(inverted_index[x]))
    result = inverted_index[sorted_query[0]]
    for query_term in sorted_query[1:]:
        result = result & inverted_index[query_term]
    return result


# ------------------------------------- #
#       Conjunctive Inverted Index      #
# ------------------------------------- #

def is_subset_of(query_set: Set[str], target_set: Set[str]):
    """
    decide if query_set is a subset of target_set
    """
    if len(query_set) > len(target_set):
        return False
    for term in query_set:
        if term not in target_set:
            return False
    return True


# ------- String ------- #
CONJUNCTIVE_SEPARATOR = "^"

def encode_terms_as_string(terms: Set[str]):
    return CONJUNCTIVE_SEPARATOR.join(sorted(terms))


def create_conjunctive_inverted_index_string(forward_index: Dict):
    """
    Create a map from conjunctive terms to doc_ids
    e.g., "rainy^city" -> [doc1, doc2, ...]
    key is conjunctive terms represented as string
    """
    conjunctive_inverted_index_string = defaultdict(set)
    for doc_id, terms in forward_index.items():
        key_string = encode_terms_as_string(terms)
        conjunctive_inverted_index_string[key_string].add(doc_id)
    return conjunctive_inverted_index_string


def query_conjunctive_inverted_index_string(query_string: str, conjunctive_inverted_index_string: Dict[str, Set]):
    result = []
    for key_string, doc_ids in conjunctive_inverted_index_string.items():
        query_terms = set(query_string.split(CONJUNCTIVE_SEPARATOR))
        key_terms = set(key_string.split(CONJUNCTIVE_SEPARATOR))
        if is_subset_of(query_terms, key_terms):
            result.extend(doc_ids)
    return result

# ------- Tuple ------- #
def encode_terms_as_tuple(terms: Set[str]):
    return tuple(sorted(terms))


def create_conjunctive_inverted_index_tuple(forward_index: Dict):
    """
    Create a map from conjunctive terms to doc_ids
    e.g., (rainy, city) -> [doc1, doc2, ...]
    key is conjunctive terms represented as tuple
    """    
    conjunctive_inverted_index_tuple = defaultdict(set)
    for doc_id, terms in forward_index.items():
        key_tuple = encode_terms_as_tuple(terms)
        conjunctive_inverted_index_tuple[key_tuple].add(doc_id)
    return conjunctive_inverted_index_tuple


def query_conjunctive_inverted_index_tuple(query_tuple: Tuple, conjunctive_inverted_index_tuple: Dict[Tuple, Set]):
    result = set()
    for key_tuple, doc_ids in conjunctive_inverted_index_tuple.items():
        if is_subset_of(set(query_tuple), set(key_tuple)):
            result.update(doc_ids)
    return result


# ------- Binary ------- #
def encode_terms_as_binary(terms: Set[str], term_to_bit_position: dict, total_terms: int):
    binary_string = ['0'] * total_terms
    for term in terms:
        bit_position = term_to_bit_position[term]
        binary_string[bit_position] = '1'
    binary_int = int(''.join(binary_string), 2)
    return binary_int


def create_conjunctive_inverted_index_binary(forward_index: Dict):
    """
    Create a map from conjunctive terms to doc_ids
    e.g., 001001 -> [doc1, doc2, ...]
    key is conjunctive terms represented as binary number (each term has a bit position)
    """    
    all_terms = set()
    for _, terms in forward_index.items():
        all_terms.update(terms)
    sorted_all_terms = sorted(all_terms)
    term_to_bit_position = {term: bit_position for bit_position, term in enumerate(sorted_all_terms)}
    conjunctive_inverted_index_binary = defaultdict(set)
    
    for doc_id, terms in forward_index.items():
        key_binary = encode_terms_as_binary(terms, term_to_bit_position, len(all_terms))
        conjunctive_inverted_index_binary[key_binary].add(doc_id)
        
    return {
        "conjunctive_inverted_index_binary": conjunctive_inverted_index_binary,
        "term_to_bit_position": term_to_bit_position,
        "sorted_all_terms": sorted_all_terms
    }


def query_conjunctive_inverted_index_binary(query_binary: int, conjunctive_inverted_index_binary: Dict):
    result = set()
    for key_binary, doc_ids in conjunctive_inverted_index_binary.items():
        if query_binary & key_binary == query_binary:
            result.update(doc_ids)
    return result

def query_conjunctive_inverted_index_binary_by_term_list(
    query_terms: List[str], 
    conjunctive_inverted_index_binary: Dict,
    term_to_bit_position: dict, 
    total_terms: int
):
    query_binary = encode_terms_as_binary(query_terms, term_to_bit_position, total_terms)
    return query_conjunctive_inverted_index_binary(query_binary, conjunctive_inverted_index_binary)


# --------------------------------------------------- #
#       Conjunctive Inverted Index (Cardinality)     #
# --------------------------------------------------- #

def create_conjunctive_inverted_index_cardinality(forward_index: Dict):
    """
    create a table maps from conjunctive terms (binary representation) to doc_ids
    the table is sorted by the cardinality of key
    e.g.,
        000001 -> [doc1, doc2, ...]
        000011 -> [doc1, doc2, ...]
        ....
        111111 -> [doc1, doc2, ...]
    this allows us to skip the unnecessary comparison between query and key
    e.g., 
        if query = 00111, we can skip the first two entries in the table
    """
    all_terms = set()
    for _, terms in forward_index.items():
        all_terms.update(terms)
    sorted_all_terms = sorted(all_terms)
    term_to_bit_position = {term: bit_position for bit_position, term in enumerate(sorted_all_terms)}
    
    conjunctive_inverted_index_cardinality = defaultdict(set)
    key_binary_to_cardinality = {}
    for doc_id, terms in forward_index.items():
        key_binary = encode_terms_as_binary(terms, term_to_bit_position, len(all_terms))
        conjunctive_inverted_index_cardinality[key_binary].add(doc_id)
        # cardinality of terms
        key_binary_to_cardinality[key_binary] = len(terms)
    
    # sort the inverted index by cardinality of the key
    conjunctive_inverted_index_cardinality_table = sorted(
        conjunctive_inverted_index_cardinality.items(), 
        key=lambda x: key_binary_to_cardinality[x[0]]
    )
    
    # record the start index of each cardinality
    cardinality_to_start_index = {}
    for i in range(len(conjunctive_inverted_index_cardinality_table)):
        key_binary = conjunctive_inverted_index_cardinality_table[i][0]
        cardinality = key_binary_to_cardinality[key_binary]
        if cardinality not in cardinality_to_start_index:
            cardinality_to_start_index[cardinality] = i
    cardinality_start_index_table = sorted(cardinality_to_start_index.items(), key=lambda x: x[0])
    
    return {
        "conjunctive_inverted_index_cardinality_table": conjunctive_inverted_index_cardinality_table,
        "cardinality_start_index_table": cardinality_start_index_table,
        "term_to_bit_position": term_to_bit_position,
        "sorted_all_terms": sorted_all_terms
    }


def find_start_index(cardinality_start_index_table, query_cardinality):
    """
    use binary search to get start_index given query cardinality
    """
    left, right = 0, len(cardinality_start_index_table) - 1
    
    while left <= right:
        mid = (left + right) // 2
        cardinality, index = cardinality_start_index_table[mid]
        
        if query_cardinality >= cardinality:
            left = mid + 1
        else:
            right = mid - 1
    
    return cardinality_start_index_table[right][1] if right >= 0 else -1


def query_conjunctive_inverted_index_cardinality(
    query_binary: int,
    query_cardinality: int,
    cardinality_start_index_table: List,
    conjunctive_inverted_index_cardinality_table: List
):
    start_index = -1
    for cardinality, index in cardinality_start_index_table:
        if query_cardinality >= cardinality:
            start_index = index
        else:
            break

    # start_index = find_start_index(cardinality_start_index_table, query_cardinality)
    
    result = set()
    for key_binary, doc_ids in conjunctive_inverted_index_cardinality_table[start_index:]:
        if query_binary & key_binary == query_binary:
            result.update(doc_ids)
    return result
    


def query_conjunctive_inverted_index_cardinality_by_term_list(
    query_terms: List[str], 
    cardinality_start_index_table: List,
    conjunctive_inverted_index_cardinality_table: List,
    term_to_bit_position: dict, 
    total_terms: int
):
    query_binary = encode_terms_as_binary(query_terms, term_to_bit_position, total_terms)
    query_cardinality = len(query_terms)
    return query_conjunctive_inverted_index_cardinality(
        query_binary,
        query_cardinality,
        cardinality_start_index_table,
        conjunctive_inverted_index_cardinality_table
    )


        
# --------------------------------------------- #
#       Conjunctive Inverted Index (Graph)     #
# --------------------------------------------- #
# TODO: Implement this


# --------------------------- #
#        Visualization        #
# --------------------------- #

def show_distribution(term_count_distribution):
    plt.figure(figsize=(4, 3))
    plt.hist(term_count_distribution, bins=range(1, max(term_count_distribution) + 2), align='left', color='skyblue', edgecolor='black')
    plt.xlabel('Number of terms')
    plt.ylabel('Frequency')
    plt.show()
    
    
def show_term_count_distribution(forward_index):
    """
    for each given doc_id, count the number of terms it contains in the forward index
    draw the count distribution
    """
    term_counts = list(map(len, forward_index.values()))
    show_distribution(term_counts)