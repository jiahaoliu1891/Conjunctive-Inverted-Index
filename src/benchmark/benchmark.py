import argparse
import json
import os
import sys
from typing import List
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
SRC_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from index.conjunctive_index_binary import ConjunctiveIndexBinary
from index.inverted_index import InvertedIndex


def read_vocabulary(vocab_path: str) -> List[str]:
    with open(vocab_path, "r", encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]


def build_indexes(docs_path: str, vocab_path: str):
    vocabulary = read_vocabulary(vocab_path)
    conj_index = ConjunctiveIndexBinary(all_terms=vocabulary)
    inv_index = InvertedIndex()

    total_docs = 0
    with open(docs_path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            record = json.loads(line)
            doc_id = record["doc_id"]
            terms = record["terms"]
            inv_index.add_document(doc_id, terms)
            conj_index.add_document(doc_id, terms)
            total_docs += 1

    return inv_index, conj_index, total_docs


def parse_args():
    parser = argparse.ArgumentParser(description="Build and save indexes from docs.jsonl")
    parser.add_argument(
        "--docs",
        type=str,
        default=os.path.join(SCRIPT_DIR, "data", "docs.jsonl"),
        help="Path to docs.jsonl file",
    )
    parser.add_argument(
        "--vocab",
        type=str,
        default=os.path.join(SCRIPT_DIR, "data", "vocab.txt"),
        help="Path to vocab.txt file",
    )
    parser.add_argument(
        "--out",
        type=str,
        default=os.path.join(SCRIPT_DIR, "index_data"),
        help="Output directory to write index files",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    os.makedirs(args.out, exist_ok=True)
    inv_index, conj_index, total_docs = build_indexes(args.docs, args.vocab)

    logger.info(f"Saving inverted index to {args.out}")
    inv_index.save(args.out)
    logger.info(f"Saving conjunctive index to {args.out}")
    conj_index.save(args.out)

    print(
        f"Built inverted and conjunctive-binary indexes for {total_docs} documents. Saved to {args.out}"
    )


if __name__ == "__main__":
    main()


