import argparse
import json
import math
import os
import random
from dataclasses import dataclass, asdict
from typing import Dict, Iterable, List, Sequence, Tuple
from tqdm import tqdm
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class GenerationConfig:
    num_documents: int
    vocab_size: int
    min_terms_per_document: int
    max_terms_per_document: int
    zipf_alpha: float
    random_seed: int
    output_directory: str
    document_id_prefix: str
    term_prefix: str


def generate_vocabulary(vocab_size: int, term_prefix: str) -> List[str]:
    return [f"{term_prefix}{i+1}" for i in range(vocab_size)]


def compute_zipf_weights(vocab_size: int, alpha: float) -> List[float]:
    # rank i in [1..V] gets weight 1 / i^alpha
    return [1.0 / (float(rank) ** alpha) for rank in range(1, vocab_size + 1)]


def sample_unique_terms(
    rng: random.Random,
    vocabulary: Sequence[str],
    weights: Sequence[float],
    k: int,
    max_attempts: int = 10,
) -> List[str]:
    """Sample k unique terms biased by Zipf weights.

    Uses with-replacement weighted sampling and deduplication, resampling as needed
    until k unique terms are collected or attempts are exhausted.
    """
    if k <= 0:
        return []

    # random.choices supports weight arguments as a sequence of relative weights
    # We may need multiple attempts to reach k unique items when collisions happen
    chosen: List[str] = []
    chosen_set = set()

    attempts = 0
    while len(chosen_set) < k and attempts < max_attempts:
        remaining = k - len(chosen_set)
        batch = rng.choices(vocabulary, weights=weights, k=max(remaining * 2, remaining))
        for term in batch:
            if term not in chosen_set:
                chosen_set.add(term)
                chosen.append(term)
                if len(chosen_set) >= k:
                    break
        attempts += 1

    # If still short, fill uniformly from remaining vocabulary to ensure size k
    if len(chosen_set) < k:
        remaining_terms = [t for t in vocabulary if t not in chosen_set]
        rng.shuffle(remaining_terms)
        needed = k - len(chosen_set)
        chosen.extend(remaining_terms[:needed])

    return chosen[:k]


def generate_documents(config: GenerationConfig) -> Tuple[List[Dict[str, object]], List[str]]:
    rng = random.Random(config.random_seed)
    vocabulary = generate_vocabulary(config.vocab_size, config.term_prefix)
    weights = compute_zipf_weights(config.vocab_size, config.zipf_alpha)

    documents: List[Dict[str, object]] = []
    for i in tqdm(range(config.num_documents), desc="Generating documents..."):
        doc_id = f"{config.document_id_prefix}{i+1:06d}"
        terms_in_doc = rng.randint(
            config.min_terms_per_document, config.max_terms_per_document
        )
        terms = sample_unique_terms(rng, vocabulary, weights, terms_in_doc)
        documents.append({"doc_id": doc_id, "terms": terms})

    return documents, vocabulary


def write_outputs(
    documents: List[Dict[str, object]],
    vocabulary: List[str],
    config: GenerationConfig,
) -> None:
    os.makedirs(config.output_directory, exist_ok=True)

    docs_path = os.path.join(config.output_directory, "docs.jsonl")
    vocab_path = os.path.join(config.output_directory, "vocab.txt")
    meta_path = os.path.join(config.output_directory, "meta.json")
    
    logger.info(f"Writing documents to {docs_path}")
    with open(docs_path, "w", encoding="utf-8") as f:
        for doc in tqdm(documents, desc="Writing documents..."):
            f.write(json.dumps(doc, ensure_ascii=False) + "\n")

    logger.info(f"Writing vocabulary to {vocab_path}")
    with open(vocab_path, "w", encoding="utf-8") as f:
        for term in tqdm(vocabulary, desc="Writing vocabulary..."):
            f.write(term + "\n")

    logger.info(f"Writing metadata to {meta_path}")
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(asdict(config), f, indent=2)


def parse_args(argv: Sequence[str] = None) -> GenerationConfig:
    parser = argparse.ArgumentParser(
        description=(
            "Generate documents with Zipf-distributed terms for indexing benchmarks."
        )
    )
    parser.add_argument(
        "--num-docs",
        type=int,
        default=100000000,
        help="Number of documents to generate",
    )
    parser.add_argument(
        "--vocab-size",
        type=int,
        default=500,
        help="Vocabulary size (number of distinct terms)",
    )
    parser.add_argument(
        "--min-terms",
        type=int,
        default=3,
        help="Minimum number of terms per document (inclusive)",
    )
    parser.add_argument(
        "--max-terms",
        type=int,
        default=12,
        help="Maximum number of terms per document (inclusive)",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=1.0,
        help="Zipf exponent alpha (>0); higher means more skew",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=os.path.join("benchmark", "data"),
        help="Output directory for generated files",
    )
    parser.add_argument(
        "--doc-prefix",
        type=str,
        default="doc",
        help="Prefix for document IDs",
    )
    parser.add_argument(
        "--term-prefix",
        type=str,
        default="t",
        help="Prefix for terms in the vocabulary",
    )

    args = parser.parse_args(argv)

    if args.min_terms < 0 or args.max_terms < 0:
        raise ValueError("min-terms and max-terms must be non-negative")
    if args.min_terms > args.max_terms:
        raise ValueError("min-terms must be <= max-terms")
    if args.vocab_size <= 0:
        raise ValueError("vocab-size must be > 0")
    if args.alpha <= 0:
        raise ValueError("alpha must be > 0")
    if args.num_docs < 0:
        raise ValueError("num-docs must be >= 0")

    return GenerationConfig(
        num_documents=args.num_docs,
        vocab_size=args.vocab_size,
        min_terms_per_document=args.min_terms,
        max_terms_per_document=args.max_terms,
        zipf_alpha=args.alpha,
        random_seed=args.seed,
        output_directory=args.output_dir,
        document_id_prefix=args.doc_prefix,
        term_prefix=args.term_prefix,
    )


def main(argv: Sequence[str] = None) -> None:
    config = parse_args(argv)
    documents, vocabulary = generate_documents(config)
    write_outputs(documents, vocabulary, config)
    print(
        f"Generated {len(documents)} documents and {len(vocabulary)} terms into {config.output_directory}"
    )


if __name__ == "__main__":
    main()
