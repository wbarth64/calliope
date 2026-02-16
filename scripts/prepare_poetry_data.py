"""
Prepare the Gutenberg Poetry Corpus for training with nanochat.

Downloads the corpus and converts it to parquet format with hybrid chunking:
- Full poems (≤20 lines) are kept as single documents
- Longer poems are chunked into ~20 line segments

Output goes to ~/.nanochat/base_data/ as parquet files.
"""

import gzip
import json
import os
import random
from collections import defaultdict
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import requests

# Configuration
CORPUS_URL = "http://static.decontextualize.com/gutenberg-poetry-v001.ndjson.gz"
CHUNK_SIZE = 20  # Max lines per document
DOCS_PER_SHARD = 20000  # Documents per parquet shard
VAL_RATIO = 0.05  # 5% for validation
RANDOM_SEED = 42


def get_data_dir():
    """Get the nanochat data directory (matches nanochat's get_base_dir)."""
    if os.environ.get("NANOCHAT_BASE_DIR"):
        base_dir = os.environ.get("NANOCHAT_BASE_DIR")
    else:
        home_dir = os.path.expanduser("~")
        cache_dir = os.path.join(home_dir, ".cache")
        base_dir = os.path.join(cache_dir, "nanochat")
    data_dir = os.path.join(base_dir, "base_data")
    os.makedirs(data_dir, exist_ok=True)
    return data_dir


def get_cache_dir():
    """Get directory for caching downloads."""
    if os.environ.get("NANOCHAT_BASE_DIR"):
        base_dir = os.environ.get("NANOCHAT_BASE_DIR")
    else:
        home_dir = os.path.expanduser("~")
        cache_dir = os.path.join(home_dir, ".cache")
        base_dir = os.path.join(cache_dir, "nanochat")
    cache_path = os.path.join(base_dir, "downloads")
    os.makedirs(cache_path, exist_ok=True)
    return cache_path


def download_corpus(cache_dir: str) -> str:
    """Download the corpus if not already cached."""
    os.makedirs(cache_dir, exist_ok=True)
    corpus_path = os.path.join(cache_dir, "gutenberg-poetry-v001.ndjson.gz")

    if os.path.exists(corpus_path):
        print(f"Corpus already downloaded at {corpus_path}")
        return corpus_path

    print(f"Downloading corpus from {CORPUS_URL}...")
    response = requests.get(CORPUS_URL, stream=True)
    response.raise_for_status()

    total_size = int(response.headers.get('content-length', 0))
    downloaded = 0

    with open(corpus_path, 'wb') as f:
        for chunk in response.iter_content(chunk_size=1024 * 1024):
            if chunk:
                f.write(chunk)
                downloaded += len(chunk)
                if total_size > 0:
                    pct = 100 * downloaded / total_size
                    print(f"\rDownloading: {pct:.1f}%", end="", flush=True)

    print(f"\nDownloaded to {corpus_path}")
    return corpus_path


def load_and_group_lines(corpus_path: str) -> dict[str, list[str]]:
    """Load corpus and group lines by Gutenberg book ID."""
    print("Loading and grouping lines by book ID...")

    books = defaultdict(list)
    line_count = 0

    with gzip.open(corpus_path, 'rt', encoding='utf-8') as f:
        for line in f:
            record = json.loads(line)
            gid = record['gid']
            text = record['s']
            books[gid].append(text)
            line_count += 1

            if line_count % 500000 == 0:
                print(f"  Processed {line_count:,} lines from {len(books):,} books...")

    print(f"Loaded {line_count:,} lines from {len(books):,} books")
    return books


def chunk_poems(books: dict[str, list[str]], chunk_size: int = CHUNK_SIZE) -> list[str]:
    """
    Apply hybrid chunking to poems.

    - Poems with ≤chunk_size lines: kept as single document
    - Longer poems: split into chunks of chunk_size lines

    Returns list of document strings (lines joined by newlines).
    """
    print(f"Applying hybrid chunking (chunk_size={chunk_size})...")

    documents = []
    full_poems = 0
    chunked_poems = 0

    for gid, lines in books.items():
        if len(lines) <= chunk_size:
            # Keep as single document
            doc = '\n'.join(lines)
            documents.append(doc)
            full_poems += 1
        else:
            # Split into chunks
            for i in range(0, len(lines), chunk_size):
                chunk = lines[i:i + chunk_size]
                doc = '\n'.join(chunk)
                documents.append(doc)
            chunked_poems += 1

    print(f"  Full poems (<={chunk_size} lines): {full_poems:,}")
    print(f"  Chunked poems (>{chunk_size} lines): {chunked_poems:,}")
    print(f"  Total documents: {len(documents):,}")

    return documents


def write_parquet_shards(documents: list[str], data_dir: str, docs_per_shard: int = DOCS_PER_SHARD):
    """Write documents to parquet shards with train/val split."""
    print(f"Writing parquet shards to {data_dir}...")

    # Shuffle documents
    random.seed(RANDOM_SEED)
    random.shuffle(documents)

    # Split into train and validation
    val_count = int(len(documents) * VAL_RATIO)
    train_docs = documents[:-val_count] if val_count > 0 else documents
    val_docs = documents[-val_count:] if val_count > 0 else []

    print(f"  Training documents: {len(train_docs):,}")
    print(f"  Validation documents: {len(val_docs):,}")

    # Clear existing parquet files
    for f in os.listdir(data_dir):
        if f.endswith('.parquet'):
            os.remove(os.path.join(data_dir, f))

    # Write training shards
    shard_idx = 0
    for i in range(0, len(train_docs), docs_per_shard):
        shard_docs = train_docs[i:i + docs_per_shard]
        table = pa.table({'text': shard_docs})

        shard_path = os.path.join(data_dir, f"shard_{shard_idx:05d}.parquet")
        pq.write_table(table, shard_path, row_group_size=1000)
        print(f"  Wrote {shard_path} ({len(shard_docs):,} documents)")
        shard_idx += 1

    # Write validation shard (last shard)
    if val_docs:
        table = pa.table({'text': val_docs})
        shard_path = os.path.join(data_dir, f"shard_{shard_idx:05d}.parquet")
        pq.write_table(table, shard_path, row_group_size=1000)
        print(f"  Wrote {shard_path} ({len(val_docs):,} validation documents)")

    print(f"Wrote {shard_idx + 1} shards total")


def compute_stats(documents: list[str]):
    """Compute and print corpus statistics."""
    print("\nCorpus statistics:")

    total_chars = sum(len(doc) for doc in documents)
    total_lines = sum(doc.count('\n') + 1 for doc in documents)
    avg_doc_lines = total_lines / len(documents) if documents else 0

    # Estimate tokens (rough: ~4 chars per token for English)
    estimated_tokens = total_chars / 4

    print(f"  Total characters: {total_chars:,}")
    print(f"  Total lines: {total_lines:,}")
    print(f"  Average lines per document: {avg_doc_lines:.1f}")
    print(f"  Estimated tokens: {estimated_tokens:,.0f}")


def main():
    # Get directories
    data_dir = get_data_dir()
    cache_dir = get_cache_dir()

    print(f"Data directory: {data_dir}")
    print(f"Cache directory: {cache_dir}")
    print()

    # Download corpus
    corpus_path = download_corpus(cache_dir)

    # Load and process
    books = load_and_group_lines(corpus_path)
    documents = chunk_poems(books)

    # Compute stats
    compute_stats(documents)

    # Write parquet files
    write_parquet_shards(documents, data_dir)

    print("\nData preparation complete!")
    print(f"Parquet files written to: {data_dir}")


if __name__ == "__main__":
    main()
