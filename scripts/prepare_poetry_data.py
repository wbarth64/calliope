"""
Prepare poetry corpora for training with nanochat.

Downloads four sources and converts them to parquet format using unified
stanza-aware chunking:

1. Gutenberg Poetry Corpus (~3M lines from Project Gutenberg)
2. Kaggle poemsdataset (michaelarman/poemsdataset, ~20K poems by form/topic)
3. HuggingFace merve/poetry (573 curated poems with author metadata)
4. PoeTree English corpus (zenodo.org/records/10907309, ~40K poems with stanza markup)

Chunking strategy:
  - Split each poem into stanzas (blank-line separated, or stanza_id for PoeTree)
  - Greedily group consecutive stanzas until the next would exceed CHARACTER_BUDGET
  - If a single stanza exceeds the budget, split it at line boundaries
  - Drop tiny fragments (< MIN_DOC_CHARS)
  - Poems that already fit under the budget are kept whole

Output goes to ~/.cache/nanochat/base_data/ as parquet files.
"""

import gzip
import json
import os
import re
import random
import zipfile
from collections import Counter, defaultdict
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import requests

# Configuration
CORPUS_URL = "http://static.decontextualize.com/gutenberg-poetry-v001.ndjson.gz"
DOCS_PER_SHARD = 20000  # Documents per parquet shard
VAL_RATIO = 0.05  # 5% for validation
RANDOM_SEED = 42

# Chunking parameters
CHARACTER_BUDGET = 1800  # ~450 tokens at ~4 chars/token; leaves packing headroom for seq_len=512
MIN_DOC_CHARS = 40       # drop fragments shorter than this


def get_data_dir():
    """Get the nanochat data directory (matches nanochat's get_base_dir)."""
    if os.environ.get("NANOCHAT_BASE_DIR"):
        base_dir = os.environ["NANOCHAT_BASE_DIR"]
    else:
        base_dir = os.path.join(os.path.expanduser("~"), ".cache", "nanochat")
    data_dir = os.path.join(base_dir, "base_data")
    os.makedirs(data_dir, exist_ok=True)
    return data_dir


def get_cache_dir():
    """Get directory for caching downloads."""
    if os.environ.get("NANOCHAT_BASE_DIR"):
        base_dir = os.environ["NANOCHAT_BASE_DIR"]
    else:
        base_dir = os.path.join(os.path.expanduser("~"), ".cache", "nanochat")
    cache_path = os.path.join(base_dir, "downloads")
    os.makedirs(cache_path, exist_ok=True)
    return cache_path


# ---------------------------------------------------------------------------
# Unified stanza-aware chunking
# ---------------------------------------------------------------------------

def clean_text(text: str) -> str:
    """Normalize text: fix line endings, strip trailing whitespace per line."""
    text = text.replace("\r\n", "\n").replace("\r", "\n").strip()
    text = "\n".join(line.rstrip() for line in text.split("\n"))
    # Collapse 4+ consecutive newlines to 3 (one blank line between stanzas)
    text = re.sub(r"\n{4,}", "\n\n\n", text)
    return text


def chunk_stanzas(stanzas: list[str], budget: int = CHARACTER_BUDGET,
                  min_chars: int = MIN_DOC_CHARS) -> list[str]:
    """
    Greedily group stanzas into documents that fit within the character budget.

    Args:
        stanzas: List of stanza strings (each may be multi-line).
        budget: Maximum characters per output document.
        min_chars: Minimum characters; smaller fragments are dropped.

    Returns:
        List of document strings.
    """
    documents = []
    current_parts = []
    current_len = 0

    def emit():
        nonlocal current_parts, current_len
        if current_parts:
            doc = "\n\n".join(current_parts)
            if len(doc) >= min_chars:
                documents.append(doc)
            current_parts = []
            current_len = 0

    for stanza in stanzas:
        stanza = stanza.strip()
        if not stanza:
            continue

        stanza_len = len(stanza)

        # If this single stanza exceeds the budget, split it at line boundaries
        if stanza_len > budget:
            emit()  # flush anything accumulated
            lines = stanza.split("\n")
            line_group = []
            group_len = 0
            for line in lines:
                # Skip individual lines that exceed the budget (prose, not poetry)
                if len(line) > budget:
                    continue
                added_len = len(line) + (1 if line_group else 0)  # +1 for newline
                if group_len + added_len > budget and line_group:
                    doc = "\n".join(line_group)
                    if len(doc) >= min_chars:
                        documents.append(doc)
                    line_group = []
                    group_len = 0
                line_group.append(line)
                group_len += added_len
            # Leftover lines become the start of the next accumulation
            if line_group:
                current_parts = ["\n".join(line_group)]
                current_len = sum(len(l) for l in line_group) + len(line_group) - 1
            continue

        # Would adding this stanza (with \n\n separator) exceed the budget?
        separator_len = 2 if current_parts else 0  # "\n\n"
        if current_len + separator_len + stanza_len > budget and current_parts:
            emit()

        current_parts.append(stanza)
        separator_len = 2 if len(current_parts) > 1 else 0
        current_len = sum(len(p) for p in current_parts) + 2 * (len(current_parts) - 1)

    emit()  # flush remainder
    return documents


def text_to_stanzas(text: str) -> list[str]:
    """Split cleaned text into stanzas at blank lines."""
    # Split on one or more blank lines
    parts = re.split(r"\n\s*\n", text)
    return [p.strip() for p in parts if p.strip()]


# ---------------------------------------------------------------------------
# Source 1: Gutenberg Poetry Corpus
# ---------------------------------------------------------------------------

def download_gutenberg(cache_dir: str) -> str:
    """Download the Gutenberg corpus if not already cached."""
    corpus_path = os.path.join(cache_dir, "gutenberg-poetry-v001.ndjson.gz")
    if os.path.exists(corpus_path):
        print(f"  Already downloaded: {corpus_path}")
        return corpus_path

    print(f"  Downloading from {CORPUS_URL}...")
    response = requests.get(CORPUS_URL, stream=True)
    response.raise_for_status()
    total_size = int(response.headers.get("content-length", 0))
    downloaded = 0
    with open(corpus_path, "wb") as f:
        for chunk in response.iter_content(chunk_size=1024 * 1024):
            if chunk:
                f.write(chunk)
                downloaded += len(chunk)
                if total_size > 0:
                    print(f"\r  Downloading: {100 * downloaded / total_size:.1f}%",
                          end="", flush=True)
    print(f"\n  Downloaded to {corpus_path}")
    return corpus_path


def load_gutenberg_documents(cache_dir: str) -> list[str]:
    """Load Gutenberg corpus, group by book, chunk by stanza."""
    print("\n--- Source 1: Gutenberg Poetry Corpus ---")
    corpus_path = download_gutenberg(cache_dir)

    # Group lines by Gutenberg book ID
    print("  Grouping lines by book ID...")
    books = defaultdict(list)
    line_count = 0
    with gzip.open(corpus_path, "rt", encoding="utf-8") as f:
        for line in f:
            record = json.loads(line)
            books[record["gid"]].append(record["s"])
            line_count += 1
            if line_count % 500000 == 0:
                print(f"    {line_count:,} lines from {len(books):,} books...")
    print(f"  Loaded {line_count:,} lines from {len(books):,} books")

    # Each "book" is an entire collection — join lines, then split into stanzas
    documents = []
    for gid, lines in books.items():
        full_text = clean_text("\n".join(lines))
        stanzas = text_to_stanzas(full_text)
        documents.extend(chunk_stanzas(stanzas))

    print(f"  Documents after chunking: {len(documents):,}")
    return documents


# ---------------------------------------------------------------------------
# Source 2: Kaggle poemsdataset
# ---------------------------------------------------------------------------

def load_kaggle_documents() -> list[str]:
    """Load poems from Kaggle poemsdataset, chunk by stanza."""
    print("\n--- Source 2: Kaggle poemsdataset ---")
    import kagglehub

    try:
        kagglehub.dataset_download("michaelarman/poemsdataset")
    except Exception:
        pass  # archive still downloaded even if Windows extraction fails

    archive_path = os.path.join(
        os.path.expanduser("~"), ".cache", "kagglehub", "datasets",
        "michaelarman", "poemsdataset", "1.archive"
    )
    if not os.path.exists(archive_path):
        print(f"  WARNING: Archive not found at {archive_path}, skipping.")
        return []

    documents = []
    seen = set()
    stats = Counter()

    with zipfile.ZipFile(archive_path, "r") as z:
        for info in z.infolist():
            if not info.filename.endswith(".txt"):
                continue
            raw = z.read(info.filename).decode("utf-8", errors="replace")
            text = clean_text(raw)
            if len(text) < MIN_DOC_CHARS:
                stats["too_short"] += 1
                continue

            key = text.lower()
            if key in seen:
                stats["duplicate"] += 1
                continue
            seen.add(key)

            stanzas = text_to_stanzas(text)
            chunks = chunk_stanzas(stanzas)
            documents.extend(chunks)
            stats["poems_kept"] += 1

    print(f"  Poems: {stats['poems_kept']:,} kept, "
          f"{stats['too_short']} too short, {stats['duplicate']} duplicates")
    print(f"  Documents after chunking: {len(documents):,}")
    return documents


# ---------------------------------------------------------------------------
# Source 3: HuggingFace merve/poetry
# ---------------------------------------------------------------------------

def load_hf_documents(global_seen: set[str]) -> list[str]:
    """Load poems from HuggingFace merve/poetry, chunk by stanza."""
    print("\n--- Source 3: HuggingFace merve/poetry ---")
    from datasets import load_dataset

    ds = load_dataset("merve/poetry", split="train")
    documents = []
    stats = Counter()

    for row in ds:
        text = clean_text(row["content"])
        if len(text) < MIN_DOC_CHARS:
            stats["too_short"] += 1
            continue

        key = text.lower()
        if key in global_seen:
            stats["duplicate"] += 1
            continue
        global_seen.add(key)

        stanzas = text_to_stanzas(text)
        chunks = chunk_stanzas(stanzas)
        documents.extend(chunks)
        stats["poems_kept"] += 1

    print(f"  Poems: {stats['poems_kept']:,} kept, "
          f"{stats['too_short']} too short, {stats['duplicate']} duplicates")
    print(f"  Documents after chunking: {len(documents):,}")
    return documents


# ---------------------------------------------------------------------------
# Source 4: PoeTree English corpus
# ---------------------------------------------------------------------------

POETREE_URL = "https://zenodo.org/records/10907309/files/en.zip?download=1"


def download_poetree(cache_dir: str) -> str:
    """Download PoeTree English corpus if not already cached."""
    zip_path = os.path.join(cache_dir, "poetree-en.zip")
    if os.path.exists(zip_path):
        print(f"  Already downloaded: {zip_path}")
        return zip_path

    # Also check repo root for a manually-placed en.zip
    repo_root_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "en.zip"
    )
    if os.path.exists(repo_root_path):
        print(f"  Using local file: {repo_root_path}")
        return repo_root_path

    print(f"  Downloading PoeTree from Zenodo...")
    response = requests.get(POETREE_URL, stream=True)
    response.raise_for_status()
    total_size = int(response.headers.get("content-length", 0))
    downloaded = 0
    with open(zip_path, "wb") as f:
        for chunk in response.iter_content(chunk_size=1024 * 1024):
            if chunk:
                f.write(chunk)
                downloaded += len(chunk)
                if total_size > 0:
                    print(f"\r  Downloading: {100 * downloaded / total_size:.1f}%",
                          end="", flush=True)
    print(f"\n  Downloaded to {zip_path}")
    return zip_path


def load_poetree_documents(cache_dir: str,
                           global_seen: set[str] | None = None) -> list[str]:
    """Load PoeTree English poems, using native stanza_id markup for chunking."""
    print("\n--- Source 4: PoeTree English corpus ---")
    zip_path = download_poetree(cache_dir)

    if global_seen is None:
        global_seen = set()

    documents = []
    stats = Counter()

    with zipfile.ZipFile(zip_path, "r") as z:
        for name in z.namelist():
            if not name.endswith(".json"):
                continue
            data = json.loads(z.read(name))

            # Skip duplicates flagged by PoeTree itself
            if data.get("duplicate", False):
                stats["poetree_flagged_dup"] += 1
                continue

            body = data.get("body", [])
            if not body:
                stats["empty"] += 1
                continue

            # Group lines by stanza_id (PoeTree provides this explicitly)
            stanza_map = defaultdict(list)
            for item in body:
                sid = item.get("stanza_id", 0)
                stanza_map[sid].append(item.get("text", ""))

            # Build stanza strings in order
            stanzas = []
            for sid in sorted(stanza_map.keys()):
                stanza_text = "\n".join(stanza_map[sid]).strip()
                if stanza_text:
                    stanzas.append(stanza_text)

            if not stanzas:
                stats["empty"] += 1
                continue

            # Check full poem text for cross-source dedup
            full_text = "\n\n".join(stanzas)
            if len(full_text) < MIN_DOC_CHARS:
                stats["too_short"] += 1
                continue

            key = full_text.lower()
            if key in global_seen:
                stats["cross_dup"] += 1
                continue
            global_seen.add(key)

            chunks = chunk_stanzas(stanzas)
            documents.extend(chunks)
            stats["poems_kept"] += 1

    print(f"  Poems: {stats['poems_kept']:,} kept, "
          f"{stats.get('poetree_flagged_dup', 0)} PoeTree-flagged duplicates, "
          f"{stats.get('cross_dup', 0)} cross-source duplicates, "
          f"{stats.get('too_short', 0)} too short, "
          f"{stats.get('empty', 0)} empty")
    print(f"  Documents after chunking: {len(documents):,}")
    return documents


# ---------------------------------------------------------------------------
# Writing and stats
# ---------------------------------------------------------------------------

def write_parquet_shards(documents: list[str], data_dir: str,
                         docs_per_shard: int = DOCS_PER_SHARD):
    """Write documents to parquet shards with train/val split."""
    print(f"\nWriting parquet shards to {data_dir}...")

    random.seed(RANDOM_SEED)
    random.shuffle(documents)

    val_count = int(len(documents) * VAL_RATIO)
    train_docs = documents[:-val_count] if val_count > 0 else documents
    val_docs = documents[-val_count:] if val_count > 0 else []

    print(f"  Training documents: {len(train_docs):,}")
    print(f"  Validation documents: {len(val_docs):,}")

    # Clear existing parquet files
    for f in os.listdir(data_dir):
        if f.endswith(".parquet"):
            os.remove(os.path.join(data_dir, f))

    shard_idx = 0
    for i in range(0, len(train_docs), docs_per_shard):
        shard_docs = train_docs[i:i + docs_per_shard]
        table = pa.table({"text": shard_docs})
        shard_path = os.path.join(data_dir, f"shard_{shard_idx:05d}.parquet")
        pq.write_table(table, shard_path, row_group_size=1000)
        print(f"  Wrote {shard_path} ({len(shard_docs):,} docs)")
        shard_idx += 1

    if val_docs:
        table = pa.table({"text": val_docs})
        shard_path = os.path.join(data_dir, f"shard_{shard_idx:05d}.parquet")
        pq.write_table(table, shard_path, row_group_size=1000)
        print(f"  Wrote {shard_path} ({len(val_docs):,} validation docs)")

    print(f"  Total shards: {shard_idx + 1}")


def compute_stats(documents: list[str], label: str = ""):
    """Print corpus statistics."""
    if not documents:
        print(f"\n{label}: 0 documents")
        return
    total_chars = sum(len(d) for d in documents)
    total_lines = sum(d.count("\n") + 1 for d in documents)
    char_lens = sorted(len(d) for d in documents)
    n = len(char_lens)
    print(f"\n{label} stats:")
    print(f"  Documents: {n:,}")
    print(f"  Characters: {total_chars:,} (~{total_chars // 4:,} tokens)")
    print(f"  Lines: {total_lines:,}")
    print(f"  Chars/doc: min={char_lens[0]}, median={char_lens[n//2]}, "
          f"p95={char_lens[int(n*0.95)]}, max={char_lens[-1]}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    data_dir = get_data_dir()
    cache_dir = get_cache_dir()
    print(f"Data directory: {data_dir}")
    print(f"Cache directory: {cache_dir}")

    # Load all sources, building a global dedup set as we go
    gutenberg_docs = load_gutenberg_documents(cache_dir)
    compute_stats(gutenberg_docs, "Gutenberg")

    kaggle_docs = load_kaggle_documents()
    compute_stats(kaggle_docs, "Kaggle")

    # Build global dedup set from Kaggle (Gutenberg is too fragmentary to dedup against)
    global_seen = set()
    for doc in kaggle_docs:
        global_seen.add(doc.lower())

    hf_docs = load_hf_documents(global_seen)
    compute_stats(hf_docs, "HuggingFace")

    poetree_docs = load_poetree_documents(cache_dir, global_seen=global_seen)
    compute_stats(poetree_docs, "PoeTree")

    # Combine
    all_docs = gutenberg_docs + kaggle_docs + hf_docs + poetree_docs
    compute_stats(all_docs, "Combined")

    # Write
    write_parquet_shards(all_docs, data_dir)
    print("\nDone!")


if __name__ == "__main__":
    main()
