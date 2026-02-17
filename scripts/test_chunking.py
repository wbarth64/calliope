"""
Spot-check the stanza-aware chunking on all four data sources.

Prints example documents from each source so you can verify:
  - Poems that fit in the budget are kept whole
  - Long poems are split at stanza boundaries (not mid-stanza)
  - Very long stanzas fall back to line-level splits
  - No documents exceed CHARACTER_BUDGET
  - No tiny fragments slip through

Run as:
    cd nanochat && uv run python ../scripts/test_chunking.py
"""

import gzip
import json
import os
import random
import zipfile
from collections import defaultdict

# Import the chunking functions from prepare_poetry_data
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from prepare_poetry_data import (
    clean_text, text_to_stanzas, chunk_stanzas,
    CHARACTER_BUDGET, MIN_DOC_CHARS,
    get_cache_dir, download_poetree,
)

DIVIDER = "=" * 72
THIN = "-" * 72


def show_doc(doc: str, label: str, index: int):
    """Pretty-print a single document with metadata."""
    lines = doc.split("\n")
    stanza_count = doc.count("\n\n") + 1
    print(f"\n{THIN}")
    print(f"  [{label} doc #{index}]  {len(doc)} chars, {len(lines)} lines, "
          f"~{stanza_count} stanzas, ~{len(doc)//4} tokens")
    if len(doc) > CHARACTER_BUDGET:
        print(f"  *** OVER BUDGET ({len(doc)} > {CHARACTER_BUDGET}) ***")
    print(THIN)
    # Show full text if short, otherwise head + tail
    if len(lines) <= 30:
        print(doc)
    else:
        for line in lines[:12]:
            print(line)
        print(f"  ... ({len(lines) - 24} lines omitted) ...")
        for line in lines[-12:]:
            print(line)


def check_invariants(documents: list[str], label: str):
    """Verify chunking invariants and print summary."""
    over_budget = [d for d in documents if len(d) > CHARACTER_BUDGET]
    under_min = [d for d in documents if len(d) < MIN_DOC_CHARS]
    char_lens = sorted(len(d) for d in documents)
    n = len(char_lens)

    print(f"\n{label} invariant check:")
    print(f"  Total documents: {n:,}")
    if n > 0:
        print(f"  Chars: min={char_lens[0]}, median={char_lens[n//2]}, "
              f"p95={char_lens[int(n*0.95)]}, max={char_lens[-1]}")
    print(f"  Over budget (>{CHARACTER_BUDGET}): {len(over_budget)}")
    print(f"  Under minimum (<{MIN_DOC_CHARS}): {len(under_min)}")
    if over_budget:
        print(f"  *** FAIL: {len(over_budget)} documents exceed budget! ***")
        for d in over_budget[:3]:
            print(f"    len={len(d)}: {d[:80]}...")
    if under_min:
        print(f"  *** FAIL: {len(under_min)} documents below minimum! ***")
    if not over_budget and not under_min:
        print(f"  PASS")


def test_gutenberg():
    """Test chunking on Gutenberg data."""
    print(f"\n{DIVIDER}")
    print("SOURCE 1: Gutenberg Poetry Corpus")
    print(DIVIDER)

    cache_dir = get_cache_dir()
    corpus_path = os.path.join(cache_dir, "gutenberg-poetry-v001.ndjson.gz")
    if not os.path.exists(corpus_path):
        print("  Gutenberg corpus not downloaded yet, skipping.")
        return

    # Load just 5 books for testing
    books = defaultdict(list)
    with gzip.open(corpus_path, "rt", encoding="utf-8") as f:
        for line in f:
            record = json.loads(line)
            books[record["gid"]].append(record["s"])
            if len(books) > 5:
                break

    all_docs = []
    for gid, lines in list(books.items())[:3]:
        full_text = clean_text("\n".join(lines))
        stanzas = text_to_stanzas(full_text)
        docs = chunk_stanzas(stanzas)
        all_docs.extend(docs)

        print(f"\nBook '{gid}': {len(lines)} lines -> {len(stanzas)} stanzas -> {len(docs)} documents")
        for i, doc in enumerate(docs[:2]):
            show_doc(doc, f"gutenberg/{gid}", i)
        if len(docs) > 2:
            print(f"  ... ({len(docs) - 2} more documents)")

    check_invariants(all_docs, "Gutenberg (sample)")


def test_kaggle():
    """Test chunking on Kaggle data."""
    print(f"\n{DIVIDER}")
    print("SOURCE 2: Kaggle poemsdataset")
    print(DIVIDER)

    archive_path = os.path.join(
        os.path.expanduser("~"), ".cache", "kagglehub", "datasets",
        "michaelarman", "poemsdataset", "1.archive"
    )
    if not os.path.exists(archive_path):
        print("  Kaggle archive not found, skipping.")
        return

    # Grab a mix: some short, some long
    poems = []
    with zipfile.ZipFile(archive_path, "r") as z:
        for info in z.infolist():
            if not info.filename.endswith(".txt"):
                continue
            raw = z.read(info.filename).decode("utf-8", errors="replace")
            text = clean_text(raw)
            if len(text) >= MIN_DOC_CHARS:
                poems.append((info.filename, text))

    # Pick examples: short, medium, long
    poems.sort(key=lambda x: len(x[1]))
    examples = [
        poems[0],                        # shortest
        poems[len(poems) // 2],          # median
        poems[int(len(poems) * 0.90)],   # 90th percentile
        poems[-1],                        # longest
    ]

    all_docs = []
    for fname, text in examples:
        stanzas = text_to_stanzas(text)
        docs = chunk_stanzas(stanzas)
        all_docs.extend(docs)
        cat = "/".join(fname.replace("\\", "/").split("/")[:2])
        print(f"\n'{cat}': {len(text)} chars -> {len(stanzas)} stanzas -> {len(docs)} documents")
        for i, doc in enumerate(docs[:2]):
            show_doc(doc, f"kaggle/{cat}", i)

    check_invariants(all_docs, "Kaggle (sample)")


def test_huggingface():
    """Test chunking on HuggingFace data."""
    print(f"\n{DIVIDER}")
    print("SOURCE 3: HuggingFace merve/poetry")
    print(DIVIDER)

    from datasets import load_dataset
    ds = load_dataset("merve/poetry", split="train")

    # Pick 4 examples at different lengths
    rows = list(ds)
    rows.sort(key=lambda r: len(r["content"]))
    picks = [rows[0], rows[len(rows)//2], rows[int(len(rows)*0.9)], rows[-1]]

    all_docs = []
    for row in picks:
        text = clean_text(row["content"])
        if len(text) < MIN_DOC_CHARS:
            print(f"\n  '{row['poem name']}' by {row['author']}: {len(text)} chars (skipped, too short)")
            continue
        stanzas = text_to_stanzas(text)
        docs = chunk_stanzas(stanzas)
        all_docs.extend(docs)
        print(f"\n'{row['poem name']}' by {row['author']}: "
              f"{len(text)} chars -> {len(stanzas)} stanzas -> {len(docs)} documents")
        for i, doc in enumerate(docs[:2]):
            show_doc(doc, f"hf/{row['author']}", i)

    check_invariants(all_docs, "HuggingFace (sample)")


def test_poetree():
    """Test chunking on PoeTree data."""
    print(f"\n{DIVIDER}")
    print("SOURCE 4: PoeTree English corpus")
    print(DIVIDER)

    cache_dir = get_cache_dir()
    try:
        zip_path = download_poetree(cache_dir)
    except Exception as e:
        print(f"  PoeTree not available ({e}), skipping.")
        return

    # Read all non-duplicate poems, pick examples at different lengths
    poems = []
    with zipfile.ZipFile(zip_path, "r") as z:
        for name in z.namelist():
            if not name.endswith(".json"):
                continue
            data = json.loads(z.read(name))
            if data.get("duplicate"):
                continue
            body = data.get("body", [])
            if not body:
                continue

            # Group by stanza_id
            stanza_map = defaultdict(list)
            for item in body:
                stanza_map[item.get("stanza_id", 0)].append(item.get("text", ""))
            stanzas = []
            for sid in sorted(stanza_map.keys()):
                s = "\n".join(stanza_map[sid]).strip()
                if s:
                    stanzas.append(s)

            full_text = "\n\n".join(stanzas)
            if len(full_text) >= MIN_DOC_CHARS:
                author = data.get("author", {})
                if isinstance(author, list):
                    author_name = author[0].get("name", "?") if author else "?"
                elif isinstance(author, dict):
                    author_name = author.get("name", "?")
                else:
                    author_name = "?"
                poems.append((data.get("title", "?"), author_name,
                              stanzas, full_text))

    print(f"  Loaded {len(poems):,} non-duplicate poems for testing")

    # Pick: short, median, 95th, longest
    poems.sort(key=lambda x: len(x[3]))
    indices = [0, len(poems)//2, int(len(poems)*0.95), len(poems)-1]
    all_docs = []
    for idx in indices:
        title, author, stanzas, full_text = poems[idx]
        docs = chunk_stanzas(stanzas)
        all_docs.extend(docs)
        print(f"\n'{title}' by {author}: "
              f"{len(full_text)} chars, {len(stanzas)} stanzas -> {len(docs)} documents")
        for i, doc in enumerate(docs[:2]):
            show_doc(doc, f"poetree/{author}", i)
        if len(docs) > 2:
            show_doc(docs[-1], f"poetree/{author}", len(docs)-1)

    check_invariants(all_docs, "PoeTree (sample)")


def test_edge_cases():
    """Test chunking edge cases directly."""
    print(f"\n{DIVIDER}")
    print("EDGE CASES")
    print(DIVIDER)

    # 1. Single stanza that exactly hits budget
    stanza = "x" * CHARACTER_BUDGET
    docs = chunk_stanzas([stanza])
    assert len(docs) == 1 and len(docs[0]) == CHARACTER_BUDGET, \
        f"Exact budget: expected 1 doc of {CHARACTER_BUDGET} chars, got {len(docs)} docs"
    print("  PASS: Single stanza exactly at budget -> 1 document")

    # 2. Single stanza over budget (line-level fallback)
    lines = ["This is line number %d of a very long stanza." % i for i in range(200)]
    big_stanza = "\n".join(lines)
    docs = chunk_stanzas([big_stanza])
    assert all(len(d) <= CHARACTER_BUDGET for d in docs), "Line-level fallback produced over-budget doc"
    assert all(len(d) >= MIN_DOC_CHARS for d in docs), "Line-level fallback produced tiny fragment"
    print(f"  PASS: Over-budget stanza ({len(big_stanza)} chars) -> {len(docs)} documents, all within budget")

    # 3. Many tiny stanzas should pack together
    tiny_stanzas = ["Short line."] * 50
    docs = chunk_stanzas(tiny_stanzas)
    assert len(docs) < 50, f"Tiny stanzas should pack, got {len(docs)} docs from 50 stanzas"
    print(f"  PASS: 50 tiny stanzas -> {len(docs)} packed documents")

    # 4. Empty input
    docs = chunk_stanzas([])
    assert docs == [], "Empty input should return empty list"
    print("  PASS: Empty input -> empty output")

    # 5. All-whitespace stanzas
    docs = chunk_stanzas(["   ", "\n\n", "  \n  "])
    assert docs == [], "Whitespace-only stanzas should be dropped"
    print("  PASS: Whitespace-only stanzas -> dropped")

    # 6. Fragment below MIN_DOC_CHARS
    docs = chunk_stanzas(["Hi"])
    assert docs == [], f"'Hi' ({len('Hi')} chars) should be dropped (min={MIN_DOC_CHARS})"
    print(f"  PASS: Tiny fragment (< {MIN_DOC_CHARS} chars) -> dropped")


def main():
    random.seed(42)
    test_edge_cases()
    test_gutenberg()
    test_kaggle()
    test_huggingface()
    test_poetree()

    print(f"\n{DIVIDER}")
    print("ALL TESTS COMPLETE")
    print(DIVIDER)


if __name__ == "__main__":
    main()
