import argparse
import json
import re
from pathlib import Path
from typing import Any

import faiss
import numpy as np
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer


RRF_K = 60

TOKEN_PATTERN = re.compile(
    r"\$?\d[\d,]*(?:\.\d+)?%?|[A-Za-z]+(?:[-'][A-Za-z]+)*"
)


def tokenize(text: str) -> list[str]:
    """Tokenize financial text for BM25."""

    return TOKEN_PATTERN.findall(text.lower())


def load_json(path: Path) -> dict[str, Any]:
    """Load a JSON object."""

    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")

    with path.open("r", encoding="utf-8") as input_file:
        return json.load(input_file)


def load_metadata(path: Path) -> list[dict[str, Any]]:
    """Load metadata while preserving FAISS row order."""

    if not path.exists():
        raise FileNotFoundError(f"Metadata not found: {path}")

    records: list[dict[str, Any]] = []

    with path.open("r", encoding="utf-8") as input_file:
        for line_number, line in enumerate(input_file, start=1):
            if not line.strip():
                continue

            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as error:
                raise ValueError(
                    f"Invalid JSON on metadata line {line_number}"
                ) from error

    if not records:
        raise ValueError(f"No metadata records found in {path}")

    return records


def retrieve_dense(
    query: str,
    model: SentenceTransformer,
    index: Any,
    candidate_count: int,
) -> list[tuple[int, float]]:
    """Retrieve semantic candidates from FAISS."""

    query_embedding = model.encode_query(
        query,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )

    query_embedding = np.asarray(
        query_embedding,
        dtype=np.float32,
    ).reshape(1, -1)

    if query_embedding.shape[1] != index.d:
        raise ValueError(
            "Query embedding dimension does not match the FAISS index."
        )

    scores, positions = index.search(
        query_embedding,
        candidate_count,
    )

    results: list[tuple[int, float]] = []

    for position, score in zip(positions[0], scores[0]):
        if position >= 0:
            results.append((int(position), float(score)))

    return results


def retrieve_sparse(
    query: str,
    metadata: list[dict[str, Any]],
    candidate_count: int,
) -> list[tuple[int, float]]:
    """Retrieve exact lexical candidates using BM25."""

    tokenized_corpus = [
        tokenize(record["embedding_text"])
        for record in metadata
    ]

    query_tokens = tokenize(query)

    if not query_tokens:
        raise ValueError("The query contains no searchable tokens.")

    bm25 = BM25Okapi(tokenized_corpus)
    scores = bm25.get_scores(query_tokens)

    ranked_positions = np.argsort(scores)[::-1][:candidate_count]

    return [
        (int(position), float(scores[position]))
        for position in ranked_positions
    ]


def reciprocal_rank_fusion(
    dense_results: list[tuple[int, float]],
    sparse_results: list[tuple[int, float]],
    metadata: list[dict[str, Any]],
    top_k: int,
) -> list[dict[str, Any]]:
    """Fuse dense and sparse rankings using RRF."""

    fused: dict[int, dict[str, Any]] = {}

    for rank, (position, score) in enumerate(
        dense_results,
        start=1,
    ):
        fused[position] = {
            "position": position,
            "rrf_score": 0.0,
            "dense_rank": None,
            "dense_score": None,
            "sparse_rank": None,
            "sparse_score": None,
            "chunk": metadata[position],
        }

        fused[position]["rrf_score"] += 1.0 / (RRF_K + rank)
        fused[position]["dense_rank"] = rank
        fused[position]["dense_score"] = score

    for rank, (position, score) in enumerate(
        sparse_results,
        start=1,
    ):
        if position not in fused:
            fused[position] = {
                "position": position,
                "rrf_score": 0.0,
                "dense_rank": None,
                "dense_score": None,
                "sparse_rank": None,
                "sparse_score": None,
                "chunk": metadata[position],
            }

        fused[position]["rrf_score"] += 1.0 / (RRF_K + rank)
        fused[position]["sparse_rank"] = rank
        fused[position]["sparse_score"] = score

    ranked_results = sorted(
        fused.values(),
        key=lambda result: result["rrf_score"],
        reverse=True,
    )

    return ranked_results[:top_k]


def hybrid_search(
    index_dir: Path,
    query: str,
    top_k: int,
    candidate_k: int,
) -> list[dict[str, Any]]:
    """Run FAISS and BM25 retrieval followed by RRF."""

    if not query.strip():
        raise ValueError("The query cannot be empty.")

    if top_k < 1:
        raise ValueError("top_k must be at least 1.")

    if candidate_k < top_k:
        raise ValueError("candidate_k must be at least top_k.")

    index_path = index_dir / "index.faiss"
    metadata_path = index_dir / "metadata.jsonl"
    manifest_path = index_dir / "manifest.json"

    if not index_path.exists():
        raise FileNotFoundError(f"FAISS index not found: {index_path}")

    index = faiss.read_index(str(index_path))
    metadata = load_metadata(metadata_path)
    manifest = load_json(manifest_path)

    if index.ntotal != len(metadata):
        raise ValueError(
            "FAISS vector count does not match metadata count."
        )

    model_id = manifest["embedding_model"]

    print(f"Loading embedding model: {model_id}")
    model = SentenceTransformer(model_id)

    candidate_count = min(candidate_k, len(metadata))

    print("Running vector search...")
    dense_results = retrieve_dense(
        query=query,
        model=model,
        index=index,
        candidate_count=candidate_count,
    )

    print("Running BM25 search...")
    sparse_results = retrieve_sparse(
        query=query,
        metadata=metadata,
        candidate_count=candidate_count,
    )

    print("Fusing rankings...")
    return reciprocal_rank_fusion(
        dense_results=dense_results,
        sparse_results=sparse_results,
        metadata=metadata,
        top_k=top_k,
    )


def display_results(
    query: str,
    results: list[dict[str, Any]],
) -> None:
    """Display hybrid search results."""

    print()
    print(f"Question: {query}")
    print(f"Retrieved {len(results)} passage(s).")
    print()

    for rank, result in enumerate(results, start=1):
        chunk = result["chunk"]

        pages = ", ".join(
            str(page)
            for page in chunk.get("page_numbers", [])
        )
        pages = pages or "unknown"

        headings = " > ".join(chunk.get("headings", []))
        headings = headings or "Unlabelled section"

        dense_rank = result["dense_rank"]
        sparse_rank = result["sparse_rank"]

        found_by: list[str] = []

        if dense_rank is not None:
            found_by.append("vector")

        if sparse_rank is not None:
            found_by.append("BM25")

        print("=" * 80)
        print(f"Result {rank}")
        print(f"RRF score:    {result['rrf_score']:.6f}")
        print(f"Found by:     {', '.join(found_by)}")
        print(f"Vector rank:  {dense_rank or '-'}")
        print(f"BM25 rank:    {sparse_rank or '-'}")

        if result["dense_score"] is not None:
            print(
                f"Cosine score: {result['dense_score']:.4f}"
            )

        if result["sparse_score"] is not None:
            print(
                f"BM25 score:   {result['sparse_score']:.4f}"
            )

        print(f"Chunk:        {chunk.get('chunk_id', 'unknown')}")
        print(f"Pages:        {pages}")
        print(f"Section:      {headings}")
        print("-" * 80)
        print(chunk.get("text", ""))
        print()

    print("=" * 80)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Hybrid search using FAISS, BM25, and RRF."
    )
    parser.add_argument(
        "query",
        help="Financial question or search query.",
    )
    parser.add_argument(
        "--index-dir",
        type=Path,
        required=True,
        help="Directory containing the FAISS index and metadata.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=5,
        help="Number of final results.",
    )
    parser.add_argument(
        "--candidate-k",
        type=int,
        default=20,
        help="Candidates retrieved by each search method.",
    )

    args = parser.parse_args()

    results = hybrid_search(
        index_dir=args.index_dir,
        query=args.query,
        top_k=args.top_k,
        candidate_k=args.candidate_k,
    )

    display_results(
        query=args.query,
        results=results,
    )


if __name__ == "__main__":
    main()