import argparse
import json
from pathlib import Path
from typing import Any

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer


def load_json(path: Path) -> dict[str, Any]:
    """Load a JSON object from disk."""

    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")

    with path.open("r", encoding="utf-8") as input_file:
        return json.load(input_file)


def load_metadata(path: Path) -> list[dict[str, Any]]:
    """Load chunk metadata while preserving FAISS row order."""

    if not path.exists():
        raise FileNotFoundError(f"Metadata file not found: {path}")

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

    return records


def search_index(
    index_dir: Path,
    query: str,
    top_k: int,
) -> list[dict[str, Any]]:
    """Search a FAISS index and return matched chunks."""

    if not query.strip():
        raise ValueError("The search query cannot be empty.")

    if top_k < 1:
        raise ValueError("top_k must be at least 1.")

    index_path = index_dir / "index.faiss"
    metadata_path = index_dir / "metadata.jsonl"
    manifest_path = index_dir / "manifest.json"

    if not index_path.exists():
        raise FileNotFoundError(f"FAISS index not found: {index_path}")

    manifest = load_json(manifest_path)
    metadata = load_metadata(metadata_path)
    index = faiss.read_index(str(index_path))

    if index.ntotal != len(metadata):
        raise ValueError(
            "FAISS vector count does not match metadata count: "
            f"{index.ntotal} vectors versus {len(metadata)} records."
        )

    expected_dimension = manifest["vector_dimension"]

    if index.d != expected_dimension:
        raise ValueError(
            "FAISS dimension does not match the manifest: "
            f"{index.d} versus {expected_dimension}."
        )

    model_id = manifest["embedding_model"]

    print(f"Loading embedding model: {model_id}")
    model = SentenceTransformer(model_id)

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
            "Query embedding dimension does not match the FAISS index: "
            f"{query_embedding.shape[1]} versus {index.d}."
        )

    result_count = min(top_k, index.ntotal)
    scores, positions = index.search(query_embedding, result_count)

    results: list[dict[str, Any]] = []

    for score, position in zip(scores[0], positions[0]):
        if position < 0:
            continue

        result = {
            "score": float(score),
            "position": int(position),
            "chunk": metadata[int(position)],
        }
        results.append(result)

    return results


def display_results(
    query: str,
    results: list[dict[str, Any]],
) -> None:
    """Display retrieved passages in a readable terminal format."""

    print()
    print(f"Question: {query}")
    print(f"Retrieved {len(results)} passage(s).")
    print()

    for rank, result in enumerate(results, start=1):
        chunk = result["chunk"]
        score = result["score"]

        pages = chunk.get("page_numbers", [])
        page_text = ", ".join(str(page) for page in pages)
        page_text = page_text or "unknown"

        headings = chunk.get("headings", [])
        section_text = " > ".join(headings)
        section_text = section_text or "Unlabelled section"

        print("=" * 80)
        print(f"Result {rank}")
        print(f"Score:   {score:.4f}")
        print(f"Chunk:   {chunk.get('chunk_id', 'unknown')}")
        print(f"Company: {chunk.get('company', 'unknown')}")
        print(f"Year:    {chunk.get('fiscal_year', 'unknown')}")
        print(f"Pages:   {page_text}")
        print(f"Section: {section_text}")
        print("-" * 80)
        print(chunk.get("text", ""))
        print()

    print("=" * 80)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Search financial-report chunks using FAISS."
    )
    parser.add_argument(
        "query",
        help="Financial question or search query.",
    )
    parser.add_argument(
        "--index-dir",
        type=Path,
        required=True,
        help="Directory containing index.faiss and its metadata.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=5,
        help="Number of passages to retrieve.",
    )

    args = parser.parse_args()

    results = search_index(
        index_dir=args.index_dir,
        query=args.query,
        top_k=args.top_k,
    )

    display_results(
        query=args.query,
        results=results,
    )


if __name__ == "__main__":
    main()