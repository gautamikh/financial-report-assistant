import argparse
import json
from pathlib import Path
from typing import Any

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer


EMBEDDING_MODEL_ID = "sentence-transformers/all-MiniLM-L6-v2"
BATCH_SIZE = 32


def load_chunks(input_path: Path) -> list[dict[str, Any]]:
    """Load and validate chunk records from a JSONL file."""

    if not input_path.exists():
        raise FileNotFoundError(f"Chunk file not found: {input_path}")

    chunks: list[dict[str, Any]] = []

    with input_path.open("r", encoding="utf-8") as input_file:
        for line_number, line in enumerate(input_file, start=1):
            if not line.strip():
                continue

            chunk = json.loads(line)
            embedding_text = chunk.get("embedding_text", "")

            if not embedding_text.strip():
                raise ValueError(
                    f"Missing embedding_text on line {line_number}"
                )

            chunks.append(chunk)

    if not chunks:
        raise ValueError(f"No chunks were found in {input_path}")

    return chunks


def save_metadata(
    chunks: list[dict[str, Any]],
    output_path: Path,
) -> None:
    """Save metadata in the same order as vectors in the FAISS index."""

    with output_path.open("w", encoding="utf-8") as output_file:
        for chunk in chunks:
            output_file.write(
                json.dumps(chunk, ensure_ascii=False) + "\n"
            )


def build_index(
    input_path: Path,
    output_dir: Path,
) -> tuple[int, int]:
    """Create and persist a normalized FAISS inner-product index."""

    chunks = load_chunks(input_path)
    texts = [chunk["embedding_text"] for chunk in chunks]

    print(f"Loaded {len(chunks)} chunks.")
    print(f"Loading embedding model: {EMBEDDING_MODEL_ID}")

    model = SentenceTransformer(EMBEDDING_MODEL_ID)

    print("Generating embeddings...")

    embeddings = model.encode_document(
        texts,
        batch_size=BATCH_SIZE,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )

    embeddings = np.asarray(embeddings, dtype=np.float32)

    if embeddings.ndim != 2:
        raise ValueError(
            f"Expected a 2D embedding matrix, received {embeddings.shape}"
        )

    norms = np.linalg.norm(embeddings, axis=1)

    if not np.allclose(norms, 1.0, atol=1e-3):
        raise ValueError("Embeddings are not correctly normalized.")

    vector_count, vector_dimension = embeddings.shape

    index = faiss.IndexFlatIP(vector_dimension)
    index.add(embeddings)

    if index.ntotal != len(chunks):
        raise RuntimeError(
            "FAISS vector count does not match the metadata count."
        )

    output_dir.mkdir(parents=True, exist_ok=True)

    index_path = output_dir / "index.faiss"
    metadata_path = output_dir / "metadata.jsonl"
    manifest_path = output_dir / "manifest.json"

    faiss.write_index(index, str(index_path))
    save_metadata(chunks, metadata_path)

    manifest = {
        "schema_version": 1,
        "embedding_model": EMBEDDING_MODEL_ID,
        "vector_dimension": vector_dimension,
        "vector_count": vector_count,
        "similarity_metric": "cosine",
        "faiss_index_type": "IndexFlatIP",
        "normalized_embeddings": True,
        "source_chunks": input_path.name,
    }

    manifest_path.write_text(
        json.dumps(manifest, indent=2),
        encoding="utf-8",
    )

    print("FAISS index created.")
    print(f"Vectors:   {vector_count}")
    print(f"Dimension: {vector_dimension}")
    print(f"Index:     {index_path}")
    print(f"Metadata:  {metadata_path}")
    print(f"Manifest:  {manifest_path}")

    return vector_count, vector_dimension


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build a FAISS index from financial-report chunks."
    )
    parser.add_argument(
        "input_path",
        type=Path,
        help="Path to the chunks JSONL file.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Directory where the FAISS index will be stored.",
    )

    args = parser.parse_args()

    build_index(
        input_path=args.input_path,
        output_dir=args.output_dir,
    )


if __name__ == "__main__":
    main()