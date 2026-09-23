import argparse
import json
from pathlib import Path
from typing import Any

from docling.chunking import HybridChunker
from docling_core.transforms.chunker.tokenizer.huggingface import (
    HuggingFaceTokenizer,
)
from docling_core.types.doc.document import DoclingDocument
from transformers import AutoTokenizer


EMBEDDING_MODEL_ID = "sentence-transformers/all-MiniLM-L6-v2"
MAX_CHUNK_TOKENS = 240


def extract_page_numbers(chunk: Any) -> list[int]:
    """Return sorted, unique source pages associated with a chunk."""

    page_numbers: set[int] = set()

    for document_item in chunk.meta.doc_items:
        for provenance in document_item.prov or []:
            page_numbers.add(provenance.page_no)

    return sorted(page_numbers)


def extract_document_references(chunk: Any) -> list[str]:
    """Return Docling references for the source items in a chunk."""

    references: list[str] = []

    for document_item in chunk.meta.doc_items:
        self_ref = getattr(document_item, "self_ref", None)

        if self_ref is not None:
            references.append(str(self_ref))

    return references


def create_chunks(
    input_path: Path,
    output_path: Path,
    company: str,
    fiscal_year: int,
    document_type: str,
) -> int:
    """Create structure-aware chunks from a Docling JSON document."""

    if not input_path.exists():
        raise FileNotFoundError(f"Docling JSON not found: {input_path}")

    if input_path.suffix.lower() != ".json":
        raise ValueError(
            f"Expected Docling JSON, received: {input_path.suffix}"
        )

    print(f"Loading Docling document: {input_path}")
    document = DoclingDocument.load_from_json(input_path)

    print(f"Loading tokenizer: {EMBEDDING_MODEL_ID}")
    hugging_face_tokenizer = AutoTokenizer.from_pretrained(
        EMBEDDING_MODEL_ID
    )

    tokenizer = HuggingFaceTokenizer(
        tokenizer=hugging_face_tokenizer,
        max_tokens=MAX_CHUNK_TOKENS,
    )

    chunker = HybridChunker(
        tokenizer=tokenizer,
        merge_peers=True,
        repeat_table_header=True,
        omit_header_on_overflow=True,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)

    chunk_count = 0

    with output_path.open("w", encoding="utf-8") as output_file:
        for index, chunk in enumerate(
            chunker.chunk(dl_doc=document),
            start=1,
        ):
            embedding_text = chunker.contextualize(chunk=chunk)
            headings = list(chunk.meta.headings or [])
            page_numbers = extract_page_numbers(chunk)
            document_references = extract_document_references(chunk)

            source_document = input_path.name

            if chunk.meta.origin is not None:
                source_document = chunk.meta.origin.filename

            record = {
                "schema_version": 1,
                "chunk_id": f"{input_path.stem}-chunk-{index:04d}",
                "chunk_index": index,
                "company": company,
                "fiscal_year": fiscal_year,
                "document_type": document_type,
                "source_document": source_document,
                "page_numbers": page_numbers,
                "headings": headings,
                "document_references": document_references,
                "text": chunk.text,
                "embedding_text": embedding_text,
                "token_count": tokenizer.count_tokens(embedding_text),
            }

            output_file.write(
                json.dumps(record, ensure_ascii=False) + "\n"
            )

            chunk_count += 1

    return chunk_count


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Create page-aware chunks from Docling JSON."
    )
    parser.add_argument(
        "input_path",
        type=Path,
        help="Path to the structured Docling JSON document.",
    )
    parser.add_argument(
        "--output-path",
        type=Path,
        required=True,
        help="Destination JSONL file.",
    )
    parser.add_argument(
        "--company",
        required=True,
        help="Company represented by the report.",
    )
    parser.add_argument(
        "--fiscal-year",
        type=int,
        required=True,
        help="Fiscal year represented by the report.",
    )
    parser.add_argument(
        "--document-type",
        default="annual-report",
        help="Document category, such as annual-report or 10-q.",
    )

    args = parser.parse_args()

    chunk_count = create_chunks(
        input_path=args.input_path,
        output_path=args.output_path,
        company=args.company,
        fiscal_year=args.fiscal_year,
        document_type=args.document_type,
    )

    print("Chunking completed.")
    print(f"Created {chunk_count} chunks.")
    print(f"Output: {args.output_path}")


if __name__ == "__main__":
    main()