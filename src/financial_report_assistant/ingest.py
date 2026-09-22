import argparse
import json
from pathlib import Path

from docling.document_converter import DocumentConverter


def convert_document(input_path: Path, output_dir: Path) -> tuple[Path, Path]:
    """Convert a PDF into Markdown and structured Docling JSON."""

    if not input_path.exists():
        raise FileNotFoundError(f"Document not found: {input_path}")

    if input_path.suffix.lower() != ".pdf":
        raise ValueError(f"Expected a PDF file, received: {input_path.suffix}")

    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Converting: {input_path}")

    converter = DocumentConverter()
    result = converter.convert(input_path)
    document = result.document

    markdown_path = output_dir / f"{input_path.stem}.md"
    json_path = output_dir / f"{input_path.stem}.json"

    markdown = document.export_to_markdown()
    structured_data = document.export_to_dict()

    markdown_path.write_text(markdown, encoding="utf-8")
    json_path.write_text(
        json.dumps(structured_data, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    return markdown_path, json_path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert a financial-report PDF using Docling."
    )
    parser.add_argument(
        "input_path",
        type=Path,
        help="Path to the PDF that should be converted.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("output"),
        help="Directory for generated Markdown and JSON files.",
    )

    args = parser.parse_args()

    markdown_path, json_path = convert_document(
        input_path=args.input_path,
        output_dir=args.output_dir,
    )

    print("Conversion completed.")
    print(f"Markdown: {markdown_path}")
    print(f"JSON:     {json_path}")


if __name__ == "__main__":
    main()