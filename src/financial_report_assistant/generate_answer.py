
import argparse
import json
import re
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from hybrid_search import hybrid_search


SYSTEM_PROMPT = """
You are an evidence-grounded financial-report analyst.

Rules:
1. Answer using only the supplied report passages.
2. Do not use outside knowledge.
3. Cite every factual claim using source labels such as [S1] or [S2].
4. Use only source labels that appear in the supplied context.
5. Preserve financial units, currencies, years, and negative values exactly.
6. Clearly distinguish millions from billions.
7. Do not make forecasts or unsupported calculations.
8. If the passages do not contain enough evidence, say:
   "I cannot answer this question from the retrieved report evidence."
9. Keep the answer concise and directly address the question.
""".strip()

CITATION_PATTERN = re.compile(r"\[(S\d+)\]")

INSUFFICIENT_EVIDENCE_MESSAGE = (
    "I cannot answer this question from the retrieved report evidence."
)

def format_context(
    results: list[dict[str, Any]],
) -> tuple[str, list[dict[str, str]]]:
    """Format retrieved passages and create citation metadata."""

    context_sections: list[str] = []
    citations: list[dict[str, str]] = []

    for source_number, result in enumerate(results, start=1):
        source_label = f"S{source_number}"
        chunk = result["chunk"]

        page_numbers = chunk.get("page_numbers", [])
        pages = ", ".join(str(page) for page in page_numbers)
        pages = pages or "unknown"

        headings = chunk.get("headings", [])
        section = " > ".join(headings)
        section = section or "Unlabelled section"

        chunk_id = chunk.get("chunk_id", "unknown")
        text = chunk.get("text", "").strip()

        context_sections.append(
            f"[{source_label}]\n"
            f"Chunk ID: {chunk_id}\n"
            f"Pages: {pages}\n"
            f"Section: {section}\n"
            f"Passage:\n{text}"
        )

        citations.append(
            {
                "label": source_label,
                "chunk_id": chunk_id,
                "pages": pages,
                "section": section,
            }
        )

    return "\n\n".join(context_sections), citations


def call_ollama(
    base_url: str,
    model: str,
    question: str,
    context: str,
) -> str:
    """Generate an answer using Ollama's local chat API."""

    user_prompt = f"""
Question:
{question}

Retrieved report evidence:
{context}

Write an evidence-grounded answer. Cite supporting sources using [S1],
[S2], and so on. If the evidence is insufficient, use the required
insufficient-evidence statement.
""".strip()

    payload = {
        "model": model,
        "stream": False,
        "messages": [
            {
                "role": "system",
                "content": SYSTEM_PROMPT,
            },
            {
                "role": "user",
                "content": user_prompt,
            },
        ],
        "options": {
            "temperature": 0.1,
        },
    }

    endpoint = f"{base_url.rstrip('/')}/api/chat"

    request = Request(
        endpoint,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    try:
        with urlopen(request, timeout=300) as response:
            response_data = json.loads(
                response.read().decode("utf-8")
            )
    except HTTPError as error:
        error_body = error.read().decode("utf-8", errors="replace")
        raise RuntimeError(
            f"Ollama returned HTTP {error.code}: {error_body}"
        ) from error
    except URLError as error:
        raise RuntimeError(
            "Could not connect to Ollama. Make sure it is running at "
            f"{base_url}."
        ) from error

    try:
        return response_data["message"]["content"].strip()
    except KeyError as error:
        raise RuntimeError(
            f"Unexpected Ollama response: {response_data}"
        ) from error

def validate_answer(
    answer: str,
    citations: list[dict[str, str]],
) -> None:
    """Ensure the answer only uses valid retrieved-source citations."""

    allowed_labels = {
        citation["label"]
        for citation in citations
    }

    used_labels = set(CITATION_PATTERN.findall(answer))
    invalid_labels = used_labels - allowed_labels

    if invalid_labels:
        invalid_text = ", ".join(sorted(invalid_labels))
        raise ValueError(
            f"Answer contains invalid citations: {invalid_text}"
        )

    is_insufficient_evidence = (
        INSUFFICIENT_EVIDENCE_MESSAGE.lower() in answer.lower()
    )

    if not is_insufficient_evidence and not used_labels:
        raise ValueError(
            "The generated answer contains no source citations."
        )

def display_answer(
    question: str,
    answer: str,
    citations: list[dict[str, str]],
) -> None:
    """Display the answer and its source mapping."""

    print()
    print("=" * 80)
    print("QUESTION")
    print("=" * 80)
    print(question)

    print()
    print("=" * 80)
    print("ANSWER")
    print("=" * 80)
    print(answer)

    print()
    print("=" * 80)
    print("RETRIEVED SOURCES")
    print("=" * 80)

    for citation in citations:
        print(
            f"[{citation['label']}] "
            f"Pages: {citation['pages']} | "
            f"Section: {citation['section']} | "
            f"Chunk: {citation['chunk_id']}"
        )


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Answer a financial-report question using retrieved "
            "evidence and a local Ollama model."
        )
    )

    parser.add_argument(
        "question",
        help="Question to answer from the financial report.",
    )
    parser.add_argument(
        "--index-dir",
        type=Path,
        required=True,
        help="Directory containing the FAISS index and metadata.",
    )
    parser.add_argument(
        "--model",
        required=True,
        help="Exact Ollama model name shown by 'ollama list'.",
    )
    parser.add_argument(
        "--ollama-url",
        default="http://localhost:11434",
        help="Base URL of the Ollama server.",
    )
    parser.add_argument(
        "--candidate-k",
        type=int,
        default=20,
        help="Candidates retrieved by vector search and BM25.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=5,
        help="Number of passages supplied to the language model.",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_arguments()

    print("Retrieving report evidence...")

    results = hybrid_search(
        index_dir=args.index_dir,
        query=args.question,
        top_k=args.top_k,
        candidate_k=args.candidate_k,
    )

    if not results:
        raise RuntimeError("No report evidence was retrieved.")

    context, citations = format_context(results)

    print(f"Generating answer with Ollama model: {args.model}")

    answer = call_ollama(
        base_url=args.ollama_url,
        model=args.model,
        question=args.question,
        context=context,
    )

    validate_answer(
    answer=answer,
    citations=citations,
    )   

    display_answer(
        question=args.question,
        answer=answer,
        citations=citations,
    )


if __name__ == "__main__":
    main()