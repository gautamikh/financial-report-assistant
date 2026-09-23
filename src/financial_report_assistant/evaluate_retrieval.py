import argparse
import json
import re
from pathlib import Path

import faiss
import numpy as np
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer


TOKEN_PATTERN = re.compile(
    r"\$?\d[\d,]*(?:\.\d+)?%?|[A-Za-z]+(?:[-'][A-Za-z]+)*"
)

RRF_K = 60


def tokenize(text: str) -> list[str]:
    """Tokenize text for BM25 lexical search."""
    return TOKEN_PATTERN.findall(text.lower())


def load_jsonl(path: Path) -> list[dict]:
    records = []

    with path.open("r", encoding="utf-8") as file:
        for line in file:
            if line.strip():
                records.append(json.loads(line))

    return records


def load_questions(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as file:
        questions = json.load(file)

    return [
        question
        for question in questions
        if question.get("status") == "verified"
        and question.get("answerable", True)
        and question.get("relevant_chunk_ids")
    ]


def vector_ranking(
    query: str,
    model: SentenceTransformer,
    index: faiss.Index,
    candidate_k: int,
) -> list[int]:
    query_embedding = model.encode_query(
        [query],
        convert_to_numpy=True,
        normalize_embeddings=True,
    ).astype("float32")

    number_of_candidates = min(candidate_k, index.ntotal)
    _, indices = index.search(query_embedding, number_of_candidates)

    return [int(index_value) for index_value in indices[0] if index_value >= 0]


def bm25_ranking(
    query: str,
    bm25: BM25Okapi,
    document_count: int,
    candidate_k: int,
) -> list[int]:
    scores = bm25.get_scores(tokenize(query))
    number_of_candidates = min(candidate_k, document_count)

    ranked_indices = np.argsort(-scores)[:number_of_candidates]
    return [int(index_value) for index_value in ranked_indices]


def reciprocal_rank_fusion(
    vector_indices: list[int],
    bm25_indices: list[int],
) -> list[int]:
    fused_scores: dict[int, float] = {}

    for ranking in (vector_indices, bm25_indices):
        for rank, index_value in enumerate(ranking, start=1):
            fused_scores[index_value] = (
                fused_scores.get(index_value, 0.0)
                + 1.0 / (RRF_K + rank)
            )

    return sorted(
        fused_scores,
        key=fused_scores.get,
        reverse=True,
    )


def calculate_metrics(
    retrieved_chunk_ids: list[str],
    relevant_chunk_ids: set[str],
) -> dict[str, float]:
    relevant_ranks = [
        rank
        for rank, chunk_id in enumerate(retrieved_chunk_ids, start=1)
        if chunk_id in relevant_chunk_ids
    ]

    hit = 1.0 if relevant_ranks else 0.0
    recall = len(set(retrieved_chunk_ids) & relevant_chunk_ids) / len(
        relevant_chunk_ids
    )
    reciprocal_rank = 1.0 / relevant_ranks[0] if relevant_ranks else 0.0

    return {
        "hit": hit,
        "recall": recall,
        "reciprocal_rank": reciprocal_rank,
    }


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate vector, BM25, and hybrid retrieval."
    )

    parser.add_argument("--index-dir", type=Path, required=True)
    parser.add_argument("--questions", type=Path, required=True)
    parser.add_argument(
        "--model-name",
        default="sentence-transformers/all-MiniLM-L6-v2",
    )
    parser.add_argument("--candidate-k", type=int, default=20)
    parser.add_argument("--top-k", type=int, default=5)

    return parser.parse_args()


def main() -> None:
    args = parse_arguments()

    index_path = args.index_dir / "index.faiss"
    metadata_path = args.index_dir / "metadata.jsonl"

    index = faiss.read_index(str(index_path))
    metadata = load_jsonl(metadata_path)
    questions = load_questions(args.questions)

    if index.ntotal != len(metadata):
        raise ValueError(
            "FAISS index size does not match the number of metadata records."
        )

    if not questions:
        raise ValueError(
            "No verified answerable questions with relevant chunks were found."
        )

    model = SentenceTransformer(args.model_name)

    tokenized_corpus = [
        tokenize(record.get("embedding_text") or record["text"])
        for record in metadata
    ]
    bm25 = BM25Okapi(tokenized_corpus)

    method_totals = {
        "vector": {"hit": 0.0, "recall": 0.0, "reciprocal_rank": 0.0},
        "bm25": {"hit": 0.0, "recall": 0.0, "reciprocal_rank": 0.0},
        "hybrid": {"hit": 0.0, "recall": 0.0, "reciprocal_rank": 0.0},
    }

    print(f"\nEvaluating {len(questions)} question(s) at top-{args.top_k}\n")

    for question in questions:
        query = question["question"]
        relevant_ids = set(question["relevant_chunk_ids"])

        vector_indices = vector_ranking(
            query,
            model,
            index,
            args.candidate_k,
        )

        bm25_indices = bm25_ranking(
            query,
            bm25,
            len(metadata),
            args.candidate_k,
        )

        hybrid_indices = reciprocal_rank_fusion(
            vector_indices,
            bm25_indices,
        )

        rankings = {
            "vector": vector_indices[: args.top_k],
            "bm25": bm25_indices[: args.top_k],
            "hybrid": hybrid_indices[: args.top_k],
        }

        print(f"Question: {query}")

        for method, indices in rankings.items():
            retrieved_ids = [
                metadata[index_value]["chunk_id"]
                for index_value in indices
            ]

            metrics = calculate_metrics(retrieved_ids, relevant_ids)

            for metric_name, value in metrics.items():
                method_totals[method][metric_name] += value

            print(
                f"  {method:<7} "
                f"Hit@{args.top_k}: {metrics['hit']:.0f}  "
                f"Recall@{args.top_k}: {metrics['recall']:.2f}  "
                f"RR: {metrics['reciprocal_rank']:.2f}"
            )

        print()

    question_count = len(questions)

    print("=" * 65)
    print("FINAL RESULTS")
    print("=" * 65)

    for method, totals in method_totals.items():
        hit_rate = totals["hit"] / question_count
        mean_recall = totals["recall"] / question_count
        mean_reciprocal_rank = totals["reciprocal_rank"] / question_count

        print(f"\n{method.upper()}")
        print(f"  Hit Rate@{args.top_k}: {hit_rate:.3f}")
        print(f"  Mean Recall@{args.top_k}: {mean_recall:.3f}")
        print(f"  MRR: {mean_reciprocal_rank:.3f}")


if __name__ == "__main__":
    main()