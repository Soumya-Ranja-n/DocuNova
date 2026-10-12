"""Run the acceptance question ledger against an already indexed demo collection."""

from __future__ import annotations

import json
import argparse
from pathlib import Path

from core.config import get_config
from core.llm.nvidia_embeddings import NvidiaEmbeddingClient
from core.llm.nvidia_reranker import NvidiaRerankerClient
from core.llm.nvidia_reasoning import NvidiaReasoningClient
from core.retrieval.answering import GroundedAnswerService
from core.retrieval.hybrid import HybridRetriever
from core.retrieval.reranking import FallbackReranker, RerankedRetriever
from core.storage.metadata_store import MetadataStore
from core.storage.qdrant_store import QdrantStore


INDIVIDUAL = [
    ("42_The Cloud Computing Book The Future of Computing Explained.pdf", "What two intertwined factors contributed to the start of the cloud paradigm?"),
    ("Introduction to Machine Learning with Python ( PDFDrive.com )-min.pdf", "What is supervised learning used for, and what makes up the training set?"),
    ("Housing.csv", "What columns are present in the Housing CSV table?"),
    ("tmdb_5000_credits.csv", "Which movie has movie_id 285, and which cast members are listed first?"),
    ("cat.jpg", "What animal is visibly present in cat.jpg?"),
    ("cat1.jpg", "What animal is visibly present in cat1.jpg?"),
    ("dog.jpg", "What animal is visibly present in dog.jpg?"),
    ("dog1.jpg", "What animal is visibly present in dog1.jpg?"),
    ("text_image.jpg", "What visible subject and text overlay are present in text_image.jpg?"),
    ("text_image1.jpg", "What exact headline is visible in text_image1.jpg?"),
]
MULTI = [
    (("42_The Cloud Computing Book The Future of Computing Explained.pdf", "Introduction to Machine Learning with Python ( PDFDrive.com )-min.pdf"), "How does the cloud book explain the motivations for cloud computing, and how does the machine-learning book define supervised learning?"),
    (("Housing.csv", "tmdb_5000_credits.csv"), "Which two different kinds of datasets are represented by Housing.csv and tmdb_5000_credits.csv?"),
    (("cat.jpg", "cat1.jpg", "dog.jpg", "dog1.jpg"), "Which selected image files show cats and which show dogs?"),
    (("text_image.jpg", "text_image1.jpg"), "What visual text-related content is shown across the two selected text images?"),
]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--group", choices=("individual", "multi", "all"), default="all")
    parser.add_argument("--files", help="Comma-separated individual filenames to run.")
    parser.add_argument("--custom-question", help="Replace the selected individual question.")
    parser.add_argument("--custom-multi-question", help="Replace every selected multi-file question.")
    parser.add_argument("--multi-index", type=int, help="Run only one zero-based multi-file question.")
    parser.add_argument("--output", help="Optional JSONL path written after each question.")
    args = parser.parse_args()
    output = Path(args.output) if args.output else None
    if output:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.unlink(missing_ok=True)
    config = get_config()
    metadata = MetadataStore(config.metadata_db_path)
    collection = next(item for item in metadata.list_collections() if item.name == "test_doc content acceptance")
    documents = {item.filename: item for item in metadata.list_documents(collection.id)}
    qdrant = QdrantStore(config.qdrant_url, config.qdrant_collection, api_key=config.qdrant_api_key)
    retriever = RerankedRetriever(
        HybridRetriever(
            metadata,
            NvidiaEmbeddingClient(config),
            qdrant,
            mode=config.retrieval_mode,
        ),
        FallbackReranker(NvidiaRerankerClient(config)),
    )
    answerer = GroundedAnswerService(retriever, NvidiaReasoningClient(config))
    groups = []
    if args.group in {"individual", "all"}:
        selected = INDIVIDUAL
        if args.files:
            wanted = set(args.files.split(","))
            selected = [item for item in INDIVIDUAL if item[0] in wanted]
        groups.append(selected)
    if args.group in {"multi", "all"}:
        groups.append([MULTI[args.multi_index]] if args.multi_index is not None else MULTI)
    for group in groups:
        for item in group:
            if len(item) == 2 and isinstance(item[0], str):
                filename, question = item
                if args.custom_question:
                    question = args.custom_question
                result = _ask(answerer, collection.id, question, [documents[filename].id], [filename])
            else:
                filenames, question = item
                if args.custom_multi_question:
                    question = args.custom_multi_question
                result = _ask(answerer, collection.id, question, [documents[name].id for name in filenames], list(filenames))
            print(json.dumps(result, ensure_ascii=False), flush=True)
            if output:
                with output.open("a", encoding="utf-8") as handle:
                    handle.write(json.dumps(result, ensure_ascii=False) + "\n")


def _ask(answerer, collection_id, question, document_ids, filenames):
    result = {"files": filenames, "question": question}
    try:
        answer = answerer.answer(collection_id, question, document_ids, top_k=6)
    except Exception as error:
        result.update({"error": type(error).__name__, "message": str(error)})
        return result
    result.update(
        {
            "answerability": answer.answerability.value,
            "verified": answer.verified,
            "citations": list(answer.citations),
            "evidence_files": sorted({item.evidence.filename for item in answer.retrieved_evidence}),
            "answer": answer.answer,
        }
    )
    return result


if __name__ == "__main__":
    main()
