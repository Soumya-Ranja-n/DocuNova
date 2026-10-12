"""Run the focused factual, descriptive, analytical, comparative, multi-part,
cross-document, and unsupported-question acceptance ledger."""

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


ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--indices", help="Comma-separated zero-based question indexes.")
    args = parser.parse_args()
    config = get_config()
    metadata = MetadataStore(config.metadata_db_path)
    collection = next(item for item in metadata.list_collections()
                      if item.name == "test_doc content acceptance")
    documents = {item.filename: item for item in metadata.list_documents(collection.id)}
    qdrant = QdrantStore(config.qdrant_url, config.qdrant_collection, api_key=config.qdrant_api_key)
    retriever = RerankedRetriever(
        HybridRetriever(metadata, NvidiaEmbeddingClient(config), qdrant, mode=config.retrieval_mode),
        FallbackReranker(NvidiaRerankerClient(config)),
    )
    answerer = GroundedAnswerService(retriever, NvidiaReasoningClient(config))
    questions = [
        (["42_The Cloud Computing Book The Future of Computing Explained.pdf"],
         "What two factors contributed to the start of the cloud paradigm?"),
        (["Introduction to Machine Learning with Python ( PDFDrive.com )-min.pdf"],
         "What is supervised learning used for, and what makes up the training set?"),
        (["Housing.csv"], "What is the first column name shown in the Housing.csv header?"),
        (["tmdb_5000_credits.csv"], "What is the title of the movie with movie_id 285?"),
        (["cat.jpg"], "What animal is visibly present in cat.jpg?"),
        (["cat1.jpg"], "What animal is visibly present in cat1.jpg?"),
        (["dog.jpg"], "What animal is visibly present in dog.jpg?"),
        (["dog1.jpg"], "What animal is visibly present in dog1.jpg?"),
        (["text_image.jpg"], "Does text_image.jpg show a woman's face?"),
        (["text_image1.jpg"], "What exact headline is visible in text_image1.jpg?"),
        (["42_The Cloud Computing Book The Future of Computing Explained.pdf",
          "Introduction to Machine Learning with Python ( PDFDrive.com )-min.pdf"],
         "Which selected PDF discusses a move to parallelism, and which selected PDF discusses supervised learning?"),
        (["42_The Cloud Computing Book The Future of Computing Explained.pdf",
          "Introduction to Machine Learning with Python ( PDFDrive.com )-min.pdf"],
         "How do the two selected books differ in subject matter?"),
        (["Housing.csv", "tmdb_5000_credits.csv"],
         "Which selected filename contains real-estate fields such as price and bedrooms, and which selected filename contains movie titles and cast members?"),
        (["cat.jpg", "cat1.jpg", "dog.jpg", "dog1.jpg"],
         "Which selected image files show cats and which show dogs?"),
        (["text_image.jpg", "text_image1.jpg"],
         "Which selected filename contains a woman's face with a photography quote, and which selected filename shows the headline BEST TOOLS TO ENHANCE TEXT IMAGE?"),
        (["Housing.csv", "tmdb_5000_credits.csv"],
         "What is the title of the movie with movie_id 285, and what is the first column name in Housing.csv?"),
        (["42_The Cloud Computing Book The Future of Computing Explained.pdf"],
         "What is the exact publication year of this book?"),
        (["42_The Cloud Computing Book The Future of Computing Explained.pdf"],
         "What was the author's favorite color?"),
    ]
    selected = questions
    if args.indices:
        selected = [questions[int(index)] for index in args.indices.split(",")]
    for filenames, question in selected:
        result = {"files": filenames, "question": question}
        try:
            answer = answerer.answer(
                collection.id, question, [documents[name].id for name in filenames], top_k=6,
            )
            result.update({
                "answerability": answer.answerability.value,
                "verified": answer.verified,
                "citations": list(answer.citations),
                "evidence_files": sorted({item.evidence.filename for item in answer.retrieved_evidence}),
                "answer": answer.answer,
                "issues": list(answer.verification_issues),
            })
        except Exception as error:
            result.update({"error": type(error).__name__, "message": str(error)})
        print(json.dumps(result, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
