"""Run the real test_doc content through ingestion, indexing, and grounded QA."""

from __future__ import annotations

import json
from pathlib import Path

from core.collections.service import CollectionService
from core.config import get_config
from core.ingestion.loader import DocumentLoader
from core.ingestion.service import IngestionService
from core.llm.nvidia_embeddings import NvidiaEmbeddingClient
from core.llm.nvidia_reranker import NvidiaRerankerClient
from core.llm.nvidia_reasoning import NvidiaReasoningClient
from core.llm.nvidia_vision import NvidiaVisionClient
from core.retrieval.answering import GroundedAnswerService
from core.retrieval.embeddings import EvidenceIndexer
from core.retrieval.hybrid import HybridRetriever
from core.retrieval.reranking import FallbackReranker, RerankedRetriever
from core.storage.document_store import DocumentStore
from core.storage.metadata_store import MetadataStore
from core.storage.qdrant_store import QdrantStore


ROOT = Path(__file__).resolve().parents[1]
DEMO_FILES = sorted((ROOT / "test_doc").iterdir())


def main() -> None:
    config = get_config()
    metadata = MetadataStore(config.metadata_db_path)
    documents = DocumentStore(config.document_store_path)
    collections = CollectionService(metadata, documents, config.max_upload_size_bytes)
    collection = collections.create_collection("test_doc content acceptance")
    uploads = [(path.name, path.read_bytes()) for path in DEMO_FILES if path.is_file()]
    stored = collections.upload_documents(collection.id, uploads)
    by_name = {document.filename: document for document in stored}

    vision = NvidiaVisionClient(config) if config.nvidia_configured else None
    processor = IngestionService(metadata, DocumentLoader(documents), vision_client=vision)
    processing = []
    for document in stored:
        result = processor.process_document(collection.id, document.id)
        processing.append(
            {
                "filename": document.filename,
                "evidence_count": result.evidence_count,
                "content_types": result.content_types,
            }
        )

    qdrant = QdrantStore(
        config.qdrant_url,
        config.qdrant_collection,
        api_key=config.qdrant_api_key,
    )
    indexer = EvidenceIndexer(metadata, NvidiaEmbeddingClient(config), qdrant)
    indexed = indexer.index_collection(collection.id)
    base = HybridRetriever(
        metadata,
        NvidiaEmbeddingClient(config),
        qdrant,
        mode=config.retrieval_mode,
    )
    reranker = FallbackReranker(NvidiaRerankerClient(config))
    answerer = GroundedAnswerService(
        RerankedRetriever(base, reranker), NvidiaReasoningClient(config)
    )

    individual = [
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
    multi = [
        (("42_The Cloud Computing Book The Future of Computing Explained.pdf", "Introduction to Machine Learning with Python ( PDFDrive.com )-min.pdf"), "How does the cloud book explain the motivations for cloud computing, and how does the machine-learning book define supervised learning?"),
        (("Housing.csv", "tmdb_5000_credits.csv"), "Which two different kinds of datasets are represented by Housing.csv and tmdb_5000_credits.csv?"),
        (("cat.jpg", "cat1.jpg", "dog.jpg", "dog1.jpg"), "Which selected image files show cats and which show dogs?"),
        (("text_image.jpg", "text_image1.jpg"), "What visual text-related content is shown across the two selected text images?"),
    ]

    questions: list[dict[str, object]] = []
    for filename, question in individual:
        questions.append(_ask(answerer, collection.id, question, [by_name[filename].id], [filename]))
    for filenames, question in multi:
        questions.append(
            _ask(answerer, collection.id, question, [by_name[name].id for name in filenames], list(filenames))
        )

    print(json.dumps({"processing": processing, "indexed": indexed.indexed_count, "questions": questions}, ensure_ascii=False))


def _ask(answerer, collection_id: str, question: str, document_ids: list[str], filenames: list[str]) -> dict[str, object]:
    result: dict[str, object] = {"files": filenames, "question": question}
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
