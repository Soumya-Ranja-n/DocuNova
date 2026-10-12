"""Resume processing and indexing of the durable test_doc acceptance collection."""

from core.config import get_config
from core.ingestion.loader import DocumentLoader
from core.ingestion.service import IngestionService
from core.llm.nvidia_embeddings import NvidiaEmbeddingClient
from core.llm.nvidia_vision import NvidiaVisionClient
from core.retrieval.embeddings import EvidenceIndexer
from core.storage.document_store import DocumentStore
from core.storage.metadata_store import MetadataStore
from core.storage.qdrant_store import QdrantStore


def main() -> None:
    config = get_config()
    metadata = MetadataStore(config.metadata_db_path)
    collection = next(item for item in metadata.list_collections()
                      if item.name == "test_doc content acceptance")
    documents = DocumentStore(config.document_store_path)
    processor = IngestionService(
        metadata,
        DocumentLoader(documents),
        vision_client=NvidiaVisionClient(config) if config.nvidia_configured else None,
    )
    for document in metadata.list_documents(collection.id):
        evidence = metadata.list_evidence(collection.id, document.id)
        needs_visual_retry = (
            document.file_type in {"png", "jpg", "jpeg"}
            and not any(item.content_type == "VISION" for item in evidence)
        )
        if document.status != "processed" or needs_visual_retry:
            print(f"processing {document.filename}", flush=True)
            processor.process_document(collection.id, document.id)
    qdrant = QdrantStore(config.qdrant_url, config.qdrant_collection, api_key=config.qdrant_api_key)
    result = EvidenceIndexer(metadata, NvidiaEmbeddingClient(config), qdrant).index_collection(collection.id)
    print({"indexed": result.indexed_count, "vector_size": result.vector_size}, flush=True)


if __name__ == "__main__":
    main()
