"""Streamlit entry point for the Phase 0 and Phase 1 application shell."""

import streamlit as st

from core.bootstrap import ensure_storage_directories
from core.collections.service import CollectionNotFoundError, CollectionService
from core.config import ConfigurationError, get_config
from core.agents.calculation_agent import CalculationOperation, CalculationRequest, calculate
from core.comparison.service import ComparisonError, DocumentComparator
from core.evaluation.evaluator import evaluate_run
from core.guardrails.contradictions import detect_contradictions
from core.agents.router import AgentRouter
from core.ingestion.loader import DocumentLoader
from core.ingestion.service import DocumentNotFoundError, IngestionService
from core.llm.nvidia_embeddings import EmbeddingServiceError, NvidiaEmbeddingClient
from core.llm.nvidia_reranker import NvidiaRerankerClient
from core.llm.nvidia_reasoning import NvidiaReasoningClient, ReasoningServiceError
from core.llm.nvidia_vision import NvidiaVisionClient
from core.retrieval.answering import GroundedAnswerService
from core.retrieval.embeddings import EvidenceIndexer
from core.retrieval.hybrid import HybridRetriever
from core.retrieval.query import ScopedRetriever
from core.retrieval.reranking import FallbackReranker, LexicalReranker, RerankedRetriever
from core.storage.document_store import DocumentStore
from core.storage.metadata_store import MetadataStore
from core.storage.qdrant_store import QdrantStore, QdrantStoreError


def _collection_service() -> CollectionService:
    config = get_config()
    return CollectionService(
        metadata_store=MetadataStore(config.metadata_db_path),
        document_store=DocumentStore(config.document_store_path),
        max_upload_size_bytes=config.max_upload_size_bytes,
    )


def _ingestion_service(service: CollectionService) -> IngestionService:
    return IngestionService(
        metadata_store=service.metadata_store,
        loader=DocumentLoader(service.document_store),
        vision_client=NvidiaVisionClient(get_config()) if get_config().nvidia_configured else None,
    )


def _evidence_indexer(service: CollectionService) -> EvidenceIndexer:
    config = get_config()
    return EvidenceIndexer(
        metadata_store=service.metadata_store,
        embedding_client=NvidiaEmbeddingClient(config),
        qdrant_store=QdrantStore(
            config.qdrant_url,
            config.qdrant_collection,
            api_key=config.qdrant_api_key,
        ),
    )


def _answer_service(service: CollectionService) -> GroundedAnswerService:
    config = get_config()
    qdrant_store = QdrantStore(
        config.qdrant_url,
        config.qdrant_collection,
        api_key=config.qdrant_api_key,
    )
    base_retriever = HybridRetriever(
        metadata_store=service.metadata_store,
        embedding_client=NvidiaEmbeddingClient(config),
        qdrant_store=qdrant_store,
        mode=config.retrieval_mode,
    )
    reranker = (
        FallbackReranker(NvidiaRerankerClient(config))
        if config.reranker_model and config.nvidia_reranker_configured
        else LexicalReranker()
    )
    return GroundedAnswerService(
        retriever=RerankedRetriever(base_retriever, reranker),
        reasoning_client=NvidiaReasoningClient(config),
    )


def main() -> None:
    """Render the setup screen without starting model-dependent work."""

    st.set_page_config(page_title="DocuNova", page_icon="📚", layout="wide")

    config = get_config()
    ensure_storage_directories(config)

    st.title("DocuNova")
    st.caption("Evidence-first multimodal document intelligence")

    st.subheader("DocuNova workspace")
    st.write(
        "Upload mixed-format documents, process them into evidence, and inspect "
        "scope-filtered retrieval, verification, contradictions, calculations, "
        "and comparisons."
    )

    col_one, col_two, col_three = st.columns(3)
    with col_one:
        st.metric("NVIDIA API key", "Configured" if config.nvidia_configured else "Missing")
    with col_two:
        st.metric("Vector store", "Qdrant configured")
    with col_three:
        st.metric("Storage", "Ready")

    if not config.nvidia_configured:
        st.warning(
            "NVIDIA_API_KEY is not configured. Model-dependent operations will "
            "remain unavailable until the operator configures it in .env."
        )

    st.subheader("Document collections")
    service = _collection_service()
    collections = service.list_collections()
    collection_options = {"Create a new collection": None}
    collection_options.update(
        {
            f"{collection.name} ({collection.id[:8]})": collection.id
            for collection in collections
        }
    )
    selected_name = st.selectbox("Collection", list(collection_options))

    new_name = ""
    if collection_options[selected_name] is None:
        new_name = st.text_input("New collection name")
    uploaded_files = st.file_uploader(
        "Upload documents",
        type=["pdf", "docx", "txt", "md", "csv", "png", "jpg", "jpeg"],
        accept_multiple_files=True,
        help="Uploads are validated and stored as untrusted source data.",
    )

    if st.button("Upload documents", type="primary"):
        try:
            collection_id = collection_options[selected_name]
            if collection_id is None:
                if not new_name.strip():
                    st.error("Enter a collection name before uploading.")
                    return
                collection_id = service.create_collection(new_name).id
            if not uploaded_files:
                st.error("Select at least one document before uploading.")
                return

            documents = service.upload_documents(
                collection_id,
                [(uploaded.name, uploaded.getvalue()) for uploaded in uploaded_files],
            )
            with st.status("Processing and indexing in progress…", expanded=True) as status:
                processor = _ingestion_service(service)
                for document in documents:
                    st.write(f"Processing {document.filename}…")
                    processor.process_document(collection_id, document.id)
                st.write("Indexing evidence in Qdrant…")
                result = _evidence_indexer(service).index_collection(collection_id)
                status.update(
                    label="Processing and indexing complete",
                    state="complete",
                    expanded=False,
                )
            st.success(
                f"Uploaded, processed, and indexed {len(documents)} document(s); "
                f"{result.indexed_count} evidence item(s) searchable."
            )
            st.dataframe(
                [
                    {
                        "filename": document.filename,
                        "type": document.file_type,
                        "size": document.size,
                        "status": document.status,
                    }
                    for document in documents
                ],
                width="stretch",
            )
        except CollectionNotFoundError as error:
            st.error(str(error))
        except (
            ConfigurationError,
            EmbeddingServiceError,
            QdrantStoreError,
            ValueError,
            OSError,
        ) as error:
            st.error(str(error))

    selected_collection_id = collection_options[selected_name]
    if selected_collection_id is not None:
        documents = service.list_documents(selected_collection_id)
        if documents:
            st.subheader("Process uploaded documents")
            document_options = {
                f"{document.filename} ({document.id[:8]})": document.id
                for document in documents
            }
            selected_documents = st.multiselect(
                "Documents to process",
                list(document_options),
            )
            if st.button("Process selected documents"):
                processor = _ingestion_service(service)
                processed = 0
                for label in selected_documents:
                    try:
                        processor.process_document(
                            selected_collection_id,
                            document_options[label],
                        )
                        processed += 1
                    except (DocumentNotFoundError, ValueError, OSError) as error:
                        st.error(str(error))
                if processed:
                    st.success(f"Processed {processed} document(s) into evidence.")

            st.subheader("Vector indexing")
            st.caption(
                "Index processed textual evidence in Qdrant using the configured "
                "NVIDIA embedding model."
            )
            if st.button("Index collection in Qdrant"):
                try:
                    result = _evidence_indexer(service).index_collection(
                        selected_collection_id
                    )
                    st.success(
                        f"Indexed {result.indexed_count} evidence item(s); "
                        f"skipped {result.skipped_count} non-text item(s)."
                    )
                except (ConfigurationError, EmbeddingServiceError, QdrantStoreError) as error:
                    st.error(str(error))
                    st.caption(
                        "If this is an NVIDIA error, verify the rotated key, embedding model, "
                        "and restart Streamlit. If this is a Qdrant error, start Docker Desktop "
                        "and the qdrant service before retrying."
                    )

            st.subheader("Grounded query")
            question = st.text_area(
                "Question",
                placeholder="Ask a question about the processed evidence...",
            )
            query_document_labels = st.multiselect(
                "Limit query to selected documents (optional)",
                list(document_options),
            )
            if st.button("Ask grounded question"):
                try:
                    selected_ids = [
                        document_options[label] for label in query_document_labels
                    ] or None
                    result = _answer_service(service).answer(
                        collection_id=selected_collection_id,
                        question=question,
                        selected_document_ids=selected_ids,
                        top_k=8,
                    )
                    route = AgentRouter().route(question)
                    st.markdown(result.answer)
                    st.caption(
                        f"Answerability: {result.answerability.value} | "
                        f"Verified: {result.verified} | "
                        f"Tools: {', '.join(tool.value for tool in route.tools)}"
                    )
                    if result.retrieved_evidence:
                        st.dataframe(
                            [
                                {
                                    "evidence_id": item.evidence.evidence_id,
                                    "filename": item.evidence.filename,
                                    "page": item.evidence.page_number,
                                    "type": item.evidence.content_type,
                                    "retrieval_score": item.retrieval_score,
                                    "vector_score": item.vector_score,
                                    "keyword_score": item.keyword_score,
                                    "fusion_score": item.fusion_score,
                                    "rerank_score": item.rerank_score,
                                    "source": item.evidence.source_reference,
                                }
                                for item in result.retrieved_evidence
                            ],
                            width="stretch",
                        )
                except (
                    ConfigurationError,
                    EmbeddingServiceError,
                    ReasoningServiceError,
                    QdrantStoreError,
                    ValueError,
                ) as error:
                    st.error(str(error))

            st.subheader("Deterministic calculation")
            operation = st.selectbox(
                "Operation",
                [operation.value for operation in CalculationOperation],
            )
            raw_values = st.text_input("Values (comma-separated)", placeholder="100, 125")
            if st.button("Calculate"):
                try:
                    values = [value.strip() for value in raw_values.split(",") if value.strip()]
                    calculation = calculate(
                        CalculationRequest.from_numbers(
                            CalculationOperation(operation),
                            values,
                        )
                    )
                    st.success(f"Result: {calculation.result}")
                    st.caption(calculation.explanation)
                except ValueError as error:
                    st.error(str(error))

            if len(documents) >= 2:
                st.subheader("Document comparison")
                comparison_options = list(document_options)
                before_label = st.selectbox("Before document", comparison_options, key="before_document")
                after_label = st.selectbox("After document", comparison_options, index=1, key="after_document")
                if st.button("Compare documents"):
                    try:
                        comparison = DocumentComparator(service.metadata_store).compare(
                            selected_collection_id,
                            document_options[before_label],
                            document_options[after_label],
                        )
                        st.dataframe(
                            [
                                {
                                    "key": difference.key,
                                    "change": difference.change_type,
                                    "before": difference.before_text,
                                    "after": difference.after_text,
                                    "before_evidence": ", ".join(difference.before_evidence_ids),
                                    "after_evidence": ", ".join(difference.after_evidence_ids),
                                }
                                for difference in comparison.differences
                            ],
                            width="stretch",
                        )
                    except ComparisonError as error:
                        st.error(str(error))

            st.subheader("Contradictions")
            contradictions = detect_contradictions(
                service.metadata_store.list_evidence(selected_collection_id)
            )
            if contradictions:
                st.warning(f"Detected {len(contradictions)} comparable conflict/consistency group(s).")
                st.json(
                    [
                        {
                            "subject": item.subject,
                            "status": item.status.value,
                            "explanation": item.explanation,
                            "evidence_ids": [value.evidence_id for value in item.values],
                        }
                        for item in contradictions
                    ]
                )
            else:
                st.caption("No comparable numeric contradiction was detected in this collection.")

            st.subheader("Evaluation runs")
            evaluations = service.metadata_store.list_evaluations(selected_collection_id)
            if evaluations:
                st.dataframe(evaluations, use_container_width=True)
            else:
                st.caption("No measured evaluation run has been recorded yet.")

    st.info(
        "Guardrails are part of the core design: evidence is required, document "
        "content is untrusted, retrieval will be scope-filtered, and unsupported "
        "questions must be answered with abstention."
    )


if __name__ == "__main__":
    main()
