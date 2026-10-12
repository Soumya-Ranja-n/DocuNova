"""FastAPI entry point for the Phase 0 through Phase 3 service layer."""

from dataclasses import asdict, replace
from contextlib import asynccontextmanager
from uuid import uuid4
import os
import time

from fastapi import BackgroundTasks, FastAPI, File, HTTPException, Query, Request, UploadFile
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware

from core.collections.service import (
    CollectionNotFoundError,
    CollectionService,
    DocumentNotFoundError as CollectionDocumentNotFoundError,
)
from core.bootstrap import ensure_storage_directories
from core.config import ConfigurationError, PROJECT_ROOT, get_config
from core.guardrails.scope import ScopeViolationError
from core.guardrails.contradictions import detect_contradictions
from core.agents.calculation_agent import (
    CalculationError,
    CalculationOperation,
    CalculationRequest,
    calculate,
)
from core.comparison.service import ComparisonError, DocumentComparator
from core.evaluation.evaluator import evaluate_run
from core.ingestion.loader import DocumentLoader
from core.ingestion.service import DocumentNotFoundError, IngestionService
from core.ingestion.workflow import IngestionWorkflow
from core.ingestion.ocr import ocr_available
from core.llm.nvidia_embeddings import EmbeddingServiceError, NvidiaEmbeddingClient
from core.llm.nvidia_reasoning import NvidiaReasoningClient, ReasoningServiceError
from core.llm.nvidia_vision import NvidiaVisionClient
from core.guardrails.pii import redact_pii
from core.retrieval.answering import GroundedAnswerService, QUERY_GUARDRAILS
from core.retrieval.embeddings import EvidenceIndexer, IndexingDocumentNotFoundError
from core.retrieval.hybrid import HybridRetriever
from core.retrieval.reranking import FallbackReranker, LexicalReranker, RerankedRetriever
from core.llm.nvidia_reranker import NvidiaRerankerClient
from core.learning.service import LearningError, LearningService
from core.learning.models import GenerateRequest, SubmitRequest, TeachAnswerRequest
from core.schemas import (
    CollectionCreateRequest,
    CollectionResponse,
    CalculationRequestSchema,
    CalculationResponse,
    ComparisonDifferenceResponse,
    ComparisonRequest,
    ComparisonResponse,
    DocumentResponse,
    EvidenceResponse,
    IndexResponse,
    ProcessResponse,
    QueryRequest,
    QueryResponse,
    RetrievedEvidenceResponse,
    EvaluationRequest,
    EvaluationResponse,
    UploadResponse,
    ProviderSettingsRequest,
)
from core.storage.document_store import DocumentStore
from core.storage.metadata_store import MetadataStore
from core.storage.qdrant_store import QdrantStore, QdrantStoreError
from core.observability import instrument_fastapi, log_event, log_span


config = get_config()
ensure_storage_directories(config)
qdrant_store = QdrantStore(
    config.qdrant_url,
    config.qdrant_collection,
    api_key=config.qdrant_api_key,
)
collection_service = CollectionService(
    metadata_store=MetadataStore(config.metadata_db_path),
    document_store=DocumentStore(config.document_store_path),
    max_upload_size_bytes=config.max_upload_size_bytes,
    qdrant_store=qdrant_store,
)
ingestion_service = IngestionService(
    metadata_store=collection_service.metadata_store,
    loader=DocumentLoader(collection_service.document_store),
    vision_client=NvidiaVisionClient(config) if config.nvidia_configured else None,
)
indexer = EvidenceIndexer(
    metadata_store=collection_service.metadata_store,
    embedding_client=NvidiaEmbeddingClient(config),
    qdrant_store=qdrant_store,
)
base_retriever = HybridRetriever(
    metadata_store=collection_service.metadata_store,
    embedding_client=NvidiaEmbeddingClient(config),
    qdrant_store=qdrant_store,
    mode=config.retrieval_mode,
)
reranker = (
    FallbackReranker(NvidiaRerankerClient(config))
    if config.reranker_model and config.nvidia_reranker_configured
    else LexicalReranker()
)
retriever = RerankedRetriever(base_retriever, reranker)
answer_service = GroundedAnswerService(
    retriever=retriever,
    reasoning_client=NvidiaReasoningClient(config),
)
learning_service = LearningService(
    metadata_store=collection_service.metadata_store,
    retriever=retriever,
    reasoning_client=NvidiaReasoningClient(config),
    config=config,
)
comparator = DocumentComparator(collection_service.metadata_store)
workflow = IngestionWorkflow(collection_service.metadata_store, ingestion_service, indexer)


@asynccontextmanager
async def lifespan(app):
    # A killed process cannot leave a collection frozen forever. The operator
    # can resume via /prepare; originals and extracted evidence are preserved.
    with collection_service.metadata_store._connect() as connection:
        connection.execute(
            "UPDATE ingestion_jobs SET status='error', detail='Preparation was interrupted. Retry preparation.' "
            "WHERE status IN ('uploading','queued','processing','indexing')"
        )
    log_event("project.started", service="docunova-api", status="running")
    yield
    log_event("project.stopped", service="docunova-api", status="stopped")

app = FastAPI(title="DocuNova", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        config.frontend_origin,
        "https://docu-nova-4ro9-6cjdno5or-soumya-ranjan-routs-projects-f780d4d2.vercel.app",
        "http://localhost:3000",
    ],
    allow_credentials=False,
    allow_methods=["GET", "POST", "PUT", "PATCH", "DELETE", "OPTIONS"],
    allow_headers=["Content-Type", "X-Request-ID", "Authorization"],
    expose_headers=["X-Request-ID"],
)


@app.middleware("http")
async def request_lifecycle_logging(request: Request, call_next):
    """Log every request lifecycle without recording bodies or credentials."""

    request_id = str(uuid4())
    started = time.perf_counter()
    log_event(
        "request.started",
        service="docunova-api",
        method=request.method,
        path=request.url.path,
        request_id=request_id,
    )
    try:
        with log_span("http.request", method=request.method, path=request.url.path, request_id=request_id):
            response = await call_next(request)
    except Exception as error:
        log_event(
            "request.failed",
            service="docunova-api",
            method=request.method,
            path=request.url.path,
            status="error",
            error_type=type(error).__name__,
        )
        raise
    log_event(
        "request.completed",
        service="docunova-api",
        method=request.method,
        path=request.url.path,
        status_code=response.status_code,
        status="ok" if response.status_code < 400 else "error",
        duration_ms=(time.perf_counter() - started) * 1000,
        request_id=request_id,
    )
    response.headers["X-Request-ID"] = request_id
    return response


async def _read_upload(upload: UploadFile) -> bytes:
    """Read one upload while enforcing the configured per-file limit."""

    chunks: list[bytes] = []
    total_size = 0
    chunk_size = 1024 * 1024
    filename = upload.filename or "unnamed file"

    while chunk := await upload.read(chunk_size):
        total_size += len(chunk)
        if total_size > config.max_upload_size_bytes:
            raise HTTPException(
                status_code=400,
                detail=(
                    f"'{filename}' exceeds the {config.max_upload_size_bytes} "
                    "byte upload limit."
                ),
            )
        chunks.append(chunk)

    return b"".join(chunks)


@app.get("/health", tags=["system"])
def health() -> dict[str, object]:
    """Return non-sensitive service status."""

    return {
        "status": "ok",
        "service": "docunova-api",
        "nvidia_configured": config.nvidia_configured,
        "providers": {
            "reranker": config.nvidia_reranker_configured,
            "ocr": ocr_available(config),
            "vision": config.nvidia_configured,
        },
        "qdrant_url": config.qdrant_url,
    }


@app.get("/settings/providers", tags=["configuration"])
def provider_settings() -> dict[str, object]:
    return {
        "reasoning": {"model": config.reasoning_model, "configured": config.nvidia_configured},
        "embedding": {"model": config.embedding_model, "configured": config.nvidia_configured},
        "reranker": {"model": config.reranker_model, "configured": config.nvidia_reranker_configured},
        "ocr": {"model": "tesseract-5", "configured": ocr_available(config)},
        "vision": {"model": config.vision_model, "configured": config.nvidia_configured},
    }


@app.post("/settings/providers", tags=["configuration"])
def save_provider_settings(payload: ProviderSettingsRequest) -> dict[str, object]:
    """Store local operator settings privately; responses contain presence only."""
    from dotenv import set_key
    global config
    if any(collection_service.metadata_store.ingestion_status(collection.id)["busy"]
           for collection in collection_service.list_collections()):
        raise HTTPException(409, "Wait for processing and indexing to finish before changing providers.")
    changes = {}
    for field in ("nvidia_api_key", "nvidia_reranker_api_key"):
        secret = getattr(payload, field)
        if secret is not None:
            value = secret.get_secret_value().strip()
            if "\n" in value or "\r" in value:
                raise HTTPException(400, "API keys must be single-line values.")
            set_key(str(PROJECT_ROOT / "backend" / ".env"), field.upper(), value)
            os.environ[field.upper()] = value
            changes[field] = value or None
    config = replace(config, **changes)
    get_config.cache_clear()
    indexer.embedding_client = NvidiaEmbeddingClient(config)
    base_retriever.embedding_client = NvidiaEmbeddingClient(config)
    retriever.reranker = (FallbackReranker(NvidiaRerankerClient(config))
                         if config.nvidia_reranker_configured else LexicalReranker())
    answer_service.reasoning_client = NvidiaReasoningClient(config)
    learning_service.reasoning_client = NvidiaReasoningClient(config)
    learning_service.config = config
    log_event("configuration.updated", service="provider_settings", status="updated",
              provider_fields=list(changes))
    return provider_settings()


@app.get("/query/guardrails", tags=["query"])
def query_guardrails() -> dict[str, object]:
    return {"guardrails": QUERY_GUARDRAILS}


@app.get("/collections/{collection_id}/ingestion", tags=["processing"])
def ingestion_status(collection_id: str):
    if collection_service.metadata_store.get_collection(collection_id) is None:
        raise HTTPException(404, "Collection was not found.")
    return collection_service.metadata_store.ingestion_status(collection_id)


@app.post("/collections/{collection_id}/prepare", status_code=202, tags=["processing"])
def prepare_collection(collection_id: str, background_tasks: BackgroundTasks):
    if collection_service.metadata_store.get_collection(collection_id) is None:
        raise HTTPException(404, "Collection was not found.")
    try:
        job = collection_service.metadata_store.create_ingestion_job(collection_id)
    except ValueError as error:
        raise HTTPException(409, str(error)) from error
    background_tasks.add_task(workflow.run, collection_id, job["id"])
    return job


@app.post("/collections", response_model=CollectionResponse, status_code=201, tags=["collections"])
def create_collection(payload: CollectionCreateRequest) -> CollectionResponse:
    try:
        collection = collection_service.create_collection(payload.name)
    except ValueError as error:
        raise HTTPException(status_code=400, detail=str(error)) from error
    return CollectionResponse.model_validate(asdict(collection))


@app.get("/collections", response_model=list[CollectionResponse], tags=["collections"])
def list_collections() -> list[CollectionResponse]:
    return [
        CollectionResponse.model_validate(asdict(collection))
        for collection in collection_service.list_collections()
    ]


@app.delete("/collections/{collection_id}", status_code=204, tags=["collections"])
def delete_collection(collection_id: str) -> None:
    try:
        collection_service.delete_collection(collection_id)
    except CollectionNotFoundError as error:
        raise HTTPException(status_code=404, detail=str(error)) from error
    except QdrantStoreError as error:
        raise HTTPException(status_code=502, detail=str(error)) from error



@app.get(
    "/collections/{collection_id}/documents",
    response_model=list[DocumentResponse],
    tags=["documents"],
)
def list_documents(collection_id: str) -> list[DocumentResponse]:
    try:
        documents = collection_service.list_documents(collection_id)
    except CollectionNotFoundError as error:
        raise HTTPException(status_code=404, detail=str(error)) from error
    return [DocumentResponse.model_validate(asdict(document)) for document in documents]


@app.delete(
    "/collections/{collection_id}/documents/{document_id}",
    status_code=204,
    tags=["documents"],
)
def delete_document(collection_id: str, document_id: str) -> None:
    try:
        collection_service.delete_document(collection_id, document_id)
    except (CollectionNotFoundError, CollectionDocumentNotFoundError) as error:
        raise HTTPException(status_code=404, detail=str(error)) from error
    except ValueError as error:
        raise HTTPException(status_code=409, detail=str(error)) from error
    except OSError as error:
        raise HTTPException(status_code=400, detail="Original source file could not be removed.") from error
    except QdrantStoreError as error:
        raise HTTPException(status_code=502, detail=str(error)) from error


@app.post(
    "/collections/{collection_id}/documents",
    response_model=UploadResponse,
    status_code=201,
    tags=["documents"],
)
async def upload_documents(
    collection_id: str,
    background_tasks: BackgroundTasks,
    files: list[UploadFile] = File(...),
) -> UploadResponse:
    job = None
    try:
        uploads = []
        for file in files:
            uploads.append((file.filename or "", await _read_upload(file)))
        collection_service.get_collection(collection_id)
        job = collection_service.metadata_store.create_ingestion_job(collection_id, "uploading")
        documents = collection_service.upload_documents(collection_id, uploads)
    except CollectionNotFoundError as error:
        raise HTTPException(status_code=404, detail=str(error)) from error
    except (ValueError, OSError) as error:
        if job:
            collection_service.metadata_store.update_ingestion_job(job["id"], "error", "Upload failed.")
        raise HTTPException(status_code=400, detail=str(error)) from error

    collection_service.metadata_store.update_ingestion_job(job["id"], "queued", "Preparing uploaded files")
    background_tasks.add_task(workflow.run, collection_id, job["id"])
    return UploadResponse(
        collection_id=collection_id,
        documents=[DocumentResponse.model_validate(asdict(document)) for document in documents],
        job_id=job["id"],
    )


@app.get("/collections/{collection_id}/documents/{document_id}/source", tags=["evidence"])
def document_source(collection_id: str, document_id: str):
    document = collection_service.metadata_store.get_document(document_id, collection_id)
    if document is None:
        raise HTTPException(404, "Document was not found in this collection.")
    path = collection_service.document_store.path_for(document.storage_path)
    if not path.is_file():
        raise HTTPException(404, "Original source file is unavailable.")
    return FileResponse(path, filename=document.filename, content_disposition_type="inline")


@app.post(
    "/collections/{collection_id}/documents/{document_id}/process",
    response_model=ProcessResponse,
    status_code=200,
    tags=["processing"],
)
def process_document(collection_id: str, document_id: str) -> ProcessResponse:
    try:
        result = ingestion_service.process_document(collection_id, document_id)
    except DocumentNotFoundError as error:
        raise HTTPException(status_code=404, detail=str(error)) from error
    except (ValueError, OSError) as error:
        raise HTTPException(status_code=400, detail=str(error)) from error
    except Exception as error:
        raise HTTPException(status_code=500, detail="Document processing failed.") from error

    return ProcessResponse(
        document=DocumentResponse.model_validate(asdict(result.document)),
        evidence_count=result.evidence_count,
        content_types=result.content_types,
    )


@app.get(
    "/collections/{collection_id}/documents/{document_id}/evidence",
    response_model=list[EvidenceResponse],
    tags=["evidence"],
)
def list_document_evidence(
    collection_id: str,
    document_id: str,
) -> list[EvidenceResponse]:
    try:
        evidence_items = ingestion_service.list_evidence(collection_id, document_id)
    except DocumentNotFoundError as error:
        raise HTTPException(status_code=404, detail=str(error)) from error
    sanitized = []
    for item in evidence_items:
        data = asdict(item)
        if data.get("text"):
            data["text"] = redact_pii(data["text"])
        sanitized.append(EvidenceResponse.model_validate(data))
    return sanitized



@app.post(
    "/collections/{collection_id}/index",
    response_model=IndexResponse,
    tags=["indexing"],
)
def index_collection(collection_id: str) -> IndexResponse:
    try:
        result = indexer.index_collection(collection_id)
    except IndexingDocumentNotFoundError as error:
        raise HTTPException(status_code=404, detail=str(error)) from error
    except ConfigurationError as error:
        raise HTTPException(status_code=503, detail=str(error)) from error
    except (EmbeddingServiceError, QdrantStoreError) as error:
        raise HTTPException(status_code=502, detail=str(error)) from error
    return IndexResponse(**asdict(result))


@app.post(
    "/collections/{collection_id}/documents/{document_id}/index",
    response_model=IndexResponse,
    tags=["indexing"],
)
def index_document(collection_id: str, document_id: str) -> IndexResponse:
    try:
        result = indexer.index_document(collection_id, document_id)
    except IndexingDocumentNotFoundError as error:
        raise HTTPException(status_code=404, detail=str(error)) from error
    except ConfigurationError as error:
        raise HTTPException(status_code=503, detail=str(error)) from error
    except (EmbeddingServiceError, QdrantStoreError) as error:
        raise HTTPException(status_code=502, detail=str(error)) from error
    return IndexResponse(**asdict(result))


@app.post(
    "/collections/{collection_id}/query",
    response_model=QueryResponse,
    tags=["query"],
)
def query_collection(
    collection_id: str,
    request: QueryRequest,
) -> QueryResponse:
    status = collection_service.metadata_store.ingestion_status(collection_id)
    if status["busy"]:
        raise HTTPException(409, "Processing and indexing are in progress. Querying is frozen.")
    try:
        result = answer_service.answer(
            collection_id=collection_id,
            question=request.question,
            selected_document_ids=request.selected_document_ids,
            top_k=request.top_k,
        )
    except ScopeViolationError as error:
        raise HTTPException(status_code=400, detail=str(error)) from error
    except ConfigurationError as error:
        raise HTTPException(status_code=503, detail=str(error)) from error
    except (EmbeddingServiceError, ReasoningServiceError, QdrantStoreError) as error:
        raise HTTPException(status_code=502, detail=str(error)) from error

    evidence = [
        RetrievedEvidenceResponse(
            evidence_id=item.evidence.evidence_id,
            collection_id=item.evidence.collection_id,
            document_id=item.evidence.document_id,
            filename=item.evidence.filename,
            page_number=item.evidence.page_number,
            content_type=item.evidence.content_type,
            source_reference=item.evidence.source_reference,
            retrieval_score=item.retrieval_score,
            vector_score=item.vector_score,
            keyword_score=item.keyword_score,
            fusion_score=item.fusion_score,
            rerank_score=item.rerank_score,
            text=item.evidence.text,
            image_reference=item.evidence.image_reference,
            section=item.evidence.section,
            metadata=item.evidence.metadata,
        )
        for item in result.retrieved_evidence
    ]
    return QueryResponse(
        answer=result.answer,
        answerability=result.answerability.value,
        verified=result.verified,
        citations=list(result.citations),
        verification_issues=list(result.verification_issues),
        evidence=evidence,
        claims=[
            {
                "claim_id": claim.claim_id,
                "text": claim.text,
                "evidence_ids": list(claim.evidence_ids),
                "verified": claim.verified,
                "contradiction_status": claim.contradiction_status.value,
            }
            for claim in result.claims
        ],
        contradictions=[
            {
                "subject": contradiction.subject,
                "status": contradiction.status.value,
                "explanation": contradiction.explanation,
                "values": [
                    {
                        "value": str(value.value),
                        "raw_value": value.raw_value,
                        "evidence_id": value.evidence_id,
                    }
                    for value in contradiction.values
                ],
            }
            for contradiction in result.contradictions
        ],
        guardrails_invoked=list(result.guardrails_invoked),
        services_invoked=list(result.services_invoked),
        response_mode=result.response_mode,
    )


def _learning_failure(error: Exception) -> HTTPException:
    if isinstance(error, LearningError):
        return HTTPException(status_code=error.status_code, detail=str(error))
    if isinstance(error, ScopeViolationError):
        return HTTPException(status_code=400, detail=str(error))
    if isinstance(error, ConfigurationError):
        return HTTPException(status_code=503, detail=str(error))
    if isinstance(error, (ReasoningServiceError, EmbeddingServiceError, QdrantStoreError)):
        return HTTPException(status_code=502, detail=str(error))
    return HTTPException(status_code=500, detail="Learning activity failed safely. Retry or narrow the topic.")


def _generate_learning(collection_id: str | None, kind: str, request: GenerateRequest) -> dict[str, object]:
    target_collection = collection_id or request.collection_id
    if not target_collection:
        raise HTTPException(status_code=422, detail="collection_id is required.")
    try:
        return learning_service.generate(target_collection, kind, request)
    except (LearningError, ScopeViolationError, ConfigurationError, ReasoningServiceError, EmbeddingServiceError, QdrantStoreError) as error:
        raise _learning_failure(error) from error


@app.get("/learning/artifacts/{collection_id}", tags=["learning"])
def list_learning_artifacts(
    collection_id: str,
    selected_document_ids: list[str] | None = Query(default=None),
) -> list[dict[str, object]]:
    try:
        return learning_service.list(collection_id, selected_document_ids)
    except (LearningError, ScopeViolationError) as error:
        raise _learning_failure(error) from error


@app.post("/learning/knowledge-graph", tags=["learning"])
def generate_knowledge_graph(request: GenerateRequest, collection_id: str | None = Query(default=None)) -> dict[str, object]:
    return _generate_learning(collection_id, "knowledge-graph", request)


@app.get("/learning/knowledge-graph/{collection_id}", tags=["learning"])
def get_knowledge_graph(
    collection_id: str,
    artifact_id: str | None = Query(default=None),
    selected_document_ids: list[str] | None = Query(default=None),
) -> dict[str, object]:
    try:
        artifacts = learning_service.list(collection_id, selected_document_ids)
        graph_artifacts = [item for item in artifacts if item["kind"] == "knowledge-graph"]
        selected = next((item for item in graph_artifacts if item["id"] == artifact_id), graph_artifacts[0] if graph_artifacts else None)
        if selected is None:
            raise LearningError("No knowledge graph has been generated for this collection.", 404)
        return learning_service.get(collection_id, str(selected["id"]), selected_document_ids)
    except (LearningError, ScopeViolationError) as error:
        raise _learning_failure(error) from error


@app.post("/learning/flashcards/generate", tags=["learning"])
def generate_flashcards(request: GenerateRequest, collection_id: str | None = Query(default=None)) -> dict[str, object]:
    return _generate_learning(collection_id, "flashcards", request)


@app.get("/learning/flashcards/{deck_id}", tags=["learning"])
def get_flashcards(deck_id: str, selected_document_ids: list[str] | None = Query(default=None)) -> dict[str, object]:
    try:
        artifact = collection_service.metadata_store.get_learning_artifact(deck_id)
        if artifact is None or artifact["artifact_type"] != "flashcards":
            raise LearningError("Flashcard deck was not found.", 404)
        return learning_service.get(artifact["collection_id"], deck_id, selected_document_ids)
    except (LearningError, ScopeViolationError) as error:
        raise _learning_failure(error) from error


@app.post("/learning/quizzes/generate", tags=["learning"])
def generate_quiz(request: GenerateRequest, collection_id: str | None = Query(default=None)) -> dict[str, object]:
    return _generate_learning(collection_id, "quizzes", request)


@app.post("/learning/quizzes/{quiz_id}/submit", tags=["learning"])
def submit_quiz(quiz_id: str, request: SubmitRequest) -> dict[str, object]:
    try:
        artifact = collection_service.metadata_store.get_learning_artifact(quiz_id)
        if artifact is None:
            raise LearningError("Quiz was not found.", 404)
        return learning_service.submit(artifact["collection_id"], quiz_id, request.answers, request.selected_document_ids)
    except (LearningError, ScopeViolationError) as error:
        raise _learning_failure(error) from error


@app.post("/learning/topic-tree", tags=["learning"])
def generate_topic_tree(request: GenerateRequest, collection_id: str | None = Query(default=None)) -> dict[str, object]:
    return _generate_learning(collection_id, "topic-tree", request)


@app.post("/learning/cheat-sheet", tags=["learning"])
def generate_cheat_sheet(request: GenerateRequest, collection_id: str | None = Query(default=None)) -> dict[str, object]:
    return _generate_learning(collection_id, "cheat-sheet", request)


@app.post("/learning/teach", tags=["learning"])
def generate_lesson(request: GenerateRequest, collection_id: str | None = Query(default=None)) -> dict[str, object]:
    return _generate_learning(collection_id, "teach", request)


@app.post("/learning/teach/{session_id}/answer", tags=["learning"])
def answer_lesson(session_id: str, request: TeachAnswerRequest) -> dict[str, object]:
    try:
        artifact = collection_service.metadata_store.get_learning_artifact(session_id)
        if artifact is None:
            raise LearningError("Teach Me lesson was not found.", 404)
        return learning_service.answer(artifact["collection_id"], session_id, request)
    except (LearningError, ScopeViolationError) as error:
        raise _learning_failure(error) from error


@app.post("/calculate", response_model=CalculationResponse, tags=["calculation"])
def calculate_values(request: CalculationRequestSchema) -> CalculationResponse:
    try:
        operation = CalculationOperation(request.operation)
        result = calculate(CalculationRequest.from_numbers(operation, request.values))
    except (ValueError, CalculationError) as error:
        raise HTTPException(status_code=400, detail=str(error)) from error
    return CalculationResponse(
        operation=result.operation.value,
        inputs=[str(value) for value in result.inputs],
        result=str(result.result),
        explanation=result.explanation,
    )


@app.post(
    "/collections/{collection_id}/compare",
    response_model=ComparisonResponse,
    tags=["comparison"],
)
def compare_documents(
    collection_id: str,
    request: ComparisonRequest,
) -> ComparisonResponse:
    try:
        result = comparator.compare(
            collection_id,
            request.before_document_id,
            request.after_document_id,
        )
    except ComparisonError as error:
        raise HTTPException(status_code=400, detail=str(error)) from error
    return ComparisonResponse(
        collection_id=result.collection_id,
        before_document_id=result.before_document_id,
        after_document_id=result.after_document_id,
        differences=[
            ComparisonDifferenceResponse(
                key=difference.key,
                change_type=difference.change_type,
                before_text=difference.before_text,
                after_text=difference.after_text,
                before_evidence_ids=list(difference.before_evidence_ids),
                after_evidence_ids=list(difference.after_evidence_ids),
            )
            for difference in result.differences
        ],
    )


@app.get("/collections/{collection_id}/contradictions", tags=["verification"])
def list_contradictions(
    collection_id: str,
    document_id: str | None = Query(default=None),
) -> list[dict[str, object]]:
    if collection_service.metadata_store.get_collection(collection_id) is None:
        raise HTTPException(status_code=404, detail="Collection was not found.")
    evidence = collection_service.metadata_store.list_evidence(collection_id, document_id)
    contradictions = detect_contradictions(evidence)
    return [
        {
            "subject": item.subject,
            "status": item.status.value,
            "explanation": item.explanation,
            "values": [
                {
                    "value": str(value.value),
                    "raw_value": value.raw_value,
                    "evidence_id": value.evidence_id,
                    "text": value.text,
                }
                for value in item.values
            ],
        }
        for item in contradictions
    ]


@app.post("/evaluation", response_model=EvaluationResponse, tags=["evaluation"])
def evaluate(request: EvaluationRequest) -> EvaluationResponse:
    if request.collection_id is not None and collection_service.metadata_store.get_collection(
        request.collection_id
    ) is None:
        raise HTTPException(status_code=404, detail="Evaluation collection was not found.")
    try:
        result = evaluate_run(
            retrieved_ids=request.retrieved_ids,
            relevant_ids=request.relevant_ids,
            ranked_runs=request.ranked_runs,
            claims=request.claims,
            contradiction_statuses=request.contradiction_statuses,
            predicted_answerability=request.predicted_answerability,
            expected_answerability=request.expected_answerability,
            latencies_ms=request.latencies_ms,
            tool_successes=request.tool_successes,
            k=request.k,
        )
    except ValueError as error:
        raise HTTPException(status_code=400, detail=str(error)) from error
    collection_service.metadata_store.save_evaluation(
        result.metrics,
        result.sample_count,
        request.collection_id,
    )
    return EvaluationResponse(**asdict(result))


@app.get("/evaluations", tags=["evaluation"])
def list_evaluations(collection_id: str | None = Query(default=None)) -> list[dict[str, object]]:
    return collection_service.metadata_store.list_evaluations(collection_id)
