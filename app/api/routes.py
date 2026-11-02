from fastapi import APIRouter, Request, Depends

from app.dto.Rag import IngestPathRequest, IngestPathResponse, QueryRequest, QueryResponse, SourceItem
from app.services.rag_service import RAGService
from utils.source_formatting import format_source

router = APIRouter()

@router.get('/health')
def health_check():
    return {'status': 'OK'}

def get_rag_service(request: Request) -> RAGService:
    return request.app.state.rag


@router.post('/ingest-path', response_model=IngestPathResponse)
def ingest_path(payload: IngestPathRequest, rag: RAGService = Depends(get_rag_service)) -> IngestPathResponse:
    ingested, skipped = rag.ingest_folder(payload.path)
    return IngestPathResponse(ingested=ingested, skipped=skipped)


@router.post('/query', response_model=QueryResponse)
def query(payload: QueryRequest, rag: RAGService = Depends(get_rag_service)) -> QueryResponse:
    answer, sources = rag.answer(payload.question)
    formatted_sources = [format_source(source) for source in sources]
    return QueryResponse(
        answer=answer,
        sources=[SourceItem(**source) for source in formatted_sources],
    )
