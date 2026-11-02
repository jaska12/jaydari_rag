from contextlib import asynccontextmanager
from fastapi import FastAPI
import logging

from app.api.routes import router
from app.core.config import Settings
from app.services.embeddings_service import EmbeddingService
from app.services.qdrant_service import QdrantService
from app.services.llm_service import LocalLLM
from app.services.rag_service import RAGService

logger = logging.getLogger("uvicorn")
logger.info("*** JAYDARI RAG STARTS ***")

@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = Settings()
    app.state.settings = settings

    logging.basicConfig(
        level=settings.log_level,
        format='%(asctime)s %(levelname)s %(name)s %(message)s'
    )

    embeddings = EmbeddingService(settings.embedding_model)
    qdrant = QdrantService(settings, embeddings)
    llm = LocalLLM(settings)
    rag = RAGService(settings, embeddings, qdrant, llm)
    app.state.rag = rag
    yield


app = FastAPI(title='JAYDARI RAG', version='1.0.0', lifespan=lifespan)


app.include_router(router)
