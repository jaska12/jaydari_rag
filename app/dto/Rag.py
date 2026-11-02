from typing import Any

from pydantic import BaseModel, Field


class IngestPathRequest(BaseModel):
    path: str = Field(min_length=5)


class IngestPathResponse(BaseModel):
    ingested: int
    skipped: int


class SourceItem(BaseModel):
    id: str
    text: str
    score: float
    metadata: dict[str, Any] | None = None


class QueryRequest(BaseModel):
    question: str = Field(min_length=10)


class QueryResponse(BaseModel):
    answer: str
    sources: list[SourceItem]
