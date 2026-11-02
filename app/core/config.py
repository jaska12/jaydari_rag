from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file='.env',
        env_file_encoding='utf-8',
        case_sensitive=False,
    )

    app_name: str = 'jaydari_rag'
    log_level: str = 'INFO'

    qdrant_url: str
    qdrant_api_key: str
    qdrant_collection: str
    qdrant_timeout: float = 10.0

    embedding_model: str
    llm_model: str
    llm_device: str = 'cpu'
    llm_max_new_tokens: int
    llm_temperature: float = 0.2
    llm_adapter_path: str | None = None

    rag_mode: str = 'auto'
    rag_enabled: bool = True
    top_k: int = 5
    max_context_chars: int = 4000
    min_score: float = 0.25
    max_score_strict: float = 0.35
    max_sources: int = 2
    chunk_size_chars: int = 200
    chunk_overlap_chars: int = 200
