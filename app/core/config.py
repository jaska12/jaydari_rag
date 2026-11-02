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
    qdrant_api_key: str | None = None
    qdrant_collection: str = 'documents'
    qdrant_timeout: float = 10.0

    embedding_model: str = 'sentence-transformers/all-MiniLM-L6-v2'

    llm_model: str = 'TinyLlama/TinyLlama-1.1B-Chat-v1.0'
    llm_device: str = 'cpu'

    rag_enabled: bool = True
    top_k: int = 5
