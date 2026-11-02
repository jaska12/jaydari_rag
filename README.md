# RAG API

Simple, production-minded RAG service using FastAPI, Pydantic, Qdrant, and TinyLlama LLM.

## Setup
1. Create a conda environment and install dependencies:

```bash
conda create -n rag_env python=3.11
conda activate rag_env
pip install -r requirements.txt
```

2. Start the server:

```bash
uvicorn app.main:app --reload
```
