import hashlib
import uuid
from pathlib import Path

from app.core.config import Settings
from app.services.embeddings_service import EmbeddingService
from app.services.llm_service import LocalLLM
from app.services.qdrant_service import QdrantService


class RAGService:
    def __init__(
        self,
        settings: Settings,
        embeddings: EmbeddingService,
        qdrant: QdrantService,
        llm: LocalLLM,
    ) -> None:
        self._settings = settings
        self._embeddings = embeddings
        self._qdrant = qdrant
        self._llm = llm

    def ingest(self, documents: list[dict]) -> int:
        return self._qdrant.upsert_documents(documents)

    def ingest_folder(self, path: str) -> tuple[int, int]:
        root = Path(path).expanduser()
        documents: list[dict] = []
        skipped = 0
        ignore_dirs = {'.git', '.venv', '__pycache__', 'node_modules'}
        for file_path in root.rglob('*'):
            if not file_path.is_file():
                continue
            if any(part in ignore_dirs for part in file_path.parts):
                continue
            try:
                text = file_path.read_text(encoding='utf-8')
            except (OSError, UnicodeDecodeError):
                skipped += 1
                continue
            if not text.strip():
                skipped += 1
                continue
            if file_path.suffix.lower() == '.py':
                chunks = self._chunk_python(text)
            else:
                chunks = self._chunk_text(text)
            if not chunks:
                skipped += 1
                continue
            for index, chunk in enumerate(chunks):
                doc_id = self._stable_id(file_path, chunk, index)
                documents.append(
                    {
                        'id': doc_id,
                        'text': chunk,
                        'metadata': {'path': str(file_path), 'chunk': index},
                    }
                )
        ingested = self._qdrant.upsert_documents(documents) if documents else 0
        return ingested, skipped

    def answer(self, question: str) -> tuple[str, list[dict]]:
        filtered: list[dict] = []
        context = ''
        if self._settings.rag_enabled and self._settings.rag_mode != 'never':
            query_vector = self._embeddings.embed_texts([question])[0]
            sources = self._qdrant.search(query_vector, limit=self._settings.top_k)
            filtered = [source for source in sources if source['score'] >= self._settings.min_score]
            use_rag = self._settings.rag_mode == 'always'
            if self._settings.rag_mode == 'auto':
                use_rag = not self._is_general_question(question)
                if filtered:
                    top_score = max(source['score'] for source in filtered)
                    if top_score < self._settings.max_score_strict:
                        use_rag = False
            if use_rag and filtered:
                max_sources = min(self._settings.top_k, self._settings.max_sources)
                filtered = filtered[:max_sources]
                context = self._build_context(filtered)
            else:
                filtered = []
                context = ''
        answer = self._llm.generate(self._build_prompt(question, context))
        answer = self._clean_answer(answer)
        result = (answer, filtered)
        return result

    def _build_context(self, sources: list[dict]) -> str:
        chunks = []
        total = 0
        for index, source in enumerate(sources, start=1):
            text = source.get('text', '').strip()
            if not text:
                continue
            metadata = source.get('metadata') or {}
            block = f"[{index}] {metadata.get('path', source.get('id', ''))}\n{text}"
            remaining = self._settings.max_context_chars - total
            if remaining <= 0:
                break
            block = block[:remaining]
            chunks.append(block)
            total += len(block)
        return '\n\n'.join(chunks)

    def _build_prompt(self, question: str, context: str) -> str:
        system = 'You are Jaydari GPT, a helpful assistant. Answer briefly and accurately.'
        if context:
            user = (
                'Answer the question using only the context below. '
                "If the context does not contain the answer, say you don't know.\n\n"
                f'Context:\n{context}\n\nQuestion: {question}'
            )
        else:
            user = question
        # TinyLlama chat format (same as jaydari_gpt/chat_template.jinja)
        return f'<|system|>\n{system}</s>\n<|user|>\n{user}</s>\n<|assistant|>\n'

    def _chunk_text(self, text: str) -> list[str]:
        text = text.strip()
        size = max(1, self._settings.chunk_size_chars)
        overlap = self._settings.chunk_overlap_chars
        # overlap >= size would never move the window forward
        step = size - overlap if size > overlap else size
        chunks = []
        start = 0
        while start < len(text):
            chunk = text[start : start + size].strip()
            if chunk:
                chunks.append(chunk)
            if start + size >= len(text):
                break
            start += step
        return chunks

    def _chunk_python(self, text: str) -> list[str]:
        blocks: list[str] = []
        preamble: list[str] = []
        current: list[str] = []

        def flush_current() -> None:
            block = '\n'.join(current).strip()
            if block:
                blocks.append(block)
            current.clear()

        for line in text.splitlines():
            is_top_level = line.startswith(('def ', 'async def ', 'class ', '@'))
            # keep decorators attached to the function/class below them
            if is_top_level and not (current and current[-1].startswith('@')):
                flush_current()
            if current or is_top_level:
                current.append(line)
            else:
                preamble.append(line)
        flush_current()
        preamble_text = '\n'.join(preamble).strip()
        if preamble_text:
            blocks.insert(0, preamble_text)
        if not blocks:
            return self._chunk_text(text)
        return blocks

    def _stable_id(self, file_path: Path, text: str, index: int) -> str:
        digest = hashlib.sha1(text.encode('utf-8', errors='ignore')).hexdigest()
        return str(uuid.uuid5(uuid.NAMESPACE_URL, f'{file_path}:{index}:{digest}'))

    def _clean_answer(self, answer: str) -> str:
        for marker in ('\nQuestion:', '\n\nQuestion:', 'Question:', '\nContext:', '\n\nContext:', 'Context:'):
            if marker in answer:
                answer = answer.split(marker, 1)[0]
        return answer.strip()

    def _is_general_question(self, question: str) -> bool:
        q = question.lower()
        project_markers = (
            'project',
            'repo',
            'codebase',
            'file',
            'path',
            'where',
            'implemented',
            'implementation',
            'class',
            'function',
            'endpoint',
            'api',
            'controller',
            'service',
            'module',
            'route',
        )
        if any(marker in q for marker in project_markers):
            return False
        if '\\' in q or '/' in q or '.py' in q:
            return False
        return True
