def format_source(source: dict, max_chars: int = 300) -> dict:
    text = (source.get('text') or '').strip()
    if len(text) > max_chars:
        text = text[:max_chars].rstrip() + '...'
    return {
        'id': str(source.get('id', '')),
        'text': text,
        'score': round(float(source.get('score', 0.0)), 4),
        'metadata': source.get('metadata') or None,
    }
