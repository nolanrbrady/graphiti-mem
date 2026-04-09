from .client import EmbedderClient

__all__ = [
    'EmbedderClient',
    'OpenAIEmbedder',
    'OpenAIEmbedderConfig',
]


def __getattr__(name: str):
    if name in {'OpenAIEmbedder', 'OpenAIEmbedderConfig'}:
        from .openai import OpenAIEmbedder, OpenAIEmbedderConfig

        return {
            'OpenAIEmbedder': OpenAIEmbedder,
            'OpenAIEmbedderConfig': OpenAIEmbedderConfig,
        }[name]
    raise AttributeError(f'module {__name__!r} has no attribute {name!r}')
