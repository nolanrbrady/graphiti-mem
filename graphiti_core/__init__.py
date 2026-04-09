__all__ = ['Graphiti', 'MemoryEngine', 'MemoryKind']


def __getattr__(name: str):
    if name == 'Graphiti':
        from .graphiti import Graphiti

        return Graphiti
    if name in {'MemoryEngine', 'MemoryKind'}:
        from .memory import MemoryEngine, MemoryKind

        return {'MemoryEngine': MemoryEngine, 'MemoryKind': MemoryKind}[name]
    raise AttributeError(f'module {__name__!r} has no attribute {name!r}')
