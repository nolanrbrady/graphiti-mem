__all__ = ['Neo4jDriver']


def __getattr__(name: str):
    if name == 'Neo4jDriver':
        from .neo4j_driver import Neo4jDriver

        return Neo4jDriver
    raise AttributeError(f'module {__name__!r} has no attribute {name!r}')
