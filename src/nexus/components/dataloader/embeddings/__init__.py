from .zero_one import zero_one_embedding
from .character_based import character_based_embedding

__all__ = [
    'zero_one_embedding',
    'character_based_embedding'
]

EMBEDDING_STRATEGIES = {
    'hd': zero_one_embedding,
    'ld': character_based_embedding,
}
