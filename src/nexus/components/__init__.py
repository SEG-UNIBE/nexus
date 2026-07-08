from nexus.components.matcher.greedy import GreedyMatcher
from nexus.components.searcher.tree import TreeSearch
from nexus.components.dataloader.model_element_loader import ModelElementDataModel


DATA_MODELS_REG = {
    'me_data': ModelElementDataModel,
}

SEARCHERS_REG = {
    'tree': TreeSearch
}

MATCHERS_REG = {
    'greedy': GreedyMatcher,
}