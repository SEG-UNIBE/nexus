import pandas as pd
from typing import List, Tuple
from dataclasses import dataclass
import numpy as np

from nexus.core.interfaces.data_model import DataModel
from nexus.utils.timing import timed
from nexus.utils.score import pairs
from collections import defaultdict
from nexus.components.dataloader.embeddings import EMBEDDING_STRATEGIES

@dataclass
class ModelElementDataModel(DataModel):
    """Software Engineering Model DataModel.
    
    Loads CSV files containing software models (e.g., UML, code models) with:
    - Hierarchy levels (e.g., model name, package, class)
    - Data attributes (e.g., method names, attributes)
    
    Embeds using configurable strategies: letter histogram, BERT, mean count, zero-one, etc.
    """
    
    def __init__(
            self, 
            data, 
            embedding_strategy: str = 'ld', 
            similarity_method: str = 'weighted',
            **embedding_params
        ):
        """
        Args:
            data: str (CSV path) or pandas DataFrame
        """
        super().__init__(embedding_strategy=embedding_strategy, **embedding_params)
        self.similarity_method = similarity_method
        self.hierarchy_levels = [0, 2]  # Column indices for hierarchy
        self.data_levels = [3]  # Column indices for data attributes
        self.attr_cache = {}  # Cache for faster attribute access
        self.embedding_fn = EMBEDDING_STRATEGIES.get(embedding_strategy)
        self.shuffle_seed = 1337
        self.subsample_fraction = 1.0
        self.df = None
        self.path = None
        if isinstance(data, str):
            self.path = data
        elif isinstance(data, pd.DataFrame):
            self.df = data.copy()
        else:
            raise ValueError("data must be a CSV path or a pandas DataFrame")
        
    def set_shuffle_seed(self, seed):
        self.shuffle_seed = seed
        np.random.seed(seed)
        self.subsample_fraction = np.random.rand()
        self.subsample_fraction = 0.5

        return self.subsample_fraction

    @timed()
    def load(self) -> 'ModelElementDataModel':
        """Load CSV file or use provided DataFrame and build unique document IDs"""
        try:
            self.df = pd.read_csv(self.path, header=None)

            self.df = self.df.sample(
                frac=self.subsample_fraction, 
                random_state=self.shuffle_seed
            ).reset_index(drop=True)

            self.df.head()
            
            model_col = self.hierarchy_levels[0]
            unique_models = self.df[model_col].dropna().unique()
            model_to_id = {model: idx for idx, model in enumerate(unique_models)}
            self.df['uniqueDocId'] = self.df[model_col].map(model_to_id).fillna(-1).astype(int)
            self.n_docs = len(unique_models)
            self.total_elements = len(self.df)
            # Build attribute cache for faster access
            self.attr_cache = {idx: self._get_attr_from_idx(idx) for idx in range(len(self.df))}
        except Exception as e:
            print(f"Error loading DataFrame: {e}")
        return self
    

    def __getitem__(self, idx) -> str:
        """Get item i from the dataset"""
        row_set = self._get_attr_from_idx(idx)
        row_str = ";".join(row_set)
        return row_str

    def __len__(self):
        """Return the number of items in the dataset"""
        return len(self.df)
    
    def get_doc_id(self, idx):
        """Extracts the document id of a given element index"""
        row = self.df.iloc[idx]
        return row[self.hierarchy_levels[0]]

    def get_document_group(self, doc_repr) -> List:
        target_mask = self.df["uniqueDocId"] == doc_repr
        return self.df[target_mask].index.tolist()

    def get_documents_from_index(self, idx_list):
        """Get document names from index list"""
        doc_names = []
        for i in idx_list:
            row = self.df.iloc[i]
            doc_names.append(row[self.hierarchy_levels[0]])
        return set(doc_names)
    
    def _get_attr_from_idx(self, idx, filter_attr=True) -> set:
        """Extract attributes for a given row index"""
        elem_data = self.df.iloc[idx][self.data_levels]
        all_attributes = []
        for attr in elem_data:
            if pd.notna(attr):
                split_attrs = str(attr).split(";")
                all_attributes.extend(split_attrs)

        # filter spaces (keep original case like Java version)
        if filter_attr:
            all_attributes = {word.strip() for word in all_attributes if word.strip()}
        return all_attributes
    
    def _jaccard_similarity(self, elements: list) -> float:
        """Compute Jaccard similarity for a group of elements"""
        # if len(elements) < 2:
        #     return 0.0
        if len(elements) == 0:
            return 0.0
        if len(elements) == 1:
            return 1.0

        prop_lists = [set(self.attr_cache[idx]) for idx in elements]
        union_props = set.union(*prop_lists)
        intersection_props = set.intersection(*prop_lists) if prop_lists else set()

        if not union_props:
            return 0.0

        return len(intersection_props) / len(union_props)
    
    def _weighted_similarity(self, elements: list) -> float:
        """Compute weighted similarity for a group of elements based on Rubin and Chechik (2013)."""
        t = len(elements)
        if t == 0:
            return 0.0

        prop_lists = [set(self.attr_cache[idx]) for idx in elements]
        all_props = set.union(*prop_lists)
        pi_t = len(all_props)
        if pi_t == 0:
            return 0.0

        prop_counts = {}
        for prop_set in prop_lists:
            for prop in prop_set:
                prop_counts[prop] = prop_counts.get(prop, 0) + 1
        
        # Formula: sum(j_p^2) / (|pi(t)| * n^2)
        # j_p = number of elements with property p
        # n = total number of models
        # numerator = sum(count ** 2 for count in prop_counts.values())
        numerator = sum(count ** 2 for count in prop_counts.values() if count >= 2)
        denominator = pi_t * (self.n_docs ** 2)
        
        if denominator == 0:
            return 0.0
            
        return numerator / denominator
    
    def compute_similarity(self, indices: list) -> float:
        """Compute similarity score for a group of elements.
        
        Args:
            indices: List of element indices
            
        Returns:
            Similarity score
        """
        if self.similarity_method == "jaccard":
            return self._jaccard_similarity(indices)
        else:  # weighted
            return self._weighted_similarity(indices)
    
    def shouldMatch(self, group1: set, group2: set, threshold: float = 0.5) -> bool:
        """Check if two groups should be merged based on similarity.
        
        For jaccard: merged similarity must exceed threshold
        For weighted: merged similarity must exceed sum of individual similarities
        
        Args:
            group1: Set of element indices in first group
            group2: Set of element indices in second group
            threshold: Minimum similarity threshold
            
        Returns:
            True if groups should be merged
        """

        merged_elements = list(set(group1) | set(group2))
        merged_sim = self.compute_similarity(merged_elements)
        
        if self.similarity_method == "jaccard":
            return merged_sim >= threshold
        else:  # weighted
            sim1 = self.compute_similarity(list(group1)) if group1 else 0.0
            sim2 = self.compute_similarity(list(group2)) if group2 else 0.0
            return merged_sim > (sim1 + sim2)

    def export_matches(self, matches: list, output_path: str) -> pd.DataFrame:
        """Export matches to a CSV file in RaQuN format.
        
        Format:
        - tuple_id: Match group ID (1-indexed)
        - tuple_size: Number of elements in the match
        - element_uuid: The UUID of the element (column 1)
        - element_model_id: Which model variant it's from (column 0)
        - element_name: The class/element name (column 2)
        - element_properties: The properties semicolon-separated (column 3)
        
        Args:
            matches: List of (set_of_indices, similarity) tuples from MatchingResult
            output_path: Path to save the CSV file
            
        Returns:
            DataFrame with the exported matches
        """
        rows = []
        tuple_id = 1
        
        for match_indices, similarity in matches:
            tuple_size = len(match_indices)
            
            for idx in match_indices:
                row = self.df.iloc[idx]
                rows.append({
                    'tuple_id': tuple_id,
                    'tuple_size': tuple_size,
                    'element_uuid': row[1] if 1 < len(row) else '',  # UUID column
                    'element_model_id': row[0] if 0 < len(row) else '',  # Model column
                    'element_name': row[2] if 2 < len(row) else '',  # Name column
                    'element_properties': row[3] if 3 < len(row) else ''  # Properties column
                })
            
            tuple_id += 1
        
        result_df = pd.DataFrame(rows)
        
        # Save to CSV
        result_df.to_csv(output_path, index=False)
        print(f"Exported {len(matches)} match groups ({len(rows)} elements) to {output_path}")
        
        return result_df
    
    def compute_metrics(self, matches: list) -> Tuple[float]:
        """
        Computes accuracy, precision, recall and f1 metrics
        """
        index_to_id = {}
        for idx, row in self.df.iterrows():
            uuid = row.iloc[1]
            model_id = row.iloc[0]
            index_to_id[idx] = (uuid, model_id)

        gt_groups = defaultdict(list)
        for idx, (uuid, model_id) in index_to_id.items():
            gt_groups[uuid].append((uuid, model_id))
            
        true_clusters = list(gt_groups.values())

        pred_clusters = []
        for m in matches:
            indices = m[0]
            cluster_ids = [index_to_id[idx] for idx in indices if idx in index_to_id]
            pred_clusters.append(set(cluster_ids))

        pred_pairs = pairs(pred_clusters)
        true_pairs = pairs(true_clusters)
        
        tp = len(pred_pairs & true_pairs)
        fp = len(pred_pairs - true_pairs)
        fn = len(true_pairs - pred_pairs)
        
        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        accuracy = tp / len(true_pairs) if true_pairs else 0.0
        f1 = 2 * (recall * precision) / (recall + precision) if (recall + precision) else 0.0
        
        return accuracy, precision, recall, f1
    

    def compute_proxy(self, matches: list, reduction: str = 'sum') -> float:
        """Compute proxy score for a set of matches
        
        Args:
            indices: List of matches to compute proxy for the matches

        Returns: 
            float: proxy score
        """
        scores = [] 
        for match in matches:
            w = self._weighted_similarity(match[0])
            scores.append(w)

        if reduction == 'mean':
            return np.mean(scores)
        elif reduction == 'max':
            return np.max(scores)
        elif reduction == 'min':
            return np.min(scores)
        else:
            return np.sum(scores)