from nexus.utils.datastructures import MatchingResult
from nexus.core.interfaces.matcher import Matcher
from nexus.core.interfaces.searcher import Searcher
from nexus.core.interfaces.data_model import DataModel


class Algo:
    """Generic algorithm that chains data_model -> searcher -> matcher
    
    Note: Vectorization is now embedded within the data_model phase.
    """
    
    def __init__(
            self, 
            data_model: DataModel, 
            searcher: Searcher, 
            matcher: Matcher, 
            name: str = "algo"
        ):
        
        self.data_model = data_model
        self.searcher = searcher
        self.matcher = matcher
        self.name = name

    
    def __call__(self, **kwargs) -> MatchingResult:
        """Execute the full algorithm pipeline"""
        timing = {}
        job_id = kwargs.get('job_id', kwargs.get('id', '?'))

        # Phase 0: Data loading
        data, load_timing = self.data_model.load()
        timing.update({f"load_{k}": v for k, v in load_timing.items()})
        load_elapsed = sum(load_timing.values())
        timing.update({"load_time": load_elapsed})
        print(f"[job {job_id}] phase=load        took={load_elapsed:.2f}s")
        
        # Phase 1: Embedding
        _, embed_timing = data.embed()
        timing.update({f"embed_{k}": v for k, v in embed_timing.items()})
        embed_elapsed = sum(embed_timing.values())
        timing.update({"embed_time": embed_elapsed})
        print(f"[job {job_id}] phase=embed       took={embed_elapsed:.2f}s")

        # Phase 2: Search
        candidates, search_timing = self.searcher(data)
        timing.update({f"search_{k}": v for k, v in search_timing.items()})
        search_elapsed = sum(search_timing.values())
        timing.update({"search_time": search_elapsed})
        print(f"[job {job_id}] phase=search      took={search_elapsed:.2f}s")
        
        # Phase 3: Matching
        result, match_timing = self.matcher(candidates, data)
        timing.update({f"match_{k}": v for k, v in match_timing.items()})
        match_elapsed = sum(match_timing.values())
        timing.update({"match_time": match_elapsed})
        print(f"[job {job_id}] phase=match       took={match_elapsed:.2f}s")

        # Attach consolidated timing to the result and compute total
        # Use only the phase-level timings to avoid double-counting
        total_elapsed = load_elapsed + embed_elapsed + search_elapsed + match_elapsed
        result.timing = timing
        result.timing["total"] = total_elapsed
        print(f"[job {job_id}] phase=total       took={total_elapsed:.2f}s")
        
        return result
    
    def exec(self, **kwargs):
        return self.__call__(**kwargs)
