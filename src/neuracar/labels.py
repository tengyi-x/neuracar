from typing import Dict, List

from .trace import Request


def next_access_gaps(requests: List[Request]) -> List[float]:
    """For each request, the time until that object is next requested (float('inf') if never again)."""
    last_pending: Dict[str, int] = {}
    gaps = [float("inf")] * len(requests)
    for i, req in enumerate(requests):
        if req.obj_id in last_pending:
            gaps[last_pending[req.obj_id]] = req.time - requests[last_pending[req.obj_id]].time
        last_pending[req.obj_id] = i
    return gaps


def reuse_labels(requests: List[Request], window: float) -> List[int]:
    """1 if the object is accessed again within `window` time units, else 0."""
    return [1 if gap <= window else 0 for gap in next_access_gaps(requests)]
