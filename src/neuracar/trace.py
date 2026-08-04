from dataclasses import dataclass
from typing import Iterator


@dataclass(frozen=True)
class Request:
    time: float
    obj_id: str
    size: float


def read_trace(path: str, delimiter: str = ",", has_header: bool = False) -> Iterator[Request]:
    """Streams a trace file of `time,obj_id,size` rows (libCacheSim's plain CSV trace format)."""
    with open(path, "r") as f:
        if has_header:
            next(f)
        for line in f:
            line = line.strip()
            if not line:
                continue
            time_str, obj_id, size_str = line.split(delimiter)[:3]
            yield Request(time=float(time_str), obj_id=obj_id, size=float(size_str))
