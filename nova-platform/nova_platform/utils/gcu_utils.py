from dataclasses import dataclass
from typing import Any

@dataclass
class GCUData:  
    tgen: Any = None
    cache_svc: Any = None
    post_stat: Any = None
    start_ref: float = 0.0
    end_ref: float = 0.0
    last_ref: float = 0.0