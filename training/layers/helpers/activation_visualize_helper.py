from framework.data_structures import DotDict
from dataclasses import dataclass
from typing import Optional, List


@dataclass
class VisInfo:
    marks: Optional[List[str]]
    start: int
    end: int


def get_vis_info(options: DotDict, tlen: int) -> VisInfo:
    marks = options.get("steplabel")
    n_steps = options.n_steps or 9999999

    if marks is not None:
        # Handle padding
        assert len(marks) <= tlen
        tlen = len(marks)

    ns1 = (tlen + n_steps) if n_steps < 0 else 0
    ns1_e = tlen if n_steps < 0 else min(n_steps, tlen)

    if marks is not None:
        marks = marks[ns1:ns1_e]

    return VisInfo(marks, ns1, ns1_e)

