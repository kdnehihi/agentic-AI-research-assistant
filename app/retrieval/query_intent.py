from __future__ import annotations


def infer_section_groups_from_query(query: str) -> tuple[str, ...]:
    lowered = query.lower()
    if any(term in lowered for term in ("main idea", "central contribution", "main takeaway", "scope")):
        return ("abstract",)
    if any(term in lowered for term in ("motivat", "research problem", "gap")):
        return ("introduction",)
    if any(term in lowered for term in ("method", "methodology", "algorithm", "approach", "system design")):
        return ("method",)
    if any(term in lowered for term in ("experimental setup", "dataset", "benchmark", "evaluation protocol")):
        return ("experiments",)
    if any(term in lowered for term in ("result", "finding", "performance conclusion", "empirical")):
        return ("results",)
    if any(term in lowered for term in ("limitation", "threat", "open challenge")):
        return ("limitations",)
    if any(term in lowered for term in ("conclusion", "future direction", "implication")):
        return ("conclusion",)
    return ()
