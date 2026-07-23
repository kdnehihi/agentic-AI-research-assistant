from __future__ import annotations


def infer_section_groups_from_query(query: str) -> tuple[str, ...]:
    """Infer likely paper section groups from query wording for soft reranking."""

    lowered = query.lower()
    if any(term in lowered for term in ("main idea", "central contribution", "main takeaway", "scope")):
        return ("abstract",)
    if any(
        term in lowered
        for term in ("intro", "introduction", "motivat", "research problem", "gap")
    ):
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


def infer_explicit_section_groups_from_query(query: str) -> tuple[str, ...]:
    """Infer section groups that are explicit enough to use as hard filters."""

    lowered = query.lower()
    matches: list[str] = []
    section_terms = [
        ("abstract", ("abstract",)),
        ("introduction", ("intro", "introduction")),
        ("method", ("method section", "methodology section", "methods section")),
        ("experiments", ("experiment section", "experiments section")),
        ("results", ("result section", "results section")),
        ("limitations", ("limitation", "limitations")),
        ("conclusion", ("conclusion", "future direction")),
        ("discussion", ("discussion",)),
        ("background", ("background", "related work")),
    ]
    for section_group, terms in section_terms:
        if any(term in lowered for term in terms):
            matches.append(section_group)
    return tuple(dict.fromkeys(matches))
