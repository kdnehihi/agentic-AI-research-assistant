from __future__ import annotations

import json
import re
from typing import Any

from app.agent.state import AgentState, SearchPlan
from app.llm.client import OpenAILLMClient, LLMClient
from app.tools.arxiv_tools import build_arxiv_query_from_terms

OPENAI_MODEL = "gpt-4.1-mini"

DEFAULT_CATEGORIES = ["cs.CL", "cs.AI", "cs.LG", "stat.ML"]
MAX_LLM_TERMS = 8


def plan_arxiv_search_query_with_llm(
    state: AgentState,
    llm_client: LLMClient | None = None,
) -> dict[str, Any]:
    """
    Use an LLM only to rewrite the user topic into structured search terms.

    This tool does not call arXiv. The actual retrieval still happens inside
    search_arxiv_papers.
    """
    llm_client = llm_client or OpenAILLMClient(model=OPENAI_MODEL)

    try:
        llm_output = llm_client.generate(_build_query_planner_prompt(state.topic))
        plan_data = _parse_llm_plan(llm_output)
        planner = "llm"
        status = "success"
        error = None
    except Exception as exc:
        plan_data = _fallback_plan_data(state.topic)
        planner = "rule_based"
        status = "partial_success"
        error = str(exc)

    arxiv_query = build_arxiv_query_from_terms(
        core_terms=plan_data["core_terms"],
        context_terms=plan_data["context_terms"],
        categories=plan_data["categories"],
    )
    search_plan = SearchPlan(
        original_query=state.topic,
        core_terms=plan_data["core_terms"],
        context_terms=plan_data["context_terms"],
        categories=plan_data["categories"],
        arxiv_query=arxiv_query,
        planner=planner,
    )
    state.set_search_plan(search_plan)

    observation = {
        "status": status,
        "planner": planner,
        "core_terms": search_plan.core_terms,
        "context_terms": search_plan.context_terms,
        "categories": search_plan.categories,
        "search_query": search_plan.arxiv_query,
        "summary": f"Planned arXiv search query using {planner} planner.",
    }
    if error:
        observation["error"] = error

    return observation


def _build_query_planner_prompt(topic: str) -> str:
    return f"""
You prepare arXiv search terms for a research assistant.

Return only valid JSON with this exact shape:
{{
  "core_terms": ["primary concept or acronym"],
  "context_terms": ["required surrounding concept"],
  "categories": ["cs.CL", "cs.AI", "cs.LG", "stat.ML"]
}}

Rules:
- Do not call arXiv or choose papers.
- core_terms should capture the main research object.
- context_terms should capture what the papers must also be about.
- Prefer short arXiv-friendly terms and common aliases.
- Use at most {MAX_LLM_TERMS} core terms and {MAX_LLM_TERMS} context terms.
- categories must be chosen from: cs.CL, cs.AI, cs.LG, cs.IR, stat.ML.

User topic:
{topic}
""".strip()


def _parse_llm_plan(llm_output: str) -> dict[str, list[str]]:
    raw_json = _extract_json_object(llm_output)
    data = json.loads(raw_json)

    core_terms = _clean_term_list(data.get("core_terms", []), limit=MAX_LLM_TERMS)
    context_terms = _clean_term_list(data.get("context_terms", []), limit=MAX_LLM_TERMS)
    categories = _clean_categories(data.get("categories", []))

    if not core_terms:
        raise ValueError("LLM query plan did not include core_terms.")

    return {
        "core_terms": core_terms,
        "context_terms": context_terms,
        "categories": categories or DEFAULT_CATEGORIES,
    }


def _extract_json_object(text: str) -> str:
    text = text.strip()
    if text.startswith("{") and text.endswith("}"):
        return text

    match = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if not match:
        raise ValueError("LLM query plan did not contain a JSON object.")

    return match.group(0)


def _clean_term_list(value: Any, limit: int) -> list[str]:
    if not isinstance(value, list):
        return []

    terms = []
    seen = set()
    for item in value:
        if not isinstance(item, str):
            continue

        term = " ".join(item.replace('"', "").split())
        term_key = term.lower()
        if not term or term_key in seen:
            continue

        seen.add(term_key)
        terms.append(term)

    return terms[:limit]


def _clean_categories(value: Any) -> list[str]:
    valid_categories = {"cs.AI", "cs.CL", "cs.LG", "cs.IR", "stat.ML"}
    if not isinstance(value, list):
        return DEFAULT_CATEGORIES

    categories = [
        category
        for category in value
        if isinstance(category, str) and category in valid_categories
    ]

    return categories[: len(valid_categories)]


def _fallback_plan_data(topic: str) -> dict[str, list[str]]:
    return {
        "core_terms": [topic],
        "context_terms": [],
        "categories": DEFAULT_CATEGORIES,
    }
