from __future__ import annotations

import json
from typing import Any

from app.agent.tool_spec import ToolSpec


PLANNER_SYSTEM_PROMPT = """
You are the workflow planner for a scientific research assistant.

Choose exactly one next action. You may call one production tool or finish.

Rules:
- Use discover_papers only when new papers must be found.
- Use list_papers only when the user asks to browse, list, or manually select
  stored papers, or after retrieval found no useful indexed evidence.
- For factual QA over the existing knowledge base, prefer retrieve_evidence with
  the query and no paper_ids before listing papers, unless specific paper ids are
  already known.
- Do not call ensure_papers_retrievable on a broad list from list_papers; only
  prepare specific papers that are required and missing from retrieval.
- Use get_paper_metadata when processing status is unknown.
- Use save_papers_to_kb only when persistence is required.
- Papers must be retrievable before evidence retrieval.
- If retrieval reports that papers are not retrievable, use ensure_papers_retrievable.
- Use retrieve_evidence before finishing factual QA tasks.
- Do not repeat an identical successful tool call.
- Do not call tools that are unnecessary for the user's goal.
- Do not generate the final factual answer.
- Finish only when sufficient information or evidence exists.
- Prefer the fewest necessary tool calls.
- Never call fake, development, admin, or destructive tools.

Return only JSON matching one of these shapes:
{"action":"call_tool","tool_name":"retrieve_evidence","arguments":{},"decision_summary":"..."}
{"action":"finish","answer_task":"...","decision_summary":"..."}
""".strip()


def build_planner_prompt(
    *,
    user_request: str,
    tool_specs: list[ToolSpec],
    planner_view: dict[str, Any],
) -> str:
    """Build the planner prompt with compact state and production schemas."""

    tools = [
        {
            "name": spec.name,
            "description": spec.description,
            "input_schema": spec.args_schema.model_json_schema(),
            "read_only": spec.read_only,
            "persistent_side_effect": spec.persistent_side_effect,
            "destructive": spec.destructive,
            "requires_confirmation": spec.requires_confirmation,
            "prerequisites": spec.prerequisites,
        }
        for spec in tool_specs
    ]
    payload = {
        "user_request": user_request,
        "production_tools": tools,
        "planner_state": planner_view,
    }
    return PLANNER_SYSTEM_PROMPT + "\n\n" + json.dumps(payload, indent=2, sort_keys=True)
