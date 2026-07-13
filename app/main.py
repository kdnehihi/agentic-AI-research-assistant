from app.agent.runner import AgentRunner, ARXIV_RAG_EVAL_WORKFLOW
from app.agent.state import AgentState
from app.llm.client import OpenAILLMClient
from app.tools.report_tools import generate_report_from_abstracts
from app.tools.llm_summary_tools import summarize_papers_with_llm
from app.tools.knowledge_base_tools import save_selected_papers_to_kb
from app.tools.registry import ToolRegistry

SEARCH_AND_FILTER_WORKFLOW = ARXIV_RAG_EVAL_WORKFLOW


TOPIC = "agentic retrieval augmented generation systems for scientific literature search and research paper summarization"
MAX_PAPERS = 5


def main():
    state = AgentState(
        topic=TOPIC,
        max_papers=MAX_PAPERS,
    )

    registry = ToolRegistry()
    runner = AgentRunner(state=state, registry=registry)

    runner.run_workflow(workflow=SEARCH_AND_FILTER_WORKFLOW)
    summarize_papers_with_llm(
        state=state,
        llm_client=OpenAILLMClient(),
    )
    generate_report_from_abstracts(state)
    save_observation = save_selected_papers_to_kb(state)

    print("\n===== FINAL REPORT =====\n")
    print(state.report)
    print("\n===== KNOWLEDGE BASE SAVE REPORT =====\n")
    print(save_observation)
    print("\n===== RETRIEVAL EVALUATION =====\n")
    print(_format_retrieval_eval(state.eval_results))


def _format_retrieval_eval(eval_results: dict | None) -> str:
    if not eval_results:
        return "No retrieval evaluation was produced."

    top_k = eval_results.get("top_k", 5)
    lines = [
        f"Cases: {eval_results.get('num_cases', 0)}",
        f"Hit Rate@{top_k}: {eval_results.get('hit_rate_at_k', 0.0):.3f}",
        f"Recall@{top_k}: {eval_results.get('mean_recall_at_k', 0.0):.3f}",
        f"Precision@{top_k}: {eval_results.get('mean_precision_at_k', 0.0):.3f}",
        f"MRR: {eval_results.get('mrr', 0.0):.3f}",
        f"nDCG@{top_k}: {eval_results.get('mean_ndcg_at_k', 0.0):.3f}",
    ]

    for result in eval_results.get("results", []):
        lines.extend(
            [
                "",
                f"Query: {result.get('query')}",
                f"Gold section: {result.get('gold_section')}",
                f"Section filters: {result.get('section_groups')}",
                f"Gold chunks: {result.get('relevant_chunk_ids')}",
                f"Retrieved top-{top_k}: {result.get('retrieved_chunk_ids')}",
                f"First relevant rank: {result.get('first_relevant_rank')}",
            ]
        )

    errors = eval_results.get("errors") or []
    if errors:
        lines.append("")
        lines.append(f"Errors: {errors}")

    return "\n".join(lines)


if __name__ == "__main__":
    main()
