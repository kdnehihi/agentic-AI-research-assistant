from __future__ import annotations

import argparse
from dataclasses import dataclass

from app.agent.state import AgentState
from app.tools.rag_answer_tools import answer_question_with_retrieval


NO_ANSWER_PHRASE = "I do not have enough evidence from the retrieved chunks to answer that."


@dataclass(frozen=True)
class AnswerEvalCase:
    query: str
    expected_no_answer: bool = False


DEFAULT_CASES = (
    AnswerEvalCase("What limitations are discussed?"),
    AnswerEvalCase("What method or approach does the paper use?"),
    AnswerEvalCase("What limitations and future work are discussed?"),
    AnswerEvalCase(
        "What GPU model and training budget were used?",
        expected_no_answer=True,
    ),
    AnswerEvalCase(
        "What exact F1 score does the paper report on GSM8K?",
        expected_no_answer=True,
    ),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate RAG answer behavior on hard and no-answer questions."
    )
    parser.add_argument(
        "--paper-id",
        default="arxiv:2505.18906v2",
        help="Paper id to evaluate.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=5,
        help="Number of retrieved chunks to put in answer context.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    state = AgentState(topic="rag answer eval", max_papers=args.top_k)

    print("===== RAG ANSWER EVAL =====")
    print(f"paper_id: {args.paper_id}")
    print(f"top_k: {args.top_k}")

    passed = 0
    for index, case in enumerate(DEFAULT_CASES, start=1):
        observation = answer_question_with_retrieval(
            state=state,
            query=case.query,
            paper_ids=(args.paper_id,),
            top_k=args.top_k,
        )
        model_refused = _model_refused(observation.get("answer", ""))
        has_citations = bool(observation.get("cited_chunk_ids"))
        pass_case = (
            model_refused
            if case.expected_no_answer
            else observation["status"] == "success" and has_citations and not model_refused
        )
        if pass_case:
            passed += 1

        print(f"\n===== CASE {index} =====")
        print(f"query: {case.query}")
        print(f"expected_no_answer: {case.expected_no_answer}")
        print(f"status: {observation['status']}")
        print(f"model_refused: {model_refused}")
        print(f"has_citations: {has_citations}")
        print(f"pass_case: {pass_case}")
        if observation["status"] != "success":
            print(f"error: {observation.get('error')}")
            continue

        print("answer:")
        print(observation["answer"])
        print("cited_chunks:")
        for evidence in observation["evidence_chunks"]:
            marker = "*" if evidence["chunk_id"] in observation["cited_chunk_ids"] else " "
            print(
                f"{marker} {evidence['evidence_id']} "
                f"chunk={evidence['chunk_id']} "
                f"section={evidence['section']} "
                f"rank={evidence['rank']}"
            )

    print("\n===== SUMMARY =====")
    print(f"passed: {passed}/{len(DEFAULT_CASES)}")


def _model_refused(answer: str) -> bool:
    normalized_answer = " ".join(answer.lower().split())
    return (
        NO_ANSWER_PHRASE.lower() in normalized_answer
        or (
            "do not have enough evidence" in normalized_answer
            and "retrieved chunks" in normalized_answer
        )
    )


if __name__ == "__main__":
    main()
