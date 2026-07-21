import pytest

from app.agent.request_intent import (
    LLMRequestIntentClassifier,
    RequestIntent,
    parse_request_intent,
)


class FakeIntentLLM:
    def __init__(self, response):
        self.response = response
        self.prompts = []

    def generate(self, prompt, **kwargs):
        self.prompts.append(prompt)
        return self.response


def test_parse_request_intent_accepts_fenced_json():
    intent = parse_request_intent(
        """
        ```json
        {
          "task_type": "discovery_only",
          "topic": "mechanistic interpretability",
          "needs_retrieval": false,
          "needs_ingestion": false,
          "finish_condition": "paper_metadata",
          "confidence": 0.91,
          "rationale": "The user only asked to find papers."
        }
        ```
        """
    )

    assert intent == RequestIntent(
        task_type="discovery_only",
        topic="mechanistic interpretability",
        needs_retrieval=False,
        needs_ingestion=False,
        finish_condition="paper_metadata",
        confidence=0.91,
        rationale="The user only asked to find papers.",
    )


def test_llm_request_intent_classifier_uses_topic_as_dynamic_data():
    llm = FakeIntentLLM(
        """
        {
          "task_type": "factual_answer",
          "topic": "causal world models for robot planning",
          "needs_retrieval": true,
          "needs_ingestion": true,
          "finish_condition": "retrieved_evidence",
          "confidence": 0.88,
          "rationale": "The user asked for findings from paper content."
        }
        """
    )
    classifier = LLMRequestIntentClassifier(llm)

    intent = classifier.classify(
        "Find recent work on causal world models for robot planning and explain the findings."
    )

    assert intent.task_type == "factual_answer"
    assert intent.topic == "causal world models for robot planning"
    assert intent.needs_retrieval is True
    assert "causal world models" in llm.prompts[0]


def test_discovery_only_intent_canonicalizes_retrieval_flags():
    intent = parse_request_intent(
        """
        {
          "task_type": "discovery_only",
          "topic": "transformer",
          "needs_retrieval": true,
          "needs_ingestion": true,
          "probe_existing_kb_first": true,
          "finish_condition": "retrieved_evidence",
          "confidence": 0.82,
          "rationale": "The user asked to find papers."
        }
        """
    )

    assert intent.task_type == "discovery_only"
    assert intent.needs_retrieval is False
    assert intent.needs_ingestion is False
    assert intent.probe_existing_kb_first is False
    assert intent.finish_condition == "paper_metadata"


def test_parse_request_intent_rejects_invalid_task_type():
    with pytest.raises(ValueError):
        parse_request_intent(
            """
            {
              "task_type": "transformer",
              "topic": "transformer",
              "needs_retrieval": false,
              "needs_ingestion": false,
              "finish_condition": "paper_metadata"
            }
            """
        )
