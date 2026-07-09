from app.llm.fake_llm import FakeLLMClient


def test_fake_llm_client_generate():
    client = FakeLLMClient()

    # Test the generate method with different prompts
    summary_prompt = "Please summarize the following papers."
    ranking_prompt = "Please rank the following papers."
    report_prompt = "Please generate a report from the following papers."
    generic_prompt = "This is a generic prompt."

    summary_response = client.generate(summary_prompt)
    ranking_response = client.generate(ranking_prompt)
    report_response = client.generate(report_prompt)
    generic_response = client.generate(generic_prompt)

    # Assert that the responses are as expected
    assert summary_response == "This is a fake summary of the papers."
    assert ranking_response == "This is a fake ranking of the papers."
    assert report_response == "This is a fake report generated from the papers."
    assert generic_response == "This is a generic fake response from the LLM."
