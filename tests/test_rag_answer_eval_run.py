from scripts.rag_answer_eval_run import _model_refused


def test_model_refused_accepts_exact_no_answer_phrase():
    assert _model_refused(
        "I do not have enough evidence from the retrieved chunks to answer that."
    )


def test_model_refused_accepts_minor_wording_variation():
    assert _model_refused(
        "I do not have enough evidence from the retrieved chunks to answer what GPU model was used."
    )


def test_model_refused_rejects_grounded_answer():
    assert not _model_refused("The paper discusses external validity [E1].")
