# Dynamic Planner Evaluation

This project treats the dynamic planner as frozen only after the deterministic
planner contract eval passes.

Run the gate:

```bash
python -m scripts.evaluate_dynamic_planner
pytest
```

The eval is offline and does not call OpenAI, arXiv, Hugging Face, Chroma, or the
network. It exercises the real `DynamicAgentRunner`, policy layer, tool executor,
observation normalization, state updates, planner view metadata, and finish
policy with queued tool responses.

Current frozen contracts:

- Existing indexed KB answer: probe `retrieve_evidence` first and finish from
  retrieved evidence.
- Missing KB answer: probe KB, observe zero evidence, discover a paper, prepare
  it, retrieve again, then finish.
- New paper request: discover first, resolve metadata, save, prepare, retrieve,
  then finish.
- Compare papers request: discover, prepare selected papers, retrieve evidence,
  then finish.

If any small planner, policy, tool schema, observation, or state update change
breaks these sequences, `scripts.evaluate_dynamic_planner` exits non-zero.
