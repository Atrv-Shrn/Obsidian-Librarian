"""Eval harnesses.

* :mod:`rag_eval`   — Ragas (non-LLM + LLM w/ ``glm-5.2:cloud`` judge) + LlamaIndex
  retrieval metrics over the golden set. Offline. No Langfuse.
* :mod:`agent_eval` — golden tasks (file assertions for writes; judged Q&A) + Langfuse.
"""