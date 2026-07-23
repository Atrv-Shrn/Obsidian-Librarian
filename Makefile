.PHONY: install install-evals sync query chat rag-serve api-serve eval eval-rag eval-agent docker-build docker-build-evals docker-up docker-down info

PY ?= python

install:
	$(PY) -m pip install -e .

# Ragas + datasets are an optional extra (~500MB); the slim runtime image omits them.
# Needed for `make eval-rag` to produce ragas scores rather than reporting them skipped.
install-evals:
	$(PY) -m pip install -e ".[evals]"

sync:
	$(PY) -m obsidian_librarian.cli sync

query:
	$(PY) -m obsidian_librarian.cli query "$(Q)"

chat:
	$(PY) -m obsidian_librarian.cli chat

rag-serve:
	$(PY) -m obsidian_librarian.cli rag-serve

api-serve:
	$(PY) -m obsidian_librarian.cli api-serve

info:
	$(PY) -m obsidian_librarian.cli info

eval: eval-rag
	@echo "Run 'make eval-agent' separately (requires Obsidian + optional Langfuse)."

eval-rag:
	$(PY) -m obsidian_librarian.evals.rag_eval

eval-agent:
	$(PY) -m obsidian_librarian.evals.agent_eval

docker-build:
	docker compose build

# Same image plus the ragas/datasets extra (~500MB larger) for running RAG evals in-container.
docker-build-evals:
	docker compose build --build-arg INSTALL_EXTRAS=[evals]

docker-up:
	docker compose up

docker-down:
	docker compose down