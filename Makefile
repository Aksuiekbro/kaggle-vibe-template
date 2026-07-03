# Kaggle Solver Template — Orchestration Commands

.PHONY: setup status submit share-start share-status share-end evaluate help

help:
	@echo "Kaggle Solver Template Commands:"
	@echo ""
	@echo "  make setup COMP=<slug>   Initialize workspace for a competition"
	@echo "  make status              Show submission registry status"
	@echo "  make evaluate AGENT=<name> FILE=<path>  Run local evaluation"
	@echo "  make submit AGENT=<name> FILE=<path> DESC=\"<text>\"  Score-gated submit"
	@echo "  make share-start         Start a sharing round"
	@echo "  make share-status        Show sharing round status"
	@echo "  make share-end           End sharing round"
	@echo ""

PYTHON ?= python3

setup:
	$(PYTHON) tools/setup.py --competition $(COMP)

status:
	$(PYTHON) tools/registry.py status

evaluate:
	$(PYTHON) tools/evaluate.py --agent $(AGENT) --file $(FILE)

submit:
	$(PYTHON) tools/submit.py --agent $(AGENT) --file $(FILE) --description "$(DESC)"

share-start:
	$(PYTHON) tools/share.py start

share-status:
	$(PYTHON) tools/share.py status

share-end:
	$(PYTHON) tools/share.py end
