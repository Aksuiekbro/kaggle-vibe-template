# Kaggle Solver Template — Orchestration Commands

.PHONY: setup status submit share-start share-status share-end evaluate help \
	lint gate-check brief memory-stats memory-sweep memory-validate skills-test amend

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
	@echo "  make lint                Run the fake-practice linter (all agents)"
	@echo "  make gate-check AGENT=<name>  Check the predict-before-read gate"
	@echo "  make brief AGENT=<name>  Generate session-start BRIEF.md"
	@echo "  make memory-stats        Trust leaderboard of memory cards"
	@echo "  make memory-sweep        Stale out unrevalidated cards"
	@echo "  make memory-validate     Schema-check all memory cards"
	@echo "  make skills-test         Re-verify every skill's self_test"
	@echo "  make amend               Consolidation amendment proposals"
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

lint:
	$(PYTHON) tools/practice_lint.py

gate-check:
	$(PYTHON) tools/writeup.py check --agent $(AGENT)

brief:
	$(PYTHON) tools/brief.py generate --agent $(AGENT)

memory-stats:
	$(PYTHON) tools/memory_cli.py stats

memory-sweep:
	$(PYTHON) tools/memory_cli.py sweep

memory-validate:
	$(PYTHON) tools/memory_cli.py validate

skills-test:
	$(PYTHON) tools/skills.py test

amend:
	$(PYTHON) tools/memory_cli.py amend-proposals

exp-next:
	$(PYTHON) tools/scheduler.py next --agent $(AGENT)

exp-status:
	$(PYTHON) tools/scheduler.py status --agent $(AGENT)

verify-data:
	$(PYTHON) tools/verifiers.py columns --train $(TRAIN) --target $(TARGET) $(if $(TEST),--test $(TEST))

calibration:
	$(PYTHON) tools/calibration.py report --write

gym-status:
	$(PYTHON) tools/gym.py status

gym-report:
	$(PYTHON) tools/gym.py report

selfcheck:
	$(PYTHON) tools/selfcheck.py
