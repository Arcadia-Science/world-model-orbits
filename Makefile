.PHONY: lint
lint:
	ruff check --exit-zero .
	ruff format --check .

.PHONY: format
format:
	ruff check --fix .
	ruff format .

.PHONY: pre-commit
pre-commit:
	pre-commit run --all-files

.PHONY: test
test:
	pytest -v .

.PHONY: execute
execute:
	# Project-wide render without executing code cells.
	# Instead, rely on pre-computed results present in `_freeze/`.
	quarto render

	# Now, render `index.ipynb` with code execution.
	# This will populate `_freeze/index/` with pre-computed results.
	quarto render index.ipynb --execute

.PHONY: draft
draft:
	# Export notebook prose + figures to draft_export/*.docx for text editing.
	# Edit the prose in Word, then paste it back into the notebook cell named
	# by the [[cell N]] marker above each block.
	python export_draft.py index.ipynb

.PHONY: preview
preview:
	quarto preview

.PHONY: execute-demo
execute-demo:
	# This command is only necessary when changes are made to the demo notebook
	# for development of the notebook pub template.
	quarto render examples/demo.ipynb --execute

.PHONY: update-theme
update-theme:
	quarto update extension Arcadia-Science/notebook-pub-theme --no-prompt
