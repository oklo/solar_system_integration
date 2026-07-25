.PHONY: validate smoke-integrate smoke-frame clean

PYTHON ?= python

validate: smoke-integrate smoke-frame
	$(PYTHON) -m py_compile scripts/*.py

smoke-integrate:
	$(PYTHON) scripts/integrate_solar_system.py \
		--years 10000 \
		--cadence-years 5000 \
		--data-dir data \
		--kernel de440s.bsp \
		--outdir outputs/validation_10k \
		--progress-every 1

smoke-frame:
	$(PYTHON) scripts/render_preview_frame.py \
		--input outputs/validation_10k/animation_primitives_inner.npz \
		--snapshot-index 1 \
		--out outputs/validation_10k/preview_frame.png \
		--dpi 120

clean:
	rm -rf outputs/validation_10k build dist *.egg-info .pytest_cache .ruff_cache
	find . -type d -name __pycache__ -prune -exec rm -rf {} +

