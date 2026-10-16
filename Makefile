# Convenience wrappers. Everything here is also a plain docker compose command
# or a plain python invocation; nothing is hidden behind the Makefile.
.PHONY: help demo bench quick comparison test tune figures results lag qp estimator \
        missions local local-test sitl sitl-mission sitl-images sitl-smoke sitl-campaign \
        sitl-results clean

help:
	@echo "make demo         run everything and verify the artifacts (docker)"
	@echo "make quick        one seed, calm air only, no tests (docker)"
	@echo "make comparison   just the controller comparison (docker)"
	@echo "make test         just the test suite (docker)"
	@echo "make tune         re-tune both controllers, rewrite config/tuned_gains.yaml"
	@echo ""
	@echo "  without docker:"
	@echo "make local        the whole pipeline, using the local Python"
	@echo "make local-test   the test suite, using the local Python"
	@echo "make figures      regenerate the figures from results/runs.csv"
	@echo "make results      regenerate docs/RESULTS.md and docs/FAILSAFE_MATRIX.md"
	@echo "make lag          re-measure the closed-loop acceleration lag"
	@echo "make qp           verify and benchmark the MPC's QP solver"
	@echo "make estimator    measure what latency compensation is worth"
	@echo "make missions     validate every mission file"
	@echo ""
	@echo "make sitl          bring up PX4 v1.17.0 SIH (headless) in the background"
	@echo "make sitl-mission  fly facade_inspection against it"
	@echo "make sitl-smoke    the per-push SITL smoke flight, as CI flies it"
	@echo "make sitl-campaign every mission by both trackers and every failsafe row"
	@echo "make sitl-results  rebuild sitl/results/summary.json, docs and README table"
	@echo "make clean        remove generated results"

demo bench:
	docker compose run --rm bench

quick:
	docker compose run --rm quick

comparison:
	docker compose run --rm comparison

test:
	docker compose run --rm test

tune:
	docker compose run --rm tune

local:
	PYTHONPATH=src bash scripts/run_all.sh

local-test:
	PYTHONPATH=src python -m pytest -q

figures:
	PYTHONPATH=src python analysis/make_figures.py

results:
	PYTHONPATH=src python analysis/make_results.py

lag:
	PYTHONPATH=src python scripts/identify_lag.py

qp:
	PYTHONPATH=src python scripts/benchmark_qp.py

estimator:
	PYTHONPATH=src python scripts/benchmark_estimator.py

missions:
	PYTHONPATH=src python scripts/validate_missions.py

sitl:
	docker compose --profile sitl up -d px4

sitl-mission:
	docker compose --profile sitl run --rm sitl-mission

sitl-images:
	docker build -t mission-control-px4-sih:latest sitl/sih
	docker build --target sitl -t mission-control-sitl:latest .

sitl-smoke: sitl-images
	PYTHONPATH=src python scripts/run_sitl_campaign.py --campaign smoke --output results/sitl-smoke
	PYTHONPATH=src python analysis/make_sitl_results.py --flights results/sitl-smoke --check-only

sitl-campaign: sitl-images
	PYTHONPATH=src python scripts/run_sitl_campaign.py --campaign full

sitl-results:
	PYTHONPATH=src python analysis/make_sitl_results.py --check --write-readme

clean:
	rm -rf results/runs.csv results/comparison_meta.json results/logs
