# Convenience wrappers. Everything here is also a plain docker compose command
# or a plain python invocation; nothing is hidden behind the Makefile.
.PHONY: help demo bench quick comparison test tune figures results lag qp estimator \
        missions local local-test sitl sitl-mission clean

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
	@echo "make sitl         bring up PX4 SITL with Gazebo"
	@echo "make sitl-mission fly a mission against PX4 SITL"
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
	docker compose --profile sitl up px4

sitl-mission:
	docker compose --profile sitl run --rm sitl-mission

clean:
	rm -rf results/runs.csv results/comparison_meta.json results/logs
