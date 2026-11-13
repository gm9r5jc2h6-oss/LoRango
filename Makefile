PYTHON ?= python
CONFIG ?= configs/train_pair.example.json

.PHONY: check validate train smoke sbom security release

check:
	bash scripts/artifact_check.sh

validate:
	$(PYTHON) train_pair.py --config $(CONFIG) --validate-only

train:
	bash scripts/train.sh $(CONFIG)

smoke:
	$(PYTHON) scripts/run_smoke.py --config $(CONFIG) --output-dir /tmp/lorango_sdxl_smoke

sbom:
	$(PYTHON) scripts/generate_sbom.py --output security/sbom.cdx.json

security:
	$(PYTHON) scripts/security_audit.py --output security/OSV_SCAN_2026-11-07.json

release:
	$(PYTHON) scripts/build_release.py --output ../LoRango_SDXL_Training_Toolkit_v1.4.0.tar.gz
