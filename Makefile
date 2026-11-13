PYTHON ?= python3
CONFIG ?= configs/train_pair.example.json
SECURITY_SCAN ?= security/OSV_SCAN_CURRENT.json

.PHONY: check validate train smoke sbom security release anonymous-release

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
	$(PYTHON) scripts/security_audit.py --output $(SECURITY_SCAN)

release:
	$(PYTHON) scripts/build_release.py --output ../LoRango_SDXL_Reproducibility_Bundle_v1.4.1.tar.gz

anonymous-release:
	$(PYTHON) scripts/build_anonymous_review.py --output ../LoRango_SDXL_Reproducibility_Bundle_v1.4.1_anonymous.tar.gz
