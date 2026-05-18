#!/bin/bash
set -e
PY=.venv/bin/python
echo "=== [$(date)] coverage-scan ==="
$PY scripts/build_satellite_dataset.py coverage-scan
echo "=== [$(date)] search ==="
$PY scripts/build_satellite_dataset.py search
echo "=== [$(date)] build ==="
$PY scripts/build_satellite_dataset.py build --workers 8
echo "=== [$(date)] satellite DONE ==="
