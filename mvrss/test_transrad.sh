#!/usr/bin/env bash
# Run from inside the mvrss folder: cd mvrss && bash test_transrad.sh
# Evaluate a trained HyperRadar model on the CARRADA test split.
# Point --cfg at the config.json that training wrote into the log folder.
set -e
DATA_DIR=/path/to/datasets
LOG_DIR=$(pwd)/carrada_logs

python -m utils.set_paths --carrada "$DATA_DIR" --logs "$LOG_DIR"
python -u test.py --cfg "$LOG_DIR/carrada/TransRadar/TransRadar_0/config.json"
