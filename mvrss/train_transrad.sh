#!/usr/bin/env bash
# Run from inside the mvrss folder: cd mvrss && bash train_transrad.sh
# Train HyperRadar on CARRADA.
# 1. Download and extract CARRADA so that <DATA_DIR>/Carrada exists.
# 2. Set the dataset and log folders (written to config_files/config.ini).
# 3. Train. Checkpoints for resuming go to <CKPT_DIR>.
set -e
DATA_DIR=/path/to/datasets
LOG_DIR=$(pwd)/carrada_logs
CKPT_DIR=$(pwd)/checkpoints

python -m utils.set_paths --carrada "$DATA_DIR" --logs "$LOG_DIR"
python -u train.py --cfg ./config_files/HyperRadar.json --cp_store "$CKPT_DIR"
