#!/bin/bash

TARGET_SEED_1=2
TARGET_DATASET="cifar10"

export TORCH_COMPILE=0
export OMP_NUM_THREADS=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TF_CPP_MIN_LOG_LEVEL=3
unset CUDA_VISIBLE_DEVICES 

echo ""
echo "--- Running Global Experiments ---"
echo " Target Dataset : $TARGET_DATASET"
echo " Seed(s)        : $TARGET_SEED_1"
echo ""

source Iris_env/bin/activate
mkdir -p workspace/output
mkdir -p workspace/logs
cd workspace

echo "[1/2] Checking FedAvg Baselines..."
python3 -u src/main.py --seed $TARGET_SEED_1 --defense fedavg --dataset $TARGET_DATASET > logs/fedavg_dataset_${TARGET_DATASET}_seed${TARGET_SEED_1}.log 2>&1

echo "[2/2] Running ALL Defenses Globally (TSFL, LASA, FLTrust, etc.)..."
python3 -u src/main.py --seed $TARGET_SEED_1 --defense "all" --dataset $TARGET_DATASET > logs/global_defenses_dataset_${TARGET_DATASET}_seed${TARGET_SEED_1}.log 2>&1

echo ""
echo " >>> Global pipeline completed! Check workspace/logs/global_defenses_dataset_${TARGET_DATASET}_seed${TARGET_SEED_1}.log"
echo ""