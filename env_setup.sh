#!/bin/bash
# chmod +x env_setup.sh


echo ">>> Setting workspace directory to current path..."
WORK_DIR=$PWD


echo ">>> Cleaning out any junk..."
rm -rf ~/.cache/pip


echo ">>> Creating custom temp folders..."
mkdir -p $WORK_DIR/pip_cache
mkdir -p $WORK_DIR/pip_tmp


echo ">>> Checking if Iris_env exists..."
if [ ! -d "Iris_env" ]; then
    echo ">>> Building fresh Iris_env..."
    python3 -m pip install --user virtualenv
    python3 -m virtualenv Iris_env
fi


echo ">>> Activating Iris_env..."
source Iris_env/bin/activate


echo ">>> Routing temporary files..."
export TMPDIR=$WORK_DIR/pip_tmp


if [ "$1" == "auto" ]; then
    echo ">>> [AUTO MODE] Step 1: Installing PyTorch 2.4.0 (Python 3.12 Compatible)..."
    pip install --cache-dir $WORK_DIR/pip_cache torch==2.4.0 torchvision==0.19.0 --index-url https://download.pytorch.org/whl/cu124
    
    echo ">>> [AUTO MODE] Step 2: Installing base requirements..."
    pip install --cache-dir $WORK_DIR/pip_cache -r requirements.txt pyod
    
    echo ">>> [AUTO MODE] Step 3: Installing PyTorch Graph libraries (No-Build-Isolation)..."
    pip install --cache-dir $WORK_DIR/pip_cache torch-scatter torch-sparse torch-geometric -f https://data.pyg.org/whl/torch-2.4.0+cu124.html --no-build-isolation
else
    echo ">>> [MANUAL MODE] Installing custom packages..."
    pip install --cache-dir $WORK_DIR/pip_cache "$@"
fi


echo ">>> Installation complete!"
