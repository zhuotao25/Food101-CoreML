#!/bin/bash
set -e

# Download model if it doesn't exist
if [ ! -f "model4b.10-0.68.hdf5" ]; then
    echo "Downloading model..."
    curl -L -o model4b.10-0.68.hdf5 \
        https://s3.amazonaws.com/stratospark/food-101/model4b.10-0.68.hdf5
fi

# Create virtual environment with Python 3
if [ ! -d "env" ]; then
    echo "Creating virtual environment..."
    python3 -m venv env
fi

# Activate virtual environment
source env/bin/activate

# Install dependencies
echo "Installing dependencies..."
pip install --upgrade pip
pip install tensorflow keras h5py coremltools

# Run conversion
echo "Converting model to CoreML..."
python food101.py

# Deactivate
deactivate

echo "Conversion complete! Food101.mlmodel has been created."
