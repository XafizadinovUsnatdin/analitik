#!/bin/bash

# Create virtual environment if it doesn't exist
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python -m venv venv
fi

# Activate virtual environment
source venv/bin/activate

# Install dependencies
echo "Installing dependencies..."
pip install -r requirements.txt

# Create necessary directories
mkdir -p data/forecasts models static/plots

# Train models if they don't exist
if [ ! "$(ls -A models)" ]; then
    echo "Training models..."
    python scripts/train_model.py
fi

# Generate forecasts if they don't exist
if [ ! "$(ls -A data/forecasts)" ]; then
    echo "Generating forecasts..."
    python scripts/forecast_model.py
fi

# Run the application
echo "Starting application..."
flask run --host=0.0.0.0

