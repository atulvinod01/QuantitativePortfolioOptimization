#!/bin/bash

# Navigate to the script's directory
cd "$(dirname "$0")"

# Check if Python 3 is installed
if ! command -v python3 &> /dev/null; then
    echo "Python 3 is not installed. Please install it first."
    exit 1
fi

# Create virtual environment if it doesn't exist
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv venv
fi

# Activate virtual environment
source venv/bin/activate

# Install dependencies
echo "Checking and installing dependencies..."
pip install -r requirements.txt

# Run the application
echo "Starting Quantitative Portfolio Optimization Engine..."
streamlit run app.py
