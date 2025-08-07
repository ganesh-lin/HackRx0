#!/bin/bash

# HackRX API Startup Script
echo "Starting HackRX Policy Analysis API..."

# Check if .env file exists
if [ ! -f ".env" ]; then
    echo "Error: .env file not found. Please create one with your configuration."
    exit 1
fi

# Load environment variables
source .env

# Check required environment variables
required_vars=("API_TOKEN" "DATABASE_URL" "HF_TOKEN" "PINECONE_API_KEY")
for var in "${required_vars[@]}"; do
    if [ -z "${!var}" ]; then
        echo "Error: $var is not set in .env file"
        exit 1
    fi
done

echo "Environment variables verified ✓"

# Create logs directory
mkdir -p logs

# Install dependencies if needed
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python -m venv venv
fi

echo "Activating virtual environment..."
source venv/bin/activate

echo "Installing/updating dependencies..."
pip install -r requirements.txt

echo "Starting the API server..."
cd app
python -m uvicorn main:app --host 0.0.0.0 --port 8000 --reload
