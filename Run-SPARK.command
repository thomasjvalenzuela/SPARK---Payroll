#!/bin/bash

# Change to the script directory
cd "$(dirname "$0")"

# Activate virtual environment
source .venv/bin/activate

# Run the SPARK application
streamlit run spark_app.py