#!/bin/bash

# Exit the script if any command fails
set -e

# Install python packages
echo "Installing packages..."
pip install -U bert_score nltk readme-ready

# Get the directory of the current script
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

# Path to validate.py in the scripts folder
SCRIPT_PATH="$SCRIPT_DIR/validate.py"

# Check if validate.py exists in the scripts folder
if [[ ! -f "$SCRIPT_PATH" ]]; then
  echo "Error: validate.py not found in the scripts folder."
  exit 1
fi

# Set the environment variables
export OPENAI_API_KEY="dummy"
export HF_TOKEN="dummy"

# Run the Python script
echo "Running validate.py:"
python "$SCRIPT_PATH"