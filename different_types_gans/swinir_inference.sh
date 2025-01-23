#!/bin/bash

# Path to the Python script
SCRIPT_PATH="/home/rishabh.mondal/Brick-Kilns-project/ijcai_2025_kilns/different_types_gans/swinir_inference.py"

# Log file path
LOG_FILE="/home/rishabh.mondal/Brick-Kilns-project/ijcai_2025_kilns/different_types_gans/test_bihar_swinir_inference_4x.log"

# Command to run the Python script and write output to the log file
python $SCRIPT_PATH > $LOG_FILE 2>&1

# Print a message to indicate the script has finished
echo "Script execution completed. Logs are saved in $LOG_FILE"
