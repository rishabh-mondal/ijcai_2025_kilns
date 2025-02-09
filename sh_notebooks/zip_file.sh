#!/bin/bash

# Path to the Python script
SCRIPT_PATH="/home/rishabh.mondal/Brick-Kilns-project/ijcai_2025_kilns/sandbox/zip.py"

# Log file path
LOG_FILE="/home/rishabh.mondal/Brick-Kilns-project/ijcai_2025_kilns/chattisgarh.log"

# Command to run the Python script in the background and write output to the log file
nohup python $SCRIPT_PATH > $LOG_FILE 2>&1 &

# Print a message to indicate the script is running in the background
echo "Script is running in the background. Logs are saved in $LOG_FILE"
