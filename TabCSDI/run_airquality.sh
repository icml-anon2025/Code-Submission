#!/bin/bash

# Start time
start_time=$(date +%s)

# Run the three files (replace with your actual files/commands)

echo "Running exe_testing_waterpot_real.py..."
python TabCSDI/exe_testing_airquality.py

echo "Running airquality_ori.py..."
python TabCSDI/airquality_ori.py

# End time
end_time=$(date +%s)

# Compute elapsed time
elapsed=$(( end_time - start_time ))

# Output total time taken
echo "Total execution time: $elapsed seconds"
