#!/bin/bash

# Start time
start_time=$(date +%s)

# Run the three files (replace with your actual files/commands)
echo "Running dataset_train_impute_breast.py..."
python dataset_train_impute_breast.py

echo "Running exe_testing_breast_real.py..."
python exe_testing_breast_real.py

# End time
end_time=$(date +%s)

# Compute elapsed time
elapsed=$(( end_time - start_time ))

# Output total time taken
echo "Total execution time: $elapsed seconds"
