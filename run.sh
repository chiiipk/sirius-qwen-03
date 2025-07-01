#!/bin/bash

# Default lambda values
LAMBDA_1=1
LAMBDA_2=1
LAMBDA_3=0.25

# Lambda_4 values for grid search
# LAMBDA_4_VALUES=(0.25 0.5 0.75 1.0 1.25 1.5 1.75 2.0)
LAMBDA_4_VALUES=(0.1 0.2 0.3 0.4 0.6 0.7 0.8 0.9)

TASK_NAME="FewRel"
NUM_K=5
TEMPERATURE=0.01
DISTANCE_THRESHOLD=0.1


echo "Starting grid search for lambda_4..."

for LAMBDA_4 in "${LAMBDA_4_VALUES[@]}"
do
  echo "----------------------------------------------------"
  echo "Running with: lambda_1=${LAMBDA_1}, lambda_2=${LAMBDA_2}, lambda_3=${LAMBDA_3}, lambda_4=${LAMBDA_4}"
  echo "----------------------------------------------------"
  python3 train.py \
    --task_name "${TASK_NAME}" \
    --num_k ${NUM_K} \
    --lambda_1 ${LAMBDA_1} \
    --lambda_2 ${LAMBDA_2} \
    --lambda_3 ${LAMBDA_3} \
    --lambda_4 ${LAMBDA_4} \
    --temperature ${TEMPERATURE} \
    --distance_threshold ${DISTANCE_THRESHOLD} >> "LAMBDA_4_${LAMBDA_4}.txt"


  echo "----------------------------------------------------"
  echo "Finished run with lambda_4=${LAMBDA_4}"
  echo "----------------------------------------------------"
done

echo "Grid search finished."
