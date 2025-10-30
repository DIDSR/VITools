#!/bin/sh
PY_ENV=$1
CMD_FILE=$2
INPUT=${3:-input.csv}
SIM_NAME=VIT-BATCH_$(date +'%m-%d-%Y_%H-%M')
LOG_DIR=${4:-"./.VIT/logs/$SIM_NAME"}

COUNT=$(cat $INPUT | wc -l)
COUNT=$(($COUNT - 1))
FIRST_ID=$(awk -F ',' 'NR==2 {split($1, a, "_"); print a[2]+0}' "$INPUT")
LAST_ID=$(awk -F ',' 'END {split($1, a, "_"); print a[2]+0}' "$INPUT")
echo Running $COUNT simulation conditions: $FIRST_ID-$LAST_ID

absolute_path=$(realpath "$LOG_DIR")
echo "See logs at $absolute_path"

START_TASK=$FIRST_ID
END_TASK=$LAST_ID
qsub -N $SIM_NAME -t $START_TASK-$END_TASK $CMD_FILE $PY_ENV $LOG_DIR $INPUT
