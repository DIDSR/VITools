#!/bin/sh
PY_ENV=$1
CMD_FILE=$2
INPUT=${3:-input.csv}
SIM_NAME=VIT-BATCH_$(date +'%m-%d-%Y_%H-%M')
LOG_DIR=../.VIT/logs/$SIM_NAME

COUNT=$(cat $INPUT | wc -l)
COUNT=$(($COUNT - 1))
echo Running $COUNT simulation conditions

absolute_path=$(realpath "$LOG_DIR")
echo "See logs at $absolute_path"

START_TASK=1
END_TASK=$COUNT
qsub -N $SIM_NAME -t $START_TASK-$END_TASK $CMD_FILE $PY_ENV  $LOG_DIR $COUNT $INPUT
