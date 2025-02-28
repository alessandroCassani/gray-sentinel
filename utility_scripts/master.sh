#!/bin/bash

BASE_DIR=$(pwd)
PID_FILE="/tmp/zookeeper.pid"

echo "build and run zookeeper..."
ZK_PID=$(./runZk.sh)

if [ -z "$PID_FILE" ]; then
    echo "failed to get zookeeper PID"
    exit 1
fi

ZK_PID=$(cat "$PID_FILE")
echo "ZK_PID received in master.sh: $ZK_PID"

./perf_monitoring.sh 

#TODO INFLUX