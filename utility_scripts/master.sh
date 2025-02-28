#!/bin/bash

BASE_DIR=$(pwd)

echo "build and run zookeeper..."
ZK_PID=$(./runZk.sh)
echo "$ZK_PID"

if [ -z "$ZK_PID" ]; then
    echo "failed to get zookeeper PID"
    exit 1
fi

echo "zookeeper running with PID: $ZK_PID"

echo "starting perf monitoring ..."

PERF_STATS=$(./perf_monitoring.sh "$ZK_PID")

#TODO INFLUX