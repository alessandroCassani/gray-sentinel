#!/bin/bash

echo "starting zookeeper monitoring..."

BASE_DIR=$(pwd)/..
ZK_HOME="${BASE_DIR}/zookeeper"
DATA_DIR="${ZK_HOME}/monitoring/data"
METRICS_DIR="${ZK_HOME}//monitoring/metrics"
LOG_DIR="${ZK_HOME}/monitoring/logs"

mkdir -p "${DATA_DIR}"
mkdir -p "${METRICS_DIR}"
mkdir -p "${LOG_DIR}"

ZK_PID=$(pgrep -f QuorumPeerMain)
if [ -z "$ZK_PID" ]; then
    echo "ZooKeeper is not running. Starting ZooKeeper..."
    cd "${ZK_HOME}/bin"
    ./zkServer.sh start
    sleep 5
    ZK_PID=$(pgrep -f QuorumPeerMain)
    if [ -z "$ZK_PID" ]; then
        echo "ERROR: Failed to start ZooKeeper. Check logs."
        exit 1
    fi
    echo "ZooKeeper started with PID: $ZK_PID"
else
    echo "ZooKeeper already running with PID: $ZK_PID"
fi


PERF_OUTPUT=$(perf stat -p $ZK_PID sleep 10)
TIMESTAMP=$(date +%s%N)

echo "$PERF_OUTPUT"