#!/bin/bash

PID_FILE="/tmp/zookeeper.pid"

if [ -z "$PID_FILE" ]; then
    echo "no PID file found ..."
    exit 1
fi

ZK_PID=$(cat "$PID_FILE")

echo "Monitoring ZooKeeper process with PID: $ZK_PID"

perf stat -p "$ZK_PID" sleep 5 2>&1 | tee perf_output.log