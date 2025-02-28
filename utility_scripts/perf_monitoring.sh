#!/bin/bash

ZK_PID=$1

if [ -z "$ZK_PID" ]; then
    echo "no PID found ..."
    exit 1
fi

echo "Monitoring ZooKeeper process with PID: $ZK_PID"

stats=$(perf stat -p "$ZK_PID" sleep 5 2>&1)

echo "$stats"