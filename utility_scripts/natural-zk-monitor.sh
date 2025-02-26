#!/bin/bash

BASE_DIR=$(pwd)/..

echo "building zookeeper first"

cd "${BASE_DIR}/zookeeper"
mvn clean install -DskipTests -Dmaven.test.skip=true
sleep 20

echo "starting zookeeper monitoring..."
ZK_HOME="${BASE_DIR}/zookeeper"

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