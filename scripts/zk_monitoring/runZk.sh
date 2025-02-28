#!/bin/bash

ROOT_DIR=$(pwd)/../..
PID_FILE="/tmp/zookeeper.pid"

cd "$ROOT_DIR"
cd "zookeeper"
mvn clean install -DskipTests -Dmaven.test.skip=true

cd "bin"

./zkServer.sh restart
sleep 5
ZK_PID=$(pgrep -f QuorumPeerMain) 
if [ -z "$ZK_PID" ]; then
    echo "ERROR: Failed to start ZooKeeper. Check logs."
    exit 1
fi

echo "$ZK_PID" > "$PID_FILE"