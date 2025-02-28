#!/bin/bash

BASE_DIR=$(pwd)/..
ZK_HOME="${BASE_DIR}/zookeeper"
PID_FILE="/tmp/zookeeper.pid"

cd "${ZK_HOME}"
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