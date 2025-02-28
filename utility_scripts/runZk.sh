#!/bin/bash

BASE_DIR=$(pwd)/..

echo "building zookeeper first"

cd "${BASE_DIR}/zookeeper"
mvn clean install -DskipTests -Dmaven.test.skip=true

ZK_HOME="${BASE_DIR}/zookeeper"

ZK_PID=$(pgrep -f QuorumPeerMain)


./zkServer.sh start
sleep 5
ZK_PID=$(pgrep -f QuorumPeerMain) 
if [ -z "$ZK_PID" ]; then
    echo "ERROR: Failed to start ZooKeeper. Check logs."
    exit 1
fi
echo "ZooKeeper started with PID: $ZK_PID"

echo "$ZK_PID"