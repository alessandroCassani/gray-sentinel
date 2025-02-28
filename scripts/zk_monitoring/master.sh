#!/bin/bash

version=$1

BASE_DIR=$(pwd)
PID_FILE="/tmp/zookeeper.pid"

cd "$BASE_DIR/.."
./runInfluxDb.sh

if [[ "$version" == "natural" ]]; then
    echo "build and run zookeeper..."
    cd "$BASE_DIR"
    ZK_PID=$(./runZk.sh)
    
    if [ ! -f "$PID_FILE" ]; then
        echo "failed to get zookeeper PID"
        exit 1
    fi
    
    ZK_PID=$(cat "$PID_FILE")
    echo "ZK_PID received in master.sh: $ZK_PID"
    
    cd ..
    ./perf_monitoring.sh
else
    cd "$BASE_DIR"
    ZK_PID=$(./runZk.sh)

    # Start performance monitoring
    cd ".."
    ./perf_monitoring.sh 
    
    ZK_DIR="../../../zookeeper" 
    ZK_VERSION="3.6.2"
    # Step 1: Instrument ZooKeeper with Legolas
    echo "Instrumenting ZooKeeper with Legolas..."
    cd "../legolas/scripts/analysis"
    ./analyze-zookeeper.sh -i "$ZK_DIR" "$ZK_VERSION"
    
    cd "../experiment"
    # Step 2: Setup the experiment
    echo "Setting up Legolas experiment..."
    ./setup-legolas-zookeeper.sh "$ZK_DIR" "$ZK_VERSION"

    # Step 3: Start the experiment
    echo "Starting Legolas fault injection experiment..."
    
    ./start-legolas-zookeeper.sh 2>&1 | tee workspace/legolas-zk/result.txt
fi
