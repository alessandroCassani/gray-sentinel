#!/bin/bash

version=$1

BASE_DIR=$(pwd)
PID_FILE="/tmp/zookeeper.pid"
ZK_VERSION="3.6.2"

cd "$BASE_DIR/.."
./runInfluxDb.sh

if [[ "$version" == "natural" ]]; then
    echo "build and run zookeeper..."
    
    #build zookeeper
    cd "../zokeeper"
    mvn clean install -DskipTests -Dmaven.test.skip=true

    cd "../scripts/zk"
    ./runZk.sh
    
    if [ ! -f "$PID_FILE" ]; then
        echo "failed to get zookeeper PID"
        exit 1
    fi
    
    ZK_PID=$(cat "$PID_FILE")
    echo "ZK_PID received in master.sh: $ZK_PID"
    
    cd ..
    ./perf_monitoring.sh
else
    cd "../zookeeper"
    #build zookeeper
    mvn clean install -DskipTests -Dmaven.test.skip=true

    ZK_DIR="../../../zookeeper"    

    # Instrument ZooKeeper with Legolas
    echo "Instrumenting ZooKeeper with Legolas..."
    cd "../legolas/scripts/analysis"
    ./analyze-zookeeper.sh -i "$ZK_DIR" "$ZK_VERSION"
    
    #Setup the experiment
    cd "../experiment"
    echo "Setting up Legolas experiment..."
    ./setup-legolas-zookeeper.sh "$ZK_DIR" "$ZK_VERSION"

    #start RMI Legolas
    if netstat -tuln | grep ":1099 " > /dev/null; then
        echo "RMI registry is already running on port 1099"
    else
        echo "Starting RMI registry..."
        cd "$LEGOLAS_DIR"
        ./bin/legolas.sh rmi 1099
        sleep 2  # Give RMI registry time to start
    fi


    # Set maxTrials to a very large number to keep orchestrator available
    if [ -f "workspace/legolas-zk.properties" ]; then
        sed -i 's/maxTrials=.*/maxTrials=999999999/' "workspace/legolas-zk.properties"
    fi

    # Start the experiment  
    cd "scripts/experiment"
    echo "Starting Legolas fault injection experiment..."

    ./start-legolas-zookeeper.sh 2>&1 | tee workspace/legolas-zk/result.txt

    
    cd "../../../scripts/zk"
    ./runZk.sh

    # Start performance monitoring
    cd ".."
    ./perf_monitoring.sh 
fi
