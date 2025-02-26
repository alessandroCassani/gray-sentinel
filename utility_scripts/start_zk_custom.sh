#!/bin/bash

# Base directory settings
ZK_HOME="/home/alessandro/PGFDS/zookeeper"
ZK_CONF="$ZK_HOME/conf"
ZK_DATA="$ZK_HOME/data"
ZK_LOGS="$ZK_HOME/logs"

# Create necessary directories
mkdir -p "$ZK_DATA"
mkdir -p "$ZK_LOGS"

# Ensure zoo.cfg exists
if [ ! -f "$ZK_CONF/zoo.cfg" ]; then
    echo "Creating zoo.cfg from sample..."
    cp "$ZK_CONF/zoo_sample.cfg" "$ZK_CONF/zoo.cfg"
    sed -i "s|dataDir=/tmp/zookeeper|dataDir=$ZK_DATA|g" "$ZK_CONF/zoo.cfg"
fi

# Build classpath without Legolas
CLASSPATH=""
for jar in $(find "$ZK_HOME/lib" -name "*.jar" | grep -v "legolas"); do
    CLASSPATH="$CLASSPATH:$jar"
done

# Set ZooKeeper specific properties
ZOOMAIN="org.apache.zookeeper.server.quorum.QuorumPeerMain"
ZOOCFG="$ZK_CONF/zoo.cfg"
ZOO_LOG_FILE="zookeeper-$USER-server-$(hostname).log"

echo "Starting ZooKeeper..."

# Start ZooKeeper directly with Java
nohup java \
    "-Dzookeeper.log.dir=$ZK_LOGS" \
    "-Dzookeeper.log.file=$ZOO_LOG_FILE" \
    "-Dzookeeper.root.logger=INFO,CONSOLE" \
    -XX:+HeapDumpOnOutOfMemoryError \
    -cp "$CLASSPATH" \
    "$ZOOMAIN" "$ZOOCFG" > "$ZK_LOGS/zookeeper.out" 2>&1 &

# Save PID
echo $! > "$ZK_DATA/zookeeper_server.pid"

# Check if ZooKeeper started
sleep 3
ZK_PID=$(cat "$ZK_DATA/zookeeper_server.pid")
if ps -p $ZK_PID > /dev/null; then
    echo "ZooKeeper started with PID: $ZK_PID"
else
    echo "Failed to start ZooKeeper. Check logs at $ZK_LOGS/zookeeper.out"
    exit 1
fi