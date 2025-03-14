#!/bin/bash

# Remove any existing faban_client containers (running or stopped)
docker rm -f faban_client 2>/dev/null

# Check if all required services are running
if ! docker ps | grep -q database_server; then
    echo "ERROR: database_server is not running"
    exit 1
fi

if ! docker ps | grep -q memcache_server; then
    echo "ERROR: memcache_server is not running"
    exit 1
fi

if ! docker ps | grep -q web_server; then
    echo "ERROR: web_server is not running"
    exit 1
fi

echo "All required services are running. Starting benchmark client..."

# Use docker inspect to find the network of the web_server container
NETWORK_NAME=$(docker inspect web_server -f '{{range $net,$v := .NetworkSettings.Networks}}{{$net}}{{end}}')

if [ -z "$NETWORK_NAME" ]; then
    echo "ERROR: Cannot determine network for web_server"
    exit 1
fi

echo "Using network: $NETWORK_NAME"

# Run the benchmark client
docker run --rm --name=faban_client --network=${NETWORK_NAME} cloudsuite/web-serving:faban_client web_server 10 --oper=run --steady=300

if [ $? -eq 0 ]; then
    echo "Benchmark completed successfully"
else
    echo "ERROR: Benchmark failed"
fi