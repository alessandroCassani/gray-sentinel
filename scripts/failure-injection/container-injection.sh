#!/bin/bash

DURATION=3000  #3000s == 50 miutes
ACTION=""
SERVICE=""
TYPE=""

if [[ "$ACTION" == "list" ]]; then
   blade status
   exit 0
fi

if [[ "$ACTION" == "clean" ]]; then
   blade destroy
   exit 0
fi


# Find container ID for the specified service
CONTAINER_ID=$(docker ps -qf name=$SERVICE)
[[ -z "$CONTAINER_ID" ]] && { echo "Error: Container '$SERVICE' not found"; exit 1; }

echo "Injecting $TYPE failure into $SERVICE for ${DURATION}s..."

case $TYPE in
   cpu)
       # CPU Load: Generates high CPU usage (80%) to simulate resource exhaustion
       # Use case: Test how application handles CPU bottlenecks and scaling policies
       blade create cri cpu fullload --cpu-percent 80 --timeout $DURATION --container-id $CONTAINER_ID
       ;;
       
   memory)
       # Memory Load: Consumes 80% of available RAM to simulate memory pressure
       # Use case: Test memory leak scenarios and out-of-memory handling
       blade create cri mem load --mode ram --mem-percent 80 --timeout $DURATION --container-id $CONTAINER_ID
       ;;
       
   network-loss)
       # Network Packet Loss: Drops 40% of network packets on eth0 interface
       # Use case: Test application resilience to unstable network connections
       blade create cri network loss --percent 40 --interface eth0 --timeout $DURATION --container-id $CONTAINER_ID
       ;;
       
   network-delay)
       # Network Latency: Adds 500ms delay (±200ms) to network traffic
       # Use case: Test timeout handling and performance under slow network conditions
       blade create cri network delay --time 500 --offset 200 --interface eth0 --timeout $DURATION --container-id $CONTAINER_ID
       ;;
       
   disk)
       # Disk I/O Stress: Performs intensive read/write operations (20MB) on root filesystem
       # Use case: Test application behavior under disk I/O bottlenecks
       blade create cri disk burn --read --write --path "/" --size 20 --timeout $DURATION --container-id $CONTAINER_ID
       ;;
esac