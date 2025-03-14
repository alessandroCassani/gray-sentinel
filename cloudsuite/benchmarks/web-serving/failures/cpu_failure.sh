#!/bin/bash

# CPU failure injection for CloudSuite web-serving benchmark
# Usage: ./cpu-failure.sh TARGET PERCENT DURATION
#   TARGET: db_server, memcached_server, web_server
#   PERCENT: CPU percentage to consume (1-100)
#   DURATION: Duration in seconds

if [ $# -lt 3 ]; then
    echo "Usage: $0 <target> <percent> <duration>"
    echo "  target: db_server, memcached_server, web_server"
    echo "  percent: CPU percentage (1-100)"
    echo "  duration: Duration in seconds"
    exit 1
fi

TARGET=$1
PERCENT=$2
DURATION=$3
BLADE_PATH="/tmp/chaosblade-1.7.4/blade"


install_chaosblade(){
    local container="$1"
    
    if ! docker exec "$container" [ -f "$BLADE_PATH" ] 2>/dev/null; then
        echo "Installing ChaosBlade in $container..."
        
        # Install ChaosBlade inside the container
        docker exec "$container" bash -c "wget https://github.com/chaosblade-io/chaosblade/releases/download/v1.7.4/chaosblade-1.7.4-linux-amd64.tar.gz -P /tmp/"
        docker exec "$container" bash -c "cd /tmp && tar -xf chaosblade-1.7.4-linux-amd64.tar.gz && chmod +x /tmp/chaosblade-1.7.4/blade"
    fi
}


run_cpu_stress(){
    local container="$1"
    install_chaosblade "$container"

    echo "Injecting CPU stress into $container..."

    # Create CPU load experiment using ChaosBlade
    local exp_id
    exp_id=$(docker exec "$container" "$BLADE_PATH" create cpu load --cpu-percent "$PERCENT" | grep -o '"result":"[^"]*"' | sed 's/"result":"//;s/"//g')

    echo "Waiting for $DURATION seconds..."
    sleep "$DURATION"

    echo "Stopping CPU stress on $container..."
    docker exec "$container" "$BLADE_PATH" destroy "$exp_id"
}


run_cpu_stress "$TARGET"
