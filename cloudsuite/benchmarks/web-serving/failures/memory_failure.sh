#!/bin/bash

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

BLADE_PATH=$"/tmp/chaosblade-1.7.4/blade"

install_chaosblade(){

    local $container="$1"

    if ! docker exec "$container" [ -f "$BLADE_PATH" ] 2>dev/null; then
        echo "Installing ChaosBlade in $container..."

        # Install ChaosBlade inside the container
        docker exec "$container" bash -c "wget https://github.com/chaosblade-io/chaosblade/releases/download/v1.7.4/chaosblade-1.7.4-linux-amd64.tar.gz -P /tmp/"
        docker exec "$container" bash -c "cd /tmp && tar -xf chaosblade-1.7.4-linux-amd64.tar.gz && chmod +x /tmp/chaosblade-1.7.4/blade"
    fi
}


run_cpu_stress(){
    local $container=$1
    install_chaosblade "$container"

    echo "Injecting memory stress into $container..."
    local exp_id
    exp_id=$(docker exec "$container" "$BLADE_PATH" create mem load --mode ram --mem-percent $PERCENT)

    sleep "$DURATION"

    echo "stopping memory stress experiment..."

    docker exec "$container" "$BLADE_PATH" destroy "$exp_id"
}

run_cpu_stress "$TARGET"