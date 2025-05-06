#!/bin/bash
# Configures gala-gopher to monitor metrics for Spring PetClinic containers

API_URL_BASE_INFO="http://localhost:9999/baseinfo"
API_URL_TCP="http://localhost:9999/tcp"
API_URL_IO="http://localhost:9999/io"
API_URL_PROC="http://localhost:9999/proc"
API_URL_THREAD="http://localhost:9999/tprofiling"

enable_probe() {
    local URL=$1
    local config=$2
    echo "Configuring probe at $URL"
    curl -s -X PUT "$URL" --data "json=$config"
    echo ""
}

echo "=== Configuring gala-gopher probes ==="

# 1. Enable system resource monitoring probes
enable_probe $API_URL_BASE_INFO '{"cmd":{"probe":["cpu", "mem", "fs", "host"]}, "state": "running"}'

# 2. Enable network monitoring probes
enable_probe $API_URL_TCP '{"cmd":{"probe":["tcp_rtt", "tcp_windows", "tcp_abnormal"]}, "snoopers":{"proc_name":[{"comm":"java"}]}, "state": "running"}'

# 3. Enable IO monitoring
enable_probe $API_URL_IO '{"cmd":{"probe":["io_trace", "io_err", "io_count"]}, "state": "running"}'

# 4. Enable process monitoring
enable_probe $API_URL_PROC '{"cmd":{"probe":["proc_syscall", "proc_io", "proc_fs"]}, "state": "running"}'

# 5. Enable thread profiling
enable_probe $API_URL_THREAD '{"cmd":{"probe":["oncpu", "syscall_file", "syscall_net", "syscall_lock", "syscall_sched"]}, "state": "running"}'

echo "Checking current probe configuration..."
curl -s http://localhost:9999

# Try to find information about Docker/container monitoring capabilities
echo "Searching for container monitoring options..."
curl -s http://localhost:8888/metrics | grep -E 'container|docker' | head -5

# Restart gala-gopher service
echo "Restarting gala-gopher service..."
sudo systemctl restart gala-gopher.service

echo "=== gala-gopher configuration complete ==="

# Check if metrics are being collected
echo "Checking if metrics are now available..."
sleep 5  # Give some time for the service to start generating metrics
curl -s http://localhost:8888/metrics | grep -E '(cpu|mem|tcp|disk|io)' | head -10

# Check if specific metrics for Java containers are being collected
echo "Checking for Java process metrics..."
curl -s http://localhost:8888/metrics | grep -E 'java|jvm' | head -5

echo "Checking for TCP metrics related to your containers..."
CONTAINER_IDS=$(sudo docker ps -q)
for ID in $CONTAINER_IDS; do
    NAME=$(sudo docker inspect --format='{{.Name}}' $ID | sed 's/\///')
    echo "Looking for metrics for container: $NAME"
    curl -s http://localhost:8888/metrics | grep -E "$NAME|$(echo $ID | cut -c1-12)" | head -3
done