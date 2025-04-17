#!/bin/bash
# Configures gala-gopher to monitor metrics 

API_URL_BASE_INFO="http://localhost:9999/baseinfo"

API_URL_TCP="http://localhost:9999/tcp"

API_URL_IO="http://localhost:9999/io"

API_URL_PROC="http://localhost:9999/proc"

API_URL_THREAD="http://localhost:9999/tprofiling"

CONFIG_FILE="/etc/gala-gopher/probes.init"

# Function to enable a probe through the API
enable_probe() {
  local URL=$1
  local config=$2

  curl -s -X PUT "$URL" \
    --data "json=$config"
}


# Main script execution
echo "=== Configuring gala-gopher probes ==="

# 1. Enable system resource monitoring probes
enable_probe $API_URL_BASE_INFO '{"cmd":{"probe":["cpu", "mem", "fs", "host"]}, "state": "running"}'

# 2. Enable network monitoring probes TODO:need to apss proc id remember!!
#enable_probe $API_URL_TCP '{"cmd":{"probe":["tcp_rtt", "tcp_windows", "tcp_abnormal"]}, "snooper":[] , "state": "running"}'

#enable IO monitoring
enable_probe $API_URL_IO  '{"cmd": {"probe": ["io_trace", "io_err", "io_count"]},"state": "running"}'

# 4. Enable process monitoring
enable_probe $API_URL_PROC '{"cmd": {"probe": ["proc_syscall", "proc_io", "proc_fs"]},"state": "running"}'

enable_probe $API_URL_THREAD '{"cmd": {"probe": ["oncpu", "syscall_file", "syscall_net", "syscall_lock", "syscall_sched"]},"state": "running"}'

systemctl restart gala-gopher.service

echo "=== gala gopher configuration complete ==="

echo "starting grafana..."
docker run -d --network="host" --name=grafana grafana/grafana-oss
echo "grafana is running!"

cd ..
cd "external/prometheus"
 ./prometheus --config.file=prometheus.yml



# Check if metrics are collected
# echo "Checking if metrics are now available..."
# curl -s http://localhost:8888/metrics | grep -E '(cpu|mem|tcp|disk|io)' | head -10