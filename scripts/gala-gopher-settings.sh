#!/bin/bash
# Configures gala-gopher to monitor metrics 

API_URL="http://localhost:9999/conf"
CONFIG_FILE="/etc/gala-gopher/probes.init"

# Function to enable a probe through the API
enable_probe() {
  local probe=$1
  local config=$2
  
  echo "Enabling $probe probe..."
  curl -s -X POST -H "Content-Type: application/json" -d "$config" $API_URL/$probe
  echo
}

# Function to update the config file for persistent configuration
update_config_file() {
  echo "Updating config file for persistent configuration..."
  
  # Backup the existing file
  cp $CONFIG_FILE ${CONFIG_FILE}.bak
  
  # Create new configuration focusing on GrayScope metrics
  cat > $CONFIG_FILE << EOF
# System metrics (CPU, memory utilization, etc.)
baseinfo {"cmd":{"probe":["cpu", "mem", "fs", "host"]}, "state": "running"}

# Network metrics (retransmissions, packet loss, etc.)
tcp {"cmd":{"probe":["tcp_rtt", "tcp_windows", "tcp_abnormal"]}, "state": "running"}
socket_trace {"cmd":{"probe":["socket_trace"]}, "state": "running"}

# Disk metrics (I/O, latency, error rates)
io_events {"cmd":{"probe":["io_events"]}, "state": "running"}
ioprobe {"cmd":{"probe":["ioprobe"]}, "state": "running"}

# Process metrics (scheduling, system call frequencies)
taskprobe {"cmd":{"probe":["taskprobe"]}, "state": "running"}

# Kernel service level indicators
ksli {"cmd":{"probe":["ksli"]}, "state": "running"}

# Memory subsystem metrics
pagecache {"cmd":{"probe":["pagecache"]}, "state": "running"}
EOF

  echo "Configuration file updated. Backup saved as ${CONFIG_FILE}.bak"
}

# Main script execution
echo "=== Configuring gala-gopher probes ==="

# 1. Enable system resource monitoring probes
enable_probe "baseinfo" '{"cmd":{"probe":["cpu", "mem", "fs", "host"]}, "state": "running"}'

# 2. Enable network monitoring probes
enable_probe "tcp" '{"cmd":{"probe":["tcp_rtt", "tcp_windows", "tcp_abnormal"]}, "state": "running"}'
enable_probe "socket_trace" '{"cmd":{"probe":["socket_trace"]}, "state": "running"}'

# 3. Enable I/O monitoring probes
enable_probe "io_events" '{"cmd":{"probe":["io_events"]}, "state": "running"}'
enable_probe "ioprobe" '{"cmd":{"probe":["ioprobe"]}, "state": "running"}'

# 4. Enable process monitoring
enable_probe "taskprobe" '{"cmd":{"probe":["taskprobe"]}, "state": "running"}'

# 5. Enable kernel service level indicators
enable_probe "ksli" '{"cmd":{"probe":["ksli"]}, "state": "running"}'

# 6. Enable memory subsystem monitoring
enable_probe "pagecache" '{"cmd":{"probe":["pagecache"]}, "state": "running"}'

# 7. Update config file for persistence
update_config_file

echo "=== Configuration complete ==="
echo "To make changes persistent, restart gala-gopher with:"
echo "systemctl restart gala-gopher.service"

# Check if metrics are collected
echo "Checking if metrics are now available..."
curl -s http://localhost:8888/metrics | grep -E '(cpu|mem|tcp|disk|io)' | head -10