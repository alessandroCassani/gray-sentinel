#!/bin/bash
# This script finds the Spark decision tree process and configures gala-gopher to monitor it

# Configuration
GOPHER_REST_API="http://localhost:9999"  # gala-gopher REST API endpoint

# First, find the Spark decision tree process
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PID=$($SCRIPT_DIR/find_dectree_process.sh | tail -1)

if [[ ! $PID =~ ^[0-9]+$ ]]; then
    echo "Failed to find a valid Spark decision tree process PID"
    exit 1
fi

echo "Configuring gala-gopher to monitor Spark decision tree process (PID: $PID)"

# Create gala-gopher probe configuration for the specific PID
cat > /tmp/dectree_probe_config.json << EOF
{
    "tcp": {
        "cmd": {
            "probe": ["tcp_rtt", "tcp_windows", "tcp_abnormal"]
        },
        "snoopers": {
            "pid": [{"pid": ${PID}}]
        },
        "params": {
            "report_event": 1
        },
        "state": "running"
    },
    "jvm": {
        "cmd": {
            "probe": ["jvm_gc", "jvm_memory", "jvm_thread"]
        },
        "snoopers": {
            "pid": [{"pid": ${PID}}]
        },
        "state": "running"
    },
    "system": {
        "cmd": {
            "probe": ["cpu", "mem", "io"]
        },
        "snoopers": {
            "pid": [{"pid": ${PID}}]
        },
        "state": "running"
    }
}
EOF

# Send configuration to gala-gopher
echo "Applying probe configuration to gala-gopher..."
curl -X POST -H "Content-Type: application/json" -d @/tmp/dectree_probe_config.json $GOPHER_REST_API/configure

if [ $? -eq 0 ]; then
    echo "Successfully configured gala-gopher to monitor process PID: $PID"
    echo "Metrics will be available through gala-gopher"
else
    echo "Failed to configure gala-gopher. Make sure it's running and the REST API is accessible."
    exit 1
fi