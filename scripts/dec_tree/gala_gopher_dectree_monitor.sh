#!/bin/bash
# Manual Process Selection for Gala-Gopher Monitoring
# This script shows all Java processes and lets you manually select which one to monitor

# Configuration
GOPHER_REST_API="http://localhost:9999"  # gala-gopher REST API endpoint
LOG_FILE="manual_monitor.log"

# Initialize log file
echo "=== Manual Process Selection for Gala-Gopher Log ===" > $LOG_FILE
echo "Started at: $(date)" >> $LOG_FILE

# Function to log messages
log() {
    local message=$1
    local timestamp=$(date '+%Y-%m-%d %H:%M:%S')
    echo "[$timestamp] $message" | tee -a $LOG_FILE
}

# Function to configure gala-gopher for a specific PID
configure_gala_gopher() {
    local pid=$1
    
    log "Configuring gala-gopher to monitor process (PID: $pid)"
    
    # Create gala-gopher probe configuration for the specific PID
    local config_file="/tmp/manual_probe_config.json"
    cat > $config_file << EOF
{
    "tcp": {
        "cmd": {
            "probe": ["tcp_rtt", "tcp_windows", "tcp_abnormal"]
        },
        "snoopers": {
            "pid": [{"pid": ${pid}}]
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
            "pid": [{"pid": ${pid}}]
        },
        "state": "running"
    },
    "system": {
        "cmd": {
            "probe": ["cpu", "mem", "io"]
        },
        "snoopers": {
            "pid": [{"pid": ${pid}}]
        },
        "state": "running"
    }
}
EOF

    log "Probe configuration created in $config_file"
    
    # Send configuration to gala-gopher
    log "Applying probe configuration to gala-gopher via REST API..."
    local response=$(curl -s -X POST -H "Content-Type: application/json" -d @$config_file $GOPHER_REST_API/configure)
    local curl_exit_code=$?
    
    if [ $curl_exit_code -eq 0 ]; then
        log "Successfully configured gala-gopher. Response: $response"
        return 0
    else
        log "Failed to configure gala-gopher. Exit code: $curl_exit_code, Response: $response"
        return 1
    fi
}

# Check if gala-gopher REST API is accessible
check_gala_gopher() {
    log "Checking if gala-gopher REST API is accessible at $GOPHER_REST_API"
    
    curl -s $GOPHER_REST_API > /dev/null
    if [ $? -ne 0 ]; then
        log "ERROR: gala-gopher REST API is not accessible at $GOPHER_REST_API"
        log "Make sure gala-gopher is running with the REST API enabled."
        return 1
    fi
    
    log "gala-gopher REST API is accessible"
    return 0
}

# Function to display Java process details
display_java_processes() {
    log "Searching for all Java processes..."
    
    echo "[DEBUG] Full Java processes (ps -ef | grep java):" | tee -a $LOG_FILE
    ps -ef | grep java | grep -v grep | tee -a $LOG_FILE
    echo "----------------------------------------------------"
    echo "PID   %CPU %MEM   COMMAND"
    echo "----------------------------------------------------"
    
    ps aux | grep java | grep -v grep | awk '{printf "%-6s %-5s %-5s   %s\n", $2, $3, $4, $11, $12, $13}' | tee -a $LOG_FILE
    
    echo "----------------------------------------------------"

    
    # Get more detailed info about specific Java processes
    log "Showing more details about potential Renaissance/Spark processes:"
    
    # Look for Renaissance processes
    local renaissance_pids=$(ps aux | grep -E "Renaissance|org.renaissance" | grep -v grep | awk '{print $2}')
    if [ -n "$renaissance_pids" ]; then
        log "Renaissance processes found:"
        for pid in $renaissance_pids; do
            echo "PID $pid details:" | tee -a $LOG_FILE
            ps -f -p $pid | tee -a $LOG_FILE
            echo "Command line: $(cat /proc/$pid/cmdline 2>/dev/null | tr '\0' ' ')" | tee -a $LOG_FILE
            echo "" | tee -a $LOG_FILE
        done
    fi
    
    # Look for Spark processes
    local spark_pids=$(ps aux | grep -E "spark" | grep -v grep | awk '{print $2}')
    if [ -n "$spark_pids" ]; then
        log "Spark processes found:"
        for pid in $spark_pids; do
            echo "PID $pid details:" | tee -a $LOG_FILE
            ps -f -p $pid | tee -a $LOG_FILE
            echo "Command line: $(cat /proc/$pid/cmdline 2>/dev/null | tr '\0' ' ' | cut -c 1-100)..." | tee -a $LOG_FILE
            echo "" | tee -a $LOG_FILE
        done
    fi
}

# Main execution
log "=== Manual Process Selection for Gala-Gopher Started ==="

# First check if gala-gopher is accessible
if ! check_gala_gopher; then
    log "Exiting due to gala-gopher accessibility issue"
    exit 1
fi

# Display all Java processes
display_java_processes

# Ask user to enter the PID to monitor
echo ""
echo "Please enter the PID of the process you want to monitor:"
read selected_pid

# Validate the PID
if ! [[ "$selected_pid" =~ ^[0-9]+$ ]]; then
    log "Invalid PID format: $selected_pid"
    exit 1
fi

# Check if PID exists
if ! ps -p $selected_pid > /dev/null; then
    log "Process with PID $selected_pid does not exist"
    exit 1
fi

# Show details of the selected process
log "Selected process (PID: $selected_pid):"
ps -f -p $selected_pid
echo "Command line: $(cat /proc/$selected_pid/cmdline 2>/dev/null | tr '\0' ' ')" | tee -a $LOG_FILE

# Confirm with user
echo ""
echo "Do you want to monitor this process with gala-gopher? (y/n)"
read confirmation

if [[ "$confirmation" =~ ^[Yy]$ ]]; then
    # Configure gala-gopher for this process
    configure_gala_gopher $selected_pid
    exit_code=$?
    
    if [ $exit_code -eq 0 ]; then
        log "Successfully set up monitoring for PID: $selected_pid"
    else
        log "Failed to set up monitoring for PID: $selected_pid"
        exit 1
    fi
else
    log "Monitoring setup cancelled by user"
    exit 0
fi

log "Monitoring setup complete. Gala-gopher is now collecting metrics from the process."