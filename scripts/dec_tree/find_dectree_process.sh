#!/bin/bash
# This script identifies the Spark executor process that runs the decision tree algorithm
# It returns the PID that gala-gopher should attach to

# Process identification patterns specific to Spark decision tree
SPARK_EXECUTOR_PATTERN="org.apache.spark.executor.CoarseGrainedExecutorBackend"
DECISION_TREE_PATTERN="DecisionTreeClassifier"

# Function to find the Spark executor process running the decision tree algorithm
find_spark_executor_pid() {
    # First try to find the specific executor that mentions decision tree operations
    local pid=$(ps aux | grep -E "$SPARK_EXECUTOR_PATTERN" | grep -E "$DECISION_TREE_PATTERN" | grep -v "grep" | awk '{print $2}' | head -1)
    
    # If not found, look for any Spark executor as fallback
    if [ -z "$pid" ]; then
        pid=$(ps aux | grep -E "$SPARK_EXECUTOR_PATTERN" | grep -v "grep" | awk '{print $2}' | head -1)
    fi
    
    echo $pid
}

# Main execution
PID=$(find_spark_executor_pid)

if [ -n "$PID" ]; then
    echo "Found Spark decision tree process with PID: $PID"
    echo "Process details:"
    ps -p $PID -o pid,ppid,user,start,etime,cmd --no-headers
    echo "$PID"  # Output just the PID for easy capture in other scripts
else
    echo "No Spark decision tree process found currently running"
    exit 1
fi