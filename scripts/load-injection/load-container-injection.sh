#!/bin/bash
# Script to run JMeter load test and then call chaos injection script after a delay

# Use the full absolute path with leading slash
JMETER_BIN="/opt/apache-jmeter-5.6.3/bin/jmeter" 
JMETER_TEST_DIR="../../external/petclinic/spring-petclinic-api-gateway/src/test/jmeter"
JMX_FILE="petclinic_test_plan.jmx" # JMeter test plan file
RESULTS_FILE="results.jtl" # JMeter results file
CHAOS_SCRIPT="../failure-injection/container-injection.sh"
TARGET_SERVICE="customers-service" # Service to inject failure into
# Type of failure (cpu, mem, network-loss, network-delay, network-corrupted, 
# disk-read, disk-write, disk-read-write)
CHAOS_TYPE="network-loss" 
DELAY_SECONDS=240 # Wait time before injecting failure
CHAOS_DURATION=600 # Duration of the chaos experiment in seconds

if [ $# -ge 1 ]; then
    JMX_FILE=$1
fi
if [ $# -ge 2 ]; then
    DELAY_SECONDS=$2
fi
if [ $# -ge 3 ]; then
    TARGET_SERVICE=$3
fi
if [ $# -ge 4 ]; then
    CHAOS_TYPE=$4
fi
if [ $# -ge 5 ]; then
    CHAOS_DURATION=$5
fi

echo "====== Test Configuration ======"
echo "JMeter Path: $JMETER_BIN"
echo "JMeter Test Plan: $JMETER_TEST_DIR/$JMX_FILE"
echo "Target Service: $TARGET_SERVICE"
echo "Chaos Type: $CHAOS_TYPE"
echo "Delay: $DELAY_SECONDS seconds"
echo "Chaos Duration: $CHAOS_DURATION seconds"
echo "=============================="

if [ ! -f "$CHAOS_SCRIPT" ]; then
    echo "Error: Chaos script not found at $CHAOS_SCRIPT"
    exit 1
fi

echo "Starting JMeter load test..."
"$JMETER_BIN" -n -t "$JMETER_TEST_DIR/$JMX_FILE" -l $RESULTS_FILE &
JMETER_PID=$!
echo "JMeter started with PID: $JMETER_PID"

echo "Waiting $DELAY_SECONDS seconds before injecting failure..."
sleep $DELAY_SECONDS

echo "Executing chaos injection script..."
"$CHAOS_SCRIPT" -s $TARGET_SERVICE -t $CHAOS_TYPE -d $CHAOS_DURATION

# Wait for JMeter to finish 
echo "Waiting for JMeter test to complete..."
wait $JMETER_PID
echo "JMeter test completed."