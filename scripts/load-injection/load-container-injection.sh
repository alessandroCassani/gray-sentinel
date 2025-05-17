#!/bin/bash
# Script to run JMeter load test and then call chaos injection script after a delay

# Use the full absolute path with leading slash
JMETER_BIN="/opt/apache-jmeter-5.6.3/bin/jmeter" 
JMETER_TEST_DIR="../../external/petclinic/spring-petclinic-api-gateway/src/test/jmeter"
JMX_FILE="petclinic_test_plan.jmx" # JMeter test plan file
RESULTS_FILE="results.jtl" # JMeter results file
CHAOS_SCRIPT="../failure-injection/container-injection.sh"
<<<<<<< HEAD
TARGET_SERVICE="discovery-server" # Service to inject failure into
=======
TARGET_SERVICE="api-gateway" # Service to inject failure into
>>>>>>> 2fd0da2c1aa6acd6272b7d9a3a2d84770274835e
# Type of failure (cpu, mem, network-loss, network-delay, network-corrupted, 
# disk-read, disk-write, disk-read-write)
CHAOS_TYPE="network-loss" 
DELAY_SECONDS=1800 # Wait time before injecting first failure 30m
CHAOS_DURATION=3000 # Duration of the chaos experiment in seconds 50m

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
echo "injection delay: $DELAY_SECONDS seconds"
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

echo "Waiting $DELAY_SECONDS1 seconds before injecting first failure..."
sleep $DELAY_SECONDS1

echo "Executing chaos injection script..."
"$CHAOS_SCRIPT" -s $TARGET_SERVICE -t $CHAOS_TYPE -d $CHAOS_DURATION

sleep $CHAOS_DURATION

echo "Cleanup: Removing tc rules from container interface..."
tc qdisc del dev veth04f87fa root  # to change hardcoded veth
echo "Chaos experiment completed and cleaned up."


# Wait for JMeter to finish 
echo "Waiting for JMeter test to complete..."
wait $JMETER_PID
echo "JMeter test completed."