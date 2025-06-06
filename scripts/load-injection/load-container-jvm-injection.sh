#!/bin/bash
# Script to run JMeter load test and then call chaos JVM injection script after a delay

JMETER_BIN="/opt/apache-jmeter-5.6.3/bin/jmeter" 
JMETER_TEST_DIR="../../external/petclinic/spring-petclinic-api-gateway/src/test/jmeter"
JMX_FILE="petclinic_test_plan.jmx"            # JMeter test plan file
RESULTS_FILE="results.jtl"                    # JMeter results file
CHAOS_SCRIPT="../failure-injection/container-jvm-injection.sh"
TARGET_SERVICE="api-gateway"                  # Service to inject failure into
# Type of failure (cpufulload, oom, codecachefilling, delay, full-gc, 
# throwCustomException, throwDeclaredException,tfl-running, tfl-wait)
CHAOS_TYPE="cpufulload"                        
DELAY_SECONDS=60                              # Wait time before injecting failure
CHAOS_DURATION=420                            # Duration of the chaos experiment in seconds

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
    echo "Chaos script not found at $CHAOS_SCRIPT"
    exit 1
fi

echo "starting jmeter load test..."
"$JMETER_BIN" -n -t "$JMETER_TEST_DIR/$JMX_FILE" -l $RESULTS_FILE &
JMETER_PID=$!
echo "Jmeter started with PID $JMETER_PID"

echo "injection delay of $DELAY_SECONDS"
sleep $DELAY_SECONDS

echo "It is time for failure injection"
"$CHAOS_SCRIPT" -s $TARGET_SERVICE -t $CHAOS_TYPE -d $CHAOS_DURATION

# Wait for JMeter to finish 
echo "Waiting for JMeter test to complete..."
wait $JMETER_PID
echo "JMeter test completed."
