#!/bin/bash
# Script to run JMeter load test and then call JVM chaos injection script after a delay
JMETER_BIN="/opt/apache-jmeter-5.6.3/bin/jmeter"
JMETER_TEST_DIR="../../external/petclinic/spring-petclinic-api-gateway/src/test/jmeter"
JMX_FILE="petclinic_test_plan.jmx"
RESULTS_FILE="results.jtl"
TARGET_SERVICE="customers-service"
CHAOS_TYPE="gc-stress"
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
echo "JVM Chaos Type: $CHAOS_TYPE"
echo "Injection delay: $DELAY_SECONDS seconds"
echo "Chaos Duration: $CHAOS_DURATION seconds"
echo "=============================="

echo "Starting JMeter load test..."
"$JMETER_BIN" -n -t "$JMETER_TEST_DIR/$JMX_FILE" -l $RESULTS_FILE &
JMETER_PID=$!
echo "JMeter started with PID: $JMETER_PID"

echo "Waiting $DELAY_SECONDS seconds before injecting first failure..."
sleep $DELAY_SECONDS

echo "Executing JVM chaos injection script..."

# Get container ID for the target service
CONTAINER_ID=$(docker ps | grep $TARGET_SERVICE | awk '{print $1}')
if [ -z "$CONTAINER_ID" ]; then
    echo "Error: Container for service '$TARGET_SERVICE' not found"
    kill $JMETER_PID
    exit 1
fi

echo "Found container: $CONTAINER_ID for service: $TARGET_SERVICE"

JAVA_PID=$(docker exec $CONTAINER_ID ps aux | grep java | grep -v grep | awk '{print $2}' | head -1)
if [ -z "$JAVA_PID" ]; then
    echo "Error: No Java process found in container $CONTAINER_ID"
    kill $JMETER_PID
    exit 1
fi
echo "Found Java process with PID: $JAVA_PID"

docker exec $CONTAINER_ID /opt/chaosblade-1.7.2/blade prepare jvm --pid 1

case $CHAOS_TYPE in
    "memory-leak")
        echo "Injecting gradual memory leak..."
        EXPERIMENT_RESULT=$(docker exec $CONTAINER_ID /opt/chaosblade-1.7.2/blade create jvm oom \
            --pid $JAVA_PID \
            --area heap \
            --interval 10000 \
            --timeout $CHAOS_DURATION)
        ;;

    
    "thread-exhaustion")
        echo "Exhausting thread pool..."
        EXPERIMENT_RESULT=$(docker exec $CONTAINER_ID /opt/chaosblade-1.7.2/blade create jvm threadfull \
            --pid $JAVA_PID \
            --running \
            --thread-count 90 \
            --timeout $CHAOS_DURATION)
        ;;
    
    "code-cache-fill")
        echo "Filling up JVM code cache..."
        EXPERIMENT_RESULT=$(docker exec $CONTAINER_ID /opt/chaosblade-1.7.2/blade create jvm CodeCacheFilling \
            --timeout $CHAOS_DURATION)
        ;;
    
    "gc-stress")
    echo "Triggering full GC stress in customers service..."
    EXPERIMENT_RESULT=$(docker exec $CONTAINER_ID /opt/chaosblade-1.7.2/blade create jvm full-gc \
        --pid $JAVA_PID \
        --effect-count 50 \
        --interval 1500 \
        --timeout $CHAOS_DURATION)
    ;;
    *)
        echo "Unknown JVM chaos type: $CHAOS_TYPE"
        kill $JMETER_PID
        exit 1
        ;;
esac

EXPERIMENT_ID=$(echo $EXPERIMENT_RESULT | grep -o '"result":"[^"]*"' | awk -F'"' '{print $4}')

if [ -z "$EXPERIMENT_ID" ]; then
    echo "Warning: Could not extract experiment ID from result"
    echo "Full result: $EXPERIMENT_RESULT"
else
    echo "JVM experiment started with ID: $EXPERIMENT_ID"
fi

sleep $CHAOS_DURATION

if [ ! -z "$EXPERIMENT_ID" ]; then
    echo "Cleaning up: Destroying JVM chaos experiment with ID: $EXPERIMENT_ID"
    DESTROY_RESULT=$(docker exec $CONTAINER_ID /opt/chaosblade-1.7.2/blade destroy $EXPERIMENT_ID)
    echo "Destroy result: $DESTROY_RESULT"
else
    echo "Warning: No experiment ID available for cleanup"
fi

echo "Waiting for JMeter test to complete..."
wait $JMETER_PID
echo "JMeter test completed."