#!/bin/bash
# Script to run JMeter load test and then call JVM chaos injection script after a delay
JMETER_BIN="/opt/apache-jmeter-5.6.3/bin/jmeter"
JMETER_TEST_DIR="../../external/petclinic/spring-petclinic-api-gateway/src/test/jmeter"
JMX_FILE="petclinic_test_plan.jmx"
RESULTS_FILE="results.jtl"
CHAOS_SCRIPT="../failure-injection/jvm-injection.sh"
TARGET_SERVICE="api-gateway"
# Type of JVM failure (oom, cpufulload, gc, throwCustomException, tde)
CHAOS_TYPE="throwCustomException"
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

if [ ! -f "$CHAOS_SCRIPT" ]; then
    echo "Error: Chaos script not found at $CHAOS_SCRIPT"
    exit 1
fi

echo "Starting JMeter load test..."
"$JMETER_BIN" -n -t "$JMETER_TEST_DIR/$JMX_FILE" -l $RESULTS_FILE &
JMETER_PID=$!
echo "JMeter started with PID: $JMETER_PID"

echo "Waiting $DELAY_SECONDS seconds before injecting first failure..."
sleep $DELAY_SECONDS

echo "Executing JVM chaos injection script..."

# Get container ID for target service
CONTAINER_ID=$(docker ps | grep $TARGET_SERVICE | awk '{print $1}')
if [ -z "$CONTAINER_ID" ]; then
    echo "Error: Container for service '$TARGET_SERVICE' not found"
    exit 1
fi

echo "Found container: $CONTAINER_ID for service: $TARGET_SERVICE"

# Execute JVM chaos injection based on type
case $CHAOS_TYPE in
    "oom")
        echo "Injecting OutOfMemoryError..."
        EXPERIMENT_RESULT=$(docker exec $CONTAINER_ID /opt/chaosblade-1.7.2/blade create jvm oom --area HEAP --timeout $CHAOS_DURATION)
        ;;
    "cpufulload")
        echo "Injecting high CPU load in JVM..."
        EXPERIMENT_RESULT=$(docker exec $CONTAINER_ID /opt/chaosblade-1.7.2/blade create jvm cpufull --timeout $CHAOS_DURATION)
        ;;
    "gc")
        echo "Triggering frequent garbage collection..."
        EXPERIMENT_RESULT=$(docker exec $CONTAINER_ID /opt/chaosblade-1.7.2/blade create jvm full-gc --effect-count 100 --interval 1000 --timeout $CHAOS_DURATION)
        ;;
    "throwCustomException")
        echo "Injecting custom exception in Gateway filter..."
        EXPERIMENT_RESULT=$(docker exec $CONTAINER_ID /opt/chaosblade-1.7.2/blade create jvm throwCustomException \
            --classname "org.springframework.cloud.gateway.filter.GatewayFilterChain" \
            --methodname "filter" \
            --exception "java.lang.RuntimeException" \
            --exception-message "PetClinic Gateway chaos failure" \
            --timeout $CHAOS_DURATION)
        ;;
    "tde")
        echo "Injecting declared exception in service discovery..."
        EXPERIMENT_RESULT=$(docker exec $CONTAINER_ID /opt/chaosblade-1.7.2/blade create jvm throwDeclaredException \
            --classname "org.springframework.cloud.netflix.eureka.EurekaDiscoveryClient" \
            --methodname "getServices" \
            --timeout $CHAOS_DURATION)
        ;;
    *)
        echo "Unknown JVM chaos type: $CHAOS_TYPE"
        echo "Available types: oom, cpufulload, gc, throwCustomException, tde"
        kill $JMETER_PID
        exit 1
        ;;
esac

echo "ChaosBlade JVM result: $EXPERIMENT_RESULT"
EXPERIMENT_ID=$(echo $EXPERIMENT_RESULT | grep -o '"result":"[^"]*"' | awk -F'"' '{print $4}')

sleep $CHAOS_DURATION

echo "Cleaning up: Destroying JVM chaos experiment with ID: $EXPERIMENT_ID"
DESTROY_RESULT=$(docker exec $CONTAINER_ID /opt/chaosblade-1.7.2/blade destroy $EXPERIMENT_ID)
echo "Destroy result: $DESTROY_RESULT"

echo "Waiting for JMeter test to complete..."
wait $JMETER_PID
echo "JMeter test completed."