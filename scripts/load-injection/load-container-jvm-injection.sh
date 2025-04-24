#!/bin/bash
# Script to run JMeter load test and then call chaos  JVM injection script after a delay

JMETER_BIN="/opt/apache-jmeter-5.6.3/bin/jmeter" 
JMETER_TEST_DIR="../../external/petclinic/spring-petclinic-api-gateway/src/test/jmeter"
JMX_FILE="petclinic_test_plan.jmx"            # JMeter test plan file
RESULTS_FILE="results.jtl"                    # JMeter results file
CHAOS_SCRIPT="../failure-injection/container-jvm-injection.sh"
TARGET_SERVICE="api-gateway"                  # Service to inject failure
CHAOS_TYPE="cpu"                              # Type of failure (cpu, memory, network, latency)
DELAY_SECONDS=60                              # Wait time before injecting failure
CHAOS_DURATION=420                            # Duration of the chaos experiment in seconds