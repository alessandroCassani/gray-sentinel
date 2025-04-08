#!/bin/bash

# Configuration variables
PETCLINIC_JAR="spring-petclinic.jar"  # Change to your actual JAR filename
PETCLINIC_PORT=8080
JMETER_PATH="/opt/jmeter/bin/jmeter"
JMETER_TEST_PLAN="petclinic-test.jmx"
RESULTS_FILE="results.jtl"
MAX_WAIT_TIME=60  # Maximum seconds to wait for PetClinic to start

# Function to check if PetClinic is up
check_petclinic_status() {
  curl -s http://localhost:${PETCLINIC_PORT} > /dev/null
  return $?
}

# Function to clean up on exit
cleanup() {
  echo "Shutting down PetClinic application..."
  kill $PETCLINIC_PID
  exit
}

# Set trap to ensure cleanup on script termination
trap cleanup SIGINT SIGTERM

# Start PetClinic application
echo "Starting PetClinic application..."
java -jar ${PETCLINIC_JAR} > petclinic.log 2>&1 &
PETCLINIC_PID=$!

# Wait for PetClinic to be ready
echo "Waiting for PetClinic to start (up to ${MAX_WAIT_TIME} seconds)..."
COUNTER=0
while ! check_petclinic_status; do
  if [ $COUNTER -ge $MAX_WAIT_TIME ]; then
    echo "Error: PetClinic failed to start within ${MAX_WAIT_TIME} seconds."
    cleanup
  fi
  
  # Check if the process is still running
  if ! ps -p $PETCLINIC_PID > /dev/null; then
    echo "Error: PetClinic process terminated unexpectedly. Check petclinic.log for details."
    cat petclinic.log
    exit 1
  fi
  
  COUNTER=$((COUNTER+1))
  sleep 1
  echo -n "."
done

echo
echo "PetClinic is up and running!"

# Start JMeter test
echo "Starting JMeter test..."
${JMETER_PATH} -n -t ${JMETER_TEST_PLAN} -l ${RESULTS_FILE}

# Check JMeter exit status
if [ $? -ne 0 ]; then
  echo "Error: JMeter test failed."
  cleanup
  exit 1
fi

echo "JMeter test completed successfully!"

# Display summary from results file
echo "Test Results Summary:"
grep "summary =" ${RESULTS_FILE} | tail -1

# Shutdown PetClinic
cleanup