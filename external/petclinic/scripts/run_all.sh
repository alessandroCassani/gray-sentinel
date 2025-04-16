#!/usr/bin/env bash
set -o errexit
set -o errtrace
set -o nounset
set -o pipefail

# Kill any running Spring Petclinic processes
pkill -9 -f spring-petclinic || echo "Failed to kill any apps"

echo "Running apps"
mkdir -p target

# Set demo key for OpenAI if no key is provided
if [ -z "${OPENAI_API_KEY:-}" ]; then
  echo "Using demo key for GenAI service"
  export OPENAI_API_KEY="demo"
fi

# Start the config server
echo "Starting config server..."
nohup java -jar spring-petclinic-config-server/target/*.jar --server.port=8888 > target/config-server.log 2>&1 &
echo "Waiting for config server to start"
sleep 20

# Start the discovery server
echo "Starting discovery server..."
nohup java -jar spring-petclinic-discovery-server/target/*.jar --server.port=8761 > target/discovery-server.log 2>&1 &
echo "Waiting for discovery server to start"
sleep 20

# Start the microservices
echo "Starting microservices..."
nohup java -jar spring-petclinic-customers-service/target/*.jar --server.port=8081 > target/customers-service.log 2>&1 &
nohup java -jar spring-petclinic-visits-service/target/*.jar --server.port=8082 > target/visits-service.log 2>&1 &
nohup java -jar spring-petclinic-vets-service/target/*.jar --server.port=8083 > target/vets-service.log 2>&1 &
nohup java -jar spring-petclinic-genai-service/target/*.jar --server.port=8084 > target/genai-service.log 2>&1 &
nohup java -jar spring-petclinic-api-gateway/target/*.jar --server.port=8080 > target/gateway-service.log 2>&1 &
nohup java -jar spring-petclinic-admin-server/target/*.jar --server.port=9090 > target/admin-server.log 2>&1 &

echo "Waiting for apps to start"
sleep 60
echo "All services should be running now. Access the application at http://localhost:8080"