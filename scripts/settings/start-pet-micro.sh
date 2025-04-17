#!/bin/bash

mkdir -p target

cd ..
cd "spring-petclinic-microservices"

# Start Discovery Server
cd "spring-petclinic-discovery-server"
nohup java -jar spring-petclinic-discovery-server/target/*.jar --server.port=8761 --spring.profiles.active=chaos-monkey > target/discovery-server.log 2>&1 &
sleep 10

# Start Config Server
cd ..
cd "spring-petclinic-config-server"
nohup java -jar spring-petclinic-config-server/target/*.jar --server.port=8888 --spring.profiles.active=chaos-monkey > target/config-server.log 2>&1 &
sleep 10

# Start Customers Service
cd ..
cd "spring-petclinic-customers-service"
nohup java -jar spring-petclinic-customers-service/target/*.jar --server.port=8081 --spring.profiles.active=chaos-monkey > target/customers-service.log 2>&1 &

sleep 10

# Start Vets Service
cd ..
cd "spring-petclinic-vets-service"
nohup java -jar spring-petclinic-vets-service/target/*.jar --server.port=8083 --spring.profiles.active=chaos-monkey > target/vets-service.log 2>&1 &
sleep 5

# Start Visits Service
cd ..
cd "spring-petclinic-visits-service"
nohup java -jar spring-petclinic-visits-service/target/*.jar --server.port=8082 --spring.profiles.active=chaos-monkey > target/visits-service.log 2>&1 &
sleep 5

# Start API Gateway
cd ..
cd "spring-petclinic-api-gateway"
nohup java -jar spring-petclinic-api-gateway/target/*.jar --server.port=8080 --spring.profiles.active=chaos-monkey > target/gateway-service.log 2>&1 &
sleep 5

# Start Admin Server
cd ..
cd "spring-petclinic-admin-server"
nohup java -jar spring-petclinic-admin-server/target/*.jar --server.port=9090 --spring.profiles.active=chaos-monkey > target/admin-server.log 2>&1 &

echo "Services started. Access URLs:"
echo "- Discovery Server:   http://localhost:8761"
echo "- Config Server:      http://localhost:8888"
echo "- Frontend (API GW):  http://localhost:8080"
echo "- Zipkin Tracing:     http://localhost:9411/zipkin/"
echo "- Admin Server:       http://localhost:9090"