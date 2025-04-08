#!/bin/bash

# Start Discovery Server
cd ..
cd "spring-petclinic-microservices/spring-petclinic-discovery-server"
../mvnw spring-boot:run &
sleep 10

# Start Config Server
cd ..
cd "spring-petclinic-config-server"
../mvnw spring-boot:run &
sleep 10

# Start Customers Service
cd ..
cd "spring-petclinic-customers-service"
../mvnw spring-boot:run &
sleep 10

# Start Vets Service
cd ..
cd "spring-petclinic-vets-service"
../mvnw spring-boot:run &
sleep 5

# Start Visits Service
cd ..
cd "spring-petclinic-visits-service"
../mvnw spring-boot:run &
sleep 5

# Start API Gateway
cd ..
cd "spring-petclinic-api-gateway"
../mvnw spring-boot:run &
sleep 5

# Start Tracing Server
cd ..
cd "spring-petclinic-tracing-server"
../mvnw spring-boot:run &
sleep 5

# Start Admin Server
cd ..
cd "spring-petclinic-admin-server"
../mvnw spring-boot:run &

echo "Services started. Access URLs:"
echo "- Discovery Server:   http://localhost:8761"
echo "- Config Server:      http://localhost:8888"
echo "- Frontend (API GW):  http://localhost:8080"
echo "- Zipkin Tracing:     http://localhost:9411/zipkin/"
echo "- Admin Server:       http://localhost:9090"