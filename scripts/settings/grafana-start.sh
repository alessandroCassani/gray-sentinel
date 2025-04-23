#!/bin/bash

echo "Creating Grafana volume..."
docker volume create grafana-storage

echo "Starting Grafana with persistent storage..."
docker run -d \
  --network="host" \
  --name=grafana \
  -v grafana-storage:/var/lib/grafana \
  grafana/grafana-oss

echo "Grafana is running!"