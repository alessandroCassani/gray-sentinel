#!/bin/bash


echo "starting grafana..."
docker run -d --network="host" --name=grafana grafana/grafana-oss
echo "grafana is running!"
