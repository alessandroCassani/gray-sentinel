#!/bin/bash

docker exec kafka kafka-topics --create --if-not-exists \
    --bootstrap-server kafka:29092 \
    --partitions 3 \
    --replication-factor 1 \
    --topic perf-metrics

