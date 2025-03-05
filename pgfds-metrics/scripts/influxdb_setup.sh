#!/bin/bash


docker exec -i influxdb influx -username "admin" -password "admin"

docker exec influxdb influx -execute "CREATE DATABASE metrics"

docker exec influxdb influx -execute "SHOW DATABASE"