#!bin/bash

CONTAINER_NAME="influxdb"
DATA_DIR="$HOME/docker-data/influxdb"  # for the volume
INFLUX_PORT=8086
DB_NAME="zookeeper_metrics"
ADMIN_USER="admin"
ADMIN_PASS="admin"
INFLUX_VERSION="1.8"

echo "creating data directory: $DATA_DIR"
mkdir -p $DATA_DIR

if docker ps -a | grep $CONTAINER_NAME -q; then
   echo "Container $CONTAINER_NAME already exists. Removing it..."
   docker stop $CONTAINER_NAME >/dev/null 2>&1
   docker rm $CONTAINER_NAME  >/dev/null 2>&1
fi

docker pull influxdb:$INFLUX_VERSION

docker run -d \
    --name: $CONTAINER_NAME \
    -p $INFLUX_PORT:8086 \
    -v $DATA_DIR:/var/lib/influxdb \
    -e INFLUX_DB:$DB_NAME \
    -e INFLUXDB_ADMIN_USER=$ADMIN_USER \
    -e INFLUXDB_ADMIN_PASSWORD=$ADMIN_PASS \


if [ $? -eq 0 ]; then
    echo "InfluxDB container started successfully!"
    echo "Container ID: $(docker ps -q -f name=$CONTAINER_NAME)"
    echo "Access URL: http://localhost:$INFLUX_PORT"
    
    echo "Waiting for InfluxDB to be ready..."
    sleep 5
    
    echo "Verifying database setup..."
    docker exec $CONTAINER_NAME influx -username $ADMIN_USER -password $ADMIN_PASS -execute "SHOW DATABASES" | grep $DB_NAME

else
    echo "Failed to start InfluxDB container. Please check Docker service is running."
    echo "Run 'sudo systemctl status docker' to check Docker status."
fi