#!bin/bash

PROTCOL="http"
MAX_PM_CHILDREN=16
WORKER_PROCESS="auto"

echo "stopping running containers..."

docker stop database_server
docker stop web_server
docker stop memcached_server

docker rm database_server
docker rm web_server
docker rm memcached_server

HOST_IP=$(hostname -I | awk '{print $1}')
echo "using IP: $HOST_IP"

echo "Starting the database server..."
docker run --net=host --name=database_server cloudsuite/web-serving:db_server

echo "Starting the Memcached server..."
docker run -dt --net=host --name=memcache_server cloudsuite/web-serving:memcached_server

echo "Starting the web server..."
docker run -dt --net=host --name=web_server cloudsuite/web-serving:web_server /etc/bootstrap.sh $PROTOCOL $HOST_IP $HOST_IP $HOST_IP $MAX_PM_CHILDREN $WORKER_PROCESS

echo "Web server accessible at: $PROTOCOL://$HOST_IP:8080"

