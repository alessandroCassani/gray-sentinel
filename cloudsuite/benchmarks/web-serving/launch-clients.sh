#!/bin/bash

if docker ps -q -f name=faban_client; then
    echo " faban_client already running."
    exit 1
fi

docker run --net=host --name=faban_client cloudsuite/web-serving:faban_client 127.0.0.1 10 --oper=run --steady=300

if [ $? -eq 0 ]; then
    echo "fabian_client succesfully running."
else
    echo "ERROR"
fi