#!/bin/bash
if [ $# -eq 1 ]; then
  port=$((10711+$1))
else
  port=10712
fi
/home/alessandro/PGFDS/zookeeper/bin/zkCli.sh -server localhost:$port $@

