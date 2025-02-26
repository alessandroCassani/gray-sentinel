#!/bin/bash
for i in 1 2 3; do
  ZOOCFGDIR=/home/alessandro/PGFDS/Legolas/workspace/legolas-zk/conf-$i ZOO_LOG_DIR=/home/alessandro/PGFDS/Legolas/workspace/legolas-zk/trials/0/logs-$i /home/alessandro/PGFDS/zookeeper/bin/zkServer.sh stop
done

