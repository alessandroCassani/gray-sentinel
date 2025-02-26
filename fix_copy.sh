#!/bin/bash
# Save this as fix_copy.sh

mkdir -p /home/alessandro/PGFDS/legolas/sootOutput/org/apache/zookeeper/server/auth/
mkdir -p /home/alessandro/PGFDS/legolas/sootOutput/org/apache/zookeeper/server/watch/
mkdir -p /home/alessandro/PGFDS/legolas/sootOutput/org/apache/zookeeper/metrics/
mkdir -p /home/alessandro/PGFDS/legolas/sootOutput/org/apache/zookeeper/audit/

cp /home/alessandro/PGFDS/zookeeper/zookeeper-server/target/classes/org/apache/zookeeper/server/auth/AuthenticationProvider.class /home/alessandro/PGFDS/legolas/sootOutput/org/apache/zookeeper/server/auth/
cp /home/alessandro/PGFDS/zookeeper/zookeeper-server/target/classes/org/apache/zookeeper/server/watch/IWatchManager.class /home/alessandro/PGFDS/legolas/sootOutput/org/apache/zookeeper/server/watch/
cp /home/alessandro/PGFDS/zookeeper/zookeeper-server/target/classes/org/apache/zookeeper/metrics/Counter.class /home/alessandro/PGFDS/legolas/sootOutput/org/apache/zookeeper/metrics/
cp /home/alessandro/PGFDS/zookeeper/zookeeper-server/target/classes/org/apache/zookeeper/audit/AuditLogger.class /home/alessandro/PGFDS/legolas/sootOutput/org/apache/zookeeper/audit/

echo "Files copied successfully!"