#!/bin/bash

BASE_DIR=$(pwd)/..

echo "building zookeeper first"

cd "${BASE_DIR}/zookeeper"
mvn clean install -DskipTests -Dmaven.test.skip=true

echo "Building and setting up Legolas"
cd "${BASE_DIR}/legolas"
mvn clean install -DskipTests

echo "instrumenting zookeeper with Legolas..."
cd "${BASE_DIR}/legolas
./bin/legolas.sh analyzer -s conf/zookeeper/3.6.2/analyzer.json -i "${BASE_DIR}/zookeeper"



# TODO understand the ralationship between this command and the one on github, to make good choice




echo "starting instrumented zookeeper monitoring..."

