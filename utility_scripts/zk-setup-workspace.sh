#!/bin/bash

# Prepare the workspace, experiment scripts, and configurations for a fault injection experiment.
# The setup scripts and configurations are generated in legolas/workspace/legolas-zk/.

BASE_DIR=$(pwd)/..

cd "${BASE_DIR}/legolas"

scripts/experiment/setup-legolas-zookeeper.sh ../zookeeper 3.6.2

# Check the generated configuration file workspace/legolas-zk/legolas-zk.properties and customize it if necessary