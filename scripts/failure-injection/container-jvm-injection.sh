#!/bin/bash

DURATION=300
ACTION=""
SERVICE=""
TYPE=""

if [[ "$ACTION" == "list" ]]; then
    echo "Active ChaosBlade experiments..."
    blade status
    exit 0
fi

if [[ "$ACTION" == "clean" ]]; then
    echo "Cleaning previous ChaosBlade experiments"
    blade destroy
    exit 0
fi


CONTAINER_ID=$(docker ps | grep $SERVICE | awk '{print $1}')
if [ -z "$CONTAINER_ID" ]; then
    echo "Error: Container for service '$SERVICE' not found. Is it running?"
    exit 1
fi

case $TYPE in
    "memory-leak")
        EXPERIMENT_RESULT=$(docker exec $CONTAINER_ID /opt/chaosblade-1.7.2/blade create jvm oom \
            --pid $JAVA_PID \
            --area heap \
            --interval 10000 \
            --timeout $CHAOS_DURATION)
        ;;

   "thread-exhaustion")
        EXPERIMENT_RESULT=$(docker exec $CONTAINER_ID /opt/chaosblade-1.7.2/blade create jvm threadfull \
            --pid $JAVA_PID \
            --running \
            --thread-count 90 \
            --timeout $CHAOS_DURATION)
        ;;

    "code-cache-fill")
        EXPERIMENT_RESULT=$(docker exec $CONTAINER_ID /opt/chaosblade-1.7.2/blade create jvm CodeCacheFilling \
            --timeout $CHAOS_DURATION)
        ;;
    
    "gc-stress")
    EXPERIMENT_RESULT=$(docker exec $CONTAINER_ID /opt/chaosblade-1.7.2/blade create jvm full-gc \
        --pid $JAVA_PID \
        --effect-count 50 \
        --interval 1500 \
        --timeout $CHAOS_DURATION)
    ;;
esac
