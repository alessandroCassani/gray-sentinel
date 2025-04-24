#!/bin/bash

function show_help() {
    echo "Usage: $0 [OPTION]"
    echo "Inject container-level failures using ChaosBlade"
    echo ""
    echo "Options:"
    echo "  -h, --help           Show this help message"
    echo "  -s, --service SERVICE Target specific service (required)"
    echo "  -t, --type TYPE      Failure type (cpu|memory|network|latency|kill)"
    echo "  -d, --duration SECONDS Experiment duration (default: 300s)"
    echo "  -l, --list           List running experiments"
    echo "  -c, --clean          Clean all experiments"
    echo ""
    echo "Examples:"
    echo "  $0 -s customers-service -t cpu -d 180"
    echo "  $0 -s api-gateway -t latency"
    echo "  $0 -c"
    exit 0
}

DURATION=300
ACTION=""
SERVICE=""
TYPE=""

while [[ $# -gt 0 ]]; do
    key="$1"
    case $key in
        -h|--help)
            show_help
            ;;
        -s|--service)
            SERVICE="$2"
            shift 2
            ;;
        -t|--type)
            TYPE="$2"
            shift 2
            ;;
        -d|--duration)
            DURATION="$2"
            shift 2
            ;;
        -l|--list)
            ACTION="list"
            shift
            ;;
        -c|--clean)
            ACTION="clean"
            shift
            ;;
        *)
            echo "Unknown option: $1"
            show_help
            ;;
    esac
done

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

# Validate inputs
if [ -z "$SERVICE" ]; then
    echo "Error: Service must be specified with -s or --service"
    show_help
fi

if [ -z "$TYPE" ]; then
    echo "Error: Failure type must be specified with -t or --type"
    show_help
fi

# Get container ID
CONTAINER_ID=$(docker ps | grep $SERVICE | awk '{print $1}')
if [ -z "$CONTAINER_ID" ]; then
    echo "Error: Container for service '$SERVICE' not found. Is it running?"
    exit 1
fi

echo "Target container found: $CONTAINER_ID for service: $SERVICE"
echo "Injecting container failure: $TYPE for $DURATION seconds..."

case $TYPE in
    cpu)
        blade create cri cpu fullload --container-id $CONTAINER_ID --cpu-percent 50 --timeout $DURATION
        ;;
    mem)
        blade create cri mem --container-id $CONTAINER_ID --mem-percent 80 --timeout $DURATION
        ;;
    network-loss)
        # Drop 30% of all packets
        blade create cri network loss --container-id $CONTAINER_ID --percent 30 --timeout $DURATION
        ;;
    network-delay)
        # Add 200ms network delay with 10ms jitter
        blade create cri network delay --container-id $CONTAINER_ID --time 200 --offset 10 --timeout $DURATION
        ;;
    network-corrupted)
        # Corrupt 30% of packets
        blade create cri network corrupt --container-id $CONTAINER_ID --percent 30 --timeout $DURATION
        ;;
    disk-read)
        # Read-only disk IO burn in the root directory
        # ATTENTION --size flag refers to block size
        blade create cri disk burn --read --path "/" --size 10 --container-id $CONTAINER_ID --timeout $DURATION
        ;;
    disk-write)
        # write-only disk IO burn in the root directory
        # ATTENTION --size flag refers to block size
        blade create cri disk burn --write --path "/" --size 10 --container-id $CONTAINER_ID --timeout $DURATION
        ;;
    disk-read-write)
        # Read and write disk IO burn in the root directory
        # ATTENTION --size flag refers to block size
        blade create cri disk burn --read --write --path "/" --size 10 --container-id $CONTAINER_ID --timeout $DURATION
        ;;
    *)
        echo "Error: Unsupported failure type: $TYPE"
        echo "Supported types: cpu, memory, network, latency, kill"
        exit 1
        ;;
esac

echo "Container chaos experiment started. It will run for $DURATION seconds."
echo "To monitor: blade status"
echo "To stop early: blade destroy <experiment_id>"