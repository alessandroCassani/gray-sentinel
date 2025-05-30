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

CONTAINER_ID=$(docker ps | grep $SERVICE | awk '{print $1}')
if [ -z "$CONTAINER_ID" ]; then
    echo "Error: Container for service '$SERVICE' not found. Is it running?"
    exit 1
fi

echo "Target container found: $CONTAINER_ID for service: $SERVICE"
echo "Injecting container failure: $TYPE for $DURATION seconds..."

case $TYPE in
    cpu)
        docker exec 8537dedffab4 /opt/chaosblade-1.7.2/blade create cpu fullload --cpu-percent 80 --timeout 300
        ;;
    mem)
        docker exec e6c809bdf4f6 /opt/chaosblade-1.7.2/blade create mem load --mode ram --mem-percent 80 
        ;;
    network-loss)
        # Drop 20% of all packets 
        blade create network loss --percent 40 --interface veth2c9f03f
        # ATTENTION u need to specify running container ports to attach network loss
        # blade create cri network loss --percent 20 --interface eth0@if31 --local-port 8080 --container-id $CONTAINER_ID  --timeout $DURATION
        ;;
    network-delay)
        # Access to native 8080 port is delayed by 0.5 seconds, and the delay time fluctuates by 0.2 second
        blade create cri network delay  --time 500 --offset 200 --interface veth38c6ce6 
        ;;
    disk-read-write)
        # Read and write disk IO burn in the root directory
        # ATTENTION --size flag refers to block size
        blade create cri disk burn --read --write --path "/" --size 20 --container-id 73a8c7bb20c7
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