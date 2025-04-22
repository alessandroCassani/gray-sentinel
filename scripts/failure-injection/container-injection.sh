#!/bin/bash

function show_help() {
    echo "Usage: $0 [OPTION]"
    echo "Inject container-level failures using ChaosBlade"
    echo ""
    echo "Options:"
    echo "  -h, --help                 Show this help message"
    echo "  -s, --service SERVICE      Target specific service (required)"
    echo "  -t, --type TYPE            Failure type (cpu|memory|network|latency|kill)"
    echo "  -d, --duration SECONDS     Experiment duration (default: 300s)"
    echo "  -l, --list                 List running experiments"
    echo "  -c, --clean                Clean all experiments"
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
    # High CPU load (80%)
    blade create docker cpu fullload --container-id $CONTAINER_ID --cpu-percent 80 --timeout $DURATION
  ;;
  
  memory)
    # Consume 80% of container memory
    blade create docker mem load --container-id $CONTAINER_ID --mem-percent 80 --timeout $DURATION
  ;;

  network)
    # Network delay and packet loss
    blade create docker network delay --container-id $CONTAINER_ID --interface eth0 --time 200 --timeout $DURATION
    blade create docker network loss --container-id $CONTAINER_ID --interface eth0 --percent 20 --timeout $DURATION
  ;;
  
  latency)
    # Add 1000ms latency to the service
    blade create docker network delay --container-id $CONTAINER_ID --interface eth0 --time 1000 --timeout $DURATION
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