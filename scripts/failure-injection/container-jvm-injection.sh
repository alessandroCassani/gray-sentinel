#!/bin/bash

function show_help() {
    echo "Usage: $0 [OPTION]"
    echo "Inject container-wide JVM failures using ChaosBlade"
    echo ""
    echo "Options:"
    echo "  -h, --help                 Show this help message"
    echo "  -s, --service SERVICE      Target specific service (required)"
    echo "  -t, --type TYPE            JVM failure type (oom|cpufull|gc|codecache)"
    echo "  -d, --duration SECONDS     Experiment duration (default: 300s)"
    echo "  -l, --list                 List running experiments"
    echo "  -c, --clean                Clean all experiments"
    echo ""
    echo "Examples:"
    echo "  $0 -s customers-service -t oom"
    echo "  $0 -s api-gateway -t gc -d 120"
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
    echo "Error: JVM failure type must be specified with -t or --type"
    show_help
fi

# Get container ID
CONTAINER_ID=$(docker ps | grep $SERVICE | awk '{print $1}')
if [ -z "$CONTAINER_ID" ]; then
    echo "Error: Container for service '$SERVICE' not found. Is it running?"
    exit 1
fi

echo "Target container found: $CONTAINER_ID for service: $SERVICE"
echo "Injecting container-wide JVM failure: $TYPE for $DURATION seconds..."

case $TYPE in
  oom)
    # Trigger OutOfMemoryError
    echo "Injecting OutOfMemoryError..."
    blade create docker jvm oom --container-id $CONTAINER_ID --timeout $DURATION
  ;;
  
  cpufull)
    # Create high CPU load within the JVM
    echo "Injecting high CPU load in JVM..."
    blade create docker jvm cpufull --container-id $CONTAINER_ID --timeout $DURATION
  ;;
  
  gc)
    # Trigger frequent garbage collection
    echo "Triggering frequent garbage collection..."
    blade create docker jvm frequent-gc --container-id $CONTAINER_ID --timeout $DURATION
  ;;
  
  codecache)
    # Fill code cache to cause JIT compilation issues
    echo "Filling code cache in JVM..."
    blade create docker jvm CodeCacheFilling --container-id $CONTAINER_ID --timeout $DURATION
  ;;
  
  *)
    echo "Error: Unsupported JVM failure type: $TYPE"
    echo "Supported JVM failure types: oom, cpufull, gc, codecache"
    exit 1
  ;;
esac

echo "JVM chaos experiment started. It will run for $DURATION seconds."
echo "To monitor: blade status"
echo "To stop early: blade destroy <experiment_id>"