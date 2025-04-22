#!/bin/bash

function show_help(){
    echo "usage $0 [OPTION]"
    echo "Inject container failures using ChaosBlade"
    echo "options:"
    echo "  -h, --help                 Show this help message"
    echo "  -s, --service SERVICE      Target specific service"
    echo "  -t, --type TYPE            Failure type (cpu|memory|network|latency|kill)"
    echo "  -d, --duration SECONDS     Experiment duration (default: 300s)"
    echo "  -l, --list                 List running experiments"
    echo "  -c, --clean                Clean all experiments"
    echo ""
    echo "Examples:"
    echo "  $0 -s customers-service -t cpu -d 180"
    echo "  $0 -s api-gateway -t network -d 120"
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
    echo "active chaosblade experiments.."
    blade status
    exit 0
fi

if [[ "$ACTION" == "clean" ]]; then
    echo "cleaning previous chaosblade experiments"
    blade destroy
    exit 0
fi

CONTAINER_ID=$(docker ps | grep $SERVICE | awk '{print $1}')
if [ -z "$CONTAINER_ID" ]; then
    echo "no container found for injecting failures"
    exit 1
fi

echo "target container found: $CONTAINER_ID"
echo "Injecting $TYPE failure for $DURATION seconds..."

case $TYPE in
  cpu)
    blade create docker cpu fullload --container-id $CONTAINER_ID --cpu-percent 80 --timeout $DURATION
  ;;
  
  memory)
    blade create docker mem load --container-id $CONTAINER_ID --mem-percent 80 --timeout $DURATION
  ;;

  # TODO separate network loss to network delay
  network)
    blade create docker network delay --container-id $CONTAINER_ID --interface eth0 --time 200 --timeout $DURATION
    blade create docker network loss --container-id $CONTAINER_ID --interface eth0 --percent 20 --timeout $DURATION
  ;;
  
  latency)
    blade create docker network delay --container-id $CONTAINER_ID --interface eth0 --time 1000 --timeout $DURATION
  ;;
  
  *)
    echo "Error: Unsupported failure type: $TYPE"
    echo "Supported types: cpu, memory, network, latency, kill"
    exit 1
  ;;
esac

echo "Chaos experiment started. It will run for $DURATION seconds."
echo "To monitor: blade status"
echo "To stop early: blade destroy <experiment_id>"