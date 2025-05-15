#!/bin/bash

function show_help() {
    echo "Usage: $0 [OPTION]"
    echo "Inject method-level failures in JVM applications using ChaosBlade (CRI version)"
    echo ""
    echo "Options:"
    echo "  -h, --help              Show this help message"
    echo "  -s, --service SERVICE   Target specific service (required)"
    echo "  -m, --method METHOD     Target method with full class path (required)"
    echo "  -t, --type TYPE         Failure type (delay|exception|default:exception)"
    echo "  -v, --value VALUE       Delay in ms or exception message"
    echo "  -d, --duration SECONDS  Experiment duration (default: 300s)"
    echo "  -l, --list              List running experiments"
    echo "  -c, --clean             Clean all experiments"
    echo ""
    echo "Examples:"
    echo "  $0 -s customers-service -m org.springframework.samples.petclinic.customers.web.OwnerResource.getOwner"
    echo "  $0 -s customers-service -m org.springframework.samples.petclinic.customers.web.OwnerResource.getOwner -t delay -v 2000"
    echo "  $0 -c"
    exit 0
}

DURATION=300
ACTION=""
SERVICE=""
METHOD=""
TYPE="exception"
VALUE=""

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
        -m|--method)
            METHOD="$2"
            shift 2
            ;;
        -t|--type)
            TYPE="$2"
            shift 2
            ;;
        -v|--value)
            VALUE="$2"
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

if [ -z "$METHOD" ]; then
    echo "Error: Method must be specified with -m or --method"
    show_help
fi

# Get container ID 
CONTAINER_ID=$(docker ps | grep $SERVICE | awk '{print $1}')
if [ -z "$CONTAINER_ID" ]; then
    echo "Error: Container for service '$SERVICE' not found"
    exit 1
fi

echo "Target container found: $CONTAINER_ID for service: $SERVICE"
echo "Target method: $METHOD"
echo "Injecting method failure: $TYPE for $DURATION seconds..."

# Extract class and method from full path
CLASS_PATH=${METHOD%.*}
METHOD_NAME=${METHOD##*.}

case $TYPE in
    delay)
        # Add latency to method execution
        DELAY_MS=${VALUE:-3000}
        echo "Injecting delay of $DELAY_MS ms into method..."
        blade create cri jvm delay \
            --container-id $CONTAINER_ID \
            --classname $CLASS_PATH \
            --methodname $METHOD_NAME \
            --time $DELAY_MS \
            --timeout $DURATION
        ;;
    exception)
        # Throw exception from method
        EXCEPTION_MSG=${VALUE:-"ChaosBlade injected exception"}
        echo "Injecting exception into method with message: $EXCEPTION_MSG"
        blade create cri jvm throwCustomException \
            --container-id $CONTAINER_ID \
            --classname $CLASS_PATH \
            --methodname $METHOD_NAME \
            --exception java.lang.RuntimeException \
            --exception-message "$EXCEPTION_MSG" \
            --timeout $DURATION
        ;;
    *)
        echo "Error: Unsupported method failure type: $TYPE"
        echo "Supported method failure types: delay, exception"
        exit 1
        ;;
esac

echo "Method chaos experiment started. It will run for $DURATION seconds."
echo "To monitor: blade status"
echo "To stop early: blade destroy <experiment_id>"