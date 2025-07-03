#!/bin/bash

function show_help() {
    echo "Usage: $0 [OPTION]"
    echo "Inject JVM failures using ChaosBlade for PetClinic services"
    echo ""
    echo "Options:"
    echo "  -h, --help        Show this help message"
    echo "  -s, --service     SERVICE Target specific service (required)"
    echo "  -t, --type        TYPE JVM failure type"
    echo "  -d, --duration    SECONDS Experiment duration (default: 300s)"
    echo "  -l, --list        List running experiments"
    echo "  -c, --clean       Clean all experiments"
    echo ""
    echo "JVM Failure Types:"
    echo "  oom               OutOfMemoryError (HEAP area)"
    echo "  cpufulload        High CPU load within JVM"
    echo "  gc                Frequent garbage collection"
    echo "  throwCustomException  Custom exception in Gateway filter"
    echo "  tde               Declared exception in service discovery"
    echo ""
    echo "Examples:"
    echo "  $0 -s api-gateway -t throwCustomException"
    echo "  $0 -s customers-service -t oom -d 120"
    echo "  $0 -s api-gateway -t gc -d 600"
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
    echo "Active ChaosBlade JVM experiments..."
    # List experiments from all PetClinic containers
    for container in api-gateway customers-service visits-service vets-service; do
        container_id=$(docker ps | grep $container | awk '{print $1}')
        if [ ! -z "$container_id" ]; then
            echo "=== $container ($container_id) ==="
            docker exec $container_id /opt/chaosblade-1.7.2/blade status 2>/dev/null || echo "No experiments running"
        fi
    done
    exit 0
fi

if [[ "$ACTION" == "clean" ]]; then
    echo "Cleaning previous ChaosBlade JVM experiments"
    for container in api-gateway customers-service visits-service vets-service; do
        container_id=$(docker ps | grep $container | awk '{print $1}')
        if [ ! -z "$container_id" ]; then
            echo "Cleaning experiments in $container..."
            docker exec $container_id /opt/chaosblade-1.7.2/blade destroy 2>/dev/null || echo "No experiments to clean"
        fi
    done
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
echo "Injecting JVM failure: $TYPE for $DURATION seconds..."

case $TYPE in
    oom)
        # Trigger OutOfMemoryError in HEAP area
        echo "Injecting OutOfMemoryError..."
        docker exec $CONTAINER_ID /opt/chaosblade-1.7.2/blade create jvm oom --area HEAP --timeout $DURATION
        ;;
    cpufulload)
        # Create high CPU load within the JVM
        echo "Injecting high CPU load in JVM..."
        docker exec $CONTAINER_ID /opt/chaosblade-1.7.2/blade create jvm cpufull --timeout $DURATION
        ;;
    gc)
        # Trigger frequent garbage collection
        # 100 full garbage collection cycles with a 1-second interval
        echo "Triggering frequent garbage collection..."
        docker exec $CONTAINER_ID /opt/chaosblade-1.7.2/blade create jvm full-gc --effect-count 100 --interval 1000 --timeout $DURATION
        ;;
    throwCustomException)
        # Throw custom exception in Gateway filter chain
        echo "Injecting custom exception in Gateway filter..."
        if [[ "$SERVICE" == "api-gateway" ]]; then
            CLASS="org.springframework.cloud.gateway.filter.GatewayFilterChain"
            METHOD="filter"
        else
            CLASS="org.springframework.web.servlet.DispatcherServlet"
            METHOD="doDispatch"
        fi
        docker exec $CONTAINER_ID /opt/chaosblade-1.7.2/blade create jvm throwCustomException \
            --classname $CLASS \
            --methodname $METHOD \
            --exception java.lang.RuntimeException \
            --exception-message "PetClinic JVM chaos failure" \
            --timeout $DURATION
        ;;
    tde|throwDeclaredException)
        # Throw the first declared exception of a method
        echo "Injecting declared exception..."
        if [[ "$SERVICE" == "api-gateway" ]]; then
            CLASS="org.springframework.cloud.netflix.eureka.EurekaDiscoveryClient"
            METHOD="getServices"
        else
            CLASS="org.springframework.web.servlet.DispatcherServlet"
            METHOD="doDispatch"
        fi
        docker exec $CONTAINER_ID /opt/chaosblade-1.7.2/blade create jvm throwDeclaredException \
            --classname $CLASS \
            --methodname $METHOD \
            --timeout $DURATION
        ;;
    *)
        echo "Unknown JVM failure type: $TYPE"
        echo "Available types: oom, cpufulload, gc, throwCustomException, tde"
        exit 1
        ;;
esac

echo "JVM chaos experiment started. It will run for $DURATION seconds."
echo "To monitor: $0 -l"
echo "To stop early: $0 -c"