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
TYPE=""
SERVICE=""
TYPE=""

while [[ $# -gt 0 ]]
