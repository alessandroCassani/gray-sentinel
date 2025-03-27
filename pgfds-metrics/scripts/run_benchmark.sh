#!/bin/bash

#####################################################
# Media Microservices Benchmark Automation Script
#####################################################

set -e

# Configuration variables (modify as needed)
WRK_BINARY="../wrk/wrk"
WRK_SCRIPT="./wrk2/scripts/media-microservices/compose-review.lua"
TARGET_HOST="localhost"
TARGET_PORT="8080"
DURATION="5m"       # Duration to run the benchmark
THREADS=4           # Number of threads for wrk2
CONNECTIONS=100     # Number of connections for wrk2
RATES=(100)         # Requests per second to test
LOG_DIR="benchmark_logs"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
FAILURE_INJECTION=0 # Set to 1 to inject failures
CHAOS_TOOL="./blade"


# Create directory for logs
mkdir -p "${LOG_DIR}/${TIMESTAMP}"

MAIN_LOG="${LOG_DIR}/${TIMESTAMP}/benchmark_run.log"

# Function to log messages
log() {
    local message="$1"
    local timestamp=$(date +"%Y-%m-%d %H:%M:%S")
    echo "${timestamp} - ${message}" | tee -a "${MAIN_LOG}"
}


start_application() {
    log "Starting Media Microservices application with Docker Compose..."
    
    docker-compose up -d
    
    log "Waiting for the application to be ready..."
    sleep 10
    
    # Register users and movie information
    log "Registering users and movie information..."
    python3 scripts/write_movie_info.py -c datasets/tmdb/casts.json -m datasets/tmdb/movies.json --server_address http://${TARGET_HOST}:${TARGET_PORT}
    ./scripts/register_users.sh
    ./scripts/register_movies.sh
    
    log "Media Microservices application is running and initialized."
}

# Function to inject failures (if enabled)
inject_failures() {
    log "Injecting failures using ChaosBlade..."
    
    if [ ! -f "${CHAOS_TOOL}" ]; then
        log "ERROR: ChaosBlade tool not found at ${CHAOS_TOOL}."
        log "Please install ChaosBlade and set the correct path."
        return 1
    fi
    
    # Example: CPU load injection to simulate partial gray failure
    log "Injecting CPU load gray failure..."
    FAILURE_ID=$(${CHAOS_TOOL} create cpu fullload --cpu-percent 50 | grep -o '"result":"[^"]*"' | cut -d'"' -f4)
    
    if [ -z "${FAILURE_ID}" ]; then
        log "WARNING: Failed to inject CPU load failure."
    else
        log "CPU load failure injected with ID: ${FAILURE_ID}"
        echo "${FAILURE_ID}" > "${LOG_DIR}/${TIMESTAMP}/failure_id.txt"
    fi
}

# Function to cleanup failures if injected
cleanup_failures() {
    if [ "${FAILURE_INJECTION}" -eq 1 ] && [ -f "${LOG_DIR}/${TIMESTAMP}/failure_id.txt" ]; then
        FAILURE_ID=$(cat "${LOG_DIR}/${TIMESTAMP}/failure_id.txt")
        log "Cleaning up injected failure with ID: ${FAILURE_ID}"
        ${CHAOS_TOOL} destroy ${FAILURE_ID}
        log "Failure cleanup completed."
    fi
}

# Function to run the benchmark
run_benchmark() {
    log "Running benchmark..."
    
    for rate in "${RATES[@]}"; do
        log "Running benchmark with ${rate} requests per second..."
        
        ${WRK_BINARY} -t ${THREADS} -c ${CONNECTIONS} -d ${DURATION} -L -s ${WRK_SCRIPT} http://${TARGET_HOST}:${TARGET_PORT}/wrk2-api/review/compose -R ${rate} > "${LOG_DIR}/${TIMESTAMP}/benchmark_${rate}_rps.log" 2>&1
        
        log "Completed benchmark with ${rate} requests per second."
    done
}


main() {
    log "Starting benchmark process for Media Microservices..."
    
    start_application
    
    if [ "${FAILURE_INJECTION}" -eq 1 ]; then
        inject_failures
    fi
    
    run_benchmark
    
    log "Benchmark process completed successfully."
    log "Logs are available in: ${LOG_DIR}/${TIMESTAMP}/"
}

main "$@"