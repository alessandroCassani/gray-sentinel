#!/bin/bash

# === CONFIG ===
CHAOSBLADE_DIR="../external/chaosblade/target/chaosblade-1.7.4"
AGENT_JAR="../external/chaosblade/target/chaosblade-1.7.4/lib/sandbox/module/chaosblade-java-agent-1.7.4.jar"
RENAISSANCE_JAR="../external/renaissance/target/renaissance.jar"
BENCHMARK="als"

CPU_PERCENT=80      # CPU load
CPU_CORE_COUNT=1    # number of CPUs to inject load

# === RUN BENCHMARK ===
echo "[*] Avvio ALS con agente ChaosBlade..."
java -javaagent:"$AGENT_JAR" -jar "$RENAISSANCE_JAR" "$BENCHMARK" &
JAVA_PID=$!

echo "[*] PID processo Java: $JAVA_PID"
sleep 5

# === INJECTION CPU PRESSURE ===
echo "[*] Iniezione pressione CPU (${CPU_PERCENT}%) su 1 core, metodo runBenchmark..."
"$CHAOSBLADE_DIR/blade" create jvm cpufull \
  --cpu-percent "$CPU_PERCENT" \
  --cpu-count "$CPU_CORE_COUNT" \
  --class org.renaissance.als.ALS \
  --method runBenchmark \
  --pid "$JAVA_PID"

# ===STATUS ===
echo "[*] Stato delle iniezioni:"
"$CHAOSBLADE_DIR/blade" status

# === WAIT TO FINISH ===
wait "$JAVA_PID"
echo "[*] Benchmark terminato."
