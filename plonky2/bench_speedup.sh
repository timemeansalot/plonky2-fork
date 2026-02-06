#!/bin/bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BASELINE_FILE="${BASELINE_FILE:-$SCRIPT_DIR/bench_baselines/merkle_cpu.txt}"
GPU_FILE="${GPU_FILE:-/tmp/plonky2_merkle_gpu.txt}"
RUN_BENCH=false

if [ "${1-}" = "--run" ]; then
    RUN_BENCH=true
fi

if [ "$RUN_BENCH" = true ]; then
    echo "Running GPU merkle benchmark (metal) and capturing output..."
    cargo bench --bench=merkle --features=metal 2>&1 | tee "$GPU_FILE"
fi

if [ ! -f "$BASELINE_FILE" ]; then
    echo "Missing CPU baseline: $BASELINE_FILE"
    echo "Run: plonky2/auto_bench.sh 3 13 16 --hash-only --no-lde --cpu-runs 10"
    exit 1
fi

if [ ! -f "$GPU_FILE" ]; then
    echo "Missing GPU log: $GPU_FILE"
    echo "Run with --run or provide GPU_FILE."
    exit 1
fi

extract_time() {
    echo "$1" | sed 's/.*\[//' | sed 's/\].*//' | awk '{print $3, $4}'
}

to_ms() {
    echo "$1" | awk '{
        if ($2 == "ms") print $1
        else if ($2 == "s") print $1 * 1000
        else if ($2 == "us") print $1 / 1000
        else if ($2 == "ns") print $1 / 1000000
        else print $1
    }'
}

echo ""
echo "=== Merkle Tree Building (CPU vs GPU) ==="
printf "%-20s | %-8s | %-12s | %-12s | %-10s\n" "Hash" "Leaves" "CPU" "GPU" "Speedup"
printf "%-20s-+-%-8s-+-%-12s-+-%-12s-+-%-10s\n" "--------------------" "--------" "------------" "------------" "----------"

current_bench=""
while IFS= read -r line; do
    if [[ "$line" == merkle-tree* ]] && [[ "$line" != *Benchmarking* ]] && [[ "$line" != *Warming* ]] && [[ "$line" != *Collecting* ]] && [[ "$line" != *Analyzing* ]]; then
        current_bench="$line"
    elif [[ "$line" == *"time:"* ]] && [[ -n "$current_bench" ]]; then
        cpu_time=$(extract_time "$line")
        cpu_ms=$(to_ms "$cpu_time")

        hash_name=$(echo "$current_bench" | sed 's/.*,[ ]*//' | sed 's/Hash>.*//' | sed 's/>//')
        size=$(echo "$current_bench" | sed 's/.*\///')

        escaped_bench=$(echo "$current_bench" | sed 's/[[\.*^$()+?{|]/\\&/g')
        gpu_line=$(grep -A1 "^${escaped_bench}$" "$GPU_FILE" | grep "time:" || true)
        if [ -n "$gpu_line" ]; then
            gpu_time=$(extract_time "$gpu_line")
            gpu_ms=$(to_ms "$gpu_time")
            if [ "$gpu_ms" != "0" ]; then
                speedup=$(awk "BEGIN {printf \"%.2fx\", $cpu_ms / $gpu_ms}")
            else
                speedup="N/A"
            fi
        else
            gpu_time="N/A"
            speedup="N/A"
        fi

        printf "%-20s | %-8s | %-12s | %-12s | %-10s\n" "$hash_name" "$size" "$cpu_time" "$gpu_time" "$speedup"
        current_bench=""
    fi
done < "$BASELINE_FILE"

