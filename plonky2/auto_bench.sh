#!/bin/bash

# Plonky2 GPU vs CPU Benchmark Script
# This script runs E2E proving benchmarks comparing true GPU vs CPU performance
# Automatically detects OS and uses appropriate GPU backend (CUDA for Linux, Metal for macOS)
#
# Usage: ./auto_bench.sh [ROUNDS] [START] [END] [--cpu-only]
#   ROUNDS    - Number of benchmark rounds (default: 3)
#   START     - Starting log_size (default: 13)
#   END       - Ending log_size (default: 18)
#   --cpu-only - Run CPU-only benchmarks (skip GPU)
#
# Examples:
#   ./auto_bench.sh              # Use defaults
#   ./auto_bench.sh 5 14 20      # 5 rounds, log_size 14-20
#   ./auto_bench.sh --cpu-only   # CPU-only mode

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Check for --cpu-only flag
CPU_ONLY=false
for arg in "$@"; do
    if [ "$arg" = "--cpu-only" ]; then
        CPU_ONLY=true
    fi
done

# Parse command line arguments (skip --cpu-only)
args=()
for arg in "$@"; do
    if [ "$arg" != "--cpu-only" ]; then
        args+=("$arg")
    fi
done

ROUNDS=${args[0]:-3}
START=${args[1]:-13}
END=${args[2]:-18}

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Detect OS and set GPU feature accordingly
OS_TYPE="$(uname -s)"
case "$OS_TYPE" in
    Linux*)
        GPU_FEATURE="cuda"
        GPU_ENV="NUM_OF_GPUS=1"
        OS_NAME="Linux"
        GPU_IMPLEMENTED=true
        ;;
    Darwin*)
        GPU_FEATURE="metal"
        GPU_ENV=""
        OS_NAME="macOS"
        GPU_IMPLEMENTED=false  # Metal not yet implemented
        ;;
    *)
        echo -e "${RED}Error: Unsupported OS: $OS_TYPE${NC}"
        exit 1
        ;;
esac

# Temp files for results
GOLD_GPU_FILE="/tmp/plonky2_gold_gpu.txt"
GOLD_CPU_FILE="/tmp/plonky2_gold_cpu.txt"
BN128_GPU_FILE="/tmp/plonky2_bn128_gpu.txt"
BN128_CPU_FILE="/tmp/plonky2_bn128_cpu.txt"
MERKLE_GPU_FILE="/tmp/plonky2_merkle_gpu.txt"
MERKLE_CPU_FILE="/tmp/plonky2_merkle_cpu.txt"
LDE_GPU_FILE="/tmp/plonky2_lde_gpu.txt"
LDE_CPU_FILE="/tmp/plonky2_lde_cpu.txt"

echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}  Plonky2 GPU vs CPU Benchmark Suite   ${NC}"
echo -e "${BLUE}========================================${NC}"
echo ""
echo -e "Configuration:"
echo -e "  OS:     ${GREEN}$OS_NAME${NC}"
echo -e "  GPU:    ${GREEN}$GPU_FEATURE${NC}"
echo -e "  ROUNDS: ${GREEN}$ROUNDS${NC}"
echo -e "  START:  ${GREEN}$START${NC}"
echo -e "  END:    ${GREEN}$END${NC}"
echo ""

# Handle Metal (not yet implemented)
if [ "$GPU_FEATURE" = "metal" ] && [ "$GPU_IMPLEMENTED" = false ] && [ "$CPU_ONLY" = false ]; then
    echo -e "${YELLOW}WARNING: Metal GPU acceleration is not yet implemented.${NC}"
    echo -e "${YELLOW}The 'metal' feature is a placeholder for future development.${NC}"
    echo -e "${YELLOW}See metal_draft.md for the migration plan.${NC}"
    echo ""
    echo -e "Running in ${GREEN}CPU-only${NC} mode..."
    echo ""
    CPU_ONLY=true
fi

# Check for GPU feature (only if not CPU-only mode)
if [ "$CPU_ONLY" = false ]; then
    echo -e "${YELLOW}Checking for $GPU_FEATURE support...${NC}"
    if ! cargo check --features=$GPU_FEATURE 2>/dev/null; then
        echo -e "${RED}Error: $GPU_FEATURE feature not available.${NC}"
        if [ "$GPU_FEATURE" = "cuda" ]; then
            echo -e "${RED}Make sure CUDA is installed and configured.${NC}"
        else
            echo -e "${RED}Make sure Metal support is available.${NC}"
        fi
        exit 1
    fi
    echo -e "${GREEN}$GPU_FEATURE support detected!${NC}"
    echo ""

    # Build all benchmarks (both CPU and GPU versions)
    echo -e "${YELLOW}Building benchmarks (GPU version with $GPU_FEATURE)...${NC}"
    cargo build --release --features=$GPU_FEATURE \
        --example bench_e2e_prove \
        --example bench_bn128 2>/dev/null
fi

echo -e "${YELLOW}Building benchmarks (CPU version)...${NC}"
cargo build --release \
    --example bench_e2e_prove \
    --example bench_bn128 2>/dev/null

echo -e "${GREEN}Build complete!${NC}"
echo ""

# ========================================
# Part 1: Criterion Benchmarks (Merkle & LDE)
# ========================================
if [ "$CPU_ONLY" = false ]; then
    echo -e "${BLUE}========================================${NC}"
    echo -e "${BLUE}  Part 1: Primitive Operations         ${NC}"
    echo -e "${BLUE}  (Merkle Tree & LDE+Merkle)           ${NC}"
    echo -e "${BLUE}========================================${NC}"
    echo ""

    echo -e "${YELLOW}Running Merkle Tree benchmark (CPU)...${NC}"
    cargo bench --bench=merkle 2>&1 | tee "$MERKLE_CPU_FILE"
    echo -e "${GREEN}Merkle CPU complete!${NC}"
    echo ""

    echo -e "${YELLOW}Running Merkle Tree benchmark (GPU with $GPU_FEATURE)...${NC}"
    env $GPU_ENV cargo bench --bench=merkle --features=$GPU_FEATURE 2>&1 | tee "$MERKLE_GPU_FILE"
    echo -e "${GREEN}Merkle GPU complete!${NC}"
    echo ""

    echo -e "${YELLOW}Running LDE+Merkle benchmark (CPU)...${NC}"
    cargo bench --bench=lde 2>&1 | tee "$LDE_CPU_FILE"
    echo -e "${GREEN}LDE CPU complete!${NC}"
    echo ""

    echo -e "${YELLOW}Running LDE+Merkle benchmark (GPU with $GPU_FEATURE)...${NC}"
    env $GPU_ENV cargo bench --bench=lde --features=$GPU_FEATURE 2>&1 | tee "$LDE_GPU_FILE"
    echo -e "${GREEN}LDE GPU complete!${NC}"
    echo ""

    # Display Criterion benchmark results
    echo -e "${BLUE}========================================${NC}"
    echo -e "${BLUE}  PRIMITIVE BENCHMARK RESULTS          ${NC}"
    echo -e "${BLUE}========================================${NC}"
    echo ""
fi

# Function to extract time from criterion output line
# Input: "                        time:   [24.711 ms 26.693 ms 27.914 ms]"
# Output: "26.693 ms" (the middle/median value)
extract_time() {
    echo "$1" | sed 's/.*\[//' | sed 's/\].*//' | awk '{print $3, $4}'
}

# Function to convert time to milliseconds
to_ms() {
    echo "$1" | awk '{
        if ($2 == "ms") print $1
        else if ($2 == "s") print $1 * 1000
        else if ($2 == "us") print $1 / 1000
        else if ($2 == "ns") print $1 / 1000000
        else print $1
    }'
}

# Function to display Merkle tree benchmark results
display_merkle_comparison() {
    local gpu_file=$1
    local cpu_file=$2

    echo -e "${GREEN}=== Merkle Tree Building ===${NC}"
    echo ""
    printf "%-20s | %-8s | %-12s | %-12s | %-10s\n" "Hash" "Leaves" "CPU" "GPU" "Speedup"
    printf "%-20s-+-%-8s-+-%-12s-+-%-12s-+-%-10s\n" "--------------------" "--------" "------------" "------------" "----------"

    # Process each benchmark from the CPU file
    local current_bench=""
    while IFS= read -r line; do
        # Match result line: starts with "merkle-tree" but NOT with "Benchmarking"
        if [[ "$line" == merkle-tree* ]] && [[ "$line" != *Benchmarking* ]] && [[ "$line" != *Warming* ]] && [[ "$line" != *Collecting* ]] && [[ "$line" != *Analyzing* ]]; then
            # Take the whole line as benchmark name (it may contain spaces)
            current_bench="$line"
        elif [[ "$line" == *"time:"* ]] && [[ -n "$current_bench" ]]; then
            local cpu_time=$(extract_time "$line")

            # Extract hash name and size from benchmark name
            # merkle-tree<GoldilocksField, PoseidonHash>/8192 -> Poseidon, 8192
            local hash_name=$(echo "$current_bench" | sed 's/.*,[ ]*//' | sed 's/Hash>.*//' | sed 's/>//')
            local size=$(echo "$current_bench" | sed 's/.*\///')

            # Find corresponding GPU time (escape special chars for grep)
            local escaped_bench=$(echo "$current_bench" | sed 's/[[\.*^$()+?{|]/\\&/g')
            local gpu_line=$(grep -A1 "^${escaped_bench}$" "$gpu_file" | grep "time:")
            if [[ -n "$gpu_line" ]]; then
                local gpu_time=$(extract_time "$gpu_line")

                local cpu_ms=$(to_ms "$cpu_time")
                local gpu_ms=$(to_ms "$gpu_time")

                if [[ -n "$cpu_ms" ]] && [[ -n "$gpu_ms" ]] && [[ "$gpu_ms" != "0" ]]; then
                    local speedup=$(awk "BEGIN {printf \"%.2f\", $cpu_ms / $gpu_ms}")
                    printf "%-20s | %-8s | %-12s | %-12s | %-10s\n" "$hash_name" "$size" "$cpu_time" "$gpu_time" "${speedup}x"
                fi
            fi
            current_bench=""
        fi
    done < "$cpu_file"
    echo ""
}

# Function to display LDE benchmark results
display_lde_comparison() {
    local gpu_file=$1
    local cpu_file=$2

    echo -e "${GREEN}=== LDE + Merkle Tree Building ===${NC}"
    echo ""
    printf "%-15s | %-12s | %-12s | %-10s\n" "LDE Size (log)" "CPU" "GPU" "Speedup"
    printf "%-15s-+-%-12s-+-%-12s-+-%-10s\n" "---------------" "------------" "------------" "----------"

    # Process each benchmark from the CPU file
    local current_bench=""
    while IFS= read -r line; do
        # Match result line: starts with "lde" but NOT with "Benchmarking"
        if [[ "$line" == lde* ]] && [[ "$line" != *Benchmarking* ]] && [[ "$line" != *Warming* ]] && [[ "$line" != *Collecting* ]] && [[ "$line" != *Analyzing* ]]; then
            # Take the whole line as benchmark name
            current_bench="$line"
        elif [[ "$line" == *"time:"* ]] && [[ -n "$current_bench" ]]; then
            local cpu_time=$(extract_time "$line")

            # Extract size and calculate log
            local size=$(echo "$current_bench" | sed 's/.*\///')
            local log_size=$(awk "BEGIN {printf \"%.0f\", log($size)/log(2)}")

            # Find corresponding GPU time (escape special chars for grep)
            local escaped_bench=$(echo "$current_bench" | sed 's/[[\.*^$()+?{|]/\\&/g')
            local gpu_line=$(grep -A1 "^${escaped_bench}$" "$gpu_file" | grep "time:")
            if [[ -n "$gpu_line" ]]; then
                local gpu_time=$(extract_time "$gpu_line")

                local cpu_ms=$(to_ms "$cpu_time")
                local gpu_ms=$(to_ms "$gpu_time")

                if [[ -n "$cpu_ms" ]] && [[ -n "$gpu_ms" ]] && [[ "$gpu_ms" != "0" ]]; then
                    local speedup=$(awk "BEGIN {printf \"%.2f\", $cpu_ms / $gpu_ms}")
                    printf "%-15s | %-12s | %-12s | %-10s\n" "$log_size" "$cpu_time" "$gpu_time" "${speedup}x"
                fi
            fi
            current_bench=""
        fi
    done < "$cpu_file"
    echo ""
}

if [ "$CPU_ONLY" = false ]; then
    display_merkle_comparison "$MERKLE_GPU_FILE" "$MERKLE_CPU_FILE"
    display_lde_comparison "$LDE_GPU_FILE" "$LDE_CPU_FILE"
fi

# ========================================
# Part 2: E2E Prove Benchmarks
# ========================================
echo -e "${BLUE}========================================${NC}"
if [ "$CPU_ONLY" = true ]; then
    echo -e "${BLUE}  CPU-Only Benchmarks                  ${NC}"
else
    echo -e "${BLUE}  Part 2: End-to-End Proving           ${NC}"
fi
echo -e "${BLUE}========================================${NC}"
echo ""

# Function to run GPU benchmark and save results
run_gpu_benchmark() {
    local name=$1
    local output_file=$2
    local example=$3

    echo -e "${YELLOW}Running $name (GPU with $GPU_FEATURE)...${NC}"
    env $GPU_ENV cargo run --release --features=$GPU_FEATURE --example "$example" -- "$ROUNDS" "$START" "$END" 2>&1 | \
        grep -E "(Circuit size|^Prove:)" > "$output_file"
    echo -e "${GREEN}$name (GPU) complete!${NC}"
}

# Function to run CPU benchmark and save results (no GPU feature)
run_cpu_benchmark() {
    local name=$1
    local output_file=$2
    local example=$3

    echo -e "${YELLOW}Running $name (CPU)...${NC}"
    cargo run --release --example "$example" -- "$ROUNDS" "$START" "$END" 2>&1 | \
        grep -E "(Circuit size|^Prove:)" > "$output_file"
    echo -e "${GREEN}$name (CPU) complete!${NC}"
}

# Run all E2E benchmarks
echo -e "${BLUE}--- Running Goldilocks Benchmarks ---${NC}"
if [ "$CPU_ONLY" = false ]; then
    run_gpu_benchmark "Goldilocks" "$GOLD_GPU_FILE" "bench_e2e_prove"
fi
run_cpu_benchmark "Goldilocks" "$GOLD_CPU_FILE" "bench_e2e_prove"

echo ""
echo -e "${BLUE}--- Running BN128 Benchmarks ---${NC}"
if [ "$CPU_ONLY" = false ]; then
    run_gpu_benchmark "BN128" "$BN128_GPU_FILE" "bench_bn128"
fi
run_cpu_benchmark "BN128" "$BN128_CPU_FILE" "bench_bn128"

echo ""
echo -e "${BLUE}========================================${NC}"
if [ "$CPU_ONLY" = true ]; then
    echo -e "${BLUE}      CPU BENCHMARK RESULTS            ${NC}"
else
    echo -e "${BLUE}      E2E BENCHMARK RESULTS            ${NC}"
fi
echo -e "${BLUE}========================================${NC}"
echo ""

# Function to display comparison table
display_comparison() {
    local title=$1
    local gpu_file=$2
    local cpu_file=$3

    echo -e "${GREEN}=== $title ===${NC}"
    echo ""
    printf "%-8s | %-12s | %-12s | %-10s\n" "Degree" "CPU Prove" "GPU Prove" "Speedup"
    printf "%-8s-+-%-12s-+-%-12s-+-%-10s\n" "--------" "------------" "------------" "----------"

    # Extract circuit sizes and prove times
    local degrees=($(grep "Circuit size" "$gpu_file" | sed 's/.*2\^\([0-9]*\).*/\1/'))
    local gpu_times=($(grep "^Prove:" "$gpu_file" | sed 's/Prove: //'))
    local cpu_times=($(grep "^Prove:" "$cpu_file" | sed 's/Prove: //'))

    for i in "${!degrees[@]}"; do
        local deg="${degrees[$i]}"
        local gpu="${gpu_times[$i]}"
        local cpu="${cpu_times[$i]}"

        # Convert times to milliseconds for speedup calculation
        local gpu_ms=$(echo "$gpu" | sed 's/ms$//' | sed 's/s$//' | awk '{
            if ($0 ~ /ms$/) { gsub(/ms$/, ""); print $0 }
            else if ($0 ~ /s$/) { gsub(/s$/, ""); print $0 * 1000 }
            else if ($0 ~ /\..*s/) { gsub(/s$/, ""); print $0 * 1000 }
            else { print $0 }
        }')
        local cpu_ms=$(echo "$cpu" | sed 's/ms$//' | sed 's/s$//' | awk '{
            if ($0 ~ /ms$/) { gsub(/ms$/, ""); print $0 }
            else if ($0 ~ /s$/) { gsub(/s$/, ""); print $0 * 1000 }
            else if ($0 ~ /\..*s/) { gsub(/s$/, ""); print $0 * 1000 }
            else { print $0 }
        }')

        # Calculate speedup
        local speedup=$(awk "BEGIN {printf \"%.2f\", $cpu_ms / $gpu_ms}")

        printf "%-8s | %-12s | %-12s | %-10s\n" "$deg" "$cpu" "$gpu" "${speedup}x"
    done
    echo ""
}

# Function to display CPU-only results
display_cpu_only() {
    local title=$1
    local cpu_file=$2

    echo -e "${GREEN}=== $title ===${NC}"
    echo ""
    printf "%-8s | %-12s\n" "Degree" "CPU Prove"
    printf "%-8s-+-%-12s\n" "--------" "------------"

    local degrees=($(grep "Circuit size" "$cpu_file" | sed 's/.*2\^\([0-9]*\).*/\1/'))
    local cpu_times=($(grep "^Prove:" "$cpu_file" | sed 's/Prove: //'))

    for i in "${!degrees[@]}"; do
        local deg="${degrees[$i]}"
        local cpu="${cpu_times[$i]}"
        printf "%-8s | %-12s\n" "$deg" "$cpu"
    done
    echo ""
}

# Display results
if [ "$CPU_ONLY" = true ]; then
    display_cpu_only "Goldilocks (64-bit field)" "$GOLD_CPU_FILE"
    display_cpu_only "BN128 (254-bit hashing)" "$BN128_CPU_FILE"
else
    display_comparison "Goldilocks (64-bit field)" "$GOLD_GPU_FILE" "$GOLD_CPU_FILE"
    display_comparison "BN128 (254-bit hashing)" "$BN128_GPU_FILE" "$BN128_CPU_FILE"
fi

echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}             SUMMARY                   ${NC}"
echo -e "${BLUE}========================================${NC}"
if [ "$CPU_ONLY" = true ]; then
    echo ""
    echo -e "${YELLOW}Note: Running in CPU-only mode (Metal not yet implemented)${NC}"
    echo "See metal_draft.md for the Metal migration plan."
    echo ""
    echo "Results saved to:"
    echo "  - $GOLD_CPU_FILE"
    echo "  - $BN128_CPU_FILE"
else
    echo "Results saved to:"
    echo "  Primitive benchmarks:"
    echo "    - $MERKLE_CPU_FILE"
    echo "    - $MERKLE_GPU_FILE"
    echo "    - $LDE_CPU_FILE"
    echo "    - $LDE_GPU_FILE"
    echo "  E2E benchmarks:"
    echo "    - $GOLD_GPU_FILE"
    echo "    - $GOLD_CPU_FILE"
    echo "    - $BN128_GPU_FILE"
    echo "    - $BN128_CPU_FILE"
fi
echo ""
echo -e "${GREEN}Benchmark complete!${NC}"
