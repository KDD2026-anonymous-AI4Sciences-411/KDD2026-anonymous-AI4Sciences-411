#!/bin/bash

# SE-RL Framework Experiment Runner
# =================================
# This script automates running experiments with different configurations

set -e  # Exit on any error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
print_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Function to show usage
show_usage() {
    echo "Usage: $0 [OPTIONS]"
    echo ""
    echo "Options:"
    echo "  -e, --experiment NAME     Experiment name (required)"
    echo "  -d, --dataset DATASET     Dataset to use (csi100|nasdaq100) [default: csi100]"
    echo "  -m, --mode MODE           Execution mode (full|component_gen|rl_training|framework) [default: full]"
    echo "  -i, --iterations N        Number of outer iterations [default: 50]"
    echo "  -l, --learning-rate LR    Learning rate [default: 3e-4]"
    echo "  -b, --batch-size BS       Batch size [default: 64]"
    echo "  -g, --gpu                 Use GPU if available"
    echo "  -c, --config FILE         Custom config file"
    echo "  -v, --verbose             Verbose output"
    echo "  -h, --help                Show this help message"
    echo ""
    echo "Examples:"
    echo "  $0 -e baseline_csi100 -d csi100 -m full"
    echo "  $0 -e nasdaq_experiment -d nasdaq100 -i 100 -g"
    echo "  $0 -e custom_test -c my_config.yaml -v"
}

# Default values
EXPERIMENT_NAME=""
DATASET="csi100"
MODE="full"
ITERATIONS=50
LEARNING_RATE="3e-4"
BATCH_SIZE=64
USE_GPU=false
CONFIG_FILE=""
VERBOSE=false

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        -e|--experiment)
            EXPERIMENT_NAME="$2"
            shift 2
            ;;
        -d|--dataset)
            DATASET="$2"
            shift 2
            ;;
        -m|--mode)
            MODE="$2"
            shift 2
            ;;
        -i|--iterations)
            ITERATIONS="$2"
            shift 2
            ;;
        -l|--learning-rate)
            LEARNING_RATE="$2"
            shift 2
            ;;
        -b|--batch-size)
            BATCH_SIZE="$2"
            shift 2
            ;;
        -g|--gpu)
            USE_GPU=true
            shift
            ;;
        -c|--config)
            CONFIG_FILE="$2"
            shift 2
            ;;
        -v|--verbose)
            VERBOSE=true
            shift
            ;;
        -h|--help)
            show_usage
            exit 0
            ;;
        *)
            print_error "Unknown option: $1"
            show_usage
            exit 1
            ;;
    esac
done

# Validate required arguments
if [[ -z "$EXPERIMENT_NAME" ]]; then
    print_error "Experiment name is required"
    show_usage
    exit 1
fi

# Validate dataset
if [[ "$DATASET" != "csi100" && "$DATASET" != "nasdaq100" ]]; then
    print_error "Dataset must be either 'csi100' or 'nasdaq100'"
    exit 1
fi

# Validate mode
if [[ "$MODE" != "full" && "$MODE" != "component_gen" && "$MODE" != "rl_training" && "$MODE" != "framework" ]]; then
    print_error "Mode must be one of: full, component_gen, rl_training, framework"
    exit 1
fi

# Check if Python is available
if ! command -v python &> /dev/null; then
    print_error "Python is not installed or not in PATH"
    exit 1
fi

# Check if required packages are installed
print_info "Checking dependencies..."
python -c "import torch, pandas, numpy, yfinance" 2>/dev/null || {
    print_error "Required packages not found. Please install them first:"
    echo "pip install -r requirements.txt"
    exit 1
}

# Set device
DEVICE="cpu"
if [[ "$USE_GPU" == true ]]; then
    if python -c "import torch; print(torch.cuda.is_available())" 2>/dev/null | grep -q "True"; then
        DEVICE="cuda"
        print_info "GPU detected and will be used"
    else
        print_warning "GPU requested but not available, using CPU"
    fi
fi

# Create experiment directory
EXPERIMENT_DIR="experiments/${EXPERIMENT_NAME}_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$EXPERIMENT_DIR"

print_info "Starting experiment: $EXPERIMENT_NAME"
print_info "Dataset: $DATASET"
print_info "Mode: $MODE"
print_info "Device: $DEVICE"
print_info "Output directory: $EXPERIMENT_DIR"

# Build command
CMD="python main.py"
CMD="$CMD --experiment_name $EXPERIMENT_NAME"
CMD="$CMD --dataset $DATASET"
CMD="$CMD --mode $MODE"
CMD="$CMD --max_iterations $ITERATIONS"
CMD="$CMD --learning_rate $LEARNING_RATE"
CMD="$CMD --batch_size $BATCH_SIZE"
CMD="$CMD --device $DEVICE"

if [[ -n "$CONFIG_FILE" ]]; then
    CMD="$CMD --config $CONFIG_FILE"
fi

if [[ "$VERBOSE" == true ]]; then
    CMD="$CMD --verbose"
fi

# Save experiment configuration
cat > "$EXPERIMENT_DIR/experiment_config.txt" << EOF
Experiment Configuration
========================
Name: $EXPERIMENT_NAME
Dataset: $DATASET
Mode: $MODE
Iterations: $ITERATIONS
Learning Rate: $LEARNING_RATE
Batch Size: $BATCH_SIZE
Device: $DEVICE
Config File: $CONFIG_FILE
Start Time: $(date)
Command: $CMD
EOF

# Run the experiment
print_info "Executing command: $CMD"
print_info "Logs will be saved to: $EXPERIMENT_DIR/experiment.log"

# Run with logging
if [[ "$VERBOSE" == true ]]; then
    $CMD 2>&1 | tee "$EXPERIMENT_DIR/experiment.log"
else
    $CMD > "$EXPERIMENT_DIR/experiment.log" 2>&1
fi

# Check exit status
if [[ $? -eq 0 ]]; then
    print_success "Experiment completed successfully!"
    print_info "Results saved in: $EXPERIMENT_DIR"
    
    # Check if results file exists
    if [[ -f "$EXPERIMENT_DIR/results.json" ]]; then
        print_info "Results summary:"
        python -c "
import json
import sys
try:
    with open('$EXPERIMENT_DIR/results.json', 'r') as f:
        results = json.load(f)
    if 'final_metrics' in results:
        metrics = results['final_metrics']
        print(f'  PA: {metrics.get(\"PA\", \"N/A\"):.4f}')
        print(f'  WR: {metrics.get(\"WR\", \"N/A\"):.4f}')
        print(f'  GLR: {metrics.get(\"GLR\", \"N/A\"):.4f}')
        print(f'  AFI: {metrics.get(\"AFI\", \"N/A\"):.4f}')
    else:
        print('  No final metrics found')
except Exception as e:
    print(f'  Could not read results: {e}')
"
    fi
else
    print_error "Experiment failed! Check logs at: $EXPERIMENT_DIR/experiment.log"
    exit 1
fi

print_info "Experiment finished at: $(date)" 