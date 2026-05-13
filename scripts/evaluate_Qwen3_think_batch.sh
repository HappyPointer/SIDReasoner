#!/bin/bash
#SBATCH --partition=gpu_h100
#SBATCH --gpus=3
#SBATCH --job-name=sid-eval-think-batch
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=24
#SBATCH --time=12:00:00
#SBATCH --output=slurm_output/%x-%j.out

set -euo pipefail

if [[ -n "${SLURM_SUBMIT_DIR:-}" && -f "${SLURM_SUBMIT_DIR}/pyproject.toml" ]]; then
    SCRIPT_DIR="${SLURM_SUBMIT_DIR}"
else
    SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
fi
cd "$SCRIPT_DIR"

source ./scripts/snellius_env.sh

# ============ CONFIGS ==============
# Override these with environment variables before sbatch/bash if needed.
cudalist="${CUDA_LIST:-0 1 2}"      # split by space
cudalist_v2="${CUDA_LIST_CSV:-0,1,2}"   # split by comma
# ===================================

exp_list=("${EXP_LIST:-./output_dir/Office_Products_stage2_reasoning_activation_Qwen3-1.7B/final_checkpoint}")
categories="${CATEGORIES:-Office_Products}"
log_file="${LOG_FILE:-./logs/evaluate_Qwen3_think_batch.log}"

mkdir -p ./logs

{
for exp_name in "${exp_list[@]}"
do
for category in ${categories}
do
    # your model path
    # exp_name="output_dir/sft_reasoning-activation_7Task-End2End-GPTGen_Qwen3-1.7B-EP3_Industrial/checkpoint-72"
    dir1=$(basename "$(dirname "$exp_name")")
    dir2=$(basename "$exp_name")
    dir0=$(basename "$(dirname "$(dirname "$exp_name")")")
    exp_name_clean="${dir0}__${dir1}__${dir2}"

    echo "Processing category: $category with model: $exp_name_clean (STANDARD MODE)"
    
    train_file=$(ls ./data/Amazon/train/${category}*.csv 2>/dev/null | head -1)
    test_file=$(ls ./data/Amazon/test/${category}*.csv 2>/dev/null | head -1)
    info_file=$(ls ./data/Amazon/info/${category}*.txt 2>/dev/null | head -1)
    
    if [[ ! -f "$test_file" ]]; then
        echo "Error: Test file not found for category $category"
        continue
    fi
    if [[ ! -f "$info_file" ]]; then
        echo "Error: Info file not found for category $category"
        continue
    fi
    
    temp_dir="./temp/${category}-${exp_name_clean}"
    echo "Creating temp directory: $temp_dir"
    mkdir -p "$temp_dir"
    
    echo "Splitting test data..."
    ${PYTHON_CMD} ./split.py --input_path "$test_file" --output_path "$temp_dir" --cuda_list ${cudalist_v2}
    
    # if [[ ! -f "$temp_dir/0.csv" ]]; then
    #     echo "Error: Data splitting failed for category $category"
    #     continue
    # fi
    
    # cudalist="4 5 6 7"  
    echo "Starting parallel evaluation (STANDARD MODE)..."
    for i in ${cudalist}
    do
        if [[ -f "$temp_dir/${i}.csv" ]]; then
            echo "Starting evaluation on GPU $i for category ${category}"
            CUDA_VISIBLE_DEVICES=$i ${PYTHON_CMD} -u ./evaluate_Qwen3_think.py \
                --base_model "$exp_name" \
                --info_file "$info_file" \
                --category ${category} \
                --test_data_path "$temp_dir/${i}.csv" \
                --item_file ./data/Amazon/index/${category}.item.json \
                --index_file ./data/Amazon/index/${category}.index.json \
                --result_json_data "$temp_dir/${i}.json" \
                --batch_size 4 \
                --num_beams 10 \
                --max_new_tokens 1024 \
                --length_penalty 0.0 &
        else
            echo "Warning: Split file $temp_dir/${i}.csv not found, skipping GPU $i"
        fi
    done
    echo "Waiting for all evaluation processes to complete..."
    wait
    
    result_files=$(ls "$temp_dir"/*.json 2>/dev/null | wc -l)
    if [[ $result_files -eq 0 ]]; then
        echo "Error: No result files generated for category $category"
        continue
    fi
    
    output_dir="./results/${exp_name_clean}"
    echo "Creating output directory: $output_dir"
    mkdir -p "$output_dir"

    actual_cuda_list=""
    for gpu in $cudalist; do
        if [[ -f "$temp_dir/${gpu}.json" ]]; then
            actual_cuda_list="${actual_cuda_list}${gpu},"
        fi
    done
    # eliminate trailing comma
    actual_cuda_list="${actual_cuda_list%,}"

    echo "Merging results from GPUs: $actual_cuda_list"
    
    ${PYTHON_CMD} ./merge.py \
        --input_path "$temp_dir" \
        --output_path "$output_dir/final_result_thinking_${category}.json" \
        --cuda_list "$actual_cuda_list"
    
    if [[ ! -f "$output_dir/final_result_thinking_${category}.json" ]]; then
        echo "Error: Result merging failed for category $category"
        continue
    fi
    
    echo "Calculating metrics..."
    ${PYTHON_CMD} ./calc.py \
        --path "$output_dir/final_result_thinking_${category}.json" \
        --item_path "$info_file"
    
    echo "Completed processing for category: $category"
    echo "Results saved to: $output_dir/final_result_thinking_${category}.json"
    echo "----------------------------------------" 
done
done
echo "All categories processed!"
} > "${log_file}" 2>&1
