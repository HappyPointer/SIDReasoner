#!/bin/bash
#SBATCH --partition=gpu_h100
#SBATCH --gpus=2
#SBATCH --job-name=sid-eval
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --time=08:00:00
#SBATCH --output=slurm_output/%x-%j.out

set -euo pipefail

if [[ -n "${SLURM_SUBMIT_DIR:-}" && -f "${SLURM_SUBMIT_DIR}/pyproject.toml" ]]; then
    SCRIPT_DIR="${SLURM_SUBMIT_DIR}"
else
    SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
fi
cd "$SCRIPT_DIR"

source ./scripts/snellius_env.sh

CATEGORY="${CATEGORY:-Office_Products}"
EXP_NAME="${EXP_NAME:-./output_dir/Office_Products_stage1_sft_Qwen3-1.7B/final_checkpoint}"
TEST_FILE="${TEST_FILE:-./data/Amazon/test/Office_Products_5_2016-10-2018-11.csv}"
INFO_FILE="${INFO_FILE:-./data/Amazon/info/Office_Products_5_2016-10-2018-11.txt}"
CUDA_LIST="${CUDA_LIST:-0 1}"
CUDA_LIST_CSV="${CUDA_LIST_CSV:-0,1}"

dir1=$(basename "$(dirname "$EXP_NAME")")
dir2=$(basename "$EXP_NAME")
exp_name_clean="${dir1}__${dir2}"

echo "Processing category: ${CATEGORY} with model: ${exp_name_clean} (STANDARD MODE)"

if [[ ! -f "${TEST_FILE}" ]]; then
    echo "Error: Test file not found for category ${CATEGORY}"
    exit 1
fi
if [[ ! -f "${INFO_FILE}" ]]; then
    echo "Error: Info file not found for category ${CATEGORY}"
    exit 1
fi

temp_dir="./temp/${CATEGORY}-${exp_name_clean}"
echo "Creating temp directory: ${temp_dir}"
mkdir -p "${temp_dir}"

echo "Splitting test data..."
${PYTHON_CMD} ./split.py --input_path "${TEST_FILE}" --output_path "${temp_dir}" --cuda_list "${CUDA_LIST_CSV}"

echo "Starting parallel evaluation (STANDARD MODE)..."
for i in ${CUDA_LIST}
do
    if [[ -f "${temp_dir}/${i}.csv" ]]; then
        echo "Starting evaluation on GPU ${i} for category ${CATEGORY}"
        CUDA_VISIBLE_DEVICES=${i} ${PYTHON_CMD} -u ./evaluate_Qwen3.py \
            --base_model "${EXP_NAME}" \
            --info_file "${INFO_FILE}" \
            --category "${CATEGORY}" \
            --test_data_path "${temp_dir}/${i}.csv" \
            --result_json_data "${temp_dir}/${i}.json" \
            --batch_size 8 \
            --num_beams 10 \
            --max_new_tokens 256 \
            --length_penalty 0.0 &
    else
        echo "Warning: Split file ${temp_dir}/${i}.csv not found, skipping GPU ${i}"
    fi
done
echo "Waiting for all evaluation processes to complete..."
wait

result_files=$(find "${temp_dir}" -maxdepth 1 -name '*.json' | wc -l)
if [[ ${result_files} -eq 0 ]]; then
    echo "Error: No result files generated for category ${CATEGORY}"
    exit 1
fi

output_dir="./results/${exp_name_clean}"
echo "Creating output directory: ${output_dir}"
mkdir -p "${output_dir}"

actual_cuda_list=""
for gpu in ${CUDA_LIST}; do
    if [[ -f "${temp_dir}/${gpu}.json" ]]; then
        actual_cuda_list="${actual_cuda_list}${gpu},"
    fi
done
actual_cuda_list="${actual_cuda_list%,}"

echo "Merging results from GPUs: ${actual_cuda_list}"

${PYTHON_CMD} ./merge.py \
    --input_path "${temp_dir}" \
    --output_path "${output_dir}/final_result_${CATEGORY}.json" \
    --cuda_list "${actual_cuda_list}"

if [[ ! -f "${output_dir}/final_result_${CATEGORY}.json" ]]; then
    echo "Error: Result merging failed for category ${CATEGORY}"
    exit 1
fi

echo "Calculating metrics..."
${PYTHON_CMD} ./calc.py \
    --path "${output_dir}/final_result_${CATEGORY}.json" \
    --item_path "${INFO_FILE}"

echo "Completed processing for category: ${CATEGORY}"
echo "Results saved to: ${output_dir}/final_result_${CATEGORY}.json"
echo "All categories processed!"
