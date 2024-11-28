#!/bin/bash 
#SBATCH --nodes=1 
#SBATCH --ntasks=1
#SBATCH --mem=0
eval "$(conda shell.bash hook)"
conda activate eedi

save_dir="eval_results/"
global_record_file="eval_results/eval_record_collection.csv"
model="./model/LLM-Research/Phi-3.5-MoE-instruct"
selected_subjects=(
    "computer science"
    "law"
    "math"
    "physics"
    "psychology"
)
gpu_util=0.9
moe_paths=(
    "../phi_mask/computer_science_ours8_global.pth"
    "../phi_mask/law_ours8_global.pth"
    "../phi_mask/math_ours8_global.pth"
    "../phi_mask/physics_ours8_global.pth"
    "../phi_mask/psychology_ours8_global.pth"
)

for moe_path in "${moe_paths[@]}"; do
    echo "Evaluating model with moe_path: $moe_path"
    for selected_subject in "${selected_subjects[@]}"; do
        echo "Evaluating subject: $selected_subject"
        /home/naifanzhang/.conda/envs/eedi/bin/python evaluate_from_local_moe_phi_task2.py \
            --selected_subjects "$selected_subject" \
            --save_dir $save_dir \
            --model $model \
            --global_record_file $global_record_file \
            --gpu_util $gpu_util \
            --moe_path "$moe_path"
    done  
    echo "Finished evaluation for: $moe_path"
    echo "----------------------------------------"
done