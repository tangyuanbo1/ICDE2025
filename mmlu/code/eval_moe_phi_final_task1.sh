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
)
gpu_util=0.9
moe_paths=(
    "../phi_mask/computer_science_baseline6.pth"
    "../phi_mask/computer_science_baseline8.pth"
    "../phi_mask/computer_science_baseline10.pth"
    "../phi_mask/computer_science_baseline12.pth"
    "../phi_mask/computer_science_baseline14.pth"
    "../phi_mask/computer_science_baseline16.pth"
    "../phi_mask/computer_science_ours6_global.pth"
    "../phi_mask/computer_science_ours8_global.pth"
    "../phi_mask/computer_science_ours10_global.pth"
    "../phi_mask/computer_science_ours12_global.pth"
    "../phi_mask/computer_science_ours14_global.pth"
    "../phi_mask/computer_science_ours16_global.pth"
)

for moe_path in "${moe_paths[@]}"; do
    echo "Evaluating model with moe_path: $moe_path"
    for selected_subject in "${selected_subjects[@]}"; do
        echo "Evaluating subject: $selected_subject"
        /home/naifanzhang/.conda/envs/eedi/bin/python evaluate_from_local_moe_phi.py \
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

save_dir="eval_results/"
global_record_file="eval_results/eval_record_collection.csv"
model="./model/LLM-Research/Phi-3.5-MoE-instruct"
selected_subjects=(
    "law"

)
gpu_util=0.9
moe_paths=(
    "../phi_mask/law_baseline6.pth"
    "../phi_mask/law_baseline8.pth"
    "../phi_mask/law_baseline10.pth"
    "../phi_mask/law_baseline12.pth"
    "../phi_mask/law_baseline14.pth"
    "../phi_mask/law_baseline16.pth"
    "../phi_mask/law_ours6_global.pth"
    "../phi_mask/law_ours8_global.pth"
    "../phi_mask/law_ours10_global.pth"
    "../phi_mask/law_ours12_global.pth"
    "../phi_mask/law_ours14_global.pth"
    "../phi_mask/law_ours16_global.pth"

)

for moe_path in "${moe_paths[@]}"; do
    echo "Evaluating model with moe_path: $moe_path"
    for selected_subject in "${selected_subjects[@]}"; do
        echo "Evaluating subject: $selected_subject"
        /home/naifanzhang/.conda/envs/eedi/bin/python evaluate_from_local_moe_phi.py \
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

save_dir="eval_results/"
global_record_file="eval_results/eval_record_collection.csv"
model="./model/LLM-Research/Phi-3.5-MoE-instruct"
selected_subjects=(
    "math"
)
gpu_util=0.9
moe_paths=(
    "../phi_mask/math_baseline6.pth"
    "../phi_mask/math_baseline8.pth"
    "../phi_mask/math_baseline10.pth"
    "../phi_mask/math_baseline12.pth"
    "../phi_mask/math_baseline14.pth"
    "../phi_mask/math_baseline16.pth"
    "../phi_mask/math_ours6_global.pth"
    "../phi_mask/math_ours8_global.pth"
    "../phi_mask/math_ours10_global.pth"
    "../phi_mask/math_ours12_global.pth"
    "../phi_mask/math_ours14_global.pth"
    "../phi_mask/math_ours16_global.pth"
)

for moe_path in "${moe_paths[@]}"; do
    echo "Evaluating model with moe_path: $moe_path"
    for selected_subject in "${selected_subjects[@]}"; do
        echo "Evaluating subject: $selected_subject"
        /home/naifanzhang/.conda/envs/eedi/bin/python evaluate_from_local_moe_phi.py \
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

save_dir="eval_results/"
global_record_file="eval_results/eval_record_collection.csv"
model="./model/LLM-Research/Phi-3.5-MoE-instruct"
selected_subjects=(
    "physics"
)
gpu_util=0.9
moe_paths=(
    "../phi_mask/physics_baseline6.pth"
    "../phi_mask/physics_baseline8.pth"
    "../phi_mask/physics_baseline10.pth"
    "../phi_mask/physics_baseline12.pth"
    "../phi_mask/physics_baseline14.pth"
    "../phi_mask/physics_baseline16.pth"
    "../phi_mask/physics_ours6_global.pth"
    "../phi_mask/physics_ours8_global.pth"
    "../phi_mask/physics_ours10_global.pth"
    "../phi_mask/physics_ours12_global.pth"
    "../phi_mask/physics_ours14_global.pth"
    "../phi_mask/physics_ours16_global.pth"
)

for moe_path in "${moe_paths[@]}"; do
    echo "Evaluating model with moe_path: $moe_path"
    for selected_subject in "${selected_subjects[@]}"; do
        echo "Evaluating subject: $selected_subject"
        /home/naifanzhang/.conda/envs/eedi/bin/python evaluate_from_local_moe_phi.py \
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

save_dir="eval_results/"
global_record_file="eval_results/eval_record_collection.csv"
model="./model/LLM-Research/Phi-3.5-MoE-instruct"
selected_subjects=(
    "psychology"
)
gpu_util=0.9
moe_paths=(
    "../phi_mask/psychology_baseline6.pth"
    "../phi_mask/psychology_baseline8.pth"
    "../phi_mask/psychology_baseline10.pth"
    "../phi_mask/psychology_baseline12.pth"
    "../phi_mask/psychology_baseline14.pth"
    "../phi_mask/psychology_baseline16.pth"
    "../phi_mask/psychology_ours6_global.pth"
    "../phi_mask/psychology_ours8_global.pth"
    "../phi_mask/psychology_ours10_global.pth"
    "../phi_mask/psychology_ours12_global.pth"
    "../phi_mask/psychology_ours14_global.pth"
    "../phi_mask/psychology_ours16_global.pth"
)

for moe_path in "${moe_paths[@]}"; do
    echo "Evaluating model with moe_path: $moe_path"
    for selected_subject in "${selected_subjects[@]}"; do
        echo "Evaluating subject: $selected_subject"
        /home/naifanzhang/.conda/envs/eedi/bin/python evaluate_from_local_moe_phi.py \
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