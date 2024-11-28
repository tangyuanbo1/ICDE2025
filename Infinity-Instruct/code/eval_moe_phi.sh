#!/bin/bash 
#SBATCH --nodes=1 
#SBATCH --ntasks=1
#SBATCH --mem=0
#SBATCH --gpus=1
eval "$(conda shell.bash hook)"
conda activate eedi

model="/data/naifanzhang/model/LLM-Research/Phi-3.5-MoE-instruct"
data_paths=(
    "../Infinity-Instruct-data/finance.parquet"
    "../Infinity-Instruct-data/medical.parquet"
    "../Infinity-Instruct-data/code.parquet"
)
moe_paths=(
    "../phi_mask/instructcoding_ours4_global.pth"
    "../phi_mask/instructcoding_ours6_global.pth"
    "../phi_mask/instructcoding_ours8_global.pth"
    "../phi_mask/instructcoding_ours10_global.pth"
    "../phi_mask/instructcoding_ours12_global.pth"
    "../phi_mask/instructcoding_ours12_global.pth"
    "../phi_mask/instructfinance_ours4_global.pth"
    "../phi_mask/instructfinance_ours6_global.pth"
    "../phi_mask/instructfinance_ours8_global.pth"
    "../phi_mask/instructfinance_ours10_global.pth"
    "../phi_mask/instructfinance_ours12_global.pth"
    "../phi_mask/instructmedical_ours4_global.pth"
    "../phi_mask/instructmedical_ours6_global.pth"
    "../phi_mask/instructmedical_ours8_global.pth"
    "../phi_mask/instructmedical_ours10_global.pth"
    "../phi_mask/instructmedical_ours12_global.pth"
    "../phi_mask/instructsafety_ours4_global.pth"
    "../phi_mask/instructsafety_ours6_global.pth"
    "../phi_mask/instructsafety_ours8_global.pth"
    "../phi_mask/instructsafety_ours10_global.pth"
    "../phi_mask/instructsafety_ours12_global.pth"
    "../phi_mask/instructsocial_ours4_global.pth"
    "../phi_mask/instructsocial_ours6_global.pth"
    "../phi_mask/instructsocial_ours8_global.pth"
    "../phi_mask/instructsocial_ours10_global.pth"
    "../phi_mask/instructsocial_ours12_global.pth"
)


# 循环遍历所有moe_path
for moe_path in "${moe_paths[@]}"; do
    echo "Evaluating model with moe_path: $moe_path"
    for data_path in "${data_paths[@]}"; do
        filename=$(basename $moe_path .pth)
        echo "Evaluating subject: $filename"
        /home/naifanzhang/.conda/envs/eedi/bin/python eval_moe_phi.py \
            --model_path $model \
            --data_path $data_path \
            --moe_path $moe_path \
            --output_prefix $filename \
            --batch_size 20
    done
    echo "Finished evaluation for: $moe_path"
    echo "----------------------------------------"
done



