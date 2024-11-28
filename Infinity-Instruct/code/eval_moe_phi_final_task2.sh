#!/bin/bash 
#SBATCH --nodes=1 
#SBATCH --ntasks=1
#SBATCH --mem=0
eval "$(conda shell.bash hook)"
conda activate eedi

model=""
data_paths=(
    "../Infinity-Instruct-data/coding.parquet"
    "../Infinity-Instruct-data/datascience.parquet"
    "../Infinity-Instruct-data/finance.parquet"
    "../Infinity-Instruct-data/math.parquet"
    "../Infinity-Instruct-data/medical.parquet"
    "../Infinity-Instruct-data/safety.parquet"
    "../Infinity-Instruct-data/social.parquet"
)
moe_paths=(
    "../phi_mask/instructcoding_ours8_global.pth"
    "../phi_mask/instructdatascience_ours8_global.pth"
    "../phi_mask/instructfinance_ours8_global.pth"
    "../phi_mask/instructmath_ours8_global.pth"
    "../phi_mask/instructmedical_ours8_global.pth"
    "../phi_mask/instructsafety_ours8_global.pth"
    "../phi_mask/instructsocial_ours8_global.pth"
)


# 循环遍历所有moe_path
for moe_path in "${moe_paths[@]}"; do
    echo "Evaluating model with moe_path: $moe_path"
    for data_path in "${data_paths[@]}"; do
        filename=$(basename $moe_path .pth)
        echo "Evaluating subject: $filename"
        /home/naifanzhang/.conda/envs/eedi/bin/python eval_moe_phi_final_task2.py \
            --model_path $model \
            --data_path $data_path \
            --moe_path $moe_path \
            --output_prefix $filename \
            --batch_size 20
    done
    echo "Finished evaluation for: $moe_path"
    echo "----------------------------------------"
done

