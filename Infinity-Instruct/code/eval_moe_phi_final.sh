#!/bin/bash 
#SBATCH --nodes=1 
#SBATCH --ntasks=1
#SBATCH --mem=0
#SBATCH --gpus=1
eval "$(conda shell.bash hook)"
conda activate eedi

model=""
data_paths=(
    "../Infinity-Instruct-data/coding.parquet"
)
moe_paths=(
    "../phi_mask/instructcoding_baseline6.pth"
    "../phi_mask/instructcoding_baseline8.pth"
    "../phi_mask/instructcoding_baseline10.pth"
    "../phi_mask/instructcoding_baseline12.pth"
    "../phi_mask/instructcoding_baseline14.pth"
    "../phi_mask/instructcoding_baseline16.pth"
    "../phi_mask/instructcoding_ours6_global.pth"
    "../phi_mask/instructcoding_ours8_global.pth"
    "../phi_mask/instructcoding_ours10_global.pth"
    "../phi_mask/instructcoding_ours12_global.pth"
    "../phi_mask/instructcoding_ours14_global.pth"
    "../phi_mask/instructcoding_ours16_global.pth"
    
)


# 循环遍历所有moe_path
for moe_path in "${moe_paths[@]}"; do
    echo "Evaluating model with moe_path: $moe_path"
    for data_path in "${data_paths[@]}"; do
        filename=$(basename $moe_path .pth)
        echo "Evaluating subject: $filename"
        /home/naifanzhang/.conda/envs/eedi/bin/python eval_moe_phi_final.py \
            --model_path $model \
            --data_path $data_path \
            --moe_path $moe_path \
            --output_prefix $filename \
            --batch_size 20
    done
    echo "Finished evaluation for: $moe_path"
    echo "----------------------------------------"
done

data_paths=(
    "../Infinity-Instruct-data/datascience.parquet"
)

moe_paths=(
    "../phi_mask/instructdatascience_baseline6.pth"
    "../phi_mask/instructdatascience_baseline8.pth"
    "../phi_mask/instructdatascience_baseline10.pth"
    "../phi_mask/instructdatascience_baseline12.pth"
    "../phi_mask/instructdatascience_baseline14.pth"
    "../phi_mask/instructdatascience_baseline16.pth"
    "../phi_mask/instructdatascience_ours6_global.pth"
    "../phi_mask/instructdatascience_ours8_global.pth"
    "../phi_mask/instructdatascience_ours10_global.pth"
    "../phi_mask/instructdatascience_ours12_global.pth"
    "../phi_mask/instructdatascience_ours14_global.pth"
    "../phi_mask/instructdatascience_ours16_global.pth"
    
)


# 循环遍历所有moe_path
for moe_path in "${moe_paths[@]}"; do
    echo "Evaluating model with moe_path: $moe_path"
    for data_path in "${data_paths[@]}"; do
        filename=$(basename $moe_path .pth)
        echo "Evaluating subject: $filename"
        /home/naifanzhang/.conda/envs/eedi/bin/python eval_moe_phi_final.py \
            --model_path $model \
            --data_path $data_path \
            --moe_path $moe_path \
            --output_prefix $filename \
            --batch_size 20
    done
    echo "Finished evaluation for: $moe_path"
    echo "----------------------------------------"
done

data_paths=(
    "../Infinity-Instruct-data/finance.parquet"
)

moe_paths=(
    "../phi_mask/instructfinance_baseline6.pth"
    "../phi_mask/instructfinance_baseline8.pth"
    "../phi_mask/instructfinance_baseline10.pth"
    "../phi_mask/instructfinance_baseline12.pth"
    "../phi_mask/instructfinance_baseline14.pth"
    "../phi_mask/instructfinance_baseline16.pth"
    "../phi_mask/instructfinance_ours6_global.pth"
    "../phi_mask/instructfinance_ours8_global.pth"
    "../phi_mask/instructfinance_ours10_global.pth"
    "../phi_mask/instructfinance_ours12_global.pth"
    "../phi_mask/instructfinance_ours14_global.pth"
    "../phi_mask/instructfinance_ours16_global.pth"
    
)


# 循环遍历所有moe_path
for moe_path in "${moe_paths[@]}"; do
    echo "Evaluating model with moe_path: $moe_path"
    for data_path in "${data_paths[@]}"; do
        filename=$(basename $moe_path .pth)
        echo "Evaluating subject: $filename"
        /home/naifanzhang/.conda/envs/eedi/bin/python eval_moe_phi_final.py \
            --model_path $model \
            --data_path $data_path \
            --moe_path $moe_path \
            --output_prefix $filename \
            --batch_size 20
    done
    echo "Finished evaluation for: $moe_path"
    echo "----------------------------------------"
done

data_paths=(
    "../Infinity-Instruct-data/math.parquet"
)

moe_paths=(
    "../phi_mask/instructmath_baseline6.pth"
    "../phi_mask/instructmath_baseline8.pth"
    "../phi_mask/instructmath_baseline10.pth"
    "../phi_mask/instructmath_baseline12.pth"
    "../phi_mask/instructmath_baseline14.pth"
    "../phi_mask/instructmath_baseline16.pth"
    "../phi_mask/instructmath_ours6_global.pth"
    "../phi_mask/instructmath_ours8_global.pth"
    "../phi_mask/instructmath_ours10_global.pth"
    "../phi_mask/instructmath_ours12_global.pth"
    "../phi_mask/instructmath_ours14_global.pth"
    "../phi_mask/instructmath_ours16_global.pth"
    
)


# 循环遍历所有moe_path
for moe_path in "${moe_paths[@]}"; do
    echo "Evaluating model with moe_path: $moe_path"
    for data_path in "${data_paths[@]}"; do
        filename=$(basename $moe_path .pth)
        echo "Evaluating subject: $filename"
        /home/naifanzhang/.conda/envs/eedi/bin/python eval_moe_phi_final.py \
            --model_path $model \
            --data_path $data_path \
            --moe_path $moe_path \
            --output_prefix $filename \
            --batch_size 20
    done
    echo "Finished evaluation for: $moe_path"
    echo "----------------------------------------"
done

data_paths=(
    "../Infinity-Instruct-data/medical.parquet"
)

moe_paths=(
    "../phi_mask/instructmedical_baseline6.pth"
    "../phi_mask/instructmedical_baseline8.pth"
    "../phi_mask/instructmedical_baseline10.pth"
    "../phi_mask/instructmedical_baseline12.pth"
    "../phi_mask/instructmedical_baseline14.pth"
    "../phi_mask/instructmedical_baseline16.pth"
    "../phi_mask/instructmedical_ours6_global.pth"
    "../phi_mask/instructmedical_ours8_global.pth"
    "../phi_mask/instructmedical_ours10_global.pth"
    "../phi_mask/instructmedical_ours12_global.pth"
    "../phi_mask/instructmedical_ours14_global.pth"
    "../phi_mask/instructmedical_ours16_global.pth"
    
)


# 循环遍历所有moe_path
for moe_path in "${moe_paths[@]}"; do
    echo "Evaluating model with moe_path: $moe_path"
    for data_path in "${data_paths[@]}"; do
        filename=$(basename $moe_path .pth)
        echo "Evaluating subject: $filename"
        /home/naifanzhang/.conda/envs/eedi/bin/python eval_moe_phi_final.py \
            --model_path $model \
            --data_path $data_path \
            --moe_path $moe_path \
            --output_prefix $filename \
            --batch_size 20
    done
    echo "Finished evaluation for: $moe_path"
    echo "----------------------------------------"
done

data_paths=(
    "../Infinity-Instruct-data/safety.parquet"
)

moe_paths=(
    "../phi_mask/instructsafety_baseline6.pth"
    "../phi_mask/instructsafety_baseline8.pth"
    "../phi_mask/instructsafety_baseline10.pth"
    "../phi_mask/instructsafety_baseline12.pth"
    "../phi_mask/instructsafety_baseline14.pth"
    "../phi_mask/instructsafety_baseline16.pth"
    "../phi_mask/instructsafety_ours6_global.pth"
    "../phi_mask/instructsafety_ours8_global.pth"
    "../phi_mask/instructsafety_ours10_global.pth"
    "../phi_mask/instructsafety_ours12_global.pth"
    "../phi_mask/instructsafety_ours14_global.pth"
    "../phi_mask/instructsafety_ours16_global.pth"
    
)


# 循环遍历所有moe_path
for moe_path in "${moe_paths[@]}"; do
    echo "Evaluating model with moe_path: $moe_path"
    for data_path in "${data_paths[@]}"; do
        filename=$(basename $moe_path .pth)
        echo "Evaluating subject: $filename"
        /home/naifanzhang/.conda/envs/eedi/bin/python eval_moe_phi_final.py \
            --model_path $model \
            --data_path $data_path \
            --moe_path $moe_path \
            --output_prefix $filename \
            --batch_size 20
    done
    echo "Finished evaluation for: $moe_path"
    echo "----------------------------------------"
done

data_paths=(
    "../Infinity-Instruct-data/social.parquet"
)

moe_paths=(
    "../phi_mask/instructsocial_baseline6.pth"
    "../phi_mask/instructsocial_baseline8.pth"
    "../phi_mask/instructsocial_baseline10.pth"
    "../phi_mask/instructsocial_baseline12.pth"
    "../phi_mask/instructsocial_baseline14.pth"
    "../phi_mask/instructsocial_baseline16.pth"
    "../phi_mask/instructsocial_ours6_global.pth"
    "../phi_mask/instructsocial_ours8_global.pth"
    "../phi_mask/instructsocial_ours10_global.pth"
    "../phi_mask/instructsocial_ours12_global.pth"
    "../phi_mask/instructsocial_ours14_global.pth"
    "../phi_mask/instructsocial_ours16_global.pth"
    
)


# 循环遍历所有moe_path
for moe_path in "${moe_paths[@]}"; do
    echo "Evaluating model with moe_path: $moe_path"
    for data_path in "${data_paths[@]}"; do
        filename=$(basename $moe_path .pth)
        echo "Evaluating subject: $filename"
        /home/naifanzhang/.conda/envs/eedi/bin/python eval_moe_phi_final.py \
            --model_path $model \
            --data_path $data_path \
            --moe_path $moe_path \
            --output_prefix $filename \
            --batch_size 20
    done
    echo "Finished evaluation for: $moe_path"
    echo "----------------------------------------"
done
