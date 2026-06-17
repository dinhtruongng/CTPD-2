#!/bin/zsh

model_name_1=$MODEL_NAME_1
model_name_2=$MODEL_NAME_2
input_dir="datasets/ultra-feedback"
output_dir="generated-data/ultra-feedback-tisdpo"
model1_template="normal"
model2_template="normal"
batch_size=4
num_gpus=8
force_sequential=false  # Set to true if multiprocessing causes issues

# Create output directory if it doesn't exist
mkdir -p $output_dir

# Run the parallel processing script
python predict_score_parallel.py \
  --model_name_1 $model_name_1 \
  --model_name_2 $model_name_2 \
  --model1_template $model1_template \
  --model2_template $model2_template \
  --input_dir $input_dir \
  --output_dir $output_dir \
  --batch_size $batch_size \
  --num_gpus $num_gpus \
  $(if $force_sequential; then echo "--force_sequential"; fi) 