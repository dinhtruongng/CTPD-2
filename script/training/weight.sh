#!/bin/zsh
set -e


cd ../..
echo "Current directory: $(pwd)"
cd src/prefkd/
echo "Current directory: $(pwd)"

model_name_1="positive_model_name"  # Replace with your positive model name
model_name_2="negative_model_name"  # Replace with your negative model name
student_model="student_model_name"  # Replace with your student model name, to get tokenizer
# input_dir="datasets/ultra-feedback"
data_path="path/to/your/data"  # Replace with your data path
output_dir="generated/weights"  # Replace with your desired output directory

batch_size=8
num_gpus=4
force_sequential=false  # Set to true if multiprocessing causes issues
split="test"
average_log_prob=0
teacher_all=0
student_all=1

# Create output directory if it doesn't exist
# mkdir -p $output_dir

# Run the parallel processing script
python3 weight.py \
  --positive_model_name $model_name_1 \
  --negative_model_name $model_name_2 \
  --student_model $student_model \
  --split=$split \
  --data_path $data_path \
  --output_dir $output_dir \
  --batch_size $batch_size \
  --num_gpus $num_gpus \
  --average_log_prob $average_log_prob \
  --teacher_all $teacher_all \
  --student_all $student_all \
  $(if $force_sequential; then echo "--force_sequential"; fi) 
