export DIM_MULTIPLIER=5.42

export CUDA_VISIBLE_DEVICES="0"
# record the start time
START_TIME=$(date +%s)

nohup python -u iglora/run_final_sft.py \
  --seed 200 \
  --dataset_name datasets/sst2 \
  --model_name_or_path models/Llama2_7B \
  --block_size 512 \
  --per_device_train_batch_size 1 \
  --per_device_eval_batch_size 1 \
  --gradient_accumulation_steps 8 \
  --num_train_epochs 10 \
  --warmup_steps 100 \
  --output_dir experiments/outputs/iglora_llama7b_sst2_200_re1 \
  --do_train false \
  --do_generation true \
  --eval_steps 100 \
  --learning_rate 2.0e-4 \
  --overwrite_output_dir \
  --apply_lora True \
  --tunable_param_names lora_A,lora_E,lora_B \
  --lora_rank 32 \
  --ranks_to_mask_path experiments/outputs/iglora_llama7b_sst2_200/ranks_to_mask.json \
  > experiments/logs/iglora_llama7b_sst2_200.log 2>&1 &

TRAIN_PID=$!

# Display process information
echo "The training task has started, PID: $TRAIN_PID"
echo "Log file: experiments/logs/iglora_llama7b_sst2_200.log"

# Wait for the training to complete and calculate the time
wait $TRAIN_PID
END_TIME=$(date +%s)
TOTAL_TIME=$((END_TIME - START_TIME))

# Write the training time to the log
echo "total train time: $(($TOTAL_TIME/3600))h$((($TOTAL_TIME%3600)/60))min$(($TOTAL_TIME%60))s" \
    >> experiments/logs/iglora_llama7b_sst2_200.log