export MODEL_TYPE=bert
export MODEL_NAME=IDEA-CCNL/Erlangshen-MegatronBert-1.3B-NLI
export CLUE_DATA_DIR=/root/CPT/finetune/classification/CLUE
export TASK_NAME=ocnli
export CLS_MODE=1
export OUT_DIR=/root/autodl-tmp/output

export CUDA_VISIBLE_DEVICES=0
python run_clue_classifier.py \
    --model_type=$MODEL_TYPE \
    --model_name_or_path=$MODEL_NAME \
    --cls_mode=$CLS_MODE \
    --task_name=$TASK_NAME \
    --do_train=True \
    --do_predict=1 \
    --no_tqdm=False \
    --data_dir=$CLUE_DATA_DIR/${TASK_NAME}/ \
    --max_seq_length=512 \
    --per_gpu_train_batch_size=4 \
    --gradient_accumulation_steps 1 \
    --per_gpu_eval_batch_size=8 \
    --weight_decay=0.1 \
    --adam_epsilon=1e-6 \
    --adam_beta1=0.9 \
    --adam_beta2=0.999 \
    --max_grad_norm=1.0 \
    --learning_rate=1e-5 \
    --power=1.0 \
    --num_train_epochs=10 \
    --warmup_steps=0.1 \
    --logging_steps=1000 \
    --save_steps=999999 \
    --output_dir=$OUT_DIR/ft/$MODEL_TYPE/${TASK_NAME}_oq_erlangshen_1e-5_nli_best/ \
    --overwrite_output_dir=True \
    --seed=42 > nohup/query_only_erlangshen_lr1e-5_nli_triple.out 2>&1 &