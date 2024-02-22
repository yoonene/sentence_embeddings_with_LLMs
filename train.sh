export WANDB_DISABLE=false

MODEL=$1
# TEMPLATE="This_sentence_:_\"*sent_0*\"_means_in_one_word:\""
TEMPLATE="이 문장_:_\"*sent_0*\"_은_한_단어로:\""

if [[ $MODEL == 42dot_LLM-PLM-1.3B ]] || [[ $MODEL == 42dot_LLM-SFT-1.3B ]]; then
    WORLD_SIZE=2 CUDA_VISIBLE_DEVICES=0,1 torchrun --nproc_per_node=2 --master_port=1234 ft_llm.py \
        --base_model 42dot/${MODEL} \
        --data_path 'data/nli_for_simcse_kor_pp.csv' \
        --batch_size 256 \
        --micro_batch_size 128 \
        --num_epochs 1 \
        --learning_rate 5e-4 \
        --cutoff_len 32 \
        --lora_r 64 \
        --lora_alpha 16 \
        --lora_dropout 0.05 \
        --output_dir ${MODEL}-kortemp-lora \
        --mask_embedding_sentence_template $TEMPLATE --use_neg_sentence --save_steps 50 --load_kbit 4
elif [[ $MODEL == open-llama-2-ko-7b ]]; then
    BASE_MODEL=beomi/open-llama-2-ko-7b
    WORLD_SIZE=6 CUDA_VISIBLE_DEVICES=2,3,4,5,6,7 torchrun --nproc_per_node=6 --master_port=1234 ft_llm.py \
        --base_model $BASE_MODEL \
        --data_path 'data/nli_for_simcse_kor_pp.csv' \
        --batch_size 32 \
        --micro_batch_size 8 \
        --num_epochs 1 \
        --learning_rate 5e-4 \
        --cutoff_len 32 \
        --lora_r 16 \
        --lora_alpha 16 \
        --lora_dropout 0.05 \
        --output_dir ${MODEL}-lora  --is_sentemb \
        --mask_embedding_sentence_template $TEMPLATE --use_neg_sentence --save_steps 50 --load_kbit 4
fi