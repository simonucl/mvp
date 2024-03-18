MODELS=(meta-llama/Llama-2-7b-hf meta-llama/Llama-2-13b-hf mistralai/Mistral-7B-v0.1 lmsys/vicuna-7b-v1.5 google/gemma-7b meta-llama/Llama-2-70b-hf mistralai/Mixtral-8x7B-v0.1)
SEEDS=(1 13 42)
ATTACKS=(textfooler textbugger swap_labels bert_attack icl_attack)
DATASETS=(rte)

BASE_MODEL=meta-llama/Llama-2-13b-hf

for MODEL in ${MODELS[@]};
do
    if [[ $MODEL == "meta-llama/Llama-2-70b-hf" ]] || [[ $MODEL == "mistralai/Mixtral-8x7B-v0.1" ]]; then
        PRECISION=int8
    else
        PRECISION=bf16
    fi
    for DATASET in ${DATASETS[@]};
    do
        for ATTACK in ${ATTACKS[@]};
        do
            for SEED in ${SEEDS[@]};
            do
                CUDA_VISIBLE_DEVICES=0 python3 src/transfer_attack.py \
                    --model $MODEL \
                    --csv_path checkpoints/rte/${BASE_MODEL}/${ATTACK}/icl-seed-${SEED}-shot-8/${ATTACK}_log.csv \
                    --attack $ATTACK \
                    --demonstration_path data/icl/${DATASET}-icl-seed-${SEED}-shot-8.pkl \
                    --precision $PRECISION

            done
        done
    done
done