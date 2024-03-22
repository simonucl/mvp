MODELS=(meta-llama/Llama-2-13b-hf mistralai/Mistral-7B-v0.1 lmsys/vicuna-7b-v1.5 google/gemma-7b meta-llama/Llama-2-70b-hf mistralai/Mixtral-8x7B-v0.1)
SEEDS=(1 13 42)
RETRIEVERS=(bm25 sbert instructor)
ATTACKS=(textfooler textbugger bert_attack icl_attack)
DATASETS=(rte)

BASE_MODEL=meta-llama/Llama-2-7b-hf

for MODEL in ${MODELS[@]};
do
    if [[ $MODEL == "meta-llama/Llama-2-70b-hf" ]] || [[ $MODEL == "mistralai/Mixtral-8x7B-v0.1" ]]; then
        PRECISION=int4
    else
        PRECISION=bf16
    fi
    for DATASET in ${DATASETS[@]};
    do
        for ATTACK in ${ATTACKS[@]};
        do
            for RETRIEVER in ${RETRIEVERS[@]};
            do
                echo model: $MODEL
                if [[ $ATTACK == "swap_labels_fix_dist" ]]; then
                echo csv_path: checkpoints/${DATASET}/${BASE_MODEL}/swap_labels/retrieval_icl-seed-1-shot-8_${RETRIEVER}_fix_dist/swap_labels_log.csv
                    python3 src/transfer_attack.py \
                        --model $MODEL \
                        --csv_path checkpoints/${DATASET}/${BASE_MODEL}/swap_labels/retrieval_icl-seed-1-shot-8_${RETRIEVER}_fix_dist/swap_labels_log.csv \
                        --attack $ATTACK \
                        --precision $PRECISION \
                        --demonstration_path data/ralm/${DATASET}_${RETRIEVER}.pkl \
                        --dataset $DATASET
                else
                    echo csv_path: checkpoints/${DATASET}/${BASE_MODEL}/icl_attack/retrieval_icl-seed-1-shot-8_${RETRIEVER}/${ATTACK}_log.csv
                    python3 src/transfer_attack.py \
                        --model $MODEL \
                        --csv_path checkpoints/${DATASET}/${BASE_MODEL}/${ATTACK}/retrieval_icl-seed-1-shot-8_${RETRIEVER}/${ATTACK}_log.csv \
                        --attack $ATTACK \
                        --precision $PRECISION \
                        --demonstration_path data/ralm/${DATASET}_${RETRIEVER}.pkl \
                        --dataset $DATASET
                fi
            done
        done
    done
done