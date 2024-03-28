DATASET=$1
MODEL=$2
MODEL_TYPE=$3 # [icl | knn_icl | retrieval_icl | retrieval_icl_attack ]
ATTACK=$4 # [textfooler | textbugger | icl_attack | swap_labels | swap_orders | irrelevant_sample]

# DATASETS=(rte sst2)
DATASETS=(sst2)

SHOTS=(8 2 4 16)
TOTAL_BATCH=32
# if [[ $DATASET == "rte" ]]; then
#     SHOTS=(8 2 4)
#     TOTAL_BATCH=8
# elif [[ $DATASET == "mnli" ]]; then
#     SHOTS=(2 4)
#     TOTAL_BATCH=8
# else
#     SHOTS=(8 2 4 16)
#     TOTAL_BATCH=32
# fi

SEEDS=(1 13 42)


if [[ $ATTACK == "swap_labels" ]]; then
    QUERY_BUDGET=250
else
    QUERY_BUDGET=-1
fi

for DATASET in ${DATASETS[@]};
do
    if [[ $ATTACK == "textfooler" ]] || [[ $ATTACK == "textbugger" ]] || [[ $ATTACK == "icl_attack" ]] || [[ $ATTACK == "bert_attack" ]]; then
        ATTACK_PRECENT=0.15
    else
        if [[ $DATASET == "sst2" ]] || [[ $DATASET == "rte" ]] || [[ $DATASET == "mr" ]] || [[ $DATASET == "cr" ]]; then
            ATTACK_PRECENT=0.5
        elif [[ $DATASET == "mnli" ]]; then
            ATTACK_PRECENT=0.33
        else
            ATTACK_PRECENT=0.2
        fi
    fi

    TEMPLATE_FILE=configs/templates_${DATASET}.yaml
    VERBALIZER_FILE=configs/verbalizer_${DATASET}.yaml

    for SHOT in ${SHOTS[@]};
    do
        for SEED in ${SEEDS[@]};
        do 
            BATCH_SIZE=$((TOTAL_BATCH / SHOT))
            if [[ $SHOT -eq 2 ]]; then
                BATCH_SIZE=$((BATCH_SIZE / 2))
            fi
            
            echo $SEED+${SHOT}+${MODEL}+"mvp"
            MODEL_TYPE='icl'
            MODEL_ID=${MODEL_TYPE}-seed-${SEED}-shot-${SHOT}
            MODELPATH=./checkpoints/${DATASET}/${MODEL}/${ATTACK}/${MODEL_ID}

            DATASET_PATH=./data/${DATASET}/${SHOT}-$SEED

            mkdir -p ${MODELPATH}
            echo ${MODELPATH}

            nohup python3 main.py \
                --mode attack \
                --attack_name ${ATTACK} \
                --num_examples 1000 \
                --dataset ${DATASET} \
                --query_budget ${QUERY_BUDGET} \
                --batch_size ${BATCH_SIZE} \
                --model_type ${MODEL_TYPE} \
                --model ${MODEL} \
                --verbalizer_file ${VERBALIZER_FILE} \
                --template_file ${TEMPLATE_FILE} \
                --seed $SEED \
                --shot ${SHOT} \
                --max_percent_words ${ATTACK_PRECENT} \
                --model_dir ${MODELPATH} \
                    > ${MODELPATH}/logs_${ATTACK}.txt
            

            KNN=$(( SHOT / 2 - 1 ))
            BETA=0.2
            KNN_T=100

            MODEL_TYPE='knn_icl'
            
            echo $SEED+${SHOT}+${MODEL}+"mvp"
            MODEL_ID=${MODEL_TYPE}-seed-${SEED}-shot-${SHOT}
            MODELPATH=./checkpoints/${DATASET}/${MODEL}/${ATTACK}/${MODEL_ID}

            DATASET_PATH=./data/${DATASET}/${SHOT}-$SEED

            mkdir -p ${MODELPATH}
            echo ${MODELPATH}

            nohup python3 main.py \
                --mode attack \
                --attack_name ${ATTACK} \
                --num_examples 1000 \
                --dataset ${DATASET} \
                --query_budget -1 \
                --batch_size ${BATCH_SIZE} \
                --model_type ${MODEL_TYPE} \
                --model ${MODEL} \
                --verbalizer_file ${VERBALIZER_FILE} \
                --template_file ${TEMPLATE_FILE} \
                --seed $SEED \
                --shot ${SHOT} \
                --max_percent_words ${ATTACK_PRECENT} \
                --model_dir ${MODELPATH} \
                --knn_T ${KNN_T} \
                --beta ${BETA} \
                --knn_k ${KNN} \
                --examples_per_label 1 \
                    > ${MODELPATH}/logs_${ATTACK}.txt

        done
    done
done
