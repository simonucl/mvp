BATCH_SIZE=$1
DATASET=$2
MODEL=$3
MODEL_TYPE=$4
EPOCHS=$5
LR=${6}
POOL_LABELS=${7}
POOL_TEMPLATES=${8}
POOL_LABELS_TEST=${9}
POOL_TEMPLATES_TEST=${10}
TEMPLATE_FILE=${11}
VERBALIZER_FILE=${12}
EXTRA_NAMES=${13}
ATTACK=${14}
MODE=${15}
NUM_TEMPLATE=${16}   
TRAIN_SIZE=${17} 
VAL_SIZE=${18}

source ~/.bashrc
echo $PWD

for SEED in 13;
do
    echo $SEED
    MODEL_ID=${MODEL_TYPE}_${SEED}_${EXTRA_NAMES}
    
    MODELPATH=./checkpoints/${DATASET}/${MODEL}/model_${MODEL_ID}/

    

    nohup nice -n10 python main.py --mode attack \
                                --path ${MODELPATH}/final_model/ \
                                --attack_name textfooler \
                                --num_examples -1 --dataset ${DATASET} \
                                --query_budget -1 --batch_size ${BATCH_SIZE} --model_type ${MODEL_TYPE} --model ${MODEL} \
                                --pool_label_words ${POOL_LABELS_TEST} --pool_templates ${POOL_TEMPLATES_TEST} \
                                --verbalizer_file ${VERBALIZER_FILE} --template_file ${TEMPLATE_FILE} \
                                --num_template ${NUM_TEMPLATE} --train_size ${TRAIN_SIZE} --val_size ${VAL_SIZE} --seed $SEED > ${MODELPATH}/logs_textfooler.txt


    nohup nice -n10 python main.py --mode attack \
                            --path ${MODELPATH}/final_model/ \
                            --attack_name textbugger \
                            --num_examples -1 --dataset ${DATASET} \
                            --query_budget -1 --batch_size ${BATCH_SIZE} --model_type ${MODEL_TYPE} --model ${MODEL} \
                            --pool_label_words ${POOL_LABELS_TEST} --pool_templates ${POOL_TEMPLATES_TEST} \
                            --verbalizer_file ${VERBALIZER_FILE} --template_file ${TEMPLATE_FILE}  \
                            --num_template ${NUM_TEMPLATE} --train_size ${TRAIN_SIZE} --val_size ${VAL_SIZE} --seed $SEED > ${MODELPATH}/logs_textbugger.txt




done