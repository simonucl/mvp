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
DATASET_PATH=${19}
SHOT=${20}

source ~/.bashrc
echo $PWD



for SEED in 13;
do
    echo $SEED
    MODEL_ID=${MODEL_TYPE}_${SEED}_${EXTRA_NAMES}
    
    MODELPATH=./checkpoints/${DATASET}/${MODEL}/model_${MODEL_ID}_${SHOT}/

    mkdir -p ${MODELPATH}
 
    python main.py  --mode $MODE \
                    --dataset $DATASET \
                    --model_type $MODEL_TYPE \
                    --model_id $MODEL_ID \
                    --batch_size $BATCH_SIZE \
                    --model $MODEL \
                    --num_epochs $EPOCHS \
                    --lr $LR  \
                    --pool_label_words $POOL_LABELS \
                    --pool_templates $POOL_TEMPLATES \
                    --verbalizer_file $VERBALIZER_FILE \
                    --template_file $TEMPLATE_FILE \
                    --num_template $NUM_TEMPLATE \
                    --train_size $TRAIN_SIZE \
                    --path None \
                    --seed $SEED \
                    --patience 10 \
                    --val_size $VAL_SIZE \
                    --dataset_path $DATASET_PATH \
                    # --model_dir $MODELPATH
done