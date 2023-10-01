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
EPSILON=${19}
NORM=${20}
NUM_ITER=${21}
DATASET_PATH=${22}
SHOT=${23}

betas=(0.2 0.5 0.8 1.0)
betas=(0.0 0.2 0.5 0.8 1.0)

source ~/.bashrc
echo $PWD
conda activate /home/co-huan1/rds/rds-qakg-2iBGk7DbOVc/jie/conda/multi

export XLA_FLAGS=--xla_gpu_cuda_data_dir=/usr/local/software/spack/spack-rhel8-20210927/opt/spack/linux-centos8-zen2/gcc-9.4.0/cuda-11.4.0-3hnxhjt2jt4ruy75w2q4mnvkw7dty72l
MODEL=roberta-large

for SEED in 13;
do
    for SHOT in 16 32 64;
    do 
        echo $SEED+${SHOT}+${MODEL}+"mvp"
        EXTRA_NAMES=mvp_seed_${SEED}

        MODEL_ID=${MODEL_TYPE}_${SEED}_${EXTRA_NAMES}_${SHOT}
        
        MODELPATH=./checkpoints/${DATASET}/${MODEL}/model_${MODEL_ID}/

        DATASET_PATH=./data/${DATASET}/${SHOT}-$SEED

        mkdir -p ${MODELPATH}

        nohup python3 main.py  --mode $MODE \
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
                --dataset_path $DATASET_PATH > ${MODELPATH}/logs_trainer.txt

        for BETA in ${betas[@]};
        do
            MODEL_TYPE=mvp_knn

            nohup python3 main.py --mode attack \
                                        --path ${MODELPATH}/final_model/ \
                                        --attack_name textfooler \
                                        --num_examples 1000 --dataset ${DATASET} \
                                        --query_budget -1 --batch_size ${BATCH_SIZE} --model_type ${MODEL_TYPE} --model ${MODEL} \
                                        --pool_label_words ${POOL_LABELS_TEST} --pool_templates ${POOL_TEMPLATES_TEST} \
                                        --verbalizer_file ${VERBALIZER_FILE} --template_file ${TEMPLATE_FILE} \
                                        --num_template ${NUM_TEMPLATE} --train_size ${TRAIN_SIZE} --val_size ${VAL_SIZE} --seed $SEED --knn_model ${MODEL} --beta ${BETA} > ${MODELPATH}/logs_textfooler_beta_${BETA}.txt

            # nohup nice -n10 python3 main.py --mode attack \
            #                             --path ${MODELPATH}/final_model/ \
            #                             --attack_name textfooler \
            #                             --num_examples 1000 --dataset ${DATASET} \
            #                             --query_budget -1 --batch_size ${BATCH_SIZE} --model_type ${MODEL_TYPE} --model ${MODEL} \
            #                             --pool_label_words ${POOL_LABELS_TEST} --pool_templates ${POOL_TEMPLATES_TEST} \
            #                             --verbalizer_file ${VERBALIZER_FILE} --template_file ${TEMPLATE_FILE} \
            #                             --num_template ${NUM_TEMPLATE} --train_size ${TRAIN_SIZE} --val_size ${VAL_SIZE} --seed $SEED > ${MODELPATH}/logs_textfooler.txt


            nohup python3 main.py --mode attack \
                                    --path ${MODELPATH}/final_model/ \
                                    --attack_name bae \
                                    --num_examples 1000 --dataset ${DATASET} \
                                    --query_budget -1 --batch_size ${BATCH_SIZE} --model_type ${MODEL_TYPE} --model ${MODEL} \
                                    --pool_label_words ${POOL_LABELS_TEST} --pool_templates ${POOL_TEMPLATES_TEST} \
                                    --verbalizer_file ${VERBALIZER_FILE} --template_file ${TEMPLATE_FILE}  \
                                    --num_template ${NUM_TEMPLATE} --train_size ${TRAIN_SIZE} --val_size ${VAL_SIZE} --seed $SEED --knn_model ${MODEL} --beta ${BETA} > ${MODELPATH}/logs_bae_beta_${BETA}.txt

            nohup python3 main.py --mode attack \
                                    --path ${MODELPATH}/final_model/ \
                                    --attack_name textbugger \
                                    --num_examples 1000 --dataset ${DATASET} \
                                    --query_budget -1 --batch_size ${BATCH_SIZE} --model_type ${MODEL_TYPE} --model ${MODEL} \
                                    --pool_label_words ${POOL_LABELS_TEST} --pool_templates ${POOL_TEMPLATES_TEST} \
                                    --verbalizer_file ${VERBALIZER_FILE} --template_file ${TEMPLATE_FILE}  \
                                    --num_template ${NUM_TEMPLATE} --train_size ${TRAIN_SIZE} --val_size ${VAL_SIZE} --seed $SEED --knn_model ${MODEL} --beta ${BETA} > ${MODELPATH}/logs_textbugger_beta_${BETA}.txt

            # nohup nice -n10 python3 main.py --mode attack \
            #                         --path ${MODELPATH}/final_model/ \
            #                         --attack_name bae \
            #                         --num_examples 1000 --dataset ${DATASET} \
            #                         --query_budget -1 --batch_size ${BATCH_SIZE} --model_type ${MODEL_TYPE} --model ${MODEL} \
            #                         --pool_label_words ${POOL_LABELS_TEST} --pool_templates ${POOL_TEMPLATES_TEST} \
            #                         --verbalizer_file ${VERBALIZER_FILE} --template_file ${TEMPLATE_FILE}  \
            #                         --num_template ${NUM_TEMPLATE} --train_size ${TRAIN_SIZE} --val_size ${VAL_SIZE} --seed $SEED > ${MODELPATH}/logs_bae.txt
            # Adversarial training and testing
            # EXTRA_NAMES=mvp_adv_${SEED}

            # DATASET_PATH=./data/${DATASET}/${SHOT}-$SEED
            # MODEL_ID=${MODEL_TYPE}_${SEED}_${EXTRA_NAMES}_${SHOT}
            # MODELPATH=./checkpoints/${DATASET}/${MODEL}/model_${MODEL_ID}/

            # mkdir -p ${MODELPATH}
            # echo $SEED+${SHOT}+${MODEL}+"mvp_adv"
            # nohup python3 main.py  --mode $MODE \
            #                 --dataset $DATASET \
            #                 --model_type $MODEL_TYPE \
            #                 --model_id $MODEL_ID \
            #                 --batch_size $BATCH_SIZE \
            #                 --model $MODEL \
            #                 --num_epochs $EPOCHS \
            #                 --lr $LR  \
            #                 --pool_label_words $POOL_LABELS \
            #                 --pool_templates $POOL_TEMPLATES \
            #                 --verbalizer_file $VERBALIZER_FILE \
            #                 --template_file $TEMPLATE_FILE \
            #                 --num_template $NUM_TEMPLATE \
            #                 --train_size $TRAIN_SIZE \
            #                 --path None \
            #                 --seed $SEED \
            #                 --patience 10 --adv_augment 1 \
            #                 --epsilon $EPSILON \
            #                 --norm $NORM \
            #                 --num_iter $NUM_ITER \
            #                 --val_size $VAL_SIZE \
            #                 --dataset_path $DATASET_PATH > ${MODELPATH}/logs_trainer.txt

            

            # nohup nice -n10 python3 main.py --mode attack \
            #                             --path ${MODELPATH}/final_model/ \
            #                             --attack_name textfooler \
            #                             --num_examples -1 --dataset ${DATASET} \
            #                             --query_budget -1 --batch_size ${BATCH_SIZE} --model_type ${MODEL_TYPE} --model ${MODEL} \
            #                             --pool_label_words ${POOL_LABELS_TEST} --pool_templates ${POOL_TEMPLATES_TEST} \
            #                             --verbalizer_file ${VERBALIZER_FILE} --template_file ${TEMPLATE_FILE} \
            #                             --num_template ${NUM_TEMPLATE} --train_size ${TRAIN_SIZE} --val_size ${VAL_SIZE} --seed $SEED > ${MODELPATH}/logs_textfooler.txt


            # nohup nice -n10 python3 main.py --mode attack \
            #                         --path ${MODELPATH}/final_model/ \
            #                         --attack_name textbugger \
            #                         --num_examples -1 --dataset ${DATASET} \
            #                         --query_budget -1 --batch_size ${BATCH_SIZE} --model_type ${MODEL_TYPE} --model ${MODEL} \
            #                         --pool_label_words ${POOL_LABELS_TEST} --pool_templates ${POOL_TEMPLATES_TEST} \
            #                         --verbalizer_file ${VERBALIZER_FILE} --template_file ${TEMPLATE_FILE}  \
            #                         --num_template ${NUM_TEMPLATE} --train_size ${TRAIN_SIZE} --val_size ${VAL_SIZE} --seed $SEED > ${MODELPATH}/logs_textbugger.txt

            # nohup nice -n10 python3 main.py --mode attack \
            #                         --path ${MODELPATH}/final_model/ \
            #                         --attack_name bae \
            #                         --num_examples -1 --dataset ${DATASET} \
            #                         --query_budget -1 --batch_size ${BATCH_SIZE} --model_type ${MODEL_TYPE} --model ${MODEL} \
            #                         --pool_label_words ${POOL_LABELS_TEST} --pool_templates ${POOL_TEMPLATES_TEST} \
            #                         --verbalizer_file ${VERBALIZER_FILE} --template_file ${TEMPLATE_FILE}  \
            #                         --num_template ${NUM_TEMPLATE} --train_size ${TRAIN_SIZE} --val_size ${VAL_SIZE} --seed $SEED > ${MODELPATH}/logs_bae.txt
        done

        # echo $SEED+${SHOT}+${MODEL}+"mvp"
        # EXTRA_NAMES=mvp_seed_${SEED}_old_template

        # MODEL_ID=${MODEL_TYPE}_${SEED}_${EXTRA_NAMES}_${SHOT}
        
        # MODELPATH=./checkpoints/${DATASET}/${MODEL}/model_${MODEL_ID}/

        # DATASET_PATH=./data/${DATASET}/${SHOT}-$SEED

        # VERBALIZER_FILE=./configs/verbalizer_sst2_old.yaml
        # TEMPLATE_FILE=./configs/templates_sst2_old.yaml

        # mkdir -p ${MODELPATH}

        # for BETA in ${betas[@]};
        # do

        #     nohup python3 main.py  --mode $MODE \
        #             --dataset $DATASET \
        #             --model_type $MODEL_TYPE \
        #             --model_id $MODEL_ID \
        #             --batch_size $BATCH_SIZE \
        #             --model $MODEL \
        #             --num_epochs $EPOCHS \
        #             --lr $LR  \
        #             --pool_label_words $POOL_LABELS \
        #             --pool_templates $POOL_TEMPLATES \
        #             --verbalizer_file $VERBALIZER_FILE \
        #             --template_file $TEMPLATE_FILE \
        #             --num_template $NUM_TEMPLATE \
        #             --train_size $TRAIN_SIZE \
        #             --path None \
        #             --seed $SEED \
        #             --patience 10 \
        #             --val_size $VAL_SIZE \
        #             --dataset_path $DATASET_PATH > ${MODELPATH}/logs_trainer.txt

        #     MODEL_TYPE=mvp_knn

        #     nohup python3 main.py --mode attack \
        #                                 --path ${MODELPATH}/final_model/ \
        #                                 --attack_name textfooler \
        #                                 --num_examples 1000 --dataset ${DATASET} \
        #                                 --query_budget -1 --batch_size ${BATCH_SIZE} --model_type ${MODEL_TYPE} --model ${MODEL} \
        #                                 --pool_label_words ${POOL_LABELS_TEST} --pool_templates ${POOL_TEMPLATES_TEST} \
        #                                 --verbalizer_file ${VERBALIZER_FILE} --template_file ${TEMPLATE_FILE} \
        #                                 --num_template ${NUM_TEMPLATE} --train_size ${TRAIN_SIZE} --val_size ${VAL_SIZE} --seed $SEED --knn_model ${MODEL} --beta ${BETA} > ${MODELPATH}/logs_textfooler_beta_${BETA}.txt
        # done
    done
done