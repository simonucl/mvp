BATCH_SIZE=$1
DATASET=$2
MODEL=$3
MODEL_TYPE=$4
TEMPLATE_FILE=${5}
VERBALIZER_FILE=${6}
ATTACK=${7}
ADV=${8}

# source ~/.bashrc
# echo $PWD
# conda activate /home/co-huan1/rds/rds-qakg-2iBGk7DbOVc/jie/conda/multi

# export XLA_FLAGS=--xla_gpu_cuda_data_dir=/usr/local/software/spack/spack-rhel8-20210927/opt/spack/linux-centos8-zen2/gcc-9.4.0/cuda-11.4.0-3hnxhjt2jt4ruy75w2q4mnvkw7dty72l

for SEED in 42;
do
    for SHOT in 32 64;
    do 
        echo $SEED+${SHOT}+${MODEL}+"mvp"

        MODEL_ID=${MODEL_TYPE}-seed-${SEED}-shot-${SHOT}
        
        ATTACK=icl_attack
        MODELPATH=./checkpoints/${DATASET}/${MODEL}/${ATTACK}/${MODEL_ID}

        DATASET_PATH=./data/${DATASET}/${SHOT}-$SEED

        mkdir -p ${MODELPATH}
        echo ${MODELPATH}

        # MODEL_TYPE=knn_icl
        KNN=4
        for M in 1 2;
        do
            nohup python3 main.py --mode attack \
                                        --attack_name ${ATTACK} --num_examples 1000 --dataset ${DATASET} \
                                        --query_budget 500 --batch_size ${BATCH_SIZE} --model_type ${MODEL_TYPE} --model ${MODEL} \
                                        --verbalizer_file ${VERBALIZER_FILE} --template_file ${TEMPLATE_FILE} \
                                        --seed $SEED --shot ${SHOT} \
                                        --adv_augment $ADV --knn_k $KNN --examples_per_label ${M} > ${MODELPATH}/logs_${ATTACK}.txt
        done
        # nohup nice -n10 python3 main.py --mode attack \
        #                             --path ${MODELPATH}/final_model/ \
        #                             --attack_name textfooler \
        #                             --num_examples 1000 --dataset ${DATASET} \
        #                             --query_budget -1 --batch_size ${BATCH_SIZE} --model_type ${MODEL_TYPE} --model ${MODEL} \
        #                             --pool_label_words ${POOL_LABELS_TEST} --pool_templates ${POOL_TEMPLATES_TEST} \
        #                             --verbalizer_file ${VERBALIZER_FILE} --template_file ${TEMPLATE_FILE} \
        #                             --num_template ${NUM_TEMPLATE} --train_size ${TRAIN_SIZE} --val_size ${VAL_SIZE} --seed $SEED > ${MODELPATH}/logs_textfooler.txt


        # nohup python3 main.py --mode attack \
        #                         --attack_name bae \
        #                         --num_examples 1000 --dataset ${DATASET} \
        #                         --query_budget -1 --batch_size ${BATCH_SIZE} --model_type ${MODEL_TYPE} --model ${MODEL} \
        #                         --pool_label_words ${POOL_LABELS_TEST} --pool_templates ${POOL_TEMPLATES_TEST} \
        #                         --verbalizer_file ${VERBALIZER_FILE} --template_file ${TEMPLATE_FILE}  \
        #                         --num_template ${NUM_TEMPLATE} --train_size ${TRAIN_SIZE} --val_size ${VAL_SIZE} --seed $SEED --knn_model ${MODEL} --beta ${BETA} > ${MODELPATH}/logs_bae_beta_${BETA}.txt

        # nohup python3 main.py --mode attack \
        #                         --attack_name textbugger \
        #                         --num_examples 1000 --dataset ${DATASET} \
        #                         --query_budget -1 --batch_size ${BATCH_SIZE} --model_type ${MODEL_TYPE} --model ${MODEL} \
        #                         --pool_label_words ${POOL_LABELS_TEST} --pool_templates ${POOL_TEMPLATES_TEST} \
        #                         --verbalizer_file ${VERBALIZER_FILE} --template_file ${TEMPLATE_FILE}  \
        #                         --num_template ${NUM_TEMPLATE} --train_size ${TRAIN_SIZE} --val_size ${VAL_SIZE} --seed $SEED --knn_model ${MODEL} --beta ${BETA} > ${MODELPATH}/logs_textbugger_beta_${BETA}.txt

    done
done
