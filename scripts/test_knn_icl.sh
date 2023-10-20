BATCH_SIZE=$1
DATASET=$2
MODEL=$3
MODEL_TYPE=$4
TEMPLATE_FILE=${5}
VERBALIZER_FILE=${6}
ATTACK=${7}
ADV=${8}
KNN_T=${9}

# source ~/.bashrc
# echo $PWD
# conda activate /home/co-huan1/rds/rds-qakg-2iBGk7DbOVc/jie/conda/multi

# export XLA_FLAGS=--xla_gpu_cuda_data_dir=/usr/local/software/spack/spack-rhel8-20210927/opt/spack/linux-centos8-zen2/gcc-9.4.0/cuda-11.4.0-3hnxhjt2jt4ruy75w2q4mnvkw7dty72l

for ATTACK in textfooler;
do
    for SHOT in 4 16 32 8;
    do 
        for SEED in 1 13 42;
        do
            BETA=0.2
            echo $SEED+${SHOT}+${MODEL}+"mvp"
            # if [[ $ADV -eq 1 ]]; then
            #     EXTRA_NAMES=adv_seed_${SEED}
            # else
            #     EXTRA_NAMES=seed_${SEED}
            # fi

            MODEL_ID=${MODEL_TYPE}-seed-${SEED}-shot-${SHOT}
            
            # ATTACK=textfooler
            MODELPATH=./checkpoints/${DATASET}/${MODEL}/${ATTACK}/${MODEL_ID}

            DATASET_PATH=./data/${DATASET}/${SHOT}-$SEED

            mkdir -p ${MODELPATH}
            echo ${MODELPATH}

            # MODEL_TYPE=knn_icl
            KNN=$(( SHOT / 2 - 1 ))
            
            BATCH_SIZE=4
            nohup python3 main.py --mode attack \
                                        --attack_name ${ATTACK} --num_examples 1000 --dataset ${DATASET} \
                                        --query_budget -1 --batch_size ${BATCH_SIZE} --model_type ${MODEL_TYPE} --model ${MODEL} \
                                        --verbalizer_file ${VERBALIZER_FILE} --template_file ${TEMPLATE_FILE} \
                                        --seed $SEED --shot ${SHOT} --path ${MODELPATH} \
                                        --adv_augment $ADV --knn_k $KNN --beta ${BETA} --max_percent_words 0.15 --examples_per_label 1 \
                                        --num_labels 2 --knn_T $KNN_T > ${MODELPATH}/logs_knn_${ATTACK}_${BETA}.txt
        done
    done
done
