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

for SEED in 13;
do
    for SHOT in 4 8 16 32;
    do 
        echo $SEED+${SHOT}+${MODEL}+"mvp"

        MODEL_ID=${MODEL_TYPE}-seed-${SEED}-shot-${SHOT}
        
        ATTACK=textfooler
        MODELPATH=./checkpoints/${DATASET}/${MODEL}/${ATTACK}/${MODEL_ID}

        DATASET_PATH=./data/${DATASET}/${SHOT}-$SEED

        mkdir -p ${MODELPATH}
        echo ${MODELPATH}

        if [[ $SHOT -eq 2 ]]; then
            Ms=(1)
        else
            Ms=(1 2)
        fi

        for M in ${Ms[@]};
        do
            for BETA in 0.5 0.8;
            do
                
                mkdir -p ${MODELPATH}/example-${M}
                echo ${MODELPATH}/example-${M}
                # MODEL_TYPE=knn_icl
                KNN=4
                nohup python3 main.py --mode attack \
                                            --attack_name ${ATTACK} --num_examples 1000 --dataset ${DATASET} \
                                            --query_budget -1 --batch_size ${BATCH_SIZE} --model_type ${MODEL_TYPE} --model ${MODEL} \
                                            --verbalizer_file ${VERBALIZER_FILE} --template_file ${TEMPLATE_FILE} \
                                            --seed $SEED --shot ${SHOT} \
                                            --adv_augment $ADV --knn_k $KNN --examples_per_label ${M} --beta ${BETA} > ${MODELPATH}/example-${M}/logs_${ATTACK}_beta_${BETA}.txt
            done
        done
    done
done
