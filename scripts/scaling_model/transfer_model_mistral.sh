DATASET=$1
MODEL=$2
MODEL_TYPE=$3 # [icl | knn_icl | retrieval_icl | retrieval_icl_attack ]
TOTAL_BATCH=$4

# replace '/' with '_'
MODEL_NAME=${MODEL//\//_}

# wait until this command is finished then next 
nohup bash scripts/ralm/attack_all_model.sh $DATASET $MODEL retrieval_icl $TOTAL_BATCH > ./logs/run_ralm_${DATASET}_${MODEL_NAME}_all_model.log 2>&1

# CUDA_VISIBLE_DEVICES=1 nohup bash scripts/ralm/attack_all_model.sh rte mistralai/Mistral-7B-v0.1 retrieval_icl 16 > ./logs/run_ralm_rte_mistralai_Mistral-7B-v0.1_all_model.log 2>&1