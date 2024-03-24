DATASET=$1
MODEL=$2
MODEL_TYPE=$3 # [icl | knn_icl | retrieval_icl | retrieval_icl_attack ]
TOTAL_BATCH=$4

# replace '/' with '_'
MODEL_NAME=${MODEL//\//_}

# wait until this command is finished then next 
nohup bash scripts/icl/attack_all_model_mistral.sh $DATASET $MODEL icl $TOTAL_BATCH > ./logs/run_icl_${DATASET}_${MODEL_NAME}_all_model.log 2>&1

# CUDA_VISIBLE_DEVICES=1 nohup bash scripts/ralm/attack_all_model.sh rte mistralai/Mistral-7B-v0.1 retrieval_icl 16 > ./logs/run_ralm_rte_mistralai_Mistral-7B-v0.1_all_model.log 2>&1

# CUDA_VISIBLE_DEVICES=0 nohup bash scripts/scaling_model/transfer_model_mistral.sh rte mistralai/Mistral-7B-v0.1 icl_attack 16 > ./logs/run_transfer_rte_mistralai_Mistral-7B-v0.1_icl_attack.log 2>&1