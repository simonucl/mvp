# ICL

## RTE
### textfooler
CUDA_VISIBLE_DEVICES=0 nohup bash scripts/icl/attack.sh rte meta-llama/Llama-2-7b-hf icl textfooler > ./logs/run_icl_rte_textfooler.log 2>&1 &

### textbugger
CUDA_VISIBLE_DEVICES=1 nohup bash scripts/icl/attack.sh rte meta-llama/Llama-2-7b-hf icl textbugger > ./logs/run_icl_rte_textbugger.log 2>&1 &

### swap_labels
CUDA_VISIBLE_DEVICES=0 nohup bash scripts/icl/attack.sh rte meta-llama/Llama-2-7b-hf icl_attack swap_labels > ./logs/run_icl_rte_swap_labels.log 2>&1 &

### swap_labels fix dist
CUDA_VISIBLE_DEVICES=0 nohup bash scripts/icl/attack_fix_dist.sh rte meta-llama/Llama-2-7b-hf icl_attack swap_labels > ./logs/run_icl_rte_swap_labels.log 2>&1 &

### swap_orders
CUDA_VISIBLE_DEVICES=1 nohup bash scripts/icl/attack_fix_dist.sh rte meta-llama/Llama-2-7b-hf icl_attack swap_orders > ./logs/run_icl_rte_swap_orders.log 2>&1 &


################################################
## SST2
### textfooler

### textbugger

### swap_labels

### swap_labels fix dist

### swap_orders

################################################
################################################
################################################

# KNN-ICL
## RTE
### textfooler
CUDA_VISIBLE_DEVICES=0 nohup bash scripts/icl/attack.sh rte meta-llama/Llama-2-7b-hf knn_icl textfooler > ./logs/run_knn_icl_rte_textfooler.log 2>&1 &

### textbugger
CUDA_VISIBLE_DEVICES=1 nohup bash scripts/icl/attack.sh rte meta-llama/Llama-2-7b-hf knn_icl textbugger > ./logs/run_knn_icl_rte_textbugger.log 2>&1 &

### swap_labels
CUDA_VISIBLE_DEVICES=0 nohup bash scripts/knn_icl/attack.sh rte meta-llama/Llama-2-7b-hf knn_icl_attack swap_labels > ./logs/run_knn_icl_rte_swap_labels.log 2>&1 &

### swap_labels fix dist

### irrelevant_sample
CUDA_VISIBLE_DEVICES=1 nohup bash scripts/knn_icl/attack.sh rte meta-llama/Llama-2-7b-hf knn_icl_attack irrelevant_sample > ./logs/run_knn_icl_rte_irrelevant_sample.log 2>&1 &

### textbugger
### icl_attack
### swap_labels
CUDA_VISIBLE_DEVICES=0 nohup bash scripts/knn_icl/attack_quantized_fix_dist.sh rte meta-llama/Llama-2-7b-hf knn_icl swap_labels > ./logs/run_knn_icl_rte_swap_labels.log 2>&1 &

### swap_orders
CUDA_VISIBLE_DEVICES=1 nohup bash scripts/knn_icl/attack.sh rte meta-llama/Llama-2-7b-hf knn_icl swap_orders > ./logs/run_knn_icl_rte_swap_orders.log 2>&1 &

### irrelevant_sample
CUDA_VISIBLE_DEVICES=1 nohup bash scripts/knn_icl/attack_quantized.sh rte meta-llama/Llama-2-7b-hf knn_icl irrelevant_sample > ./logs/run_knn_icl_rte_irrelevant_sample.log 2>&1 &

################################################
################################################
################################################


## Retrieval-ICL

### textbugger
### icl_attack
### swap_labels
CUDA_VISIBLE_DEVICES=0 nohup bash scripts/ralm/attack_quantized_fix_dist.sh rte meta-llama/Llama-2-7b-hf retrieval_icl swap_labels > ./logs/run_retrieval_icl_rte_swap_labels.log 2>&1 &
### swap_orders
CUDA_VISIBLE_DEVICES=1 nohup bash scripts/ralm/attack.sh rte meta-llama/Llama-2-7b-hf retrieval_icl swap_orders > ./logs/run_retrieval_icl_rte_swap_orders.log 2>&1 &

### irrelevant_sample

################################################
################################################
################################################


## Retrieval-ICL-attack

### textbugger
### icl_attack
### swap_labels
CUDA_VISIBLE_DEVICES=1 nohup bash scripts/ralm/attack_quantized_fix_dist.sh rte meta-llama/Llama-2-7b-hf retrieval_icl_attack swap_labels > ./logs/run_retrieval_icl_attack_rte_swap_labels.log 2>&1 &
### swap_orders
CUDA_VISIBLE_DEVICES=1 nohup bash scripts/ralm/attack.sh rte meta-llama/Llama-2-7b-hf retrieval_icl_attack swap_orders > ./logs/run_retrieval_icl_attack_rte_swap_orders.log 2>&1 &

### irrelevant_sample