CUDA_VISIBLE_DEVICES=0 nohup bash scripts/icl/attack_rte.sh rte meta-llama/Llama-2-7b-hf icl_attack swap_labels > ./logs/run_icl_rte_swap_labels.log 2>&1

CUDA_VISIBLE_DEVICES=0 nohup bash scripts/ralm/attack_rte.sh rte meta-llama/Llama-2-7b-hf retrieval_icl swap_labels > ./logs/run_retrieval_icl_rte_swap_labels.log 2>&1