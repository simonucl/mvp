## ICL
CUDA_VISIBLE_DEVICES=1 bash scripts/test_icl.sh 8 sst2 meta-llama/Llama-2-7b-hf icl configs/templates_sst2_icl.yaml configs/verbalizer_sst2.yaml textfooler 0
## ICL (nohup)
CUDA_VISIBLE_DEVICES=1 nohup bash scripts/test_icl.sh 8 sst2 meta-llama/Llama-2-7b-hf icl configs/templates_sst2_icl.yaml configs/verbalizer_sst2.yaml textfooler 0 > ./logs/run_icl.log 2>&1 &

# KNN+ICL
CUDA_VISIBLE_DEVICES=2 bash scripts/test_knn_icl.sh 8 sst2 meta-llama/Llama-2-7b-hf knn_icl configs/templates_sst2_icl.yaml configs/verbalizer_sst2.yaml textfooler 0

CUDA_VISIBLE_DEVICES=1 nohup bash scripts/test_knn_icl.sh 8 sst2 meta-llama/Llama-2-7b-hf knn_icl configs/templates_sst2_icl.yaml configs/verbalizer_sst2.yaml textfooler 0 > ./logs/run_knn_icl.log 2>&1 &

# Retrieval ICL
CUDA_VISIBLE_DEVICES=0 nohup bash scripts/test_knn_llama_retrieval.sh 8 sst2 meta-llama/Llama-2-7b-hf retrieval_icl configs/templates_sst2_icl.yaml configs/verbalizer_sst2.yaml textfooler 0 > ./logs/run_retrieval_icl.log 2>&1 &

CUDA_VISIBLE_DEVICES=0 bash scripts/test_knn_llama_retrieval.sh 8 sst2 meta-llama/Llama-2-7b-hf retrieval_icl configs/templates_sst2_icl.yaml configs/verbalizer_sst2.yaml textfooler 0

# Retrieval ICL (Pre-computed)
CUDA_VISIBLE_DEVICES=1 nohup bash scripts/test_knn_llama_retrieval.sh 8 sst2 meta-llama/Llama-2-7b-hf retrieval_icl_attack configs/templates_sst2_icl.yaml configs/verbalizer_sst2.yaml textfooler 0 > ./logs/run_retrieval_icl_attack.log 2>&1 &

CUDA_VISIBLE_DEVICES=1 bash scripts/test_knn_llama_retrieval.sh 8 sst2 meta-llama/Llama-2-7b-hf retrieval_icl_attack configs/templates_sst2_icl.yaml configs/verbalizer_sst2.yaml textfooler 0

# Retrieval ICL (Pre-computed) attack
CUDA_VISIBLE_DEVICES=2 bash scripts/test_knn_llama_retrieval_attack.sh 8 sst2 meta-llama/Llama-2-7b-hf retrieval_icl_attack configs/templates_sst2_icl.yaml configs/verbalizer_sst2.yaml textfooler 0 > ./logs/run_retrieval_icl_attack_attack.log 2>&1 &

CUDA_VISIBLE_DEVICES=2 bash scripts/test_knn_llama_retrieval_attack.sh 8 sst2 meta-llama/Llama-2-7b-hf retrieval_icl_attack configs/templates_sst2_icl.yaml configs/verbalizer_sst2.yaml textfooler 0


# Seed 42, Shot 4,8, KNN-ICL for SST-2 (nohup)
CUDA_VISIBLE_DEVICES=0 nohup bash scripts/test_knn_icl.sh 8 sst2 meta-llama/Llama-2-7b-hf knn_icl configs/templates_sst2_icl.yaml configs/verbalizer_sst2.yaml textfooler 0 > ./logs/run_knn_icl_0.log 2>&1 &

# Seed 42, Shot 16, KNN-ICL for SST-2 (nohup)
CUDA_VISIBLE_DEVICES=1 nohup bash scripts/test_knn_icl_1.sh 8 sst2 meta-llama/Llama-2-7b-hf knn_icl configs/templates_sst2_icl.yaml configs/verbalizer_sst2.yaml textfooler 0 > ./logs/run_knn_icl_1.log 2>&1 &

# Seed 42, Shot 32, KNN-ICL for SST-2 (nohup)
CUDA_VISIBLE_DEVICES=2 nohup bash scripts/test_knn_icl_2.sh 8 sst2 meta-llama/Llama-2-7b-hf knn_icl configs/templates_sst2_icl.yaml configs/verbalizer_sst2.yaml textfooler 0 > ./logs/run_knn_icl_2.log 2>&1 &

# Seed 42, Shot 64, KNN-ICL (icl_attack) for SST-2
CUDA_VISIBLE_DEVICES=0 bash scripts/test_knn_icl_attack.sh 8 sst2 meta-llama/Llama-2-7b-hf knn_icl configs/templates_sst2_icl.yaml configs/verbalizer_sst2.yaml icl_attack 0

# Seed 1, Shot 4, ICL for RTE 
CUDA_VISIBLE_DEVICES=1 bash scripts/test_icl.sh 8 mnli meta-llama/Llama-2-7b-hf icl configs/templates_mnli.yaml configs/verbalizer_mnli.yaml textfooler 0

CUDA_VISIBLE_DEVICES=0 nohup bash scripts/test_icl_quantized_bound.sh 8 rte meta-llama/Llama-2-7b-hf icl_attack configs/templates_rte.yaml configs/verbalizer_rte.yaml swap_labels 0 > ./logs/run_icl_quantized_bound.log 2>&1 &

CUDA_VISIBLE_DEVICES=1 nohup bash scripts/test_icl_quantized.sh 8 rte meta-llama/Llama-2-7b-hf icl_attack configs/templates_rte.yaml configs/verbalizer_rte.yaml swap_labels 0 > ./logs/run_icl_quantized.log 2>&1 &

CUDA_VISIBLE_DEVICES=0 nohup bash scripts/test_icl_quantized_test.sh 8 rte meta-llama/Llama-2-7b-hf icl_attack configs/templates_rte.yaml configs/verbalizer_rte.yaml swap_orders 0 > ./logs/run_icl_swap_orders.log 2>&1 &

CUDA_VISIBLE_DEVICES=1 nohup bash scripts/test_icl_quantized_test.sh 8 rte meta-llama/Llama-2-7b-hf icl_attack configs/templates_rte.yaml configs/verbalizer_rte.yaml irrelevant_sample 0 > ./logs/run_icl_irrelevant_sample.log 2>&1 &

CUDA_VISIBLE_DEVICES=0 bash scripts/test_icl_quantized_test.sh 8 rte meta-llama/Llama-2-7b-hf knn_icl_attack configs/templates_rte.yaml configs/verbalizer_rte.yaml swap_labels 0

# RTE, KNN-ICL, quantized, swap_labels attack
CUDA_VISIBLE_DEVICES=0 bash scripts/test_knn_icl_quantized_swap_labels.sh 8 rte meta-llama/Llama-2-7b-hf knn_icl_attack configs/templates_rte.yaml configs/verbalizer_rte.yaml swap_labels 0 100

# RTE, KNN-ICL, quantized, irrelevant_sample attack
CUDA_VISIBLE_DEVICES=1 bash scripts/test_knn_icl_quantized_swap_labels.sh 8 rte meta-llama/Llama-2-7b-hf knn_icl_attack configs/templates_rte.yaml configs/verbalizer_rte.yaml irrelevant_sample 0 100



# RTE, RALM, quantized, swap_labels attack
CUDA_VISIBLE_DEVICES=0 bash scripts/test_ralm_1.sh 8 rte meta-llama/Llama-2-7b-hf retrieval_icl configs/templates_rte.yaml configs/verbalizer_rte.yaml swap_labels 0

# RTE, RALM-attack, quantized, swap_labels attack
CUDA_VISIBLE_DEVICES=1 bash scripts/test_ralm_1.sh 8 rte meta-llama/Llama-2-7b-hf retrieval_icl_attack configs/templates_rte.yaml configs/verbalizer_rte.yaml swap_labels 0







CUDA_VISIBLE_DEVICES=2 nohup bash scripts/test_icl.sh 8 rte meta-llama/Llama-2-7b-hf icl_attack configs/templates_rte.yaml configs/verbalizer_rte.yaml swap_labels 0 > ./logs/run_icl.log 2>&1 &

CUDA_VISIBLE_DEVICES=2 bash scripts/test_icl_precision.sh 8 rte meta-llama/Llama-2-7b-hf icl_attack configs/templates_rte.yaml configs/verbalizer_rte.yaml swap_labels 0 > ./logs/run_icl_precision.log 2>&1 &

CUDA_VISIBLE_DEVICES=1 bash scripts/test_icl_precision_1.sh 8 rte meta-llama/Llama-2-7b-hf icl_attack configs/templates_rte.yaml configs/verbalizer_rte.yaml swap_labels 0 > ./logs/run_icl_precision_1.log 2>&1 &


CUDA_VISIBLE_DEVICES=1 bash scripts/test_knn_icl_quantized.sh 8 trec meta-llama/Llama-2-7b-hf knn_icl configs/templates_trec.yaml configs/verbalizer_trec.yaml textfooler 0 100

CUDA_VISIBLE_DEVICES=0 bash scripts/test_icl.sh 8 rte meta-llama/Llama-2-7b-hf icl configs/templates_rte.yaml configs/verbalizer_rte.yaml textfooler 0

# Seed 1, Shot 4, ICL for MNLI
CUDA_VISIBLE_DEVICES=0 nohup bash scripts/test_icl.sh 8 mnli meta-llama/Llama-2-7b-hf icl configs/templates_mnli.yaml configs/verbalizer_mnli.yaml textfooler 0 > ./logs/run_icl_mnli.log 2>&1 &

CUDA_VISIBLE_DEVICES=0 nohup bash scripts/test_icl.sh 8 rte meta-llama/Llama-2-7b-hf icl configs/templates_rte.yaml configs/verbalizer_rte.yaml textfooler 0 > ./logs/run_icl_rte.log 2>&1 &

CUDA_VISIBLE_DEVICES=1 nohup bash scripts/test_knn_icl.sh 8 mnli meta-llama/Llama-2-7b-hf knn_icl configs/templates_mnli.yaml configs/verbalizer_mnli.yaml textfooler 0 100 > ./logs/run_knn_icl_mnli.log 2>&1 &

CUDA_VISIBLE_DEVICES=1 nohup bash scripts/test_knn_icl.sh 8 rte meta-llama/Llama-2-7b-hf knn_icl configs/templates_rte.yaml configs/verbalizer_rte.yaml textfooler 0 100 > ./logs/run_knn_icl_rte.log 2>&1 &

CUDA_VISIBLE_DEVICES=2 nohup bash scripts/test_ralm_1.sh 8 mnli meta-llama/Llama-2-7b-hf retrieval_icl configs/templates_mnli.yaml configs/verbalizer_mnli.yaml textfooler 0 > ./logs/run_ralm_mnli.log 2>&1 &

CUDA_VISIBLE_DEVICES=2 nohup bash scripts/test_ralm_1.sh 8 rte meta-llama/Llama-2-7b-hf retrieval_icl configs/templates_rte.yaml configs/verbalizer_rte.yaml textfooler 0 > ./logs/run_ralm_rte.log 2>&1 &

CUDA_VISIBLE_DEVICES=3 nohup bash scripts/test_ralm_1_1.sh 8 mnli meta-llama/Llama-2-7b-hf retrieval_icl configs/templates_mnli.yaml configs/verbalizer_mnli.yaml textfooler 0 > ./logs/run_ralm_mnli_1.log 2>&1 &

CUDA_VISIBLE_DEVICES=3 nohup bash scripts/test_ralm_1_1.sh 8 rte meta-llama/Llama-2-7b-hf retrieval_icl configs/templates_rte.yaml configs/verbalizer_rte.yaml textfooler 0 > ./logs/run_ralm_rte_1.log 2>&1 &

# Seed 1, Shot 4, ICL for SUBJ
CUDA_VISIBLE_DEVICES=0 bash scripts/test_icl.sh 8 subj meta-llama/Llama-2-7b-hf icl configs/templates_subj.yaml configs/verbalizer_subj.yaml textfooler 0

# Random
CUDA_VISIBLE_DEVICES=0 bash scripts/test_knn_icl_test.sh 8 sst2 meta-llama/Llama-2-7b-hf knn_icl configs/templates_sst2_icl.yaml configs/verbalizer_sst2.yaml textfooler 0 500

CUDA_VISIBLE_DEVICES=0 nohup bash scripts/test_knn_icl_test.sh 32 sst2 meta-llama/Llama-2-7b-hf knn_icl configs/templates_sst2_icl.yaml configs/verbalizer_sst2.yaml textfooler 0 100 > ./logs/run_knn_icl_test.log 2>&1 &

CUDA_VISIBLE_DEVICES=1 nohup bash scripts/test_knn_icl_test_1.sh 8 sst2 meta-llama/Llama-2-7b-hf knn_icl configs/templates_sst2_icl.yaml configs/verbalizer_sst2.yaml textfooler 0 100 > ./logs/run_knn_icl_test_1.log 2>&1 &

CUDA_VISIBLE_DEVICES=2 nohup bash scripts/test_knn_icl_test_2.sh 8 sst2 meta-llama/Llama-2-7b-hf knn_icl configs/templates_sst2_icl.yaml configs/verbalizer_sst2.yaml textfooler 0 100 > ./logs/run_knn_icl_test_2.log 2>&1 &

CUDA_VISIBLE_DEVICES=2 nohup bash scripts/test_icl_attack.sh 8 sst2 meta-llama/Llama-2-7b-hf icl_attack configs/templates_sst2_icl.yaml configs/verbalizer_sst2.yaml icl_attack 0 > ./logs/run_icl_attack.log 2>&1 &

CUDA_VISIBLE_DEVICES=0 nohup bash scripts/test_icl.sh 32 sst2 meta-llama/Llama-2-7b-hf icl configs/templates_sst2_icl.yaml configs/verbalizer_sst2.yaml textfooler 0 > ./logs/run_icl_flash.log 2>&1 &

CUDA_VISIBLE_DEVICES=0 nohup bash scripts/test_knn_icl_test_instruction.sh 8 sst2 meta-llama/Llama-2-7b-hf icl configs/templates_sst2_icl.yaml configs/verbalizer_sst2.yaml textfooler 0 100 > ./logs/run_icl_wo_instruction.log 2>&1 &

CUDA_VISIBLE_DEVICES=1 nohup bash scripts/test_icl_attack.sh 8 sst2 meta-llama/Llama-2-7b-hf icl_attack configs/templates_sst2_icl.yaml configs/verbalizer_sst2.yaml icl_attack 0 > ./logs/run_icl_attack.log 2>&1 &

CUDA_VISIBLE_DEVICES=1 nohup bash scripts/test_icl_attack.sh 8 sst2 meta-llama/Llama-2-7b-hf icl_attack configs/templates_sst2_icl.yaml configs/verbalizer_sst2.yaml icl_attack_word 0 > ./logs/run_icl_attack_word.log 2>&1 &

CUDA_VISIBLE_DEVICES=0 nohup bash scripts/test_ralm.sh 32 sst2 meta-llama/Llama-2-7b-hf retrieval_icl configs/templates_sst2_icl.yaml configs/verbalizer_sst2.yaml textfooler 0 > ./logs/run_ralm.log 2>&1 &

CUDA_VISIBLE_DEVICES=1 nohup bash scripts/test_ralm_shot_2.sh 32 sst2 meta-llama/Llama-2-7b-hf retrieval_icl configs/templates_sst2_icl.yaml configs/verbalizer_sst2.yaml textfooler 0 > ./logs/run_ralm_shot_2.log 2>&1 &

CUDA_VISIBLE_DEVICES=1 nohup bash scripts/test_ralm_1.sh 32 sst2 meta-llama/Llama-2-7b-hf retrieval_icl_attack configs/templates_sst2_icl.yaml configs/verbalizer_sst2.yaml textfooler 0 > ./logs/run_ralm_1.log 2>&1 &

CUDA_VISIBLE_DEVICES=0 nohup bash scripts/test_knn_icl_attack.sh 32 sst2 meta-llama/Llama-2-7b-hf knn_icl configs/templates_sst2_icl.yaml configs/verbalizer_sst2.yaml icl_attack 0 > ./logs/run_knn_icl_attack.log 2>&1 &

CUDA_VISIBLE_DEVICES=0 nohup bash scripts/test_icl.sh 32 sst2 meta-llama/Llama-2-7b-hf icl configs/templates_sst2_icl.yaml configs/verbalizer_sst2.yaml icl_attack 0 > ./logs/run_knn_icl_attack.log 2>&1 &
