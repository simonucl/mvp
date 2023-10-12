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

CUDA_VISIBLE_DEVICES=0 bash scripts/test_knn_icl_test.sh 8 sst2 meta-llama/Llama-2-7b-hf knn_icl configs/templates_sst2_icl.yaml configs/verbalizer_sst2.yaml textfooler 0 500

CUDA_VISIBLE_DEVICES=0 nohup bash scripts/test_knn_icl_test.sh 8 sst2 meta-llama/Llama-2-7b-hf knn_icl configs/templates_sst2_icl.yaml configs/verbalizer_sst2.yaml textfooler 0 100 > ./logs/run_knn_icl_test.log 2>&1 &

CUDA_VISIBLE_DEVICES=1 nohup bash scripts/test_knn_icl_test_1.sh 8 sst2 meta-llama/Llama-2-7b-hf knn_icl configs/templates_sst2_icl.yaml configs/verbalizer_sst2.yaml textfooler 0 100 > ./logs/run_knn_icl_test_1.log 2>&1 &

CUDA_VISIBLE_DEVICES=2 nohup bash scripts/test_knn_icl_test_2.sh 8 sst2 meta-llama/Llama-2-7b-hf knn_icl configs/templates_sst2_icl.yaml configs/verbalizer_sst2.yaml textfooler 0 100 > ./logs/run_knn_icl_test_2.log 2>&1 &

CUDA_VISIBLE_DEVICES=0 nohup bash scripts/test_icl_attack.sh 8 sst2 meta-llama/Llama-2-7b-hf icl_attack configs/templates_sst2_icl.yaml configs/verbalizer_sst2.yaml icl_attack 0 > ./logs/run_icl_attack.log 2>&1 &

CUDA_VISIBLE_DEVICES=1 nohup bash scripts/test_icl.sh 8 sst2 meta-llama/Llama-2-7b-hf icl configs/templates_sst2_icl.yaml configs/verbalizer_sst2.yaml textfooler 0 > ./logs/run_icl.log 2>&1 &

CUDA_VISIBLE_DEVICES=2 nohup bash scripts/test_icl_attack.sh 8 sst2 meta-llama/Llama-2-7b-hf icl_attack configs/templates_sst2_icl.yaml configs/verbalizer_sst2.yaml icl_attack_word 0 > ./logs/run_icl_attack_word.log 2>&1 &
