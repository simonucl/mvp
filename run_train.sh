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
