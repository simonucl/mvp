## ICL
CUDA_VISIBLE_DEVICES=1 bash scripts/test_icl.sh 8 sst2 meta-llama/Llama-2-7b-hf icl configs/templates_sst2_icl.yaml configs/verbalizer_sst2.yaml textfooler 0
## ICL (nohup)
CUDA_VISIBLE_DEVICES=1 nohup bash scripts/test_icl.sh 8 sst2 meta-llama/Llama-2-7b-hf icl configs/templates_sst2_icl.yaml configs/verbalizer_sst2.yaml textfooler 0 > ./logs/run_icl.log 2>&1 &

# KNN+ICL
CUDA_VISIBLE_DEVICES=2 bash scripts/test_knn_icl.sh 8 sst2 meta-llama/Llama-2-7b-hf knn_icl configs/templates_sst2_icl.yaml configs/verbalizer_sst2.yaml textfooler 0

CUDA_VISIBLE_DEVICES=0 nohup bash scripts/test_knn_icl.sh 4 sst2 meta-llama/Llama-2-7b-hf knn_icl configs/templates_sst2_icl.yaml configs/verbalizer_sst2.yaml textfooler 0 > ./logs/run_knn_icl.log 2>&1 &

# KNN with hidden states as features
CUDA_VISIBLE_DEVICES=2 bash scripts/test_knn_icl.sh 4 sst2 meta-llama/Llama-2-7b-hf knn_features configs/templates_sst2_icl.yaml configs/verbalizer_sst2.yaml textfooler 0

CUDA_VISIBLE_DEVICES=1 bash scripts/test_icl_test.sh 4 sst2 meta-llama/Llama-2-7b-hf icl configs/templates_sst2_icl.yaml configs/verbalizer_sst2.yaml textfooler 0
