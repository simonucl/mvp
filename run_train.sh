# Adversarial training for SST-2
# CUDA_VISIBLE_DEVICES=0 bash scripts/train_adv_1_seed.sh 32 sst2 roberta-base mvp 20 1e-5 max mean max mean configs/templates_sst2.yaml configs/verbalizer_sst2.yaml mvp_adv textfooler train -1 0.95 0.05 1 l2 1

# Non adversarial training for SST-2
# CUDA_VISIBLE_DEVICES=0 bash scripts/train_1_seed.sh 32 sst2 roberta-base mvp 20 1e-5 max mean max mean configs/templates_sst2.yaml configs/verbalizer_sst2.yaml mvp_seed_0 textfooler train -1 0.95 0.05

# Few shot training for SST-2
# CUDA_VISIBLE_DEVICES=0 bash scripts/train_1_seed_fewshot.sh 8 sst2 roberta-base mvp 20 1e-5 max mean max mean configs/templates_sst2.yaml configs/verbalizer_sst2.yaml mvp_seed_0 textfooler train -1 0.95 0.05 ./data/sst2/64-13 64

# Adversarial training for SST-2
CUDA_VISIBLE_DEVICES=0 bash scripts/train_adv_and_test_few_shot.sh 64 sst2 roberta-large mvp 20 1e-5 max mean max mean configs/templates_sst2.yaml configs/verbalizer_sst2.yaml mvp_adv textfooler train -1 0.95 0.05 1 l2 1 ./data/sst2/64-13 64

nohup bash scripts/train_adv_and_test_few_shot.sh 64 sst2 roberta-large mvp 20 1e-5 max mean max mean configs/templates_sst2.yaml configs/verbalizer_sst2.yaml mvp_adv textfooler train -1 0.95 0.05 1 l2 1 ./data/sst2/64-13 64 > ./logs/run.log 2>&1 & 

bash scripts/train_adv_and_test_few_shot.sh 64 sst2 roberta-large knn_cli 20 1e-5 max mean max mean configs/templates_sst2.yaml configs/verbalizer_sst2.yaml mvp_adv textfooler train -1 0.95 0.05 1 l2 1 ./data/sst2/64-13 64

CUDA_VISIBLE_DEVICES=0 nohup bash scripts/test_knn_k.sh 64 sst2 roberta-large knn_cli 20 1e-5 max mean max mean configs/templates_sst2.yaml configs/verbalizer_sst2.yaml mvp_adv textfooler train -1 0.95 0.05 1 l2 1 ./data/sst2/64-13 64 0 > ./logs/run_cuda_0.log 2>&1 &

CUDA_VISIBLE_DEVICES=1 bash scripts/test_knn_k.sh 64 sst2 roberta-large knn_icl 20 1e-5 max mean max mean configs/templates_sst2.yaml configs/verbalizer_sst2.yaml mvp_adv textfooler train -1 0.95 0.05 1 l2 1 ./data/sst2/64-13 64 0

CUDA_VISIBLE_DEVICES=1 nohup bash scripts/test_knn_k.sh 64 sst2 roberta-large knn_cli 20 1e-5 max mean max mean configs/templates_sst2.yaml configs/verbalizer_sst2.yaml mvp_adv textfooler train -1 0.95 0.05 1 l2 1 ./data/sst2/64-13 64 1 > ./logs/run_cuda_1.log 2>&1 &

CUDA_VISIBLE_DEVICES=1 bash scripts/test_knn.sh 64 sst2 roberta-large knn_icl 20 1e-5 max mean max mean configs/templates_sst2.yaml configs/verbalizer_sst2.yaml mvp_adv textfooler train -1 0.95 0.05 1 l2 1 ./data/sst2/64-13 64 1

bash scripts/test_knn.sh 64 sst2 roberta-large knn_cli 20 1e-5 max mean max mean configs/templates_sst2.yaml configs/verbalizer_sst2.yaml mvp_adv textfooler train -1 0.95 0.05 1 l2 1 ./data/sst2/64-13 64 1

bash scripts/test_knn.sh 64 sst2 roberta-large knn_icl 20 1e-5 max mean max mean configs/templates_sst2.yaml configs/verbalizer_sst2.yaml mvp_adv textfooler train -1 0.95 0.05 1 l2 1 ./data/sst2/64-13 64 1

CUDA_VISIBLE_DEVICES=1 nohup bash scripts/test_knn_icl_attack.sh 64 sst2 roberta-large icl_attack 20 1e-5 max mean max mean configs/templates_sst2.yaml configs/verbalizer_sst2.yaml mvp_adv icl_attack train -1 0.95 0.05 1 l2 1 ./data/sst2/64-13 64 0 > ./logs/run_cuda_mask.log 2>&1 &

nohup bash scripts/test_knn.sh 64 sst2 roberta-large knn_icl 20 1e-5 max mean max mean configs/templates_sst2.yaml configs/verbalizer_sst2.yaml mvp_adv textfooler train -1 0.95 0.05 1 l2 1 ./data/sst2/64-13 64 > ./logs/run.log 2>&1 &

CUDA_VISIBLE_DEVICES=1 bash scripts/test_knn.sh 64 sst2 meta-llama/Llama-2-13b-hf knn_cli 20 1e-5 max mean max mean configs/templates_sst2.yaml configs/verbalizer_sst2.yaml mvp_adv textfooler train -1 0.95 0.05 1 l2 1 ./data/sst2/64-13 64 0

CUDA_VISIBLE_DEVICES=0 nohup bash scripts/test_knn_k_icl.sh 64 sst2 roberta-large knn_icl 20 1e-5 max mean max mean configs/templates_sst2.yaml configs/verbalizer_sst2.yaml mvp_adv textfooler train -1 0.95 0.05 1 l2 1 ./data/sst2/64-13 64 0 > ./logs/run_knn_icl.log 2>&1 &
CUDA_VISIBLE_DEVICES=0 bash scripts/test_knn_k_icl.sh 64 sst2 roberta-large knn_icl 20 1e-5 max mean max mean configs/templates_sst2.yaml configs/verbalizer_sst2.yaml mvp_adv textfooler train -1 0.95 0.05 1 l2 1 ./data/sst2/64-13 64 0

CUDA_VISIBLE_DEVICES=1 nohup bash scripts/test_knn_k_icl.sh 64 sst2 roberta-large knn_icl 20 1e-5 max mean max mean configs/templates_sst2.yaml configs/verbalizer_sst2.yaml mvp_adv textfooler train -1 0.95 0.05 1 l2 1 ./data/sst2/64-13 64 1 > ./logs/run_knn_icl_adv.log 2>&1 &

CUDA_VISIBLE_DEVICES=0 nohup bash scripts/test_knn_mask.sh 64 sst2 roberta-large knn_cli 20 1e-5 max mean max mean configs/templates_sst2.yaml configs/verbalizer_sst2.yaml mvp_adv textfooler train -1 0.95 0.05 1 l2 1 ./data/sst2/64-13 64 1 > ./logs/run_knn_mask.log 2>&1 &

bash scripts/test_knn_icl_attack.sh 64 sst2 roberta-large icl_attack 20 1e-5 max mean max mean configs/templates_sst2.yaml configs/verbalizer_sst2.yaml mvp_adv icl_attack train -1 0.95 0.05 1 l2 1 ./data/sst2/64-13 64 0

CUDA_VISIBLE_DEVICES=0 nohup bash scripts/test_knn_beta.sh 64 sst2 roberta-large knn_cli 20 1e-5 max mean max mean configs/templates_sst2.yaml configs/verbalizer_sst2.yaml mvp_adv textfooler train -1 0.95 0.05 1 l2 1 ./data/sst2/64-13 64 0 > ./logs/run_knn_cli_beta.log 2>&1 &
CUDA_VISIBLE_DEVICES=1 nohup bash scripts/test_knn_beta.sh 32 sst2 roberta-large knn_icl 20 1e-5 max mean max mean configs/templates_sst2.yaml configs/verbalizer_sst2.yaml mvp_adv textfooler train -1 0.95 0.05 1 l2 1 ./data/sst2/64-13 64 0 > ./logs/run_knn_icl_beta.log 2>&1 &

CUDA_VISIBLE_DEVICES=0 nohup bash scripts/test_knn_k_icl.sh 64 sst2 roberta-large knn_icl 20 1e-5 max mean max mean configs/templates_sst2.yaml configs/verbalizer_sst2.yaml mvp_adv textfooler train -1 0.95 0.05 1 l2 1 ./data/sst2/64-13 64 0 > ./logs/run_knn_icl.log 2>&1 &

CUDA_VISIBLE_DEVICES=0 bash scripts/test_knn.sh 64 sst2 roberta-large icl 20 1e-5 max mean max mean configs/templates_sst2.yaml configs/verbalizer_sst2.yaml mvp_adv textfooler train -1 0.95 0.05 1 l2 1 ./data/sst2/64-13 64 0

CUDA_VISIBLE_DEVICES=0 nohup bash scripts/test_knn_llama.sh 4 sst2 meta-llama/Llama-2-7b-hf icl 20 1e-5 max mean max mean configs/templates_sst2_icl.yaml configs/verbalizer_sst2.yaml mvp_adv textfooler train -1 0.95 0.05 1 l2 1 ./data/sst2/64-13 64 0 > ./logs/run_knn_llama.log 2>&1 &

CUDA_VISIBLE_DEVICES=0 nohup bash scripts/test_knn_llama.sh 4 sst2 meta-llama/Llama-2-7b-hf knn_icl 20 1e-5 max mean max mean configs/templates_sst2_icl.yaml configs/verbalizer_sst2.yaml mvp_adv textfooler train -1 0.95 0.05 1 l2 1 ./data/sst2/64-13 64 0 > ./logs/run_knn_llama.log 2>&1 &

CUDA_VISIBLE_DEVICES=2 nohup bash scripts/test_knn_llama_example.sh 4 sst2 meta-llama/Llama-2-7b-hf knn_icl 20 1e-5 max mean max mean configs/templates_sst2_icl.yaml configs/verbalizer_sst2.yaml mvp_adv textfooler train -1 0.95 0.05 1 l2 1 ./data/sst2/64-13 64 0 > ./logs/run_knn_llama_example.log 2>&1 &

CUDA_VISIBLE_DEVICES=1 nohup bash scripts/test_knn_llama_beta.sh 4 sst2 meta-llama/Llama-2-7b-hf knn_icl 20 1e-5 max mean max mean configs/templates_sst2_icl.yaml configs/verbalizer_sst2.yaml mvp_adv textfooler train -1 0.95 0.05 1 l2 1 ./data/sst2/64-13 64 0 > ./logs/run_knn_llama_beta.log 2>&1 &

CUDA_VISIBLE_DEVICES=1 nohup bash scripts/test_knn_llama_example.sh 4 sst2 mistralai/Mistral-7B-v0.1 knn_icl 20 1e-5 max mean max mean configs/templates_sst2_icl.yaml configs/verbalizer_sst2.yaml mvp_adv textfooler train -1 0.95 0.05 1 l2 1 ./data/sst2/64-13 64 0 > ./logs/run_knn_mistral.log 2>&1 &

CUDA_VISIBLE_DEVICES=0 nohup bash scripts/test_knn_llama_mvp.sh 4 sst2 gpt2-xl mvp 20 1e-5 max mean max mean configs/templates_sst2_old.yaml configs/verbalizer_sst2_old.yaml mvp_adv textfooler train -1 0.95 0.05 1 l2 1 ./data/sst2/64-13 64 0 > ./logs/run_mvp.log 2>&1 &