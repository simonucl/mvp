# Adversarial training for SST-2
# CUDA_VISIBLE_DEVICES=0 bash scripts/train_adv_1_seed.sh 32 sst2 roberta-base mvp 20 1e-5 max mean max mean configs/templates_sst2.yaml configs/verbalizer_sst2.yaml mvp_adv textfooler train -1 0.95 0.05 1 l2 1

# Non adversarial training for SST-2
# CUDA_VISIBLE_DEVICES=0 bash scripts/train_1_seed.sh 32 sst2 roberta-base mvp 20 1e-5 max mean max mean configs/templates_sst2.yaml configs/verbalizer_sst2.yaml mvp_seed_0 textfooler train -1 0.95 0.05

# Few shot training for SST-2
# CUDA_VISIBLE_DEVICES=0 bash scripts/train_1_seed_fewshot.sh 8 sst2 roberta-base mvp 20 1e-5 max mean max mean configs/templates_sst2.yaml configs/verbalizer_sst2.yaml mvp_seed_0 textfooler train -1 0.95 0.05 ./data/sst2/64-13 64

# Adversarial training for SST-2
# CUDA_VISIBLE_DEVICES=0 bash scripts/train_adv_and_test_few_shot.sh 32 sst2 roberta-base mvp 20 1e-5 max mean max mean configs/templates_sst2.yaml configs/verbalizer_sst2.yaml mvp_adv textfooler train -1 0.95 0.05 1 l2 1 ./data/sst2/64-13 64

CUDA_VISIBLE_DEVICES=0 bash scripts/train_adv_and_test_few_shot_roberta.sh 32 sst-5 roberta-large mvp 20 1e-5 max mean max mean configs/templates_sst2.yaml configs/verbalizer_sst5.yaml mvp_adv textfooler train -1 0.95 0.05 1 l2 1 ./data/sst-5/64-13 64
