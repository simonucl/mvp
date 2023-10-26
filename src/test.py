from __future__ import absolute_import
import sys, os
# sys.path.append("textattack_lib/textattack/.")
sys.path.append("customattacks/.")
from textattack.attacker import Attacker
from customattacks import TextFoolerCustom, TextBuggerCustom
# from TextBuggerCustom import TextBuggerCustom
from textattack.attack_recipes import TextFoolerJin2019, TextBuggerLi2018, ICLTextAttack, ICLTextAttackWord, SwapLabel2023, SwapOrderAttack
from src.utils.funcs import *
from src.models import get_model
from textattack import AttackArgs
from textattack.datasets import HuggingFaceDataset
from textattack.attack_recipes import BAEGarg2019
from src.utils.anchor import subsamplebyshot
from collections import OrderedDict
from time import time

import tensorflow as tf
gpus = tf.config.list_physical_devices('GPU')
if gpus:
  try:
    # Currently, memory growth needs to be the same across GPUs
    for gpu in gpus:
      tf.config.experimental.set_memory_growth(gpu, True)
    logical_gpus = tf.config.list_logical_devices('GPU')
    print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
  except RuntimeError as e:
    # Memory growth must be set before GPUs have been initialized
    print(e)

# CUDA_VISIBLE_DEVICES=1 python main.py --mode attack --path models/imdb/bert-base-uncased/model_unk_rand_0.25_no-fill_limit_lm/final_model/ --attack_name pruthi --swap_size 10000 --infill no-fill

def ood_evaluation_loop(args, model):
    ood_datasets = {"movie_rationales":[], "rotten_tomatoes":[],"emotion2":[],"amazon_polarity":[]}
    model.mode = "eval"
    for dataset in ood_datasets.keys():
        args.dataset = dataset
        eval_results = evaluate_model(model, args)
        ood_datasets[dataset] = eval_results
        print(eval_results)
    import json

    with open(f'{args.model_dir}/ood.json', 'w') as fp:
        json.dump(ood_datasets, fp)


def convert_to_icl(data, icl_examples, verbalizer=None):
    if verbalizer is None:
        verbalizer =  {0:["negative"], 1:["positive"]}
    outputs = OrderedDict()
    
    if type(icl_examples) == list:
        idx = data['idx']
        icl_example = icl_examples[idx]
        for i, e in enumerate(icl_example):
            if 'sentence' in e.keys():
                outputs[f"Example_{i}"] = e['sentence']
            elif 'premise' in e.keys():
                outputs[f"Premise_{i}"] = e['premise']
                outputs[f"Hypothesis_{i}"] = e['hypothesis']
            outputs[f"Label_{i}"] = verbalizer[e['label']][0]
    else:
        num_labels = len(icl_examples.keys())
        num_samples_per_label = len(list(icl_examples.values())[0])
        for i in range(num_samples_per_label):
            j = 0
            for k, v in icl_examples.items():
                if 'sentence' in v[i].keys():
                    outputs[f"Example_{i*num_labels + j}"] = v[i]['sentence']
                elif 'premise' in v[i].keys():
                    outputs[f"Premise_{i*num_labels + j}"] = v[i]['premise']
                    outputs[f"Hypothesis_{i*num_labels + j}"] = v[i]['hypothesis']
                outputs[f"Label_{i*num_labels + j}"] = k
                j += 1
    if 'sentence' in data.keys():
        outputs["sentence"] = data['sentence']
        outputs.move_to_end("sentence")
    elif 'premise' in data.keys():
        outputs["premise"] = data['premise']
        outputs["hypothesis"] = data['hypothesis']
        outputs.move_to_end("premise")
        outputs.move_to_end("hypothesis")
    outputs["label"] = data['label']
    # print('Outputs', outputs)
    return outputs

def attacker(args):
    print(args)
    
    #ipdb.set_trace()
    if not os.path.exists(args.model_dir): os.makedirs(args.model_dir)
    file = open(f"{args.model_dir}/eval.txt", "a")  
    def myprint(a): print(a); file.write(a); file.write("\n"); file.flush()
    chkpt_name = os.path.basename(args.path)
    #train dataset is needed to get the right vocabulary for the problem
    my_dataset, tokenizer, data_collator = prepare_huggingface_dataset(args)
    verbalizer, templates = get_prompts(args)
    model = None

    model = get_model(args, my_dataset, tokenizer, data_collator, verbalizer = verbalizer, template = templates)

    print('Finished loading model')
    
    split = args.split
    args.num_examples = min(my_dataset[split].num_rows, args.num_examples)

    if args.model_type in ["icl_attack", "knn_icl_attack", "retrieval_icl_attack"] or args.attack_name in ["swap_labels"]:
        if 'gpt' in args.model:
            num_tokens = 1
        elif ('opt' in args.model) or ('Llama' in args.model):
            num_tokens = 2
        else:
            num_tokens = 3

        if args.model_type in ["icl", "icl_attack", "retrieval_icl_attack", "knn_icl_attack"]:
            examples_per_label = args.shot
        elif args.model_type in ["retrieval_icl"]:
            examples_per_label = 0
        else:
            examples_per_label = args.examples_per_label

        label_set = []
        for k,v in verbalizer.items():
            for word in v:
                if (not is_causal_model(args.model)) or ("gpt" in args.model):
                    word = " " + word
                if (len(tokenizer(word)["input_ids"]) == num_tokens):
                    label_set.append(k)
                else:
                    assert len(tokenizer(word)["input_ids"]) == num_tokens, f"Verbalizer word {word} has {len(tokenizer(word)['input_ids'])} tokens, but model has {num_tokens} tokens"
        # if args.model_type in ["retrieval_icl_attack"]:
        #     anchor_subsample, _ = subsamplebyshot(my_dataset['train'], args.seed, label_set, verbalizer, args.shot, 0)
        #     icl_examples = model.indexEmbedder.subsamplebyretrieval(anchor_subsample, my_dataset[split]['sentence'], args.examples_per_label)
        # else:
        _, icl_examples = subsamplebyshot(my_dataset['train'], args.seed, label_set, verbalizer, args.shot, examples_per_label)

        verbalizer = model.verbalizer if model is not None else None
        # if 'sentence' in my_dataset[0].keys():
        #     remove_columns = 'sentence'
        # elif 'premise' in my_dataset[0].keys():
        #     remove_columns = ['premise', 'hypothesis']
        my_dataset = my_dataset[split].map(lambda x: convert_to_icl(x, icl_examples, verbalizer), batched=False)
        print('ICL examples')
        print(my_dataset[0])
    else:
        my_dataset = my_dataset[split]
    
    columns_name = my_dataset.column_names
    if 'label' in columns_name:
        columns_name.remove('label')
    if 'idx' in columns_name:
        columns_name.remove('idx')
    print(my_dataset[0])
    dataset = HuggingFaceDataset(my_dataset, dataset_columns=(columns_name, 'label'))

    attack_name = args.attack_name

    if attack_name == "none":
        model.mode = "eval"
        eval_results = evaluate_model(model, args)
        myprint (f"Results: {eval_results}")
    
    elif attack_name == "ood":
        ood_evaluation_loop(args, model)
    
    else:
        model.mode = "attack"
        attack_name_mapper = {
            # "textfooler":TextFoolerJin2019, 
            # "textbugger":TextBuggerLi2018,
                            "textfooler": TextFoolerCustom,
                            "textbugger": TextBuggerCustom,
                            "bae": BAEGarg2019,
                            "icl_attack": ICLTextAttack,
                            "icl_attack_word": ICLTextAttackWord,
                            "swap_labels": SwapLabel2023,
                            "swap_orders": SwapOrderAttack
                            }
                            
        attack_class = attack_name_mapper[attack_name]
        log_to_csv=f"{args.model_dir}/{attack_name}_log.csv"

        print(templates)
        print(verbalizer)

        attack = attack_class.build(model)
        
        if args.query_budget < 0: args.query_budget = None 
        attack_args = AttackArgs(num_examples=args.num_examples, 
                                log_to_csv=log_to_csv,
                                disable_stdout=True,
                                enable_advance_metrics=True,
                                query_budget = args.query_budget,parallel=False)
        attacker = Attacker(attack, dataset, attack_args)

        #set batch size of goal function
        attacker.attack.goal_function.batch_size = args.batch_size
        #set max words pertubed constraint
        if attack_name in ["icl_attack", "icl_attack_word"]:
            max_percent_words = 0.1
        elif attack_name in ["swap_labels"]:
            max_percent_words = 0.7
        else:
            max_percent_words = 0.15
        #flag = 0
        
        if args.max_percent_words > 0:
            max_percent_words = args.max_percent_words

        for i,constraint in enumerate(attacker.attack.constraints):
            if type(constraint) == textattack.constraints.overlap.max_words_perturbed.MaxWordsPerturbed:
                attacker.attack.constraints[i].max_percent = max_percent_words
            
        print(attacker)
       
        attack_start_time = time()

        attacker.attack_dataset()

        attack_end_time = time()
        print(f"Attack time: {attack_end_time - attack_start_time}")
        myprint(f"Attack time: {attack_end_time - attack_start_time}")
