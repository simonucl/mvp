from random import triangular
from ..utils.model_utils import *
from ..utils.knn import HuggingFaceSentimentAnalysisPipelineWrapper
import os, pickle, copy
from torch_scatter import scatter_max, scatter_mean
import torch
from torch import nn
from .model_wrapper import ModelWrapper
from ..utils.anchor import AnchorStore
from ..utils.dataset import *
from tqdm import tqdm

SST2_LABELS2ID = {'0': 0, '1': 1}

class KNN_CLI(ModelWrapper):
    def __init__(self, args, model, tokenizer, data_collator, dataset, verbalizer = None, template=None):  
        '''
        args: args object from argparse
        model: huggingface model
        tokenizer: huggingface tokenizer
        data_collator: huggingface data collator
        verbalizer: dictoionary of verbalizer
        template: list of templates

        This is the MVP model
        '''
        label_words = []
        label_set = []
        self.verbalizer = verbalizer
        self.tokenizer = tokenizer
        self.knn_T = args.knn_T
        self.knn_k = args.knn_k

        num_tokens = 1 if 'gpt' in args.model else 3 # bert tokenizes into cls, word, sep. we want the word to be a single token

        # only keep those words that are tokenized into a single token
        for k,v in self.verbalizer.items():
            for word in v:
                if "roberta" in args.model:
                    word = " " + word
                if(len(self.tokenizer(word)["input_ids"]) == num_tokens):
                    label_set.append(k)
                    label_words.append(word)
                else:
                    print(word)
        self.label_set = torch.tensor(label_set)
        toks = self.tokenizer(label_words)["input_ids"]

        if args.dataset == "sst2":
            self.label2id = SST2_LABELS2ID
        else:
            raise NotImplementedError
        
        if 'gpt' not in args.model:
            new_toks = [t for t in toks if len(t) == num_tokens]
            self.label_word_ids = torch.tensor(new_toks)[:,1]
        else:
            new_toks = [t for t in toks]
            self.label_word_ids = torch.tensor(new_toks)[:,0]
        self.template_ids = []
        self.len_templates = []
        for prompt in template:
            used_prompt = prompt.replace("[MASK]", tokenizer.mask_token)
            if used_prompt.split(" ")[0] == "[SEP]":
                used_prompt = " ".join(used_prompt.split(" ")[1:])
            self.len_templates.append(1+len(tokenizer(used_prompt)["input_ids"][1:-1]))

        anchor_data = dataset['train']
        
        # TODO Creation of anchor data
        self.anchor_store = AnchorStore(K=anchor_data.__len__(),
                               dim=model.config.vocab_size,
                               knn=args.knn_k,
                               n_class=len(args.num_labels))
        
        anchor_subsample = self.subsamplebyshot(anchor_data, args.seed, args.shot)
        for ins in tqdm(anchor_subsample, total=anchor_subsample.__len__()):
            labels = self.label2id[ins['label']]
            gen_logits = self.get_logits(ins['text']).detach().cpu()
            self.anchor_store.enqueue(torch.softmax(gen_logits, dim=-1), torch.tensor(labels))

        super(KNN_CLI, self).__init__(args, model, tokenizer, data_collator, verbalizer = verbalizer, template=template)

    def subsamplebyshot(self, anchor_data, seed, shot=1):
        '''
        anchor_data: list of anchor data
        seed: seed for random
        shot: number of examples per class

        returns: subsampled anchor data
        '''
        random.seed(seed)
        anchor_data = copy.deepcopy(anchor_data)
        new_anchor_data = []
        for label in self.label_set:
            label_data = [d for d in anchor_data if d['label'] == label]
            random.shuffle(label_data)
            new_anchor_data.extend(label_data[:shot])
        return new_anchor_data
    
    def get_logits(self, input_ids, attention_mask=None, outputs=None):
        '''
        input_ids: torch tensor of shape (1, seq_len)
        attention_mask: torch tensor of shape (1, seq_len)
        '''
        if outputs is None:
            input_ids, attention_mask = self.get_updated_input_ids(input_ids, attention_mask)
            with torch.no_grad():
                outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs.logits                             # (B * num_templates, seq_len, vocab_size)
        batchid, indices = torch.where(input_ids == self.tokenizer.mask_token_id) # See how the word is inserted
        if 'gpt' in self.args.model:
            # it predicts next word
            indices = indices -1

        mask_logits = logits[batchid, indices,:]         # (B * num_templates, vocab_size)
        # print('Mask logits shape: ', mask_logits.shape)
        label_words_logits = mask_logits[:, self.label_word_ids]    # (B * num_templates, num_candidates)

        # self.label_set = self.label_set.to(input_ids.device)
        # if self.args.pool_label_words == "max":
        #     label_words_logits = scatter_max(label_words_logits, self.label_set)[0] # (B * num_templates, num_classes)
        # elif self.args.pool_label_words == "mean":
        #     label_words_logits = scatter_mean(label_words_logits, self.label_set)   # (B * num_templates, num_classes)
        num_templates = 1 if (self.args.num_template == -2 and self.mode == "train") else len(self.template)
        template_mask = (torch.arange(label_words_logits.shape[0])/(num_templates)).to(torch.long)
        y = torch.stack([template_mask]*label_words_logits.shape[1],dim=1)
        y = y.to(input_ids.device)
        
        if self.args.pool_templates == "mean":
            label_words_logits = scatter_mean(label_words_logits, y, dim=0)   # (B, vocab_size)
        elif self.args.pool_templates == "max":
            label_words_logits = scatter_max(label_words_logits, y, dim=0)[0]  # (B, vocab_size)

        return label_words_logits # (1, vocab_size)
    
    def outs_to_logits(self, input_ids, outputs):
        '''
        input_ids: torch tensor of shape (batch_size, seq_len)
        outputs: output of the model
        raw_inputs: torch tensor of shape (batch_size, seq_len)

        returns logits of shape (batch_size, num_classes)
        '''
        query_logits = self.get_logits(input_ids, outputs=outputs) # (batch_size, vocab_size)
        # Directly return the logits
        # kl_dists = self.anchor_store.knn_infer(query_logits) # [B, K+1]
        # scaled_dists = -1.0 / self.knn_T * kl_dists

        # top_dists, top_indices = torch.topk(scaled_dists, self.knn_k) # [B, K+1], [B, K+1]
        # new_vals = values.unsqueeze(0).repeat(self.args.batch_size, 1) # [B, L]
        # top_values = torch.gather(new_vals, 1, top_indices).unsqueeze(-1)  # [B, K, 1]
        # knn_weight = torch.softmax(top_dists, dim=-1).unsqueeze(-1)  # [B, K, 1]
        
        # # init knn-prob
        # knn_tgt_prob = torch.zeros([self.args.batch_size, self.knn_k, self.args.num_labels], dtype=torch.float32, device=keys.device)
        # knn_tgt_prob.scatter_(2, top_values, knn_weight) # The scatter function is used to scatter the values in knn_weight to the corresponding positions in knn_tgt_prob. 
        # # The 2 in the first parameter means that the scatter is performed on the third dimension of knn_tgt_prob, and the second parameter is the index of the position to be scattered. The third parameter is the value to be scattered.
        # # the final dimension is [B, K, V]
        # prob = knn_tgt_prob.sum(dim=-2)  # [B, V]
        prob = self.anchor_store.knn_calibrate(query_logits)
        
        return prob
    
        return label_words_logits