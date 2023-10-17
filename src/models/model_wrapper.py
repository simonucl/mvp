from random import triangular
from ..utils import *
import os, pickle, copy
import torch
from torch import nn
from time import time

class ModelWrapper(torch.nn.Module):
    def __init__(self, args, model, tokenizer, data_collator, verbalizer = None, template=None):  
        '''
        args: args object from argparse
        model: huggingface model
        tokenizer: huggingface tokenizer
        data_collator: huggingface data collator
        verbalizer: verbalizer object
        template: list of templates

        This is a wrapper class for the huggingface model. It is used to add the verbalizer and template to the model.
        '''

        super().__init__()
        self.model = model
        self.tokenizer = tokenizer
        self.mode = args.mode
        self.data_collator = data_collator
        self.args = args
        self.verbalizer = verbalizer
        self.template = template
        self.icl_examples = None
        if model_utils.is_causal_model(args.model):
            self.model.config.pad_token_id = self.tokenizer.pad_token_id
            self.model.config.mask_token_id = self.tokenizer.mask_token_id
        self.config = self.model.config
        self.model.resize_token_embeddings(len(tokenizer))
        self.indexEmbedder = None
        if self.args.model_type in ["retrieval_icl", "retrieval_icl_attack"]:
            self.indexEmbedder = IndexEmbedder("sentence-transformers/all-MiniLM-L6-v2")
    
    def text_to_ids(self, text):
        '''
        text: list of strings

        returns: input_ids, attention_mask
        '''
        inputs_dict = self.tokenizer(text,
            add_special_tokens=True,
            padding=False,
            truncation=True,
        )
        inputs_dict = self.data_collator(inputs_dict)
        input_ids = inputs_dict["input_ids"].cuda()
        attention_mask = inputs_dict["attention_mask"].cuda()
        return input_ids, attention_mask

    
    def get_updated_input_ids(self, input_ids, attention_mask, is_knn=False, **kwargs):
        '''
        input_ids: torch tensor of shape (batch_size, seq_len)
        attention_mask: torch tensor of shape (batch_size, seq_len)
        kwargs: additional arguments

        returns: updated input_ids, attention_mask after adding the verbalizer and template in case of mvp
        '''
        ## if we receive tokenized text, untokenize it:
        is_icl_attack = False
        if type(input_ids) != type(torch.ones(1)):
            text_input_list = []
            icl_examples = []
            for t in input_ids:
                examples = {}
                if type(t) is not tuple:
                    text_input_list.append(t.lower())
                elif len(t) > 2:
                    # text_input_list = [t[-1]]
                    # the len(t) must be odd number
                    assert len(t) % 2 == 1
                    no_examples = (len(t) - 1) // 2
                    for i in range(no_examples):
                        example, label = t[2*i], t[2*i+1]
                        if label not in examples:
                            examples[label] = []
                        examples[label].append({'sentence': example})
                    inference_input = t[-1]
                    text_input_list.append(inference_input)
                    is_icl_attack = True
                    icl_examples.append(examples)
                    # for i in range(no_examples):
                    #     text_input.append((t[2*i], t[2*i+1]))
                    # text_input.append(inference_input)
                    # is_icl_attack = True
                    # text_input_list.append(text_input)
                elif self.args.dataset == "boolq":
                    text_input_list.append((t[1]+"</s></s>"+t[0]+"?").lower()) 
                else:
                    text_input_list.append((t[0]+"</s></s>"+t[1]).lower())

            if self.args.model_type in ["retrieval_icl"]:
                icl_examples = self.indexEmbedder.subsamplebyretrieval(self.anchor_subsample, text_input_list, self.args.examples_per_label)
                self.icl_examples = icl_examples

            if not is_icl_attack:
                input_ids, attention_mask = self.text_to_ids(text_input_list)
            else:
                self.icl_examples = icl_examples
                input_ids, attention_mask = self.text_to_ids(text_input_list)
        

        # print('ICL examples', self.icl_examples)

        input_indices = None

        if is_icl_attack or (self.args.model_type in ['icl', 'knn_icl', 'icl_attack', 'knn_icl_attack', "retrieval_icl", "retrieval_icl_attack"]):
            # input_ids, attention_mask, input_indices = craft_tokenized_prompts(self.tokenizer, self.args.model, text_input_list, self.template, self.len_templates, use_all = (self.args.num_template != -2) or self.mode!="train", icl_examples = self.icl_examples)
            input_ids, attention_mask, input_indices = insert_icl_prompts(self, self.tokenizer, self.args.model_type, input_ids, self.template, self.len_templates, use_all = (self.args.num_template != -2) or self.mode!="train", icl_examples = self.icl_examples, is_knn=is_knn)
        # elif self.args.model_type in ['icl', 'knn_icl', 'icl_attack', 'knn_icl_attack']:
            # input_ids, attention_mask, input_indices = insert_icl_prompts(self, self.tokenizer, self.args.model, input_ids, self.template, self.len_templates, use_all = (self.args.num_template != -2) or self.mode!="train", icl_examples = self.icl_examples)
        elif self.args.model_type in ["mvp", "untrained_mvp", "mvp_knn", "knn_cli"]:
        # elif self.args.model_type == "mvp" or self.args.model_type == "untrained_mvp" or self.args.model_type == "mvp_knn" or self.args.model_type == "knn_cli" or self.args.model_type == "knn_icl" or self.args.model_type == "icl_attack" or 
            input_ids, attention_mask, input_indices = insert_tokenized_prompts(self.tokenizer, self.args.model, input_ids, self.template, self.len_templates, use_all = (self.args.num_template != -2) or self.mode!="train", icl_examples = self.icl_examples)
        return input_ids, attention_mask, input_indices
    

    def forward(self, input_ids, attention_mask=None, **kwargs):
        '''
        input_ids: torch tensor of shape (batch_size, seq_len)
        attention_mask: torch tensor of shape (batch_size, seq_len)
        kwargs: additional arguments

        returns: logits
        '''
        if 'label' in kwargs.keys(): 
            #for gpt2 model
            kwargs['labels'] = kwargs['label']
        import copy
        raw_input = copy.deepcopy(input_ids)

        start_time = time()
        # Step that consist adding prompt and verbalizer
        input_ids, attention_mask, _  = self.get_updated_input_ids(input_ids, attention_mask, **kwargs)

        end_time = time()

        # print('Get updated input ids time', end_time - start_time)

        # print('ICL input example', self.tokenizer.decode(input_ids[0]))

        input_ids, attention_mask = input_ids.to(self.model.device), attention_mask.to(self.model.device)
        if self.args.adv_augment and self.mode=="train":
            # This part is to attack it (generate adversarial examples)
            embedding_outs = pgd_attack(self, input_ids, attention_mask, kwargs['labels'], self.args, norm = self.args.norm)
            adv_inputs = {}
            adv_inputs['inputs_embeds'] = embedding_outs
            adv_inputs['attention_mask'] = attention_mask
            adv_inputs['output_attentions'] = True
            outputs = self.model(**adv_inputs)
        else:
            start_time = time()
            outputs = self.model(input_ids=input_ids, attention_mask = attention_mask, output_hidden_states = True, output_attentions = True)
            end_time = time()
            # print('Model forward time', end_time - start_time)

        start_time = time()
        if (self.args.model_type == "mvp_knn") and (self.args.mode == "attack"):
            logits = self.outs_to_logits(input_ids, outputs, raw_input)
        else:
            logits = self.outs_to_logits(input_ids, outputs)
        end_time = time()

        # print('Outs to logits time', end_time - start_time)
        
        if 'labels' in kwargs.keys() and self.mode in ["train", "eval"]:
            loss = F.cross_entropy(logits, kwargs['labels'])
            return loss, logits
        else:
            assert (self.mode == "attack")
            return logits.detach()