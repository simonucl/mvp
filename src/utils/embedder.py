from sentence_transformers import SentenceTransformer, util
from typing import Dict, List, Tuple, Union
import torch
# from pyserini.search import SimpleSearcher
from rank_bm25 import BM25Okapi


class IndexEmbedder(torch.nn.Module):
    def __init__(self, model_name) -> None:
        super().__init__()
        self.embedder = SentenceTransformer(model_name)

    def encode(self, queries: List[str]) -> torch.Tensor:
        return self.embedder.encode(queries, convert_to_tensor=True)

    def bm25subsmple(self, anchor_data : List[str], original_anchor, query : str, top_k=1, num_labels=2) -> List[str]:
        '''
        queries: list of query, Shape: [B, seq_len]
        corpus: list of corpus, Shape: [B, seq_len]
        returns: top-k retrieved corpus
        '''

        retrieved_examples = []

        tokenized_corpus = [doc.split(" ") for doc in anchor_data]
        bm25 = BM25Okapi(tokenized_corpus)
        tokenized_query = query.split(" ")
        doc_scores = bm25.get_scores(tokenized_query)
        top_results = sorted(range(len(doc_scores)), key=lambda i: doc_scores[i])[-top_k*num_labels:]
        for idx in top_results:
            retrieved_examples.append(original_anchor[idx])
        return retrieved_examples
    
    def sbert_subsample(self, anchor_data : List[str], original_anchor, query : str, top_k=1, num_labels=2) -> List[str]:
        '''
        queries: list of query, Shape: [B, seq_len]
        corpus: list of corpus, Shape: [B, seq_len]
        returns: top-k retrieved corpus
        '''
        retrieved_examples = []

        anchor_data_embeddings = self.embedder.encode(anchor_data, convert_to_tensor=True)
        query_embedding = self.embedder.encode(query, convert_to_tensor=True)
        cos_scores = util.pytorch_cos_sim(query_embedding, anchor_data_embeddings)[0]
        top_results = torch.topk(cos_scores, k=top_k*num_labels)
        for score, idx in zip(top_results[0], top_results[1]):
            retrieved_examples.append(original_anchor[idx.item()])
        return retrieved_examples

    def subsamplebyretrieval(self, anchor_data : Union[List[Dict], List[List[Dict]]], text_input_list, top_k=1, num_labels=2, retrieve_method='sbert') -> List[List[Dict]]:
        '''
        anchor_data: list of anchor data, [{'sentence': 'text', 'label': 0}, ...]
        text_input_list: list of input text, Shape: [B, seq_len]
        returns: top-k retrieved anchor data
        '''
        # print(len(anchor_data))
        retrieved_examples = [[] for _ in range(len(text_input_list))]

        if type(anchor_data[0]) is list:
            assert (len(anchor_data) == len(text_input_list)), f'Length of anchor data {len(anchor_data)} and text input list {len(text_input_list)} must be the same'
        
        for i, text in enumerate(text_input_list):
            if type(text) is tuple:
                text = text[0]
            if type(anchor_data[0]) is list:
                anchor_data_idx = anchor_data[i]
                original_anchor_data = anchor_data[i]
            else:
                anchor_data_idx = anchor_data
                original_anchor_data = anchor_data

            if type(anchor_data_idx[0]) is dict:
                anchor_data_idx = list(map(lambda x: x['sentence'] if 'sentence' in x else x['premise'], anchor_data_idx))
            elif type(anchor_data_idx[0]) is tuple:
                anchor_data_idx = list(map(lambda x: x[0], anchor_data_idx))

            # anchor_data_idx = list(map(lambda x: x['sentence'] if 'sentence' in x else x['premise'], anchor_data_idx))
            if retrieve_method == 'sbert':
                retrieved_examples[i] = self.sbert_subsample(anchor_data_idx, original_anchor_data, text, top_k, num_labels)
            elif retrieve_method == 'bm25':
                retrieved_examples[i] = self.bm25subsmple(anchor_data_idx, original_anchor_data, text, top_k, num_labels)
        # else:
        #     unique_labels = set(list(map(lambda x: x['label'], anchor_data)))
        #     label2data = {label: [] for label in unique_labels}
        #     for data in anchor_data:
        #         if type(data) is dict:
        #             data = data['sentence']
        #         elif type(data) is tuple:
        #             data = data[0]
        #         label2data[data['label']].append(data)
        #     # train_data = [x['sentence'] for x in anchor_data]
        #     retrieved_examples = [[] for _ in range(len(text_input_list))]
        #     for label, data in label2data.items():
        #         train_data_embeddings = self.embedder.encode(data, convert_to_tensor=True)
        #         for i, text in enumerate(text_input_list):
        #             if type(text) is tuple:
        #                 text = text[0]
        #             query_embedding = self.embedder.encode(text, convert_to_tensor=True) 
        #             cos_scores = util.pytorch_cos_sim(query_embedding, train_data_embeddings)[0]
        #             top_results = torch.topk(cos_scores, k=top_k)
        #             for score, idx in zip(top_results[0], top_results[1]):
        #                 retrieved_examples[i].append(data[idx.item()])
        return retrieved_examples


        # train_data_embeddings = self.embedder.encode(train_data, convert_to_tensor=True)
        # icl_examples = []
        # for text in text_input_list:
        #     query_embedding = self.embedder.encode(text, convert_to_tensor=True) 
        #     cos_scores = util.pytorch_cos_sim(query_embedding, train_data_embeddings)[0] # 
        #     top_results = torch.topk(cos_scores, k=top_k)
        #     retrieved_examples = []
        #     for score, idx in zip(top_results[0], top_results[1]):
        #         retrieved_examples.append(anchor_data[idx.item()])
        #     icl_examples.append(retrieved_examples)
        # return icl_examples