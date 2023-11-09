from sentence_transformers import SentenceTransformer, util
from typing import Dict, List, Tuple, Union
import torch
# from pyserini.search import SimpleSearcher
from rank_bm25 import BM25Okapi
from transformers import AutoTokenizer
from InstructorEmbedding import INSTRUCTOR

instructor_suffix = (' for retrieval: ', ' for retrieving support documents: ')

instructor_prefix = {
    'sst2': 'Represent the sentence',
    'rte': 'Represent the document',
}


class IndexEmbedder(torch.nn.Module):
    def __init__(self, model_name, task_name) -> None:
        super().__init__()
        self.embedder = SentenceTransformer(model_name)
        self.bert_tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
        self.instructor = INSTRUCTOR('hkunlp/instructor-large')
        self.task_name = task_name

    def encode(self, queries: List[str]) -> torch.Tensor:
        return self.embedder.encode(queries, convert_to_tensor=True)

    def bm25subsmple(self, anchor_data : List[str], original_anchor, query : str, top_k=1, num_labels=2) -> List[str]:
        '''
        queries: list of query, Shape: [B, seq_len]
        corpus: list of corpus, Shape: [B, seq_len]
        returns: top-k retrieved corpus
        '''

        retrieved_examples = []

        tokenized_corpus = [self.bert_tokenizer.tokenize(doc) for doc in anchor_data]
        bm25 = BM25Okapi(tokenized_corpus)
        tokenized_query = self.bert_tokenizer.tokenize(query)
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

    def instructor_subsample(self, 
                             anchor_data : List[str],
                                original_anchor,
                                query : str,
                                top_k=1,
                                num_labels=2,
                             ) -> List[str]:
        '''
        queries: list of query, Shape: [B, seq_len]
        corpus: list of corpus, Shape: [B, seq_len]
        returns: top-k retrieved corpus
        '''
        retrieved_examples = []

        anchor_data = [[instructor_prefix[self.task_name] + instructor_suffix[0], doc] for doc in anchor_data]
        query = [instructor_prefix[self.task_name] + instructor_suffix[1], query]

        anchor_data_embeddings = self.instructor.encode(anchor_data)
        query_embedding = self.instructor.encode(query)
        cos_scores = util.pytorch_cos_sim(query_embedding, anchor_data_embeddings)[0]
        top_results = torch.topk(cos_scores, k=top_k*num_labels)
        for score, idx in zip(top_results[0], top_results[1]):
            retrieved_examples.append(original_anchor[idx.item()])
        return retrieved_examples
    
    def subsamplebyretrieval(self, 
                             anchor_data : Union[List[Dict], List[List[Dict]]], 
                             text_input_list, 
                             top_k=1, 
                            num_labels=2,
                            retrieve_method='sbert') -> List[List[str]]:
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
            elif retrieve_method == 'instructor':
                retrieved_examples[i] = self.instructor_subsample(anchor_data_idx, original_anchor_data, text, top_k, num_labels)
                
        return retrieved_examples