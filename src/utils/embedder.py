from sentence_transformers import SentenceTransformer, util
from typing import Dict, List, Tuple, Union
import torch
# from pyserini.search import SimpleSearcher
from rank_bm25 import BM25Okapi
from transformers import AutoTokenizer
from InstructorEmbedding import INSTRUCTOR
from tqdm import tqdm
import multiprocessing
import os
import pickle
from torch.multiprocessing import Pool, Process, set_start_method

instructor_suffix = (' for retrieval: ', ' for retrieving support documents: ')

instructor_prefix = {
    'sst2': 'Represent the sentence',
    'rte': 'Represent the document',
    'mnli': 'Represent the document',
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

    def bm25subsample(self, anchor_data : List[str], original_anchor, query : str, top_k=1, num_labels=2) -> List[str]:
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

    def process_text(self, i, text, anchor_data_idx, original_anchor_data, retrieve_method, top_k, num_labels):
        if retrieve_method == 'sbert':
            result = self.sbert_subsample(anchor_data_idx, original_anchor_data, text, top_k, num_labels)
        elif retrieve_method == 'bm25':
            result = self.bm25subsample(anchor_data_idx, original_anchor_data, text, top_k, num_labels)
        elif retrieve_method == 'instructor':
            result = self.instructor_subsample(anchor_data_idx, original_anchor_data, text, top_k, num_labels)
        return i, result
    
    def subsamplebyretrieval(self, 
                             anchor_data : Union[List[Dict], List[List[Dict]]], 
                             text_input_list, 
                             top_k=1, 
                            num_labels=2,
                            retrieve_method='sbert',
                            save_path=None) -> List[List[str]]:
        '''
        anchor_data: list of anchor data, [{'sentence': 'text', 'label': 0}, ...]
        text_input_list: list of input text, Shape: [B, seq_len]
        returns: top-k retrieved anchor data
        '''
        try:
            set_start_method('spawn', force=True)
        except RuntimeError:
            pass

        # print(len(anchor_data))
        retrieved_examples = None
        if save_path is None or not os.path.exists(save_path):
        
            num_processes = min(1, multiprocessing.cpu_count())
            pool = Pool(processes=num_processes)

            print(f'Using {num_processes} processes for retrieval')

            retrieved_examples = [[] for _ in range(len(text_input_list))]
            results = []
            if type(anchor_data[0]) is list:
                assert (len(anchor_data) == len(text_input_list)), f'Length of anchor data {len(anchor_data)} and text input list {len(text_input_list)} must be the same'
            
            for i, text in enumerate(tqdm(text_input_list, desc='Retrieving anchor data')):
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

                retrieved_examples[i] = self.process_text(i, text, anchor_data_idx, original_anchor_data, retrieve_method, top_k, num_labels)[1]
                # result = pool.apply_async(self.process_text, args=(i, text, anchor_data_idx, original_anchor_data, retrieve_method, 64, num_labels))
                # results.append(result)

            # pool.close()
            # pool.join()

            # results = [r.get() for r in results]
            # results.sort(key=lambda x: x[0])

            # retrieved_examples = [r[1] for r in results]

            with open(save_path, 'wb') as f:
                print(f'Saving retrieved examples to {save_path}')
                pickle.dump(retrieved_examples, f)

        if retrieved_examples is None:
            with open(save_path, 'rb') as f:
                print(f'Loading retrieved examples from {save_path}')
                retrieved_examples = pickle.load(f)
        assert len(retrieved_examples) == len(text_input_list), f'Length of retrieved examples {len(retrieved_examples)} and text input list {len(text_input_list)} must be the same'
        
        return [retrieved_example[:top_k*num_labels] for retrieved_example in retrieved_examples]