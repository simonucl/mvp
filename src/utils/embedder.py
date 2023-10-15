from sentence_transformers import SentenceTransformer, util
from typing import Dict, List, Tuple
import torch

class IndexEmbedder(torch.nn.Module):
    def __init__(self, model_name) -> None:
        super().__init__()
        self.embedder = SentenceTransformer(model_name)

    def encode(self, queries: List[str]) -> torch.Tensor:
        return self.embedder.encode(queries, convert_to_tensor=True)

    def subsamplebyretrieval(self, anchor_data, text_input_list, top_k=1):
        '''
        anchor_data: list of anchor data, [{'sentence': 'text', 'label': 0}, ...]
        text_input_list: list of input text, Shape: [B, seq_len]
        returns: top-k retrieved anchor data
        '''
        print(len(anchor_data))
        unique_labels = set(list(map(lambda x: x['label'], anchor_data)))
        label2data = {label: [] for label in unique_labels}
        for data in anchor_data:
            label2data[data['label']].append(data)
        # train_data = [x['sentence'] for x in anchor_data]
        retrieved_examples = [[] for _ in range(len(text_input_list))]
        for label, data in label2data.items():
            train_data_embeddings = self.embedder.encode(data, convert_to_tensor=True)
            for i, text in enumerate(text_input_list):
                query_embedding = self.embedder.encode(text, convert_to_tensor=True) 
                cos_scores = util.pytorch_cos_sim(query_embedding, train_data_embeddings)[0]
                top_results = torch.topk(cos_scores, k=top_k)
                for score, idx in zip(top_results[0], top_results[1]):
                    retrieved_examples[i].append(data[idx.item()])
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