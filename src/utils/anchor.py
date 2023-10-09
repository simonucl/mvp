import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
import random
import copy
from collections import defaultdict

class AnchorStore(nn.Module):

    def __init__(self, K=1024, dim=50257, knn=1, n_class=2, knn_T=0.05):
        super(AnchorStore, self).__init__()

        self.register_buffer("queue_anchor", torch.randn(K, dim))
        self.register_buffer("queue_label", torch.zeros(K, dtype=torch.long))
        self.queue_label.fill_(-1)
        self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))
        # self.datastore_keys = np.zeros([K, dim], dtype=np.float32)
        # self.datastore_vals = np.zeros([K], dtype=np.int64)
        # self.datastore_ptr = 0

        self.knn = knn
        self.knn_T = knn_T
        self.n_class = n_class

    def enqueue(self, anchors, labels):

        ptr = int(self.queue_ptr)
        bs = anchors.shape[0]

        self.queue_anchor[ptr:ptr + bs, :] = anchors
        self.queue_label[ptr:ptr + bs] = labels
        self.queue_ptr[0] = ptr + bs

        # ptr = self.datastore_ptr
        # bs = anchors.shape[0]

        # self.datastore_keys[ptr:ptr + bs, :] = anchors.cpu().numpy()
        # self.datastore_vals[ptr:ptr + bs] = labels.cpu().numpy()
        # self.datastore_ptr = ptr + bs

    def knn_calibrate(self, logits):
        '''
        logits: [B, dim]
        '''
        # print('The shape of queue anchor: ', self.queue_anchor.shape)
        # print('The shape of logits: ', logits.shape)

        # print('Queue anchor: ', self.queue_anchor)
        # print('Logits: ', logits)
        self.queue_anchor = self.queue_anchor.to(logits.device)
        # print('Queue anchor: ', self.queue_anchor)
        # print('Logits: ', logits.shape)

        dists = torch.mean(self.queue_anchor[:, None, :] * (self.queue_anchor[:, None, :].log() - logits.log()), dim=2).transpose(1, 0)
        l2_dists = ((self.queue_anchor.unsqueeze(0) - logits.unsqueeze(1)) ** 2).sum(dim=-1)

        # print('KL dists: ', dists)
        # print('L2 dists: ', l2_dists)
        scaled_dists = -1.0 / self.knn_T * dists
        
        # print sorted scaled dists
        # print('Sorted scaled dists: ', torch.sort(scaled_dists, dim=-1))
              
        # print('Scaled dists: ', scaled_dists)
        top_dists, top_indices = torch.topk(scaled_dists, self.knn) # [B, K+1], [B, K+1]
        values = torch.tensor(self.queue_label, dtype=torch.int64, device=logits.device)
        new_vals = values.unsqueeze(0).repeat(logits.shape[0], 1) # [B, L]
        top_values = torch.gather(new_vals, 1, top_indices).unsqueeze(-1)  # [B, K, 1]
        knn_weight = torch.softmax(top_dists, dim=-1).unsqueeze(-1)  # [B, K, 1]

        # print('KNN weight: ', knn_weight)
        # Check the errors here

        # init knn-prob
        knn_prob = torch.zeros((logits.shape[0], self.knn, self.n_class), device=logits.device)
        knn_prob.scatter_(2, top_values, knn_weight)
        knn_prob = knn_prob.sum(dim=-2) # [B, n_class]

        # print('KNN prob', knn_prob.shape)
        return knn_prob
    
    def knn_infer(self, query):

        # query: [1, 1, dim]
        # kl_div.shape = [1, len(self.queue_anchor)]
        kl_distance = torch.mean(self.queue_anchor[:, None, :] * (self.queue_anchor[:, None, :].log() - query.log()), dim=2).transpose(1, 0)
        
        l2_distance = torch.mean((self.queue_anchor[:, None, :] - query) ** 2, dim=2).transpose(1, 0)

        if self.knn == 1:
            # directly return the nearest neighbor
            return self.queue_label[kl_distance.argmin(dim=1)].tolist()
        else:
            values, indices = torch.topk(kl_distance, self.knn, dim=1, largest=False)
            # count for each category within k nearest neighbors, and return the dominant category
            # knn_cnt.shape = [1, self.n_class]
            knn_cnt = torch.zeros((query.shape[0], self.n_class))
            for i in range(self.n_class):
                knn_cnt[:, i] = (self.queue_label[indices] == i).sum(dim=1)
            return knn_cnt.argmax(dim=1).tolist()



def subsamplebyshot(anchor_data, seed, label_set, verbalizer, shot=1, examples_per_class=1):
    '''
    anchor_data: list of anchor data
    seed: seed for random
    shot: number of examples per class

    returns: subsampled anchor data
    '''
    random.seed(seed)
    anchor_data = copy.deepcopy(anchor_data)
    new_anchor_data = []
    icl_example = {}
    for label in label_set:
        label_data = [d for d in anchor_data if d['label'] == label]
        random.shuffle(label_data)
        new_anchor_data.extend(label_data[examples_per_class: shot-examples_per_class])
        # how to get the item from tensor? 
        # check if label is tensor
        if torch.is_tensor(label):
            label = label.item()
        #     icl_example[verbalizer[label][0]] = label_data[shot:shot+examples_per_class]
        # else:
        icl_example[verbalizer[label][0]] = label_data[:examples_per_class]
    return new_anchor_data, icl_example