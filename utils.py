import numpy as np
import torch


def attention_inference(dataloader, backbone, attention, pool, hash_layer, hash_length, device):
    hashcodes = list()
    labels = list()
    backbone.eval()
    hash_layer.eval()
    attention.eval()
    threshold = np.array([0.0] * hash_length)  # 0.5
    with torch.no_grad():
        for imgs, labels_ in dataloader:
            labels.append(labels_.view(labels_.size()[0], ).numpy())
            # print('imgs:', imgs.size())
            features = backbone(imgs.to(device))
            features_atten = features * attention(features)
            features_pool = pool(features_atten)
            # features_pool = pool(features)
            h, _ = hash_layer(features_pool)
            hashcodes.append(h.cpu().numpy())
    # print hashcodes-threshold
    return (np.sign(np.concatenate(hashcodes) - threshold)).astype(np.int8), np.concatenate(labels)

def inference(dataloader, backbone, pool, hash_layer, hash_length, device):
    hashcodes = list()
    labels = list()
    backbone.eval()
    hash_layer.eval()
    threshold = np.array([0.0] * hash_length)  # 0.5
    with torch.no_grad():
        for imgs, labels_ in dataloader:
            labels.append(labels_.view(labels_.size()[0], ).numpy())
            # print('imgs:', imgs.size())
            features = backbone(imgs.to(device))
            features_pool = pool(features)
            h, _ = hash_layer(features_pool)
            hashcodes.append(h.cpu().numpy())
    # print hashcodes-threshold
    return (np.sign(np.concatenate(hashcodes) - threshold)).astype(np.int8), np.concatenate(labels)



def compute_MAP(db_binary, db_label, test_binary, test_label):
    AP = []
    Ns = np.array(range(1, db_binary.shape[0] + 1)).astype(np.float32)
    for i in range(test_binary.shape[0]):
        query_binary = test_binary[i]
        query_label = test_label[i]
        query_result = np.argsort(np.sum((query_binary != db_binary), axis=1))
        correct = (query_label == db_label[query_result])
        P = np.cumsum(correct, axis=0) / Ns
        AP.append(np.sum(P * correct) / np.sum(correct))
    MAP = np.mean(np.array(AP))
    # return round(MAP,5)
    return MAP


def compute_topk_mAP(db_binary, db_label, test_binary, test_label, k):
    AP = []
    Ns = np.array(range(1, k + 1)).astype(np.float32)
    for i in range(test_binary.shape[0]):
        query_binary = test_binary[i]
        query_label = test_label[i]
        query_result = np.argsort(np.sum((query_binary != db_binary), axis=1))
        correct = (query_label == db_label[query_result[0:k]])
        P = np.cumsum(correct, axis=0) / Ns
        if np.sum(correct) == 0:
            AP.append(0.)
        else:
            AP.append(np.sum(P * correct) / np.sum(correct))
    topk_MAP = np.mean(np.array(AP))
    # return round(MAP,5)
    return topk_MAP


def compute_MAP_mutli(db_binary, db_label, test_binary, test_label):
    AP = []
    Ns = np.array(range(1, db_binary.shape[0] + 1)).astype(np.float32)
    for i in range(test_binary.shape[0]):
        query_binary = test_binary[i]
        query_label = test_label[i]
        query_result = np.argsort(np.sum((query_binary != db_binary), axis=1))
        # correct=(query_label==db_label[query_result])
        correct = (np.dot(db_label[query_result, query_label]) > 0)
        P = np.cumsum(correct, axis=0) / Ns
        AP.append(np.sum(P * correct) / np.sum(correct))
    MAP = np.mean(np.array(AP))
    # return round(MAP,5)
    return MAP
