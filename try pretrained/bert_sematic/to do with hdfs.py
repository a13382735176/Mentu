import ast
import torch
import torch.nn as nn
import torch.optim as optim
from transformers import RobertaModel, RobertaTokenizer
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm
import os
from sklearn.metrics import accuracy_score, classification_report, recall_score, f1_score
from torch.optim.lr_scheduler import ReduceLROnPlateau
import umap
import matplotlib.pyplot as plt
from xlstm import xLSTMBlockStack, xLSTMBlockStackConfig, mLSTMBlockConfig, mLSTMLayerConfig, sLSTMBlockConfig, sLSTMLayerConfig, FeedForwardConfig
import subprocess
import csv
import torch
import random
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.decomposition import TruncatedSVD


def set_seed(seed=45):
    torch.manual_seed(seed)  # 为CPU设置种子
    torch.cuda.manual_seed(seed)  # 为当前GPU设置种子
    torch.cuda.manual_seed_all(seed)  # 为所有GPU设置种子（如果有多个）
    np.random.seed(seed)  # 为NumPy设置种子
    random.seed(seed)  # 为Python内置的random库设置种子
    torch.backends.cudnn.deterministic = True  # 确保每次卷积操作相同
    torch.backends.cudnn.benchmark = False  # 禁用优化，确保每次计算结果相同

# 调用 set_seed 函数
set_seed(44)  # set seed [42,43,44,45,46]

# choose device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")#xlstm must need cuda
np.random.seed(44)#set seed [42,43,44,45,46] the same with set_seed 

data_path = '/home/user10/tao/dataset/preprocessed/HDFS/Modified_hdfs_log_structed_with_translated_event.csv'
if os.path.exists(data_path):
    data = pd.read_csv(data_path)
else:
    raise FileNotFoundError("Data file does not exist")

# "Convert the 'Status' column to a numeric type."
data['Status'] = data['Status'].apply(lambda x: 1 if x == 'Success' else 0)
#data['Status'] = data['Status'].apply(lambda x: 1 if x == 0 else 0)
# RoBERTa embedding
tokenizer = RobertaTokenizer.from_pretrained('/home/user10/tao/pretrained_models/roberta-base')
roberta_model = RobertaModel.from_pretrained('/home/user10/tao/pretrained_models/roberta-base')
roberta_model = roberta_model.to(device)

embeddings_path = '/home/user10/tao/dataset/preprocessed/HDFS/modified_dim400_embeddings.npy'

def generate_embeddings(texts, tokenizer, model, device, batch_size=32):
    embeddings = []
    for i in tqdm(range(0, len(texts), batch_size)):
        batch_texts = texts[i:i+batch_size].tolist()
        inputs = tokenizer(batch_texts, return_tensors='pt', padding=True, truncation=True, max_length=512)
        inputs = {key: val.to(device) for key, val in inputs.items()}
        with torch.no_grad():
            outputs = model(**inputs)
        batch_embeddings = outputs.last_hidden_state.mean(dim=1).cpu().numpy()
        embeddings.append(batch_embeddings)
    return np.concatenate(embeddings, axis=0)

# RoBERTa embedded generation (for similarity calculation)
if os.path.exists(embeddings_path):
    embeddings = np.load(embeddings_path)
else:
    embeddings = generate_embeddings(data['TranslatedEvent'], tokenizer, roberta_model, device)
    #embeddings = generate_embeddings(data['Content'], tokenizer, roberta_model, device)
    np.save(embeddings_path, embeddings)

data['Embeddings'] = list(embeddings)
print("Embedding generation is complete.")

train_data, test_data = data[:int(len(data) * 0.8)], data[int(len(data) * 0.8):]

test_data_Eventlist=test_data['EventList'].values

normal_indices = np.array(train_data[train_data['Status'] == 1].index)
anomalous_indices = np.array(train_data[train_data['Status'] == 0].index)

# Ensure that the number of samples selected does not exceed the number actually available
max_normal_samples = min(2000, len(normal_indices))
max_anomalous_samples = min(2000, len(anomalous_indices))
print(max_anomalous_samples)

selected_normal_indices = np.random.choice(normal_indices, max_normal_samples, replace=False)
# get correct templates
selected_normal_indices_eventlist = train_data.loc[selected_normal_indices, 'EventList']
print(selected_normal_indices_eventlist)
normal_list = []
for c in selected_normal_indices_eventlist:
    event_list = ast.literal_eval(c) if isinstance(c, str) else c  # Convert to list if necessary
    normal_list.extend(event_list)  # Flatten the lists into one

# Create a set of unique events
normal_set = set(normal_list)
print(f"Normal set length : {len(normal_set)}")

# Similarly for anomalous indices
selected_anomalous_indices = np.random.choice(anomalous_indices, max_anomalous_samples, replace=False)
selected_anomalous_indices_eventlist = train_data.loc[selected_anomalous_indices, 'EventList']

abnormal_list = []
for c in selected_anomalous_indices_eventlist:
    event_list = ast.literal_eval(c) if isinstance(c, str) else c  # Convert to list if necessary
    abnormal_events = set(event_list) - normal_set
    # 如果不在正常集中的事件数量为 1，或所有剩余事件都是同一类型，则认为这些是错误的日志模板
    if len(abnormal_events) == 1 or len(abnormal_events) > 0 and len(set(abnormal_events)) == 1:
        abnormal_list.extend(abnormal_events)  

# Create a set of unique events
probility_abnormal_set = set(abnormal_list)
print(f"Abnormal set length: {len(probility_abnormal_set)}")
print(probility_abnormal_set)
# get failure eventlist for hdfs
error_eventlist=[]
for c in selected_anomalous_indices_eventlist:
    if all(event in normal_set for event in ast.literal_eval(c)):     
        error_eventlist.append(c)
print(len(error_eventlist))
print(error_eventlist)
storage_indices = np.concatenate([selected_normal_indices, selected_anomalous_indices])



# Gets the corresponding storage embeddings and labels
#valid_storage_indices = np.array([idx for idx in selected_indices if idx in train_data.index])
valid_storage_indices = np.array([idx for idx in storage_indices if idx in train_data.index])
storage_embeddings = embeddings[valid_storage_indices]
storage_labels = train_data.loc[valid_storage_indices, 'Status'].values

# Find the few log coding blocks that are closest to the new log block
def find_most_similar(new_embedding, storage_embeddings, top_k=5):
    # The stored embeddings and new embeddings are converted to Tensor and moved to the GPU
    storage_embeddings_tensor = torch.tensor(storage_embeddings).to(device)
    new_embedding_tensor = torch.tensor(new_embedding).to(device)
    # Normalized embedding vector
    storage_embeddings_tensor = torch.nn.functional.normalize(storage_embeddings_tensor, p=2, dim=1)
    new_embedding_tensor = torch.nn.functional.normalize(new_embedding_tensor, p=2, dim=0)


    # Calculate the cosine similarity
    similarities = torch.matmul(storage_embeddings_tensor, new_embedding_tensor).cpu().numpy()
    
    # Find the most similar top_k
    top_k_indices = np.argsort(similarities)[-top_k:]
    return top_k_indices,similarities[top_k_indices]#similarities similarities[top_k_indices]


def compute_reconstruction_errors(xlstm_stack, X, batch_size=64):
    reconstruction_errors = []
    xlstm_stack.eval()
    
    with torch.no_grad():
        for i in tqdm(range(0, X.shape[0], batch_size)):
            batch_X = X[i:i + batch_size]
            xlstm_reconstruction = xlstm_stack(batch_X)
            
            # 计算每个样本的重构误差
            reconstruction_error = (xlstm_reconstruction - batch_X).abs()
            
            # 获取每个样本误差中最大的三个值
            # topk 返回的是 (values, indices)，我们只关心 values
            top3_errors = torch.topk(reconstruction_error, k=3, dim=2)[0]  # 取最后一个维度上最大的3个误差值
            
            # 分别乘以 20, 50, 100
            weights = torch.tensor([50.0, 70.0, 100.0]).to(top3_errors.device)  # 权重 [20, 50, 100]
            
            # 按照权重加权三个误差
            weighted_errors = top3_errors * weights
            
            # 将这三个加权后的误差相加，得到每个样本的整体重构误差
            final_error = weighted_errors.sum(dim=2)  # 对最后一维求和
            
            # Flatten the batch dimension to ensure consistency with top_k_probabilities_test
            reconstruction_errors.extend(final_error.cpu().numpy().flatten())
    
    return np.array(reconstruction_errors)

#reconstruction_errors_train = compute_reconstruction_errors(xlstm_stack, X_train)
#reconstruction_errors_test = compute_reconstruction_errors(xlstm_stack, X_test)

def find_top_k_probabilities(storage_embeddings, storage_labels, embeddings, top_k=100):
    probabilities = []
    all_similarities = []  # 用来存储每次的similarity
    for new_embedding in tqdm(embeddings, desc="Calculating probabilities"):
        top_k_indices, similarities = find_most_similar(new_embedding, storage_embeddings, top_k=top_k)
        most_similar_labels = storage_labels[top_k_indices]
        
        # Sort similarities for storage and analysis
        sorted_similarities = np.sort(similarities)  # 对 similarity 排序
        all_similarities.append(sorted_similarities)  # 存储排序后的 similarity
        m=100*(1-most_similar_labels.mean())
        probabilities.append(m)
    
    np.savetxt('prob.txt', probabilities)
    np.savetxt('similarities.txt', np.array(all_similarities))  # 保存所有的 similarities
    return np.array(probabilities)

#top_k_probabilities_train = find_top_k_probabilities(storage_embeddings, storage_labels, np.array(train_data['Embeddings']))
top_k_probabilities_test = find_top_k_probabilities(storage_embeddings, storage_labels, np.array(test_data['Embeddings']))

#combined_features_test = torch.tensor(np.stack([top_k_probabilities_test,reconstruction_errors_test], axis=1), dtype=torch.float32).to(device)
labels_test = test_data['Status'].values  # labels_test is now defined as a NumPy array

import ast

c = []
re_error_list=[]
for pro in top_k_probabilities_test:
    pro_cpu = pro # 将 prob 移动到 CPU 并转换为 Python 标量
    c.append(pro_cpu)




weighted_scores_test = np.array(c)  # to np


#print(re_error_max)
throld = np.percentile(weighted_scores_test, 99.3)  # set 99.3 for hdfs，97.35 for hadoop2.,96.3 for hadoop3.,100 for spark2 and spark3(not need bert)

m = []
l = 0  # 记录符合条件的 event_list 数量

for event_list_str, score in zip(test_data_Eventlist, c):
    score = float(score)
    throld = float(throld)
    try:
        #event_list_str = event_list_str.strip('"')
        event_list = ast.literal_eval(event_list_str)  # 将字符串解析为列表
    except (ValueError, SyntaxError) as e:
        continue  # 如果解析失败，跳过当前 event_list

    # 1. 优先检查 event_list 是否在 error_eventlist 中
    if any(event_list_str == error_event for error_event in error_eventlist):
        m.append(0)  # 异常，跳过其他检查
    # 2. 如果没有在 error_eventlist 中，则检查是否在正常集合中
    elif any(event in probility_abnormal_set for event in event_list):
        m.append(0)  # 异常
    elif score>=throld:
        m.append(0)  # 异常
        print(1)
    elif all(event in normal_set for event in event_list):
        m.append(1)  # 正常
        l += 1
    # 4. 最后比较分数是否超过阈值
    else:
        m.append(1)  # 正常


# 获取测试集标签
labels_test = test_data['Status'].values
print(f"Number of event_lists in normal_set: {l}")

# 打印分类报告
from sklearn.metrics import classification_report
print(classification_report(labels_test, m, digits=3))
