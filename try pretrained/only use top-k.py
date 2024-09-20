import torch
import torch.nn as nn
import torch.optim as optim
from transformers import AutoTokenizer, AutoModel
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from tqdm import tqdm
import os
from sklearn.metrics import accuracy_score, recall_score, f1_score
from torch.optim.lr_scheduler import ReduceLROnPlateau
from xlstm import xLSTMBlockStack, xLSTMBlockStackConfig, mLSTMBlockConfig, mLSTMLayerConfig, sLSTMBlockConfig, sLSTMLayerConfig, FeedForwardConfig
import subprocess
from torch.cuda.amp import autocast
from sklearn.cluster import KMeans
from transformers import RobertaModel, RobertaTokenizer


# choose device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# dataload
data_path = '/home/user10/tao/dataset/preprocessed/Thunderbird/200l_Thunderbird.csv'

if os.path.exists(data_path):
    data = pd.read_csv(data_path)
else:
    raise FileNotFoundError("Data file does not exist")

# Convert the 'Status' column to a numeric type
data['Status'] = data['Status'].apply(lambda x: 1 if x == 'success' else 0)

# Conan-embedding 模型替换 RoBERTa
tokenizer = RobertaTokenizer.from_pretrained('/home/user10/tao/pretrained_models/roberta-base')
roberta_model = RobertaModel.from_pretrained('/home/user10/tao/pretrained_models/roberta-base')
roberta_model = roberta_model.to(device)

embeddings_path = '/home/user10/tao/dataset/preprocessed/Thunderbird/200l_dim400_embeddings with no translated.npy'

# 嵌入生成函数
def generate_embeddings(texts, tokenizer, model, device, batch_size=200):
    embeddings = []
    for i in tqdm(range(0, len(texts), batch_size)):
        batch_texts = texts[i:i+batch_size].tolist()
        inputs = tokenizer(batch_texts, return_tensors='pt', padding=True, truncation=True, max_length=400)
        inputs = {key: val.to(device) for key, val in inputs.items()}
        with torch.no_grad():
            with autocast():
                outputs = model(**inputs)
        batch_embeddings = outputs.last_hidden_state.mean(dim=1).cpu().numpy()
        embeddings.append(batch_embeddings)
    return np.concatenate(embeddings, axis=0)

# Conan 嵌入生成
if os.path.exists(embeddings_path):
    embeddings = np.load(embeddings_path)
else:
    embeddings = generate_embeddings(data['Content'], tokenizer, roberta_model, device)
    np.save(embeddings_path, embeddings)

data['Embeddings'] = list(embeddings)
print("Embedding generation with Conan is complete.")

'''# TF-IDF encoding of the event sequence
vectorizer = TfidfVectorizer(max_features=400)
event_sequences = data['EventList'].apply(eval).apply(lambda x: ' '.join(map(str, x)))
tfidf_matrix = vectorizer.fit_transform(event_sequences).toarray()
'''
# Split dataset into training and test sets
train_data, test_data = data[:int(len(data) * 0.8)], data[int(len(data) * 0.8):]
'''train_tfidf, test_tfidf = tfidf_matrix[:int(len(tfidf_matrix) * 0.8)], tfidf_matrix[int(len(tfidf_matrix) * 0.8):]

# Ensure that the length of the input data is 400
max_length = 400

def pad_or_truncate(sequence, max_length):
    if len(sequence) < max_length:
        sequence.extend([0] * (max_length - len(sequence)))
    return sequence[:max_length]

train_tfidf = np.array([pad_or_truncate(list(seq), max_length) for seq in train_tfidf])
test_tfidf = np.array([pad_or_truncate(list(seq), max_length) for seq in test_tfidf])

# 将训练集和测试集的数据转换为 Tensor
X_train = torch.tensor(train_tfidf, dtype=torch.float32).to(device)
X_test = torch.tensor(test_tfidf, dtype=torch.float32).to(device)

# Change the dimension of the input data from (B, S) to (B, S, 1)
X_train = X_train.unsqueeze(-1).transpose(1, 2)
X_test = X_test.unsqueeze(-1).transpose(1, 2)

print("Ready to start training with xlstm .")

# xLSTM model configuration
cfg = xLSTMBlockStackConfig(
    mlstm_block=mLSTMBlockConfig(
        mlstm=mLSTMLayerConfig(
            conv1d_kernel_size=4, qkv_proj_blocksize=4, num_heads=4
        )
    ),
    slstm_block=sLSTMBlockConfig(
        slstm=sLSTMLayerConfig(
            backend="cuda",
            num_heads=4,
            conv1d_kernel_size=4,
            bias_init="powerlaw_blockdependent",
        ),
        feedforward=FeedForwardConfig(proj_factor=1.3, act_fn="gelu"),
    ),
    context_length=256,
    num_blocks=7,
    embedding_dim=400,
    slstm_at=[1],
)

# the train of xlstm
xlstm_stack = xLSTMBlockStack(cfg).to(device)
optimizer = optim.Adam(xlstm_stack.parameters(), lr=0.001)
scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=5)
num_epochs = 5
batch_size = 64
early_stopping_patience = 10
best_loss = 0.0001
patience_counter = 0

for epoch in range(num_epochs):
    xlstm_stack.train()
    epoch_loss = 0
    for i in range(0, X_train.shape[0], batch_size):
        batch_X = X_train[i:i + batch_size]
        optimizer.zero_grad()
        outputs = xlstm_stack(batch_X)
        loss = ((outputs - batch_X) ** 2).mean()  # Reconstruction error
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()

    epoch_loss /= (X_train.shape[0] // batch_size)
    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss:.4f}')

    scheduler.step(epoch_loss)

    # Early stopping logic
    if epoch_loss < best_loss:
        best_loss = epoch_loss
        patience_counter = 0
    else:
        patience_counter += 1

    if patience_counter >= early_stopping_patience:
        print("Early stopping triggered.")
        break
'''
# 使用KMeans生成伪标签
def generate_pseudo_labels_kmeans(embeddings, n_clusters=2, sample_size=10000):
    np.random.seed(21)
    if len(embeddings) > sample_size:
        sampled_indices = np.random.choice(len(embeddings), size=sample_size, replace=False)
        embeddings = embeddings[sampled_indices]
    else:
        sampled_indices = np.arange(len(embeddings))

    # KMeans 聚类
    kmeans = KMeans(n_clusters=n_clusters, random_state=42).fit(embeddings)
    
    # 计算每个样本到其簇中心的距离
    distances = np.linalg.norm(embeddings - kmeans.cluster_centers_[kmeans.labels_], axis=1)
    
    # 定义一个距离阈值，删除难以归类的样本（比如距离超过90分位数）
    threshold = np.percentile(distances, 70)#70use to bgl 100l
    valid_indices = sampled_indices[distances < threshold]
    
    return kmeans.labels_[distances < threshold], valid_indices

# 从训练集中生成伪标签，选择10000条训练样本
# 使用KMeans生成伪标签
pseudo_labels_train, valid_train_indices = generate_pseudo_labels_kmeans(np.array(train_data['Embeddings'].tolist()), n_clusters=2)

# 替换真实标签为伪标签，只对 valid_train_indices 生效
train_data.loc[valid_train_indices, 'Status'] = pseudo_labels_train


# 将有效的嵌入和标签保存下来
valid_train_embeddings = np.array(train_data['Embeddings'].tolist())[valid_train_indices]
train_data_filtered = train_data.iloc[valid_train_indices]

# 假设标签较多的类为正常样本，较少的为异常样本
label_counts = np.bincount(pseudo_labels_train)
normal_label = np.argmax(label_counts)
anomalous_label = np.argmin(label_counts)

# 分别获取正常样本和异常样本的索引
normal_indices = np.where(pseudo_labels_train == normal_label)[0]
anomalous_indices = np.where(pseudo_labels_train == anomalous_label)[0]

#use to top-k
normal_indices_useto_top_k = normal_indices[:int(0.5 * len(normal_indices))]
anomalous_indices_useto_top_k = anomalous_indices[:int(0.5 * len(anomalous_indices))]
storage_indices = np.concatenate([normal_indices_useto_top_k, anomalous_indices_useto_top_k])

#use to find threshold with fake label
normal_indices_useto_find_threshold = normal_indices[int(0.5 * len(normal_indices)):]
anomalous_indices_useto_find_threshold = anomalous_indices[int(0.5 * len(anomalous_indices)):]
threshold_indices = np.concatenate([normal_indices_useto_find_threshold, anomalous_indices_useto_find_threshold])

# 存储正常样本和异常样本的嵌入
storage_embeddings = embeddings[storage_indices]
storage_labels = pseudo_labels_train[storage_indices]  # 使用伪标签代替真实标签

def find_most_similar(new_embedding, storage_embeddings, top_k=20):
    storage_embeddings_tensor = torch.tensor(storage_embeddings).to(device)
    new_embedding_tensor = torch.tensor(new_embedding).to(device)

    # Normalized embedding vector
    storage_embeddings_tensor = torch.nn.functional.normalize(storage_embeddings_tensor, p=2, dim=1)
    new_embedding_tensor = torch.nn.functional.normalize(new_embedding_tensor, p=2, dim=0)

    # Calculate the cosine similarity
    similarities = torch.matmul(storage_embeddings_tensor, new_embedding_tensor).cpu().numpy()

    # Find the most similar top_k
    top_k_indices = np.argsort(similarities)[-top_k:]
    return top_k_indices, similarities[top_k_indices]

# 基于伪标签计算正样本（正常样本）概率
def find_top_k_probabilities(storage_embeddings, storage_labels, embeddings, top_k=20):
    probabilities = []
    for new_embedding in tqdm(embeddings, desc="Calculating probabilities"):
        top_k_indices, _ = find_most_similar(new_embedding, storage_embeddings, top_k=top_k)
        most_similar_labels = storage_labels[top_k_indices]
        probabilities.append(most_similar_labels.mean())
    return np.array(probabilities)

# 计算 top-k 概率
top_k_probabilities_train = find_top_k_probabilities(storage_embeddings, storage_labels, np.array(train_data['Embeddings']))
top_k_probabilities_test = find_top_k_probabilities(storage_embeddings, storage_labels, np.array(test_data['Embeddings']))

# Calculate reconstruction errors on the training and test sets
def compute_reconstruction_errors(xlstm_stack, X, batch_size=128):
    reconstruction_errors = []
    xlstm_stack.eval()
    with torch.no_grad():
        for i in tqdm(range(0, X.shape[0], batch_size)):
            batch_X = X[i:i + batch_size]
            xlstm_reconstruction = xlstm_stack(batch_X)
            reconstruction_error = ((xlstm_reconstruction - batch_X) ** 2).mean(dim=[1, 2]).cpu().numpy()
            reconstruction_errors.extend(reconstruction_error)
    return np.array(reconstruction_errors)

'''reconstruction_errors_train = compute_reconstruction_errors(xlstm_stack, X_train)
reconstruction_errors_test = compute_reconstruction_errors(xlstm_stack, X_test)

# Normalized reconstruction error
min_reconstruction_error_train = reconstruction_errors_train.min()
max_reconstruction_error_train = reconstruction_errors_train.max()
normalized_reconstruction_errors_train = (reconstruction_errors_train - min_reconstruction_error_train) / (max_reconstruction_error_train - min_reconstruction_error_train)

min_reconstruction_error_test = reconstruction_errors_test.min()
max_reconstruction_error_test = reconstruction_errors_test.max()
normalized_reconstruction_errors_test = (reconstruction_errors_test - min_reconstruction_error_test) / (max_reconstruction_error_test - min_reconstruction_error_test)
'''
# 提取后50%索引对应的 top-k 概率和重构误差信息
top_k_probabilities_threshold = top_k_probabilities_train[threshold_indices]
#normalized_reconstruction_errors_threshold = normalized_reconstruction_errors_train[threshold_indices]

# 将这些数据组合成训练注意力机制的输入
#combined_features_train = torch.tensor(np.stack([top_k_probabilities_threshold, normalized_reconstruction_errors_threshold], axis=1), dtype=torch.float32).to(device)

# 使用伪标签进行训练
labels_train = torch.tensor(pseudo_labels_train[threshold_indices], dtype=torch.float32).to(device)


# Prepare the data for the test set
#combined_features_test = torch.tensor(np.stack([top_k_probabilities_test, normalized_reconstruction_errors_test], axis=1), dtype=torch.float32).to(device)
labels_test = test_data['Status'].values  # 使用真标签
best_threshold, best_f1 = 0, 0

for threshold in np.linspace(min(top_k_probabilities_threshold), max(top_k_probabilities_threshold), 100):
    predictions = (top_k_probabilities_threshold > threshold).astype(int)
    # 将 labels_train 转移到 CPU，然后转换为 NumPy 数组
    f1 = f1_score(labels_train.cpu().numpy(), predictions)
    if f1 > best_f1:
        best_f1 = f1
        best_threshold = threshold

# Apply the optimal threshold to the test set
final_predictions_test = (top_k_probabilities_test > best_threshold).astype(int)
accuracy_test = accuracy_score(labels_test, final_predictions_test)
recall_test = recall_score(labels_test, final_predictions_test)
f1_test = f1_score(labels_test, final_predictions_test)

print(f'Ablation Experiment - top-k probability ignored (w1 = 0):')
print(f'Optimal threshold: {best_threshold:.4f}')
print(f'Joint Accuracy: {accuracy_test:.4f}')
print(f'Joint Recall: {recall_test:.4f}')
print(f'Joint F1-score: {f1_test:.4f}')



