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
from sklearn.metrics import accuracy_score, recall_score, f1_score
from torch.optim.lr_scheduler import ReduceLROnPlateau
import umap
import matplotlib.pyplot as plt
from xlstm import xLSTMBlockStack, xLSTMBlockStackConfig, mLSTMBlockConfig, mLSTMLayerConfig, sLSTMBlockConfig, sLSTMLayerConfig, FeedForwardConfig

# choose device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")#xlstm must need cuda

# dataload
data_path = '/root/try pretrained/dataset/preprocessed/Spirit/200l_Spirit.csv'
if os.path.exists(data_path):
    data = pd.read_csv(data_path)
else:
    raise FileNotFoundError("Data file does not exist")

# "Convert the 'Status' column to a numeric type."
data['Status'] = data['Status'].apply(lambda x: 1 if x == 'success' else 0)

# RoBERTa 嵌入生成（用于相似度计算）
tokenizer = RobertaTokenizer.from_pretrained('/root/try pretrained/pretrained_models/roberta-base')
roberta_model = RobertaModel.from_pretrained('/root/try pretrained/pretrained_models/roberta-base')
roberta_model = roberta_model.to(device)

embeddings_path = '/root/try pretrained/dataset/preprocessed/Spirit/200l_dim400_embeddings with no translated.npy'

def generate_embeddings(texts, tokenizer, model, device, batch_size=16):
    embeddings = []
    for i in tqdm(range(0, len(texts), batch_size)):
        batch_texts = texts[i:i+batch_size].tolist()
        inputs = tokenizer(batch_texts, return_tensors='pt', padding=True, truncation=True, max_length=400)
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
    embeddings = generate_embeddings(data['Content'], tokenizer, roberta_model, device)
    np.save(embeddings_path, embeddings)

data['Embeddings'] = list(embeddings)
print("Embedding generation is complete.")

# TF-IDF encoding of the event sequence
vectorizer = TfidfVectorizer(max_features=400)
event_sequences = data['EventList'].apply(eval).apply(lambda x: ' '.join(map(str, x)))
tfidf_matrix = vectorizer.fit_transform(event_sequences).toarray()

# Divide the data set into a training set and a test set
train_data, test_data = train_test_split(data, test_size=0.2, random_state=42)
train_tfidf, test_tfidf = train_test_split(tfidf_matrix, test_size=0.2, random_state=42)

# Ensure that the length of the input data is 400以上翻译结果来自有道神经网络翻译（YNMT）· 通用场景
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

# Change the dimension of the input data from (B, S) to (B, S, 1)以上翻译结果来自有道神经网络翻译（YNMT）· 通用场景
X_train = X_train.unsqueeze(-1).transpose(1, 2)
X_test = X_test.unsqueeze(-1).transpose(1, 2)
print("Ready to start xLSTM training.")

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


# Select samples from the training set
normal_indices = np.array(train_data[train_data['Status'] == 1].index)
anomalous_indices = np.array(train_data[train_data['Status'] == 0].index)

# Ensure that the number of samples selected does not exceed the number actually available
max_normal_samples = min(1000, len(normal_indices))
max_anomalous_samples = min(1000, len(anomalous_indices))

selected_normal_indices = np.random.choice(normal_indices, max_normal_samples, replace=False)
selected_anomalous_indices = np.random.choice(anomalous_indices, max_anomalous_samples, replace=False)
storage_indices = np.concatenate([selected_normal_indices, selected_anomalous_indices])

# Gets the corresponding storage embeddings and labels
valid_storage_indices = np.array([idx for idx in storage_indices if idx in train_data.index])
storage_embeddings = embeddings[valid_storage_indices]
storage_labels = train_data.loc[valid_storage_indices, 'Status'].values

# Find the few log coding blocks that are closest to the new log block
def find_most_similar(new_embedding, storage_embeddings, top_k=20):
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
    return top_k_indices, similarities[top_k_indices]


# Calculate reconstruction errors on the training and test sets
def compute_reconstruction_errors(xlstm_stack, X, batch_size=64):
    reconstruction_errors = []
    xlstm_stack.eval()
    with torch.no_grad():
        for i in tqdm(range(0, X.shape[0], batch_size)):
            batch_X = X[i:i + batch_size]
            xlstm_reconstruction = xlstm_stack(batch_X)
            reconstruction_error = ((xlstm_reconstruction - batch_X) ** 2).mean(dim=[1, 2]).cpu().numpy()
            reconstruction_errors.extend(reconstruction_error)
    return np.array(reconstruction_errors)

reconstruction_errors_train = compute_reconstruction_errors(xlstm_stack, X_train)
reconstruction_errors_test = compute_reconstruction_errors(xlstm_stack, X_test)

# Normalized reconstruction error
min_reconstruction_error_train = reconstruction_errors_train.min()
max_reconstruction_error_train = reconstruction_errors_train.max()
normalized_reconstruction_errors_train = (reconstruction_errors_train - min_reconstruction_error_train) / (max_reconstruction_error_train - min_reconstruction_error_train)

min_reconstruction_error_test = reconstruction_errors_test.min()
max_reconstruction_error_test = reconstruction_errors_test.max()
normalized_reconstruction_errors_test = (reconstruction_errors_test - min_reconstruction_error_test) / (max_reconstruction_error_test - min_reconstruction_error_test)

# 查找与新日志块最相近的几个日志块，并计算概率
'''def find_top_k_probabilities(storage_embeddings, storage_labels, embeddings, top_k=20):
    probabilities = []
    for new_embedding in tqdm(embeddings, desc="Calculating probabilities"):
        top_k_indices, _ = find_most_similar(new_embedding, storage_embeddings, top_k=top_k)
        most_similar_labels = storage_labels[top_k_indices]
        probabilities.append(most_similar_labels.mean())
    return np.array(probabilities)'''
def find_top_k_probabilities(storage_embeddings, storage_labels, embeddings, top_k=20):
    probabilities = []
    for new_embedding in tqdm(embeddings, desc="Calculating probabilities"):
        top_k_indices, _ = find_most_similar(new_embedding, storage_embeddings, top_k=top_k)
        most_similar_labels = storage_labels[top_k_indices]
        probabilities.append(most_similar_labels.mean())
    return np.array(probabilities)


top_k_probabilities_train = find_top_k_probabilities(storage_embeddings, storage_labels, np.array(train_data['Embeddings']))
top_k_probabilities_test = find_top_k_probabilities(storage_embeddings, storage_labels, np.array(test_data['Embeddings']))

# 注意力机制
class Attention(nn.Module):
    def __init__(self, input_dim):
        super(Attention, self).__init__()
        self.attention_weights = nn.Parameter(torch.randn(input_dim))
        self.softmax = nn.Softmax(dim=0)

    def forward(self, x):
        weights = self.softmax(self.attention_weights)
        return torch.sum(x * weights, dim=1)

attention = Attention(2).to(device)

# Prepare the data to train the attention mechanism
combined_features_train = torch.tensor(np.stack([top_k_probabilities_train, normalized_reconstruction_errors_train], axis=1), dtype=torch.float32).to(device)
labels_train = torch.tensor(train_data['Status'].values, dtype=torch.float32).to(device)

# Prepare the data for the test set
combined_features_test = torch.tensor(np.stack([top_k_probabilities_test, normalized_reconstruction_errors_test], axis=1), dtype=torch.float32).to(device)
labels_test = test_data['Status'].values  # labels_test is now defined as a NumPy array
# Define loss functions and optimizers
criterion = nn.BCELoss()
attention_optimizer = optim.Adam(attention.parameters(), lr=0.001)
scheduler_attention = ReduceLROnPlateau(attention_optimizer, mode='min', factor=0.1, patience=5)
early_stopping_patience_attention = 10
best_attention_loss = float('inf')
patience_counter_attention = 0
# Training attention mechanism
num_attention_epochs = 50
for epoch in range(num_attention_epochs):
    attention.train()
    attention_optimizer.zero_grad()
    weighted_scores_train = attention(combined_features_train)
    loss = criterion(weighted_scores_train, labels_train)
    loss.backward()
    attention_optimizer.step()
    
    print(f'Attention Epoch [{epoch+1}/{num_attention_epochs}], Loss: {loss.item():.4f}')
    
    scheduler_attention.step(loss.item())

    # Early stopping logic for attention mechanism
    if loss.item() < best_attention_loss:
        best_attention_loss = loss.item()
        patience_counter_attention = 0
    else:
        patience_counter_attention += 1

    if patience_counter_attention >= early_stopping_patience_attention:
        print("Early stopping for attention mechanism triggered.")
        break
final_weights = attention.softmax(attention.attention_weights).detach().cpu().numpy()
w1, w2 = final_weights[0], final_weights[1]
print(f'Final w1: {w1:.4f}, w2: {w2:.4f}')
# Find the optimal threshold on the training set
best_threshold, best_f1 = 0, 0

# Disconnect the calculation diagram and convert Tensor to a NumPy array
weighted_scores_train_np = weighted_scores_train.detach().cpu().numpy()
labels_train_np = labels_train.detach().cpu().numpy()

for threshold in np.linspace(min(weighted_scores_train_np), max(weighted_scores_train_np), 100):
    predictions = (weighted_scores_train_np > threshold).astype(int)
    f1 = f1_score(labels_train_np, predictions)
    if f1 > best_f1:
        best_f1 = f1
        best_threshold = threshold

# Apply the optimal threshold to the test set
attention.eval()
with torch.no_grad():
    weighted_scores_test = attention(combined_features_test).cpu().numpy()

final_predictions_test = (weighted_scores_test > best_threshold).astype(int)
accuracy_test = accuracy_score(labels_test, final_predictions_test)
recall_test = recall_score(labels_test, final_predictions_test)
f1_test = f1_score(labels_test, final_predictions_test)

print(f'Optimal threshold: {best_threshold:.4f}')
print("训练集上的f1-SCORE",best_f1)
print(f'joint-Accuracy: {accuracy_test:.4f}')
print(f'joint-Recall: {recall_test:.4f}')
print(f'joint-F1-score: {f1_test:.4f}')

# UMAP 参数调整
reducer = umap.UMAP(n_neighbors=50, min_dist=0.01, metric='cosine')

# 进行降维
log_embeddings_2d = reducer.fit_transform(embeddings)

# 提取不同数据集的嵌入
train_embeddings_2d = log_embeddings_2d[train_data.index]
test_embeddings_2d = log_embeddings_2d[test_data.index]
random_sample_embeddings_2d = log_embeddings_2d[valid_storage_indices]

# 可视化并保存图片
plt.figure(figsize=(12, 8))

# 绘制训练集点
plt.scatter(train_embeddings_2d[:, 0], train_embeddings_2d[:, 1], c='blue', label='Train', alpha=0.6)

# 绘制测试集点
plt.scatter(test_embeddings_2d[:, 0], test_embeddings_2d[:, 1], c='green', label='Test', alpha=0.6)

# 绘制随机样本点
plt.scatter(random_sample_embeddings_2d[:, 0], random_sample_embeddings_2d[:, 1], c='red', label='Random Sample', alpha=0.6)

# 设置图例和标题
plt.legend()
plt.title("2D Visualization of Log Embeddings using UMAP")
plt.xlabel("Component 1")
plt.ylabel("Component 2")

# 保存图片
plt.savefig('/root/try pretrained/umap_visualization.png')
plt.close()

print("UMAP visualization saved as 'umap_visualization.png'")
