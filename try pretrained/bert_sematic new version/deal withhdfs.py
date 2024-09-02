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

# 选择设备：CUDA 或 CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 数据加载和处理
data_path = '/home/user10/tao/dataset/preprocessed/HDFS/Modified_hdfs_log_structed_with_translated_event.csv'
if os.path.exists(data_path):
    data = pd.read_csv(data_path)
else:
    raise FileNotFoundError("Data file does not exist")

# 将"Status"列转换为数值类型
data['Status'] = data['Status'].apply(lambda x: 1 if x == 'Success' else 0)

# RoBERTa 嵌入生成（用于相似度计算）
tokenizer = RobertaTokenizer.from_pretrained('/home/user10/tao/pretrained_models/roberta-base')
roberta_model = RobertaModel.from_pretrained('/home/user10/tao/pretrained_models/roberta-base')
roberta_model = roberta_model.to(device)

embeddings_path = '/home/user10/tao/dataset/preprocessed/HDFS/modified_dim400_embeddings.npy'

def generate_embeddings(texts, tokenizer, model, device, batch_size=64):
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

# 检查嵌入文件是否存在
if os.path.exists(embeddings_path):
    embeddings = np.load(embeddings_path)
else:
    embeddings = generate_embeddings(data['TranslatedEvent'], tokenizer, roberta_model, device)
    np.save(embeddings_path, embeddings)

data['Embeddings'] = list(embeddings)
print("Embedding generation is complete.")

# 事件序列的TF-IDF编码
vectorizer = TfidfVectorizer(max_features=400)
event_sequences = data['EventList'].apply(eval).apply(lambda x: ' '.join(map(str, x)))
tfidf_matrix = vectorizer.fit_transform(event_sequences).toarray()

# 获取块大小的最大长度
max_length = max(data['EventList'].apply(lambda x: len(eval(x))))

def pad_or_truncate(sequence, max_length):
    if len(sequence) < max_length:
        sequence.extend([0] * (max_length - len(sequence)))
    return sequence[:max_length]

# 将数据集划分为训练集和测试集
#train_data, test_data = train_test_split(data, test_size=0.2, random_state=42)
train_data, test_data = data[:int(len(data) * 0.8)], data[int(len(data) * 0.8):]
train_tfidf = np.array([pad_or_truncate(list(seq), max_length) for seq in vectorizer.fit_transform(train_data['EventList'].apply(eval).apply(lambda x: ' '.join(map(str, x)))).toarray()])
test_tfidf = np.array([pad_or_truncate(list(seq), max_length) for seq in vectorizer.transform(test_data['EventList'].apply(eval).apply(lambda x: ' '.join(map(str, x)))).toarray()])

# 将训练集和测试集的数据转换为 Tensor
X_train = torch.tensor(train_tfidf, dtype=torch.float32).to(device)
X_test = torch.tensor(test_tfidf, dtype=torch.float32).to(device)

# 将输入数据的维度从 (B, S) 改为 (B, S, 1)
X_train = X_train.unsqueeze(-1).transpose(1, 2)
X_test = X_test.unsqueeze(-1).transpose(1, 2)
print("Ready to start xLSTM training.")

# xLSTM模型配置
cfg = xLSTMBlockStackConfig(
    mlstm_block=mLSTMBlockConfig(
        mlstm=mLSTMLayerConfig(
            conv1d_kernel_size=4, qkv_proj_blocksize=4, num_heads=4
        )
    ),
    slstm_block=sLSTMBlockConfig(
        slstm=sLSTMLayerConfig(
            backend="cuda",
            num_heads=2,
            conv1d_kernel_size=4,
            bias_init="powerlaw_blockdependent",
        ),
        feedforward=FeedForwardConfig(proj_factor=1.3, act_fn="gelu"),
    ),
    context_length=256,
    num_blocks=7,
    embedding_dim=298,
    slstm_at=[1],
)

# xLSTM模型训练
xlstm_stack = xLSTMBlockStack(cfg).to(device)
optimizer = optim.Adam(xlstm_stack.parameters(), lr=0.001)
scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=5)
num_epochs = 2
batch_size = 128
early_stopping_patience = 10
best_loss = float('inf')
patience_counter = 0

for epoch in range(num_epochs):
    xlstm_stack.train()
    epoch_loss = 0
    for i in range(0, X_train.shape[0], batch_size):
        batch_X = X_train[i:i + batch_size]
        optimizer.zero_grad()
        outputs = xlstm_stack(batch_X)
        loss = ((outputs - batch_X) ** 2).mean()  # 重建误差
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

# 从训练集中选择样本
normal_indices = np.array(train_data[train_data['Status'] == 1].index)
anomalous_indices = np.array(train_data[train_data['Status'] == 0].index)
np.random.seed(42)
# 确保选择的样本数量不会超过实际可用的数量
max_normal_samples = min(800, len(normal_indices))
max_anomalous_samples = min(1000, len(anomalous_indices))

selected_normal_indices = np.random.choice(normal_indices, max_normal_samples, replace=False)
selected_anomalous_indices = np.random.choice(anomalous_indices, max_anomalous_samples, replace=False)
storage_indices = np.concatenate([selected_normal_indices, selected_anomalous_indices])

# 获取对应的存储嵌入和标签
valid_storage_indices = np.array([idx for idx in storage_indices if idx in train_data.index])
storage_embeddings = embeddings[valid_storage_indices]
storage_labels = train_data.loc[valid_storage_indices, 'Status'].values

# 查找与新日志块最相近的几个日志编码块
def find_most_similar(new_embedding, storage_embeddings, top_k=20):
    # 将存储的嵌入和新嵌入转换为Tensor并移动到GPU
    storage_embeddings_tensor = torch.tensor(storage_embeddings).to(device)
    new_embedding_tensor = torch.tensor(new_embedding).to(device)

    # 归一化嵌入向量
    storage_embeddings_tensor = torch.nn.functional.normalize(storage_embeddings_tensor, p=2, dim=1)
    new_embedding_tensor = torch.nn.functional.normalize(new_embedding_tensor, p=2, dim=0)

    # 计算余弦相似度
    similarities = torch.matmul(storage_embeddings_tensor, new_embedding_tensor).cpu().numpy()

    # 找到最相似的 top_k
    top_k_indices = np.argsort(similarities)[-top_k:]
    return top_k_indices, similarities[top_k_indices]


# 计算训练集和测试集上的重建误差
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

reconstruction_errors_train = compute_reconstruction_errors(xlstm_stack, X_train)
reconstruction_errors_test = compute_reconstruction_errors(xlstm_stack, X_test)

# 归一化重建误差
min_reconstruction_error_train = reconstruction_errors_train.min()
max_reconstruction_error_train = reconstruction_errors_train.max()
normalized_reconstruction_errors_train = (reconstruction_errors_train - min_reconstruction_error_train) / (max_reconstruction_error_train - min_reconstruction_error_train)

min_reconstruction_error_test = reconstruction_errors_test.min()
max_reconstruction_error_test = reconstruction_errors_test.max()
normalized_reconstruction_errors_test = (reconstruction_errors_test - min_reconstruction_error_test) / (max_reconstruction_error_test - min_reconstruction_error_test)

# 查找与新日志块最相近的几个日志块，并计算概率
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

# 准备训练注意力机制的数据
combined_features_train = torch.tensor(np.stack([top_k_probabilities_train, normalized_reconstruction_errors_train], axis=1), dtype=torch.float32).to(device)
labels_train = torch.tensor(train_data['Status'].values, dtype=torch.float32).to(device)

# 准备测试集的数据
combined_features_test = torch.tensor(np.stack([top_k_probabilities_test, normalized_reconstruction_errors_test], axis=1), dtype=torch.float32).to(device)
labels_test = test_data['Status'].values  # labels_test 现在已经被定义为 NumPy 数组

# 定义损失函数和优化器
criterion = nn.BCELoss()
attention_optimizer = optim.Adam(attention.parameters(), lr=0.001)
scheduler_attention = ReduceLROnPlateau(attention_optimizer, mode='min', factor=0.1, patience=5)
early_stopping_patience_attention = 10
best_attention_loss = float('inf')
patience_counter_attention = 0

# 训练注意力机制
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

# 在训练集上找到最优阈值
best_threshold, best_f1 = 0, 0

# 断开计算图并将Tensor转换为NumPy数组
weighted_scores_train_np = weighted_scores_train.detach().cpu().numpy()
labels_train_np = labels_train.detach().cpu().numpy()

for threshold in np.linspace(min(weighted_scores_train_np), max(weighted_scores_train_np), 100):
    predictions = (weighted_scores_train_np > threshold).astype(int)
    f1 = f1_score(labels_train_np, predictions)
    if f1 > best_f1:
        best_f1 = f1
        best_threshold = threshold

# 将最优阈值应用于测试集
attention.eval()
with torch.no_grad():
    weighted_scores_test = attention(combined_features_test).cpu().numpy()

final_predictions_test = (weighted_scores_test > best_threshold).astype(int)
accuracy_test = accuracy_score(labels_test, final_predictions_test)
recall_test = recall_score(labels_test, final_predictions_test)
f1_test = f1_score(labels_test, final_predictions_test)

print(f'Optimal threshold: {best_threshold:.4f}')
print(f'joint-Accuracy: {accuracy_test:.4f}')
print(f'joint-Recall: {recall_test:.4f}')
print(f'joint-F1-score: {f1_test:.4f}')

# 生成并保存 UMAP 图
def plot_and_save_umap(embeddings, labels, file_path, title="UMAP plot"):
    labels = np.array(labels)
    
    # 确保 embeddings 是一个二维数组
    embeddings = np.array([np.array(embedding).flatten() for embedding in embeddings])
    
    reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, metric='cosine')
    umap_embeddings = reducer.fit_transform(embeddings)

    plt.figure(figsize=(10, 7))
    plt.scatter(umap_embeddings[labels == 1, 0], umap_embeddings[labels == 1, 1], c='yellow', label='Normal predictions', alpha=0.5)
    plt.scatter(umap_embeddings[labels == 0, 0], umap_embeddings[labels == 0, 1], c='blue', label='Anomaly predictions', alpha=0.5)
    plt.scatter(umap_embeddings[:len(train_data), 0], umap_embeddings[:len(train_data), 1], c='pink', label='Training samples', alpha=0.5)
    plt.title(title)
    plt.legend(loc='best')
    plt.savefig(file_path)  # 保存图片
    plt.close()


# 调用函数，保存 UMAP 图
plot_and_save_umap(np.array(data['Embeddings']), data['Status'], file_path="/root/try_pretrained/umap_plot.png", title="UMAP plot of Thunderbird semantic vectors")
