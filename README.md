# "A log anomaly detector based on Template level"
The goal of log anomaly detection is to accurately distinguish between normal and anomalous logs. However, due to the highly imbalanced distribution of normal and anomalous logs in publicly available datasets (HDFS: 2.93%, Spirit: 15.29%, Thunderbird: 0.49%, BGL: 7.34%), many log anomaly detectors tend to report inflated metrics (F1 scores over 0.9). While these detectors perform well in identifying positive samples, they often struggle to effectively recognize negative samples. **In Hoang's paper**, "**How Far Are We**?", when advanced detectors are evaluated using fixed windows and session-based segmentation, many issues arise, primarily because the positive-to-negative sample ratio becomes more balanced under these conditions, increasing the detection difficulty.

This observation leads us to consider whether it's possible to design an anomaly detector that maintains robust generalization performance even when the positive-to-negative sample ratio is more balanced, as in session-based or fixed-length segmentation. To address this, we have developed an anomaly diagnosis mechanism that not only significantly improves detection accuracy in high-fault-rate scenarios but also automates the precise threshold search, **eliminating the need for tedious manual tuning**.

Specifically, we select a subset of log blocks from the training set and store them in a **container**. Then, we compare each log block under inspection with those in the container using cosine similarity. The **top-k** most similar log blocks are chosen, and a probability value is generated based on this comparison. Following this, we use an XLSTM model to calculate the **reconstruction error**. We then apply additive attention to combine the **probability value and the reconstruction error**, creating a threshold objective function. Finally, we search for the optimal threshold within this objective function and apply it to the test set. Experimental results have demonstrated the effectiveness of this method.

# Work flow
![image](https://github.com/user-attachments/assets/fc405aed-28e8-40fb-97b8-f7b941036ce4)
Specifically, after segmenting the log blocks by fixed length or session, for each log block (assuming a fixed length of 20), we take blockid 1 as an example. We only use the "Content" column and input it into the Roberta model for encoding. Similarly, for blockid 1, we use the "Eventlist" column, apply TF-IDF for encoding, and pass the encoded result to the Xlstm model.

# Setup
The environment only needs to ensure that Python is greater than **3.11** and **xlstm==1.0.3** and torch==2.3.1+cu118. The rest of the environment should be automatically installed on the server. Special attention is required for CUDA (I used versions **11.8 and 12.2**, both of which work).
# Table of Contents
![image](https://github.com/user-attachments/assets/3a498686-120d-4a5a-8b69-cd538ef5b333)

If you want to run the HDFS experiment, please open the  "**deal_with_hdfs" file in the "bert_sematic**" folder and run. I’ve already completed the Roberta encoding, so it can be run with a single command. You just need to update the file path.
   **detect anomaly in Tb Sp BGL with no translated event.py**，This is for performing anomaly detection on three other datasets.
   The "**BGL and Tb spirit slice with no translated content.py**" script is used for fixed-length segmentation of the datasets, which is a preprocessing step. However, you don’t need to run it, as I’ve already completed the segmentation.
    Thank you.

# Usage
```
data_path = '/root/try pretrained/dataset/preprocessed/Spirit/200l_Spirit.csv' 
if os.path.exists(data_path):
    data = pd.read_csv(data_path)
else:
    raise FileNotFoundError("Data file does not exist")

data['Status'] = data['Status'].apply(lambda x: 1 if x == 'success' else 0)
tokenizer = RobertaTokenizer.from_pretrained('/root/try pretrained/pretrained_models/roberta-base')
roberta_model = RobertaModel.from_pretrained('/root/try pretrained/pretrained_models/roberta-base')
roberta_model = roberta_model.to(device)

embeddings_path = '/root/try pretrained/dataset/preprocessed/Spirit/200l_dim400_embeddings with no translated.npy
```
# You only need to replace `data_path` and `embedding_path` to run the experiment and observe the results under different segmentation lengths.
---
# Dataset
https://figshare.com/articles/thesis/dataset_zip/26885611?file=48909394

"**After extracting the dataset file, simply drag it into the directory.**"
![23b4a343-7823-4984-a809-4dc4cc3b451f](https://github.com/user-attachments/assets/18e4f719-9950-4089-a663-e313176f7373)


