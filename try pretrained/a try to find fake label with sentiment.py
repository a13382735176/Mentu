import pandas as pd
import os
import numpy as np

# Path to the dataset
data_path = '/home/user10/tao/dataset/preprocessed/Spirit/20l_Spirit.csv'

if os.path.exists(data_path):
    data = pd.read_csv(data_path)
else:
    raise FileNotFoundError("Data file does not exist")

# Define the negative sentiment keywords
storage_negative_sentiment = ['panic', 'interrupt','error']
storage_good_sentiment=['corrected','detected']                              

# Split the data into training set (80%)
data_train = data[int(0.8 * len(data)):]

# Extract necessary columns
storage_content = data_train['Content']
storage_contentid = data_train['BlockID']
storage_status = data_train['Status']  # Assuming 'Status' is the real label

# Create a dictionary from BlockID and Content
contentandid_dict = dict(zip(storage_contentid, storage_content))

# Initialize variables to store fake labels and real labels
storage_normal_fake_labels = []
storage_anom_fake_labels = []
real_normal_labels = []
real_anom_labels = []

# Loop through content and generate fake labels
for id, every_content in contentandid_dict.items():
    k = 0
    # Split content by the '|' delimiter
    every_content_split_token = every_content.split('|')
    
    # Check for negative sentiment keywords
    for c in every_content_split_token:
        for negative_word in storage_negative_sentiment:
            if negative_word in c:                
                k += 1
        for good_word in storage_good_sentiment:
            if good_word in c:
                k-=1
    # Assign fake label based on keyword matching
    if k == 0:
        storage_normal_fake_labels.append(id)
    else:
        storage_anom_fake_labels.append(id)

# Calculate true normal and anomaly labels based on 'Status' column
for id, status in zip(storage_contentid, storage_status):
    if status == 'success':  # Assuming 'normal' is the label for normal logs
        real_normal_labels.append(id)
    else:
        real_anom_labels.append(id)

# Calculate accuracy
# Convert lists to sets for easier comparison
fake_normal_set = set(storage_normal_fake_labels)
fake_anom_set = set(storage_anom_fake_labels)
real_normal_set = set(real_normal_labels)
real_anom_set = set(real_anom_labels)

# True positives and negatives for normal
correct_normal = len(fake_normal_set.intersection(real_normal_set))
# True positives and negatives for anomaly
correct_anom = len(fake_anom_set.intersection(real_anom_set))

# Calculate accuracy for normal and anomaly labels
normal_accuracy = correct_normal / len(real_normal_set) if len(real_normal_set) > 0 else 0
anom_accuracy = correct_anom / len(real_anom_set) if len(real_anom_set) > 0 else 0

print(f"Normal label accuracy: {normal_accuracy * 100:.2f}%")
print(f"Anomaly label accuracy: {anom_accuracy * 100:.2f}%")

      

