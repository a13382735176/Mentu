import pandas as pd

# 读取CSV文件
log_file = '/home/user10/tao/dataset/preprocessed/MULT single/processed_logs_part.csv'  # 替换为你的CSV日志文件路径
log_df = pd.read_csv(log_file,low_memory=False,memory_map=True)

# 提取Content列
log_content = log_df['Content']

# 保存Content列到新的文件
log_content.to_csv('/home/user10/tao/dataset/preprocessed/MULT single/content_log.txt', index=False, header=False)
