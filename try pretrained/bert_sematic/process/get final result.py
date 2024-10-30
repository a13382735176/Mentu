import pandas as pd

# 读取CSV文件
log_file = '/home/user10/tao/parse_node1.csv/parse_node1_with_templates.csv'  # 替换为你的CSV日志文件路径
log_df = pd.read_csv(log_file,low_memory=False,memory_map=True)
first_content_by_eventid = log_df.groupby('EventId')['TemplateId'].first().to_dict()
# 提取Content列
file_2='/home/user10/tao/parse_node1.csv/content_log.txt_structured.csv'
df2 = pd.read_csv(file_2,low_memory=False,memory_map=True)
file_3='/home/user10/tao/dataset/preprocessed/MULT single/processed_logs_part.csv'
df3 = pd.read_csv(file_3,low_memory=False,memory_map=True)
# 保存Content列到新的文件
df3['TemplateId']=df2['EventId'].map(first_content_by_eventid)
output_file = '/home/user10/tao/dataset/preprocessed/MULT single/content_log_with_templates.csv'
df3.to_csv(output_file, index=False)

print(f"文件已保存到: {output_file}")