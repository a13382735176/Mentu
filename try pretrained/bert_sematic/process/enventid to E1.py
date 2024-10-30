import os
import pandas as pd
import numpy as np
file='/home/user10/tao/parse_node1.csv/content_log.txt_templates.csv'
fd1=pd.read_csv(file,low_memory=False,memory_map=True)
K=len(fd1)
fd1['TemplateId']=['E'+ str(i + 1) for i in range(len(fd1))]
print(fd1.head())

# 保存到新文件（可选）
output_file = '/home/user10/tao/parse_node1_with_templates.csv'
fd1.to_csv(output_file, index=False)

print(f"文件已保存到: {output_file}")