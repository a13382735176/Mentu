import re
import pandas as pd

# 定义一个函数处理每一行日志
def process_log_line(line):
    # 正则表达式匹配日志的时间戳、级别和内容
    log_pattern = re.compile(r'([\-]?)\s*(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2},\d{3}) \[.*?\] (INFO|ERROR|WARN|DEBUG)\s*(.*)')
    match = log_pattern.match(line)
    if match:
        is_normal = match.group(1) == '-'  # 检查行是否以“-”开头
        timestamp = match.group(2)  # 提取时间戳
        log_level = match.group(3)  # 提取日志级别
        content = match.group(4)  # 提取日志内容
        status = "success" if is_normal else "fail"  # 根据是否有“-”标记正常或异常
        return {"Timestamp": timestamp, "Level": log_level, "Content": content, "Status": status}
    return None

# 逐行读取大文件
log_file_path = '/home/user10/tao/dataset/preprocessed/MULT single/label1.log'  # 替换为实际日志文件路径
processed_logs = []

with open(log_file_path, 'r', encoding='utf-8') as file:
    for line in file:
        result = process_log_line(line)
        if result:
            processed_logs.append(result)
        # 每处理一定数量的行，写入到文件或数据库，避免内存溢出
        if len(processed_logs) >= 1000000:  # 每处理100万行，保存一次
            log_df = pd.DataFrame(processed_logs)
            log_df.to_csv('/home/user10/tao/dataset/preprocessed/MULT single/processed_logs_part.csv', mode='a', index=False, header=False)  # 追加写入CSV文件
            processed_logs = []  # 清空内存中的已处理日志

# 处理剩余的日志
if processed_logs:
    log_df = pd.DataFrame(processed_logs)
    log_df.to_csv('/home/user10/tao/dataset/preprocessed/MULT single/processed_logs_part.csv', mode='a', index=False, header=False)

# 输出最终结果
import ace_tools as tools# type: ignore
log_df = pd.read_csv('/mnt/data/processed_logs_part.csv', nrows=100)  # 显示前100行结果
tools.display_dataframe_to_user(name="Processed Logs", dataframe=log_df)
