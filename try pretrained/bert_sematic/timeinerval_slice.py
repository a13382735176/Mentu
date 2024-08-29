import pandas as pd
import numpy as np


def parse_date(date_str):
    date_str = str(date_str).zfill(6)  # 确保日期字符串为6位，例如：81109 -> 081109
    if len(date_str) == 6:
        return pd.to_datetime(date_str, format='%y%m%d')
    else:
        raise ValueError(f"Unexpected date format: {date_str}")


def parse_time(time_str):
    time_str = str(time_str).zfill(6)  # 确保时间字符串为6位，例如：203518
    if len(time_str) == 6:
        return pd.to_timedelta(f"{time_str[:2]}:{time_str[2:4]}:{time_str[4:]}")
    else:
        raise ValueError(f"Unexpected time format: {time_str}")


def split_log_by_time_interval(input_file, interval_seconds, step_seconds, output_file):
    # 读取CSV文件
    df = pd.read_csv(input_file)

    # 解析日期和时间
    df['datetime'] = df['Date'].astype(str).apply(parse_date) + df['Time'].astype(str).apply(parse_time)
    df = df.sort_values(by='datetime').reset_index(drop=True)

    start_time = df['datetime'].min()
    end_time = start_time + pd.Timedelta(seconds=interval_seconds)
    block_id = 1
    results = []

    while start_time < df['datetime'].max():
        block = df[(df['datetime'] >= start_time) & (df['datetime'] < end_time)].copy()

        if not block.empty:
            block['BlockID'] = block_id
            block['Status'] = 'success' if all(block['Level'] == 'INFO') else 'fail'
            block['TranslatedEvent'] = '|'.join(block['Translated_event'])
            block['EventIntervals'] = (
                block['datetime'].diff().fillna(pd.Timedelta(seconds=0)).dt.total_seconds()).tolist()
            results.append(block)
            block_id += 1

        start_time += pd.Timedelta(seconds=step_seconds)
        end_time = start_time + pd.Timedelta(seconds=interval_seconds)

    result_df = pd.concat(results)
    result_df.to_csv(output_file, index=False)


# 调用函数


# 调用函数


# 调用函数
split_log_by_time_interval(
    input_file='/root/try pretrained/dataset/preprocessed/HDFS/Modified_HDFS_log_structed_with_translated_event.csv',
    interval_seconds=1800,  # 每块日志的时间间隔（例如10分钟）
    step_seconds=1800,  # 步长，即每次前进的秒数（例如5分钟）
    output_file='/root/try pretrained/dataset/preprocessed/HDFS/0.5h_10min_HDFS_log_structed_with_translated_event.csv'
)
