import pandas as pd

def process_log_to_csv(file_path, output_path, L):
    # Load the CSV file
    df = pd.read_csv(file_path,low_memory=False,memory_map=True)

    # Ensure 'Date' and 'Time' columns are treated as strings
    #df['Date'] = df['Date'].astype(str)
    #df['Time'] = df['Time'].astype(str)

    # Initialize lists to store the results
    block_ids = []
    block_statuses = []
    event_intervals = []
    event_lists = []
    translated_events = []

    # Loop through the logs in chunks of L
    for block_index in range(0, len(df), L):
        # Extract the block of logs
        block = df.iloc[block_index:block_index + L]

        # Block ID
        block_id = block_index // L + 1

        # Determine the block status
        block_status = 'success' if all(block['Status'].str.lower() == 'success') else 'fail'
        #block_status = 'success' if all(block['Label'].str.lower() == 'success') else 'fail'
        # Calculate time intervals
        '''try:
            # Combine Date and Time into a single datetime column
            times = pd.to_datetime(block['Date'] + ' ' + block['Time'], format='%Y.%m.%d %H:%M:%S', errors='coerce')
            intervals = [0.0] + (times.diff().dt.total_seconds().fillna(0).tolist())[1:]
        except Exception as e:
            print(f"Error processing datetime conversion: {e}")
            continue'''

        # Create event list and concatenated translated events
        event_list = block['TemplateId'].tolist()
        Content = ' | '.join(block['Content'].tolist())  # Using '|' as a separator

        # Append results to the lists
        block_ids.append(block_id)
        block_statuses.append(block_status)
        #event_intervals.append(intervals)
        event_lists.append(event_list)
        translated_events.append(Content)

    # Create a DataFrame for the output
    output_df = pd.DataFrame({
        'BlockID': block_ids,
        'Status': block_statuses,
        'EventList': event_lists,
        'Content': translated_events
    })

    # Save the DataFrame to a CSV file
    output_df.to_csv(output_path, index=False)

# Example usage:
file_path = '/home/user10/tao/dataset/preprocessed/MULT single/content_log_with_templates.csv'
# Example usage:'
output_path = '/home/user10/tao/dataset/preprocessed/MULT single/20l_node1.csv'
L = 20  # The desired block length
process_log_to_csv(file_path, output_path, L)
