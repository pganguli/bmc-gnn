from subprocess import call
import os
import glob
import cosine_similarity as cs
import pandas as pd
def extract_frame_time(frame_no, path):
    files = [f for f in os.listdir(path) if f.endswith('.csv')]
    results = []
    for filename in files:
        data = pd.read_csv(f"{path}/{filename}")
        if 'Frame' in data.columns and 'Time' in data.columns:
            filtered_data = data[data['Frame'] == frame_no]
            if not filtered_data.empty:
                times = filtered_data['Time'].tolist()
                results.append((filename, frame_no, times))
            else:
                results.append((filename, frame_no, f"No 'Frame = {frame_no}' found"))
        else:
            results.append((filename, "Missing required columns"))
    return results
