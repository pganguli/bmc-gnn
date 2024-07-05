import os
import pandas as pd


def extract_frame_time(frame_no, path, best_friend):
    results = []

    for root, dirs, files in os.walk(path):
        for dir in dirs:
            subdir_path = os.path.join(root, dir)
            best_friend_csv = f"{best_friend}.csv"
            file_path = os.path.join(subdir_path, best_friend_csv)

            if os.path.isfile(file_path):
                data = pd.read_csv(file_path)

                if "F" in data.columns and "Time" in data.columns:
                    filtered_data = data[data["F"] == frame_no]
                    if not filtered_data.empty:
                        times = filtered_data["Time"].tolist()
                        results.append((dir, frame_no, times))

    return results
