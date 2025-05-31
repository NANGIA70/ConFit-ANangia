import json
import pandas as pd


def load_job_descriptions(file_path: str) -> pd.DataFrame:
    try:
        df = pd.read_csv(file_path)
        print(f"Loaded {len(df)} job descriptions from {file_path}")
        return df
    except Exception as e:
        raise ValueError(f"Error loading job descriptions: {str(e)}")

def load_labels(file_path: str) -> dict:
    try:
        with open(file_path, 'r') as f:
            labels = json.load(f)
        print(f"Loaded labels for {len(labels)} job descriptions from {file_path}")
        return labels
    except Exception as e:
        raise ValueError(f"Error loading labels: {str(e)}")

def load_rank_resume(file_path: str) -> dict:
    try:
        with open(file_path, 'r') as f:
            rank_resume = json.load(f)
        print(f"Loaded rank resume data for {len(rank_resume)} job descriptions")
        return rank_resume
    except Exception as e:
        raise ValueError(f"Error loading rank resume data: {str(e)}")

def load_all_labels_csv(file_path: str) -> pd.DataFrame:
    try:
        df = pd.read_csv(file_path)
        print(f"Loaded all labels data: {len(df)} entries")
        return df
    except Exception as e:
        raise ValueError(f"Error loading all labels data: {str(e)}")

def build_all_labels_dict(rank_resume: dict, all_labels_df: pd.DataFrame) -> dict:
    all_labels_dict = {}
    for _, row in all_labels_df.iterrows():
        job_id = row['jd_no']
        user_id = row['user_id']
        if job_id not in rank_resume:
            continue
            
        if job_id in all_labels_dict:
            all_labels_dict[job_id]["user_ids"].append(user_id)
            
            satisfied_value = row["satisfied"]
            if satisfied_value == 0:
                satisfied_value = 0.5
                
            all_labels_dict[job_id]["satisfied"].append(satisfied_value)
        
        else:
            all_labels_dict[job_id] = {
                "user_ids": [],
                "satisfied": []
            }
            
            all_labels_dict[job_id]["user_ids"].append(user_id)

            satisfied_value = row["satisfied"]
            if satisfied_value == 0:
                satisfied_value = 0.5
                
            all_labels_dict[job_id]["satisfied"].append(satisfied_value)
    return all_labels_dict