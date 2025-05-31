import json
import os
import pandas as pd


def save_final_results(
    output_dir: str,
    metrics: list[dict],
    model_provider: str,
    model_name: str,
    bm25_k_values: list[int],
    eval_k_values: list[int]
):
    df = pd.DataFrame(metrics) if metrics else pd.DataFrame()
    
    final_results = {
        "model_provider": model_provider,
        "model_name": model_name,
        "num_jobs": len(df["job_id"].unique()) if not df.empty else 0,
        "precision": {},
        "recall": {},
        "ndcg1": {},
        "map1": 0.0,
        "mrr1": 0.0,
        "recall_at_all1": 0.0,
        "recallmap1": 0.0,
        "ndcg1@all": 0.0,
        "average Retrieval Number of Resumes": 0.0,
        "average_query_length": 0.0
    }
    
    for k in eval_k_values:
        k_key = f"top{k}"
        
        for metric_name in ["precision", "recall"]:
            values = []
            for m in metrics:
                if metric_name in m and k_key in m[metric_name]:
                    v = m[metric_name][k_key]
                    if v is not None:
                        values.append(v)
                elif f"{metric_name}_{k_key}" in m:
                    v = m[f"{metric_name}_{k_key}"]
                    if v is not None:
                        values.append(v)
            
            if values:
                final_results[metric_name][k_key] = sum(values) / len(values)
            else:
                final_results[metric_name][k_key] = 0.0
                
        ndcg_key = f"@{k}"
        
        for metric_name in ["ndcg1"]:
            values = []
            for m in metrics:
                if metric_name in m and ndcg_key in m[metric_name]:
                    v = m[metric_name][ndcg_key]
                    if v is not None:
                        values.append(v)
            
            if values:
                final_results[metric_name][ndcg_key] = sum(values) / len(values)
            else:
                final_results[metric_name][ndcg_key] = 0.0
    
    for metric_name in ["map1", "mrr1", "recall_at_all1", "recallmap1", "ndcg1@all"]:
        values = []
        for m in metrics:
            if metric_name in m and isinstance(m[metric_name], (int, float)):
                v = m[metric_name]
                if v is not None:
                    values.append(v)
        
        if values:
            final_results[metric_name] = sum(values) / len(values)
        else:
            final_results[metric_name] = 0.0
    
    values = []
    for m in metrics:
        if "hits_count" in m:
            v = m["hits_count"]
            if v is not None:
                values.append(v)
    if values:
        final_results["average Retrieval Number of Resumes"] = sum(values) / len(values)
    else:
        final_results["average Retrieval Number of Resumes"] = 0.0
        
    values = []
    for m in metrics:
        if "query_length" in m:
            v = m["query_length"]
            if v is not None:
                values.append(v)
    if values:
        final_results["average_query_length"] = sum(values) / len(values)
    else:
        final_results["average_query_length"] = 0.0

    with open(os.path.join(output_dir, "final_results.json"), "w") as f:
        json.dump(final_results, f, indent=2)

    print("\nFinal Results:")
    print(f"Model: {final_results['model_provider']}/{final_results['model_name']}")
    print(f"Number of jobs: {final_results['num_jobs']}")
    
    print("\nRelevance Metrics:")
    for k in eval_k_values:
        k_key = f"top{k}"
        # print(f"Precision@{k}: {final_results['precision'][k_key]:.4f}")
        print(f"Recall@{k}: {final_results['recall'][k_key]:.4f}")
        ndcg_key = f"@{k}"
        print(f"NDCG1@{k}: {final_results['ndcg1'][ndcg_key]:.4f}")
        print("")
    
    
    print(f"MAP1 (only sat=1.0): {final_results['map1']:.4f}")
    print(f"MRR1 (only sat=1.0): {final_results['mrr1']:.4f}")
    print(f"Recall@All1 (only sat=1.0): {final_results['recall_at_all1']:.4f}")
    print(f"RecallMAP1 (only sat=1.0): {final_results['recallmap1']:.4f}")
    
    print(f"NDCG1@all: {final_results['ndcg1@all']:.4f}")
    print(f"Average Retrieval Number of Resumes: {final_results['average Retrieval Number of Resumes']:.4f}")
    print(f"Average Query Length: {final_results['average_query_length']:.4f}")
    return


def save_intermediate_results(
    output_dir: str,
    keyword_results: list[dict],
    query_results: list[dict],
    search_results: list[dict],
    metrics: list[dict]
):
    # Save keywords
    with open(os.path.join(output_dir, "keywords.json"), "w") as f:
        json.dump(keyword_results, f, indent=2)
    
    # Save queries
    with open(os.path.join(output_dir, "queries.json"), "w") as f:
        json.dump(query_results, f, indent=2)
    
    # Save search results (might be large, so save separately)
    with open(os.path.join(output_dir, "search_results.json"), "w") as f:
        json.dump(search_results, f, indent=2)
    
    # Save metrics
    with open(os.path.join(output_dir, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)