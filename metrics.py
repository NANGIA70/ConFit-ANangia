from typing import Optional
import re
import numpy as np


def calculate_query_length(keyword_dict: dict) -> int:
    total_length = 0
    
    normal_fields = ["title", "location", "industryKeywords", "jobFunctionKeywords", "languages"]
    
    for field in normal_fields:
        if field in keyword_dict:
            if isinstance(keyword_dict[field], list):
                total_length += len(keyword_dict[field])
            elif isinstance(keyword_dict[field], str) and keyword_dict[field].strip():
                total_length += 1
    
    if "yearOfWork" in keyword_dict and keyword_dict["yearOfWork"]:
        if isinstance(keyword_dict["yearOfWork"], dict):
            if "gte" in keyword_dict["yearOfWork"] and keyword_dict["yearOfWork"]["gte"] is not None:
                total_length += 1
            if "lte" in keyword_dict["yearOfWork"] and keyword_dict["yearOfWork"]["lte"] is not None:
                total_length += 1
    
    if "skillBooleanString" in keyword_dict and isinstance(keyword_dict["skillBooleanString"], list):
        for boolean_expr in keyword_dict["skillBooleanString"]:
            if not boolean_expr or not isinstance(boolean_expr, str):
                continue
            
            def count_terms(expr: str) -> int:
                expr = expr.upper()
                if " AND " in expr and " OR " in expr:
                    or_blocks = re.split(r'\s+(?i:OR)\s+', expr)
                    count = 0
                    
                    for block in or_blocks:
                        if " AND " in block.upper():
                            and_parts = re.split(r'\s+(?i:AND)\s+', block)
                            count += len(and_parts)
                        else:
                            count += 1
                    
                    return count
                elif " AND " in expr:
                    and_parts = re.split(r'\s+(?i:AND)\s+', expr)
                    return len(and_parts)
                elif " OR " in expr:
                    or_parts = re.split(r'\s+(?i:OR)\s+', expr)
                    return len(or_parts)
                else:
                    return 1
            
            total_length += count_terms(boolean_expr)
    return total_length


def calculate_top_k_avg_bm25(search_results: dict, k: int = 10) -> Optional[float]:
    if not search_results or 'hits' not in search_results or 'hits' not in search_results['hits']:
        return 0.0
    
    hits = search_results['hits']['hits']
    top_k_hits = hits[:min(k, len(hits))]
    
    if not top_k_hits:
        return 0.0
    
    total_score = sum(hit.get('_score', 0) for hit in top_k_hits)
    avg_score = total_score / len(top_k_hits) if len(top_k_hits) > 0 else 0.0
    
    return avg_score

def calculate_precision_recall_f1(
    search_results: dict,
    job_id: str,
    labels: dict,
    k: int = 10
) -> dict:
    """Calculate precision, recall, and F1 score for the top k search results."""
    if not search_results or 'hits' not in search_results or 'hits' not in search_results['hits']:
        return {
            "precision": 0.0,
            "recall": 0.0,
            "f1": 0.0
        }
    
    if job_id not in labels:
        return {
            "precision": 0.0,
            "recall": 0.0,
            "f1": 0.0
        }
    
    # Get the top k search results
    hits = search_results['hits']['hits']
    top_k_hits = hits[:min(k, len(hits))]
    
    # If no results, return zeros
    if not top_k_hits:
        return {
            "precision": 0.0,
            "recall": 0.0,
            "f1": 0.0
        }
    
    # Get the IDs of the top k search results
    top_k_ids = [hit.get('_id') for hit in top_k_hits]
    
    # Get the relevant resumes for this job
    relevant_ids = []
    for i, satisfied in enumerate(labels[job_id]['satisfied']):
        if satisfied == 1.0:  # Use 1.0 for "interviewed" status
            relevant_ids.append(labels[job_id]['user_ids'][i])
    
    # If no relevant documents, return zeros
    if not relevant_ids:
        return {
            "precision": 0.0,
            "recall": 0.0,
            "f1": 0.0
        }
    
    # Calculate true positives (resumes that are both relevant and retrieved)
    true_positives = len(set(top_k_ids).intersection(set(relevant_ids)))
    
    # Calculate precision, recall, and F1
    precision = true_positives / len(top_k_ids) if top_k_ids else 0.0
    recall = true_positives / len(relevant_ids) if relevant_ids else 0.0
    
    if precision + recall > 0:
        f1 = 2 * (precision * recall) / (precision + recall)
    else:
        f1 = 0.0
    
    return {
        "precision": precision,
        "recall": recall,
        "f1": f1
    }

def calculate_map1(search_results: dict, job_id: str, labels: dict) -> float:
    """Calculate MAP considering only documents with satisfied == 1.0"""
    if not search_results or 'hits' not in search_results or 'hits' not in search_results['hits']:
        return 0.0
    
    if job_id not in labels:
        return 0.0
    
    hits = search_results['hits']['hits']

    result_ids = [hit.get('_id') for hit in hits]
    
    # Get the relevant resumes for this job
    relevant_ids = []
    for i, satisfied in enumerate(labels[job_id]['satisfied']):
        if satisfied == 1.0:  # Use 1.0 for "interviewed" status
            relevant_ids.append(labels[job_id]['user_ids'][i])
    
    # If no relevant documents, return zero
    if not relevant_ids:
        return 0.0
    
    # Calculate average precision
    precision_sum = 0.0
    relevant_count = 0
    
    for i, doc_id in enumerate(result_ids):
        if doc_id in relevant_ids:
            relevant_count += 1
            precision_at_i = relevant_count / (i + 1)
            precision_sum += precision_at_i
    
    # Normalize by the total number of relevant documents
    if len(relevant_ids) > 0:
        return precision_sum / len(relevant_ids)
    else:
        return 0.0

def calculate_map2(search_results: dict, job_id: str, labels: dict) -> float:
    """Calculate MAP considering documents with satisfied == 1.0 OR satisfied == 0.5"""
    if not search_results or 'hits' not in search_results or 'hits' not in search_results['hits']:
        return 0.0
    
    if job_id not in labels:
        return 0.0
    
    hits = search_results['hits']['hits']

    result_ids = [hit.get('_id') for hit in hits]
    
    # Get the relevant resumes for this job (satisfied == 1.0 OR satisfied == 0.5)
    relevant_ids = []
    for i, satisfied in enumerate(labels[job_id]['satisfied']):
        if satisfied == 1.0 or satisfied == 0.5:
            relevant_ids.append(labels[job_id]['user_ids'][i])
    
    # If no relevant documents, return zero
    if not relevant_ids:
        return 0.0
    
    # Calculate average precision
    precision_sum = 0.0
    relevant_count = 0
    
    for i, doc_id in enumerate(result_ids):
        if doc_id in relevant_ids:
            relevant_count += 1
            precision_at_i = relevant_count / (i + 1)
            precision_sum += precision_at_i
    
    # Normalize by the total number of relevant documents
    if len(relevant_ids) > 0:
        return precision_sum / len(relevant_ids)
    else:
        return 0.0

def calculate_mrr1(search_results: dict, job_id: str, labels: dict) -> float:
    """Calculate MRR considering only documents with satisfied == 1.0"""
    if not search_results or 'hits' not in search_results or 'hits' not in search_results['hits']:
        return 0.0
    
    if job_id not in labels:
        return 0.0

    hits = search_results['hits']['hits']

    result_ids = [hit.get('_id') for hit in hits]
    
    # Get the relevant resumes for this job
    relevant_ids = []
    for i, satisfied in enumerate(labels[job_id]['satisfied']):
        if satisfied == 1.0:  # Use 1.0 for "interviewed" status
            relevant_ids.append(labels[job_id]['user_ids'][i])
    
    # If no relevant documents, return zero
    if not relevant_ids:
        return 0.0
    
    # Find the rank of the first relevant document
    for i, doc_id in enumerate(result_ids):
        if doc_id in relevant_ids:
            return 1.0 / (i + 1)
    
    # If no relevant document is found
    return 0.0

def calculate_mrr2(search_results: dict, job_id: str, labels: dict) -> float:
    """Calculate MRR considering documents with satisfied == 1.0 OR satisfied == 0.5"""
    if not search_results or 'hits' not in search_results or 'hits' not in search_results['hits']:
        return 0.0
    
    if job_id not in labels:
        return 0.0

    hits = search_results['hits']['hits']

    result_ids = [hit.get('_id') for hit in hits]
    
    # Get the relevant resumes for this job (satisfied == 1.0 OR satisfied == 0.5)
    relevant_ids = []
    for i, satisfied in enumerate(labels[job_id]['satisfied']):
        if satisfied == 1.0 or satisfied == 0.5:
            relevant_ids.append(labels[job_id]['user_ids'][i])
    
    # If no relevant documents, return zero
    if not relevant_ids:
        return 0.0
    
    # Find the rank of the first relevant document
    for i, doc_id in enumerate(result_ids):
        if doc_id in relevant_ids:
            return 1.0 / (i + 1)
    
    # If no relevant document is found
    return 0.0


def calculate_recall_at_all1(search_results: dict, job_id: str, labels: dict) -> float:
    """Calculate recall over all search results, only considering satisfied == 1.0."""
    if not search_results or 'hits' not in search_results or 'hits' not in search_results['hits']:
        return 0.0
    
    if job_id not in labels:
        return 0.0
    
    hits = search_results['hits']['hits']
    
    if not hits:
        return 0.0
    
    all_result_ids = [hit.get('_id') for hit in hits]
    
    # Get the relevant resumes for this job
    relevant_ids = []
    for i, satisfied in enumerate(labels[job_id]['satisfied']):
        if satisfied == 1.0:  # Use 1.0 for "interviewed" status
            relevant_ids.append(labels[job_id]['user_ids'][i])
    
    # If no relevant documents, return zero
    if not relevant_ids:
        return 0.0
    
    # Calculate recall at all (how many relevant docs are in the entire result set)
    found_relevant = len(set(all_result_ids).intersection(set(relevant_ids)))
    total_relevant = len(relevant_ids)
    
    recall_at_all = found_relevant / total_relevant if total_relevant > 0 else 0.0
    
    return recall_at_all

def calculate_recall_at_all2(search_results: dict, job_id: str, labels: dict) -> float:
    """Calculate recall over all search results, considering satisfied == 1.0 OR satisfied == 0.5."""
    if not search_results or 'hits' not in search_results or 'hits' not in search_results['hits']:
        return 0.0
    
    if job_id not in labels:
        return 0.0
    
    hits = search_results['hits']['hits']
    
    if not hits:
        return 0.0
    
    all_result_ids = [hit.get('_id') for hit in hits]
    
    # Get the relevant resumes for this job (satisfied == 1.0 OR satisfied == 0.5)
    relevant_ids = []
    for i, satisfied in enumerate(labels[job_id]['satisfied']):
        if satisfied == 1.0 or satisfied == 0.5:
            relevant_ids.append(labels[job_id]['user_ids'][i])
    
    # If no relevant documents, return zero
    if not relevant_ids:
        return 0.0
    
    # Calculate recall at all (how many relevant docs are in the entire result set)
    found_relevant = len(set(all_result_ids).intersection(set(relevant_ids)))
    total_relevant = len(relevant_ids)
    
    recall_at_all = found_relevant / total_relevant if total_relevant > 0 else 0.0
    
    return recall_at_all

def calculate_ndcg_at_k(search_results: dict, job_id: str, labels: dict, 
                       k: int = None, total_doc_count: int = 3112, 
                       use_alternative_relevance: bool = False) -> float:

    all_result_ids = [hit.get('_id') for hit in search_results['hits']['hits']]

    result_ids = all_result_ids
    if k is not None:
        result_ids = all_result_ids[:min(k, len(all_result_ids))]

    position_map = {doc_id: pos + 1 for pos, doc_id in enumerate(all_result_ids)}

    relevance_map = {}
    
    for i, user_id in enumerate(labels[job_id]['user_ids']):
        satisfied_value = labels[job_id]['satisfied'][i]

        if use_alternative_relevance:
            if satisfied_value == 1.0:
                relevance_map[user_id] = 1 
            elif satisfied_value == 0.5:
                relevance_map[user_id] = 0.5 
            else:
                relevance_map[user_id] = 0 
        else:
            if satisfied_value == 1.0:
                relevance_map[user_id] = 1 
            else:
                relevance_map[user_id] = 0 
    
    dcg = 0.0
    
    for doc_id in result_ids:
        relevance = relevance_map.get(doc_id, 0)
        position = position_map[doc_id]
        dcg += relevance / np.log2(position + 1)
    
    sorted_relevance = sorted([relevance_map.get(doc_id, 0) for doc_id in labels[job_id]['user_ids'] 
                               if relevance_map.get(doc_id, 0) > 0], reverse=True)
    
    if k is not None:
        sorted_relevance = sorted_relevance[:min(k, len(sorted_relevance))]
    
    idcg = 0.0
    for i, relevance in enumerate(sorted_relevance):
        position = i + 1
        idcg += relevance / np.log2(position + 1)
    
    if idcg > 0:
        ndcg = dcg / idcg
    else:
        ndcg = 0.0
    
    return ndcg


def calculate_metrics(search_result: dict, job_id: str, labels: dict, 
                      bm25_k_values: list, eval_k_values: list) -> dict:
    metrics_data = {
        # "bm25_scores": {},
        "precision": {},
        "recall": {},
        # "f1": {},
        "ndcg1": {},
        # "ndcg2": {}
    }
    
    # # Calculate BM25 scores
    # for k in bm25_k_values:
    #     k_key = f"top{k}"
    #     metrics_data["bm25_scores"][k_key] = calculate_top_k_avg_bm25(search_result, k)
    
    # Calculate precision, recall, F1 for each k value
    for k in eval_k_values:
        k_key = f"top{k}"
        
        # Calculate precision, recall, F1
        precision_recall_f1 = calculate_precision_recall_f1(search_result, job_id, labels, k)
        metrics_data["precision"][k_key] = precision_recall_f1["precision"]
        metrics_data["recall"][k_key] = precision_recall_f1["recall"]
        # metrics_data["f1"][k_key] = precision_recall_f1["f1"]
        ndcg_key = f"@{k}"
        metrics_data["ndcg1"][ndcg_key] = calculate_ndcg_at_k(
            search_result, job_id, labels, k=k, use_alternative_relevance=False
        )
        # metrics_data["ndcg2"][ndcg_key] = calculate_ndcg_at_k(
        #     search_result, job_id, labels, k=k, use_alternative_relevance=True
        # )
    
    metrics_data["map1"] = calculate_map1(search_result, job_id, labels)
    metrics_data["mrr1"] = calculate_mrr1(search_result, job_id, labels)
    
    # metrics_data["map2"] = calculate_map2(search_result, job_id, labels)
    # metrics_data["mrr2"] = calculate_mrr2(search_result, job_id, labels)
    
    metrics_data["recall_at_all1"] = calculate_recall_at_all1(search_result, job_id, labels)
    # metrics_data["recall_at_all2"] = calculate_recall_at_all2(search_result, job_id, labels)
    
    metrics_data["recallmap1"] = metrics_data["map1"] * metrics_data["recall_at_all1"]
    # metrics_data["recallmap2"] = metrics_data["map2"] * metrics_data["recall_at_all2"]
    
    metrics_data["ndcg1@all"] = calculate_ndcg_at_k(
        search_result, job_id, labels, k=None, use_alternative_relevance=False
    )
    # metrics_data["ndcg2@all"] = calculate_ndcg_at_k(
    #     search_result, job_id, labels, k=None, use_alternative_relevance=True
    # )
    
    return metrics_data