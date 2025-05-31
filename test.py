import argparse
import json
import os
import re
import random
import pandas as pd
import diskcache
import itertools
from pydantic import BaseModel, Field
from typing import Optional
from elasticsearch import Elasticsearch
from openai import OpenAI
from metrics import calculate_metrics, calculate_query_length
from utils import save_final_results, save_intermediate_results
from dataloader import (
    load_job_descriptions,
    load_rank_resume,
    load_all_labels_csv,
    build_all_labels_dict
)


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Resume Search")
    
    parser.add_argument("--jd_file", type=str, required=True, help="Path to job descriptions CSV file")
    parser.add_argument("--rank_resume_file", type=str, 
                        help="Path to rank_resume.json file")
    parser.add_argument("--all_labels_csv", type=str, 
                        help="Path to all-labels.csv file")
    parser.add_argument("--prompt_template", type=str, default="dataset/prompt_template.json", help="Path to prompt template JSON file")
    parser.add_argument("--output_dir", type=str, default="results", help="Directory to save results")
    parser.add_argument("--model_provider", type=str, choices=["local"], required=True, 
                        help="LLM provider to use")
    parser.add_argument("--model_name", type=str, required=True, 
                        help="Model name (e.g., gpt-4o, claude-3-7-sonnet-latest)")
    parser.add_argument("--local_url",type=str,help="Local path that hosts the model")
    parser.add_argument("--num_jds", type=int, default=None, 
                        help="Number of job descriptions to process (None for all)")
    parser.add_argument("--prompts_per_jd", type=int, default=1, 
                        help="Number of prompts to generate per job description")
    parser.add_argument("--es_host", type=str, default="http://adaptation.cs.columbia.edu:55190", 
                        help="Elasticsearch host URL")
    parser.add_argument("--es_user", type=str, default="elastic", help="Elasticsearch username")
    parser.add_argument("--es_password", type=str, default="confitv3", help="Elasticsearch password")
    parser.add_argument("--es_index", type=str, default="resume_0303_2025", help="Elasticsearch index")
    parser.add_argument("--bm25_k_values", type=str, default="5,10", 
                        help="Comma-separated list of k values for BM25 metrics")
    parser.add_argument("--eval_k_values", type=str, default="10,30,50,100", 
                        help="Comma-separated list of k values for precision/recall/F1 metrics")
    parser.add_argument("--few_shot_file", type=str, default="dataset/mapped_best_keywords.jsonl",
                        help="Path to few-shot examples file (JSONL format)")
    parser.add_argument("--max_fewshot_examples", type=int, default=6, 
                        help="Maximum number of few-shot examples to include in prompt")
    parser.add_argument("--random_select", type=int, default=-1, 
                        help="Number of retreived results randomly selected")
    parser.add_argument("--cache_dir", type=str, default="./search_cache", 
                        help="directory of search cache")
    return parser.parse_args()


def save_all_labels_dict(all_labels_dict: dict, output_path: str):
    rows = []
    
    for job_id, data in all_labels_dict.items():
        for user_id, satisfied in zip(data["user_ids"], data["satisfied"]):
            rows.append({
                "job_id": job_id,
                "user_id": user_id,
                "satisfied": satisfied
            })
    
    df = pd.DataFrame(rows)
    df.to_csv(output_path, index=False)
    print(f"Saved all labels data to {output_path}")


def load_prompt_template(file_path: str) -> dict:
    try:
        with open(file_path, 'r') as f:
            template = json.load(f)
        return template
    except Exception as e:
        raise ValueError(f"Error loading prompt template: {str(e)}")


def load_few_shot_examples(file_path: str, max_examples: int) -> list[dict]:
    examples = []
    try:
        with open(file_path, 'r') as f:
            for line in f:
                example_json = json.loads(line.strip())
                if "query" in example_json and "llm_output" in example_json["query"]:
                    example = {
                        "jid": example_json.get("jd_no", ""),
                        "keyword_example": example_json["query"]["llm_output"]
                    }
                    examples.append(example)
                    if len(examples) >= max_examples:
                        break
        print(f"Loaded {len(examples)} few-shot examples from {file_path}")
        return examples
    except Exception as e:
        print(f"Error loading few-shot examples: {str(e)}")
        return []


def enrich_prompt_with_fewshot(template: dict, examples: list[dict], jd_df: pd.DataFrame) -> dict:
    if not examples:
        return template
    
    new_template = template.copy()
    
    output_fields_emphasis = (
        "Please ensure your output includes the following fields:\n"
        "- 'title': List of relevant job titles\n"
        "- 'location': List of relevant locations\n"
        "- 'industryKeywords': List of relevant industry keywords\n"
        "- 'jobFunctionKeywords': List of relevant job function keywords\n"
        "- 'languages': List of required languages\n"
        "- 'yearOfWork': Work experience range in years\n"
        "- 'skillBooleanString': List of boolean expressions combining skills with AND/OR operators. Create multiple expressions to cover different skill areas. Use AND when both skills are required together, and OR when any of the skills is acceptable.\n"
    )
    
    few_shot_text = f"\n\n{output_fields_emphasis}\n\nHere are some examples of well-formatted keyword queries:\n\n"
    
    for i, example in enumerate(examples):
        if "jid" in example and "keyword_example" in example:
            jid = example["jid"]
            matching_rows = jd_df[jd_df['jd_no'] == jid]
            job_text = None
            if not matching_rows.empty:
                job_text = matching_rows.iloc[0]['job_text']
            
            if job_text:
                few_shot_text += f"Example {i+1} Job Description:\n{job_text}\n\n"
                few_shot_text += f"Example {i+1} Keywords:\n{json.dumps(example['keyword_example'], indent=2)}\n\n"
            else:
                few_shot_text += f"Example {i+1}:\n{json.dumps(example['keyword_example'], indent=2)}\n\n"
    
    if "system_message" in new_template:
        new_template["system_message"] += few_shot_text
    
    return new_template


class YearOfWork(BaseModel):
    gte: Optional[int]
    lte: Optional[int]


class KeywordResult(BaseModel):
    title: list[str]
    location: list[str]
    industryKeywords: list[str]
    jobFunctionKeywords: list[str]
    languages: Optional[list[str]] = []
    yearOfWork: Optional[YearOfWork] = None
    skillBooleanString: list[str] = Field(
        # default_factory=list,
        description=(
            "Boolean expressions combining skills with AND/OR operators. Each expression represents a search query where terms with AND require both terms to be present, "
            "and terms with OR require any term to be present. Example: 'Russian AND Chinese OR Russian AND English' is interpreted as (Russian AND Chinese) OR (Russian AND English). "
            "Create multiple expressions to cover different skill areas."
        )
    )

def generate_search_keywords_openai(
    client: OpenAI,
    job_description: str,
    template: dict,
    model: str
) -> dict:
    """given an input job description, prompt an LLM to generate search queries to search on the ES index and find matching candidates

    Args:
        client (OpenAI): OpenAI client connected to the LLM
        job_description (str): input job description
        template (dict): prompt template
        model (str): LLM's model name

    Returns:
        dict: LLM generated search keywords
    """
    prompt = template["prompt_template"].replace("{job_description}", job_description)
    system_message = template.get(
        "system_message",
        "You are a helpful assistant that extracts relevant search keywords from job descriptions to find matching candidates."
    )
    ### START OF YOUR CODE
    # prompt the LLM to generate an ES search query, which can be used to search on the ES index
    # you should use the beta.chat.completions.parse API to generate the search query, with response_format set to the KeywordResult object above
    # for more details on structured output, please refer to https://docs.vllm.ai/en/latest/features/structured_outputs.html#experimental-automatic-parsing-openai-api

    try:
        response = client.beta.chat.completions.parse(
            model=model,
            messages=[
                {"role": "system", "content": system_message},
                {"role": "user", "content": prompt}
            ],
            response_format=KeywordResult,
            temperature=0.0,
            max_tokens=1024,
        )

        # Debugging
        # print(response.choices[0].message)
        # print(response.choices[0].message.content)
        # print(response.choices[0].message.parsed)

        parsed_choice: KeywordResult = response.choices[0].message.parsed

        parsed_response = parsed_choice.model_dump(mode="json")

        # Debugging
        # print(f"LLM reply (parsed) for JD:\n{json.dumps(parsed_response, indent=2)}")
        # print(f"type(parsed_response): {type(parsed_response)}")
    except Exception as e:
        try:
            print(f"Error generating keywords for job description: {str(e)}")
            fallback_prompt = (
                prompt 
                + "\n\nNOTE: Only output valid JSON matching the fields: 'title', 'location', 'industryKeywords', 'jobFunctionKeywords', 'languages', 'yearOfWork', 'skillBooleanString'."
            )
            response = client.beta.chat.completions.parse(
                model=model,
                messages=[
                    {"role": "system", "content": system_message},
                    {"role": "user",   "content": fallback_prompt}
                ],
                response_format=KeywordResult,
                temperature=0.0,
            )
            parsed_choice: KeywordResult = response.choices[0].parsed

            # 3) Convert the Pydantic model (KeywordResult) into a plain dict.
            parsed_response = parsed_choice.model_dump(mode="json")
        except Exception as e2:
            print(f"Fallback also failed: {str(e2)}")
            return {
                "title": [],
                "location": [],
                "industryKeywords": [],
                "jobFunctionKeywords": [],
                "languages": [],
                "yearOfWork": {"gte": None, "lte": None},
                "skillBooleanString": []
            }
        
    
    #### END OF YOUR CODE
    return parsed_response


def generate_keywords(
    job_id: str,
    job_description: str,
    model_provider: str,
    model_name: str,
    prompt_template: dict,
    openai_client: Optional[OpenAI] = None,
    num_prompts: int = 1,
) -> list[dict]:
    results = []
    keyword_dict = {}
    for i in range(num_prompts):
        if model_provider == "local":
            try:
                if not openai_client:
                    raise ValueError("Local OpenAI client not provided")
                keyword_dict = generate_search_keywords_openai(
                    openai_client, job_description, prompt_template, 
                    model_name
                )
            except Exception as e:
                print(f"Error generating keywords for job {job_id}, prompt {i+1}: {str(e)}")
        else:
            raise ValueError(f"Unsupported model provider: {model_provider}")
        
        results.append({
            "job_id": job_id,
            "prompt_num": i + 1,
            "keywords": keyword_dict
        })
    return results


def build_es_query_from_dict(keyword_dict: dict) -> dict:
    '''for finetuned model using search_train_data.json'''
    query = {
        "from": 0,
        "size": 10000,
        "query": {
            "bool": {
                "should": []
            }
        }
    }
    
    field_mapping = {
        "title": "titles",
        "location": "location",
        "industryKeywords": "industries",
        "jobFunctionKeywords": "jobFunctions",
        "languages": "languages",
        "yearOfWork": "yearOfWork"
    }
    
    def clean_term(term: str) -> str:
        term = term.replace("\\", "")
        term = term.strip()
        if term.startswith('('):
            term = term[1:]
        if term.endswith(')'):
            term = term[:-1]
        term = term.strip()
        if (term.startswith('"') and term.endswith('"')) or (term.startswith("'") and term.endswith("'")):
            term = term[1:-1]
        return term.strip().lower() 
    
    for model_field, value in keyword_dict.items():
        if model_field == "skillBooleanString":
            continue
            
        es_field = field_mapping.get(model_field)
        if es_field is None:
            continue
        
        if model_field == "yearOfWork":
            if isinstance(value, str):
                value = value.strip().strip("[]")
                parts = re.split(r'(?i)\s+to\s+', value.strip())
                range_query = {"range": {es_field: {}}}
                if len(parts) == 2:
                    min_years = parts[0].strip()
                    max_years = parts[1].strip()
                    
                    if min_years != "" and min_years.isdigit():
                        range_query["range"][es_field]["gte"] = int(min_years)
                    
                    if max_years != "" and max_years.isdigit():
                        range_query["range"][es_field]["lte"] = int(max_years)
                
                if range_query["range"][es_field] != {}:
                    query["query"]["bool"]["filter"] = range_query
            elif isinstance(value, dict):
                range_query = {"range": {es_field: {}}}
                if "gte" in value and value["gte"] is not None:
                    range_query["range"][es_field]["gte"] = value["gte"]
                if "lte" in value and value["lte"] is not None:
                    range_query["range"][es_field]["lte"] = value["lte"]
                
                if range_query["range"][es_field] != {}:
                    query["query"]["bool"]["filter"] = range_query
        
        elif isinstance(value, list) and value:
            or_clauses = []
            for item in value:
                if isinstance(item, str) and item.strip():
                    cleaned_term = clean_term(item)
                    if cleaned_term:
                        or_clauses.append({"match": {es_field: cleaned_term}})
            
            if or_clauses:
                query["query"]["bool"]["should"].append({
                    "bool": {
                        "should": or_clauses,
                        "minimum_should_match": 1
                    }
                })
        
        elif isinstance(value, str) and value.strip():
            query["query"]["bool"]["should"].append({"match": {es_field: clean_term(value)}})
    
    if "skillBooleanString" in keyword_dict and isinstance(keyword_dict["skillBooleanString"], list):
        search_content_clauses = []
        
        for boolean_expr in keyword_dict["skillBooleanString"]:
            if not boolean_expr or not isinstance(boolean_expr, str):
                continue
                
            def parse_boolean_expr(expr: str):
                if " AND " in expr.upper():
                    and_parts = re.split(r'\s+(?i:AND)\s+', expr)
                    and_clauses = []
                    
                    for part in and_parts:
                        cleaned_part = clean_term(part)
                        if cleaned_part:
                            and_clauses.append({"match": {"searchContent": cleaned_part}})
                    
                    if and_clauses:
                        return {
                            "bool": {
                                "must": and_clauses
                            }
                        }
                elif " OR " in expr.upper():
                    or_parts = re.split(r'\s+(?i:OR)\s+', expr)
                    or_clauses = []
                    
                    for part in or_parts:
                        cleaned_part = clean_term(part)
                        if cleaned_part:
                            or_clauses.append({"match": {"searchContent": cleaned_part}})
                    
                    if or_clauses:
                        return {
                            "bool": {
                                "should": or_clauses,
                                "minimum_should_match": 1
                            }
                        }
                else:
                    cleaned_expr = clean_term(expr)
                    if cleaned_expr:
                        return {"match": {"searchContent": cleaned_expr}}
                
                return None
            
            def analyze_complex_expr(expr: str):
                if " AND " in expr.upper() and " OR " in expr.upper():
                    or_blocks = re.split(r'\s+(?i:OR)\s+', expr)
                    or_clauses = []
                    
                    for block in or_blocks:
                        if " AND " in block.upper():
                            and_parts = re.split(r'\s+(?i:AND)\s+', block)
                            and_clauses = []
                            
                            for part in and_parts:
                                cleaned_part = clean_term(part)
                                if cleaned_part:
                                    and_clauses.append({"match": {"searchContent": cleaned_part}})
                            
                            if and_clauses:
                                or_clauses.append({
                                    "bool": {
                                        "must": and_clauses
                                    }
                                })
                        else:
                            cleaned_block = clean_term(block)
                            if cleaned_block:
                                or_clauses.append({"match": {"searchContent": cleaned_block}})
                    
                    if or_clauses:
                        return {
                            "bool": {
                                "should": or_clauses,
                                "minimum_should_match": 1
                            }
                        }
                else:
                    return parse_boolean_expr(expr)
                
                return None
            
            parsed_expr = analyze_complex_expr(boolean_expr)
            if parsed_expr:
                search_content_clauses.append(parsed_expr)
        
        if search_content_clauses:
            query["query"]["bool"]["should"].append({
                "bool": {
                    "should": search_content_clauses,
                    "minimum_should_match": 1,
                    # "boost": 2.0  
                }
            })
    
    if len(query["query"]["bool"]["should"]) == 0:
        print("Warning: No valid query conditions were created from keywords")
    return query


def run_elasticsearch_query(
    es_client: Elasticsearch,
    query: dict,
    cache_dir: diskcache.Cache,
    index: str = "resume_0417_2025_small",
    random_select: int = -1
) -> dict:
    """Run Elasticsearch query and cache the result (for efficiency)

    Args:
        es_client (Elasticsearch): Elasticsearch client
        query (dict): Elasticsearch query generated by an LLM
        cache_dir (diskcache.Cache): Cache directory
        index (str, optional): Elasticsearch index to search on. Defaults to "resume_0417_2025_small".
        random_select (int, optional): ignore this for now 

    Returns:
        dict: Elasticsearch search result
    """
    key= (json.dumps(query),index,random_select)
    if key in cache_dir:
        return cache_dir[key]
    
    #### START OF YOUR CODE
    # use the es_client to search on the index with the generated query
    response = es_client.search(index=index, body=query)
    #### END OF YOUR CODE
    
    if not hasattr(response, 'body'):
        return {'hits': {'total': {'value': 0}, 'hits': []}}
    else:
        search_result = response.body
        cache_dir[key] = search_result
    return search_result
    


def main():
    """Main function to run the pipeline."""
    args = parse_arguments()
    search_cache = diskcache.Cache(args.cache_dir)
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    #### 1. load data and prepare inference endpoint (e.g., few shot examples)
    api_key = "sk-xxx"
    openai_client = OpenAI(base_url=args.local_url,api_key=api_key,)
    
    es_client = Elasticsearch(
        args.es_host,
        basic_auth=(args.es_user, args.es_password)
    )
    jd_df = load_job_descriptions(args.jd_file)
    
    if args.rank_resume_file and args.all_labels_csv:
        print("Using new data format with rank_resume and all_labels_csv...")
        rank_resume = load_rank_resume(args.rank_resume_file)
        all_labels_df = load_all_labels_csv(args.all_labels_csv)
        labels = build_all_labels_dict(rank_resume, all_labels_df)
    
    prompt_template = load_prompt_template(args.prompt_template)
    
    # few shot examples
    few_shot_examples = []
    if args.few_shot_file:
        few_shot_examples = load_few_shot_examples(args.few_shot_file, args.max_fewshot_examples)
        if few_shot_examples:
            prompt_template = enrich_prompt_with_fewshot(prompt_template, few_shot_examples, jd_df)
    
    # metrics
    bm25_k_values = [int(k) for k in args.bm25_k_values.split(",")]
    eval_k_values = [int(k) for k in args.eval_k_values.split(",")]
    
    jd_ids = list(labels.keys())
    if args.num_jds is not None:
        if args.num_jds > len(jd_ids):
            print(f"Warning: Requested {args.num_jds} jobs but only {len(jd_ids)} are available. Using all available jobs.")
            args.num_jds = len(jd_ids)
        jd_ids = jd_ids[:args.num_jds]
    
    print(f"Searcing {len(jd_ids)} job descriptions...")
    print(f"Using model: {args.model_provider}/{args.model_name}")
    print(f"Few-shot examples: {len(few_shot_examples)}")

    ##### 2. search against ES database
    all_keyword_results = []
    all_query_results = []
    all_search_results = []
    all_metrics = []
    
    for job_id in jd_ids:
        print(f"Processing job ID: {job_id}")
        job_text = None
        
        matching_rows = jd_df[jd_df['jd_no'] == job_id]
        
        if not matching_rows.empty:
            job_text = matching_rows.iloc[0]['job_text']
        
        if not job_text:
            print(f"No job description found for ID {job_id}, skipping...")
            continue
        
        ## 2.1 prompt the LLM to generate an ES search query
        keyword_results = generate_keywords(
            job_id=job_id,
            job_description=job_text,
            model_provider=args.model_provider,
            model_name=args.model_name,
            prompt_template=prompt_template,
            openai_client=openai_client,
            num_prompts=args.prompts_per_jd,
        )
        
        
        assert len(keyword_results) == 1
        keyword_result = keyword_results[0]

        #### START OF YOUR CODE
        # 2.2. parse LLM output (keyword_results['keywords']) to executable ES query using build_es_query_from_dict
        # 2.3. execute the ES query using run_elasticsearch_query
        query = build_es_query_from_dict(keyword_result["keywords"])
        search_result = run_elasticsearch_query(
            es_client,
            query,
            search_cache,
            index=args.es_index,
            random_select=args.random_select
        )
        #### END OF YOUR CODE
        
        ### 2.4. calculate performance metrics
        query_length = calculate_query_length(keyword_result["keywords"]) if len(keyword_result["keywords"]) > 0 else 0
        query_result = {
            "job_id": job_id,
            "prompt_num": keyword_result["prompt_num"],
            "query": query
        }
        all_query_results.append(query_result)
        all_keyword_results.extend(keyword_results)
        
        if query_length == 0:
            print(f"Skipping due to empty query...")
            continue
        
        if search_result:
            search_data = {
                "job_id": job_id,
                "prompt_num": keyword_result["prompt_num"],
                "search_result": search_result
            }
            
            all_search_results.append(search_data)
            
            metrics = calculate_metrics(search_result, job_id, labels, bm25_k_values, eval_k_values)
            
            metrics_data = {
                "job_id": job_id,
                "prompt_num": keyword_result["prompt_num"],
                "model_provider": args.model_provider,
                "model_name": args.model_name,
                "hits_count": search_result['hits']['total']['value'],
                "query_length": query_length,
                **metrics
            }
            
            all_metrics.append(metrics_data)
        
        if job_id == jd_ids[0] or (job_id == jd_ids[-1]) or (jd_ids.index(job_id) % 10 == 0):
            save_intermediate_results(
                args.output_dir,
                all_keyword_results,
                all_query_results,
                all_search_results,
                all_metrics
            )

    save_final_results(
        args.output_dir,
        all_metrics,
        args.model_provider,
        args.model_name,
        bm25_k_values,
        eval_k_values
    )
    print(f"Evaluation complete. Results saved to {args.output_dir}")
    return


if __name__ == "__main__":
    main()