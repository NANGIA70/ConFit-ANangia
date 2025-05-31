set -x
all_labels_csv=dataset/all_labels_pos.csv
rank_resume_file=dataset/rank_resume.json
jd_file=dataset/all_jd_texts.csv

output_dir=./results
cache_dir=./cache
## use num_jds=10 for testing, and num_jds=50 for full evaluation
python test.py \
--jd_file $jd_file \
--rank_resume_file $rank_resume_file \
--all_labels_csv $all_labels_csv \
--prompt_template dataset/prompt_template.json \
--few_shot_file dataset/mapped_best_keywords.jsonl \
--output_dir $output_dir \
--cache_dir $cache_dir \
--model_provider local \
--model_name Qwen/Qwen2.5-7B-Instruct \
--max_fewshot_examples 0 \
--local_url http://adaptation.cs.columbia.edu:30055/v1 \
--es_index resume_0417_2025_small \
--num_jds 50 \
--eval_k_values 300,1000 \
--prompts_per_jd 1