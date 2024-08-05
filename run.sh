OIE_LLM=meta-llama/Meta-Llama-3-8B-Instruct
SD_LLM=mistralai/Mistral-7B-Instruct-v0.2
SC_LLM=mistralai/Mistral-7B-Instruct-v0.3
DATASET=example

python run.py \
    --oie_llm $OIE_LLM \
    --oie_few_shot_example_file_path "./few_shot_examples/${DATASET}/oie_few_shot_examples.txt" \
    --sd_llm $SD_LLM \
    --sd_few_shot_example_file_path "./few_shot_examples/${DATASET}/sd_few_shot_examples.txt" \
    --sc_llm $SC_LLM \
    --input_text_file_path "./datasets/${DATASET}.txt" \
    --target_schema_path "./schemas/${DATASET}_schema.csv" \
    --output_dir "./output/${DATASET}_target_alignment" \
    --logging_verbose