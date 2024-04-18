#!/bin/bash
python run.py \
    --oie_llm gpt-4 \
    --sd_llm gpt-3.5-turbo \
    --sc_llm gpt-3.5-turbo \
    --ee_llm gpt-3.5-turbo \
    --oie_few_shot_example_file_path ./edc/few_shot_examples/webnlg/oie_few_shot_examples.txt \
    --sd_few_shot_example_file_path ./edc/few_shot_examples/webnlg/sd_few_shot_examples.txt \
    --oie_refine_few_shot_example_file_path ./edc/few_shot_examples/webnlg/oie_few_shot_refine_examples.txt \
    --ee_few_shot_example_file_path ./edc/few_shot_examples/webnlg/ee_few_shot_examples.txt \
    --input_text_file_path ./edc/datasets/webnlg.txt \
    --target_schema_path ./edc/schemas/webnlg_schema.csv \
    --output_dir ./output/webnlg
