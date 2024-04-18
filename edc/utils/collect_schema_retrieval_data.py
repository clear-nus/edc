import llm_utils
import pandas as pd
import os
from tqdm import tqdm
import csv
import json
from datasets import load_dataset, Dataset, DatasetDict
import random

from collections import Counter


def read_tekgen(tekgen_path):
    json_dict_list = []
    with open(tekgen_path, "r") as f:
        lines = f.readlines()
        for l in tqdm(lines):
            line_json_dict = json.loads(l)
            triples = line_json_dict["triples"]
            skip_flag = False
            for triple in triples:
                # skip quadruples
                if len(triple) != 3:
                    skip_flag = True
            if not skip_flag:
                json_dict_list.append(line_json_dict)
    return json_dict_list


def crawl_relation_definitions(result_csv_path):
    json_dict_list = []
    with open("../../datasets/tekgen/quadruples-test.tsv", "r") as f:
        lines = f.readlines()
        for l in tqdm(lines):
            line_json_dict = json.loads(l)
            json_dict_list.append(line_json_dict)

    collected_relations = set()

    if not os.path.exists(result_csv_path):
        result_csv = open(result_csv_path, "w")
        csv_writer = csv.writer(result_csv)
        csv_writer.writerow(["text", "triplets", "descriptions"])
    else:
        result_csv = open(result_csv_path, "a")
        csv_writer = csv.writer(result_csv)

    progress_bar = tqdm(total=5000)
    for json_dict in json_dict_list:
        if len(collected_relations) >= 5000:
            break
        triples = json_dict["triples"]
        skip_flag = False
        for triple in triples:
            # skip quadruples
            if len(triple) != 3:
                skip_flag = True
            relation = triple[1]
            if relation in collected_relations:
                # This is already collected, skip
                skip_flag = True
        if skip_flag:
            continue
        else:
            for triple in triples:
                relation = triple[1]
                if relation not in collected_relations:
                    collected_relations.add(relation)
                    progress_bar.update()
            text = json_dict["sentence"]
            triples = json_dict["triples"]

            filled_first_prompt = first_prompt_relation_webnlg.format(
                text=text, triples=triples
            )
            # print(filled_first_prompt)
            output = llm_utils.openai_chat_completion(
                "gpt-3.5-turbo",
                system_prompt=None,
                history=[{"role": "user", "content": filled_first_prompt}],
            )
            csv_writer.writerow([text, triples, output])
            result_csv.flush()


def collect_samples(entries, relation_definitions: dict):
    # entries: list of dicts containing text and triples
    # relation_definitions: dict from relation to definitions
    collected_samples = []

    for entry in tqdm(entries):
        text = entry["text"]
        triples = entry["triplets"]

        positive_relations = set()
        
        relation_triple_dict = {}

        for triple in triples:
            subject = triple[0]
            relation = triple[1]
            object = triple[2]
            
            # check if subject and object are present in text
            # if subject.lower() not in text.lower() or object.lower() not in text.lower():
            #     print(f"{triple} not explicitly in {text}")
            #     continue
            
            if relation in relation_definitions:
                positive_relations.add(relation)
                if relation not in relation_triple_dict:
                    relation_triple_dict[relation] = [triple]
                else:
                    relation_triple_dict[relation].append(triple)

        negative_relations = set(relation_definitions.keys()) - positive_relations

        negative_relations = random.sample(negative_relations, len(positive_relations))

        positive_relations = list(positive_relations)
        negative_relations = list(negative_relations)

        assert len(positive_relations) == len(negative_relations)

        for idx in range(len(negative_relations)):
            if idx >= 2:
                # Max 3 samples per sentence
                break
            sample = {
                "sentence": text,
                "positive": f"{positive_relations[idx]}: {relation_definitions[positive_relations[idx]]}",
                "negative": f"{negative_relations[idx]}: {relation_definitions[negative_relations[idx]]}",
                "positive_relation": positive_relations[idx],
                "negative_relation": negative_relations[idx],
                "positive_triple": relation_triple_dict[positive_relations[idx]]
            }
            # print(sample)
            collected_samples.append(sample)
            print(sample)
            if len(collected_samples) >= 50000:
                return collected_samples
    return collected_samples
    


if __name__ == "__main__":
    # entries = read_tekgen("../../datasets/tekgen/quadruples-test.tsv")
    
    entries = KGCDataset('rebel').entries
    
    relation_description_csv_path = (
        "./relation_descriptions/gpt3.5_relation_description_rebel_gt_new.csv"
    )

    # if not os.path.exists(relation_description_csv_path):
    #     crawl_relation_definitions(relation_description_csv_path)
    
    relation_description_dict, _ = llm_canonicalization_description.parse_relation_description_df(pd.read_csv(relation_description_csv_path), webnlg=False)
    
    for relation in relation_description_dict.keys():
        # if len(relation_description_dict[relation]) != 1:
        #     print(relation)
        #     print(relation_description_dict[relation])
        if len(relation_description_dict[relation]) != 1:
            descr = relation_description_dict[relation]      
            counter = Counter(descr)
            most_common_descr, freq = counter.most_common(1)[0]
            relation_description_dict[relation] = most_common_descr
        else:
            relation_description_dict[relation] = relation_description_dict[relation][0]

    collected_samples = collect_samples(entries, relation_description_dict)
    # print(collected_samples)
    print(len(collected_samples))
    
    data = Dataset.from_list(collected_samples)
    
    train_test_split = data.train_test_split()
    test_valid = train_test_split['test'].train_test_split(test_size=0.5)
    train_test_valid_dataset = DatasetDict({
        'train': train_test_split['train'],
        'test': test_valid['test'],
        'valid': test_valid['train']})

    print(train_test_valid_dataset)
    train_test_valid_dataset.save_to_disk("schema_dataset_sub-rebel-name/")