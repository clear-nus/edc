import numpy as np
from transformers import T5ForConditionalGeneration, T5Tokenizer
import torch
import pandas as pd
import csv
from typing import List
import os
import openai
import time
from packaging import version
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer
from edc.utils.e5_mistral_utils import MistralForSequenceEmbedding

if version.parse(transformers.__version__) >= version.parse("4.30"):
    from transformers import LlamaForCausalLM, LlamaTokenizer
import json
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
import random
import ast
import requests
from torch import Tensor


def last_token_pool(last_hidden_states: Tensor,
                 attention_mask: Tensor) -> Tensor:
    left_padding = (attention_mask[:, -1].sum() == attention_mask.shape[0])
    if left_padding:
        return last_hidden_states[:, -1]
    else:
        sequence_lengths = attention_mask.sum(dim=1) - 1
        batch_size = last_hidden_states.shape[0]
        return last_hidden_states[torch.arange(batch_size, device=last_hidden_states.device), sequence_lengths]

def get_detailed_instruct(task_description: str, query: str) -> str:
    return f'Instruct: {task_description}\nQuery: {query}'

def get_embedding_e5mistral(model, tokenizer, sentence, task=None):
    model.eval()
    device = model.device
    
    if task != None:
        # It's a query to be embed
        sentence = get_detailed_instruct(task, sentence)
    
    sentence = [sentence]

    max_length = 4096
    # Tokenize the input texts
    batch_dict = tokenizer(sentence, max_length=max_length - 1, return_attention_mask=False, padding=False, truncation=True)
    # append eos_token_id to every input_ids
    batch_dict['input_ids'] = [input_ids + [tokenizer.eos_token_id] for input_ids in batch_dict['input_ids']]
    batch_dict = tokenizer.pad(batch_dict, padding=True, return_attention_mask=True, return_tensors='pt')
    
    batch_dict.to(device)

    embeddings = model(**batch_dict).detach().cpu()
        
    assert len(embeddings) == 1
    
    return embeddings[0]

def parse_raw_triplets(raw_triplets: str):
    # Look for enclosing brackets
    unmatched_left_bracket_indices = []
    matched_bracket_pairs = []
    
    collected_triples = []
    for c_idx, c in enumerate(raw_triplets):
        if c == '[':
            unmatched_left_bracket_indices.append(c_idx)
        if c == ']':
            if len(unmatched_left_bracket_indices) == 0:
                continue
            # Found a right bracket, match to the last found left bracket
            matched_left_bracket_idx = unmatched_left_bracket_indices.pop()
            matched_bracket_pairs.append((matched_left_bracket_idx, c_idx))
    for (l, r) in matched_bracket_pairs:
        bracketed_str = raw_triplets[l:r+1]
        try:
            parsed_triple = ast.literal_eval(bracketed_str)
            if len(parsed_triple) == 3 and all([isinstance(t, str) for t in parsed_triple]):
                if all([e != '' and e != '_' for e in parsed_triple]):
                    collected_triples.append(parsed_triple)
            elif not all([type(x)==type(parsed_triple[0]) for x in parsed_triple]):
                for e_idx, e in enumerate(parsed_triple):
                    if isinstance(e, list):
                        parsed_triple[e_idx] = ', '.join(e)
                # print(parsed_triple)
                collected_triples.append(parsed_triple)
        except Exception as e:
            print(raw_triplets)
            print(str(e))
            print('ERROR!')
            pass
    return collected_triples

def parse_relation_definition(raw_definitions: str):
    descriptions = raw_definitions.split("\n")
    relation_definition_dict = {}
    
    for description in descriptions:
            if ":" not in description:
                continue
            index_of_colon = description.index(":")
            relation = description[:index_of_colon].strip()
            
            relation_description = description[index_of_colon + 1 :].strip()
            
            if relation == "Answer":
                continue
            
            
            
            if relation not in relation_definition_dict:
                relation_definition_dict[relation] = [relation_description]
            else:
                relation_definition_dict[relation].append(relation_description)
    return relation_definition_dict


def tokenize_input(tokenizer, input: list, remove_eos=False, add_special_tokens=True):
    if isinstance(input, str):
        input = [input]
    tokenized_input = tokenizer(
        input, return_tensors="pt", padding=True, add_special_tokens=add_special_tokens
    ).input_ids
    return tokenized_input


def transformer_completion_remote(url, message):
    try:
        response = requests.post(url, json=message, timeout=300)

        # Check if the request was successful (status code 200)
        if response.status_code == 200:
            print(f"Success! Response from server: {response.text}")
        else:
            print(f"Request failed with status code: {response.status_code}")
        return response.text
    except requests.RequestException as e:
        print(f"Request failed: {e}")


def is_model_openai(model_name):
    return "gpt" in model_name


def load_model(model_name, output_attentions=False):
    if model_name == "mistral7b":
        model = AutoModelForCausalLM.from_pretrained("mistralai/Mistral-7B-Instruct-v0.2", device_map="auto")
        tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-Instruct-v0.2", device_map="auto")
    elif model_name == "llama7B":
        tokenizer = LlamaTokenizer.from_pretrained(os.path.expanduser("~/llama_weights_hf/7B"), device_map="auto")
        model = LlamaForCausalLM.from_pretrained(
            os.path.expanduser("~/llama_weights_hf/7B"), device_map="auto", output_hidden_states=True
        )
    elif model_name == "llama13B":
        tokenizer = LlamaTokenizer.from_pretrained(os.path.expanduser("~/llama_weights_hf/13B"), device_map="auto")
        model = LlamaForCausalLM.from_pretrained(
            os.path.expanduser("~/llama_weights_hf/13B"), device_map="auto", output_hidden_states=True
        )
    elif model_name == "llama30B":
        tokenizer = LlamaTokenizer.from_pretrained(os.path.expanduser("~/llama_weights_hf/30B"), device_map="auto")
        model = LlamaForCausalLM.from_pretrained(
            os.path.expanduser("~/llama_weights_hf/30B"), device_map="auto", output_hidden_states=True
        )
    elif model_name == "llama65B":
        tokenizer = LlamaTokenizer.from_pretrained(os.path.expanduser("~/llama_weights_hf/65B"), device_map="auto")
        model = LlamaForCausalLM.from_pretrained(
            os.path.expanduser("~/llama_weights_hf/65B"), device_map="auto", output_hidden_states=True
        )
    elif model_name == "llama2-7B":
        tokenizer = LlamaTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf")
        model = LlamaForCausalLM.from_pretrained(
            "meta-llama/Llama-2-7b-hf", device_map="auto", output_hidden_states=True
        )
    elif model_name == "llama2-7B-chat":
        tokenizer = LlamaTokenizer.from_pretrained("meta-llama/Llama-2-7b-chat-hf")
        model = LlamaForCausalLM.from_pretrained(
            "meta-llama/Llama-2-7b-chat-hf", device_map="auto", output_hidden_states=True
        )
    elif model_name == "llama2-13B":
        tokenizer = LlamaTokenizer.from_pretrained("meta-llama/Llama-2-13b-hf")
        model = LlamaForCausalLM.from_pretrained(
            "meta-llama/Llama-2-13b-hf", device_map="auto", output_hidden_states=True
        )
    elif model_name == "llama2-13B-chat":
        tokenizer = LlamaTokenizer.from_pretrained("meta-llama/Llama-2-13b-chat-hf")
        model = LlamaForCausalLM.from_pretrained(
            "meta-llama/Llama-2-13b-chat-hf", device_map="auto", output_hidden_states=True
        )
    elif model_name == "llama2-70B":
        tokenizer = LlamaTokenizer.from_pretrained("meta-llama/Llama-2-70b-hf")
        model = LlamaForCausalLM.from_pretrained(
            "meta-llama/Llama-2-70b-hf", device_map="auto", output_hidden_states=True
        )
    elif model_name == "llama2-70B-chat":
        tokenizer = LlamaTokenizer.from_pretrained("meta-llama/Llama-2-70b-chat-hf")
        model = LlamaForCausalLM.from_pretrained(
            "meta-llama/Llama-2-70b-chat-hf", device_map="auto", output_hidden_states=True
        )
    elif model_name == "gpt2":
        model = AutoModelForCausalLM.from_pretrained("gpt2", device_map="auto", output_hidden_states=True)
        tokenizer = AutoTokenizer.from_pretrained("gpt2")
    elif model_name == "gpt-neo":
        model = AutoModelForCausalLM.from_pretrained(
            "EleutherAI/gpt-neo-2.7B", device_map="auto", output_hidden_states=True
        )
        tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-neo-2.7B")
    elif model_name == "gpt-neox":
        model = AutoModelForCausalLM.from_pretrained(
            "EleutherAI/gpt-neox-20b", device_map="auto", output_hidden_states=True
        )
        tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-neox-20b")
    elif model_name == "t5":
        model = T5ForConditionalGeneration.from_pretrained("google/flan-t5-xxl", device_map="auto")
        tokenizer = T5Tokenizer.from_pretrained("google/flan-t5-xxl")
    elif model_name == "falcon7B":
        model = AutoModelForCausalLM.from_pretrained(
            os.path.expanduser("tiiuae/falcon-7b"), device_map="auto", output_hidden_states=True, trust_remote_code=True
        )
        tokenizer = AutoTokenizer.from_pretrained("tiiuae/falcon-7b")
    elif model_name == "falcon7B-I":
        model = AutoModelForCausalLM.from_pretrained(
            os.path.expanduser("tiiuae/falcon-7b-instruct"),
            device_map="auto",
            output_hidden_states=True,
            trust_remote_code=True,
        )
        tokenizer = AutoTokenizer.from_pretrained("tiiuae/falcon-7b-instruct")
    elif model_name == "falcon40B":
        model = AutoModelForCausalLM.from_pretrained(
            os.path.expanduser("tiiuae/falcon-40b"),
            device_map="auto",
            output_hidden_states=True,
            trust_remote_code=True,
        )
        tokenizer = AutoTokenizer.from_pretrained("tiiuae/falcon-40b")
    elif model_name == "falcon40B-I":
        model = AutoModelForCausalLM.from_pretrained(
            os.path.expanduser("tiiuae/falcon-40b-instruct"),
            device_map="auto",
            output_hidden_states=True,
            trust_remote_code=True,
        )
        tokenizer = AutoTokenizer.from_pretrained("tiiuae/falcon-40b-instruct")
    elif model_name == "flan_t5":
        model = T5ForConditionalGeneration.from_pretrained(
            "google/flan-t5-xxl", output_attentions=output_attentions, device_map="auto"
        )
        tokenizer = T5Tokenizer.from_pretrained("google/flan-t5-xxl")
    else:
        raise Exception(f"Invalid model name {model_name} provided!")
    tokenizer.pad_token = tokenizer.eos_token
    for param in model.parameters():
        param.requires_grad = False
    return model, tokenizer


def generate_completion_transformers(
    input: list, model: AutoModelForCausalLM, tokenizer: AutoTokenizer, device, batch_size=1, max_new_token=256, answer_prepend="",
):
    completions = []
    if isinstance(input, str):
        input = [input]
    for i in range(0, len(input), batch_size):
        batch = input[i : i + batch_size]
        model_inputs = [tokenizer.apply_chat_template(entry, add_generation_prompt=True, tokenize=False) + answer_prepend for entry in batch]
        print(model_inputs)
        model_inputs = tokenizer(model_inputs, return_tensors="pt", padding=True, add_special_tokens=False).to(device)

        generated_ids = model.generate(**model_inputs, max_new_tokens=max_new_token, do_sample=False, pad_token_id=tokenizer.eos_token_id)[:, model_inputs['input_ids'].shape[1]:]
        generated_texts = tokenizer.batch_decode(generated_ids, skip_special_tokens=False)
        completions += generated_texts
    return completions

def generate_completion_transformers_no_batch(
    input: list, model: AutoModelForCausalLM, tokenizer: AutoTokenizer, device, batch_size=1, max_new_token=256
):
    completions = []
    if isinstance(input, str):
        input = [input]
    for i in range(0, len(input)):
        entry = input[i]
        print(entry)
        model_inputs = tokenizer.apply_chat_template(entry, add_generation_prompt=True, return_tensors="pt")
        # model_inputs = tokenizer.apply_chat_template(batch, return_tensors="pt", add_generation_prompt=True).to("cuda")
        print(model_inputs)
        print(tokenizer.batch_decode(model_inputs))

        generated_ids = model.generate(model_inputs, max_new_tokens=max_new_token, do_sample=False)[:, model_inputs.shape[1]:]
        # output_ids = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
        generated_texts = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
        completions += generated_texts
    return completions




def openai_chat_completion(model, system_prompt, history, temperature=0, max_tokens=512):
    openai.api_key = "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"
    response = None
    if system_prompt is not None:
        messages = [{"role": "system", "content": system_prompt}] + history
    else:
        messages = history
    while response is None:
        try:
            response = openai.chat.completions.create(
                model=model, messages=messages, temperature=temperature, max_tokens=max_tokens
            )
        except Exception as e:
            print(str(e), flush=True)
            time.sleep(5)
    # Save the query to log
    if not os.path.exists("./log/gpt_query_history.csv"):
        os.makedirs("./log")
        f = open("./log/gpt_query_history.csv", "w")
        writer = csv.writer(f)
        writer.writerow(["model", "messages", "temperature", "response"])
    else:
        f = open("./log/gpt_query_history.csv", "a")
        writer = csv.writer(f)
    writer.writerow([model, messages, temperature, response])
    return response.choices[0].message.content
