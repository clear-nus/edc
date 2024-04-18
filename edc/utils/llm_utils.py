import os
import openai
import time
from transformers import AutoModelForCausalLM, AutoTokenizer
import ast


def get_detailed_instruct(task_description: str, query: str) -> str:
    return f"Instruct: {task_description}\nQuery: {query}"


def get_embedding_e5mistral(model, tokenizer, sentence, task=None):
    model.eval()
    device = model.device

    if task != None:
        # It's a query to be embed
        sentence = get_detailed_instruct(task, sentence)

    sentence = [sentence]

    max_length = 4096
    # Tokenize the input texts
    batch_dict = tokenizer(
        sentence, max_length=max_length - 1, return_attention_mask=False, padding=False, truncation=True
    )
    # append eos_token_id to every input_ids
    batch_dict["input_ids"] = [input_ids + [tokenizer.eos_token_id] for input_ids in batch_dict["input_ids"]]
    batch_dict = tokenizer.pad(batch_dict, padding=True, return_attention_mask=True, return_tensors="pt")

    batch_dict.to(device)

    embeddings = model(**batch_dict).detach().cpu()

    assert len(embeddings) == 1

    return embeddings[0]


def parse_raw_entities(raw_entities: str):
    left_bracket_idx = raw_entities.index("[")
    right_bracket_idx = raw_entities.index("]")
    return ast.literal_eval(raw_entities[left_bracket_idx : right_bracket_idx + 1])


def parse_raw_triplets(raw_triplets: str):
    # Look for enclosing brackets
    unmatched_left_bracket_indices = []
    matched_bracket_pairs = []

    collected_triples = []
    for c_idx, c in enumerate(raw_triplets):
        if c == "[":
            unmatched_left_bracket_indices.append(c_idx)
        if c == "]":
            if len(unmatched_left_bracket_indices) == 0:
                continue
            # Found a right bracket, match to the last found left bracket
            matched_left_bracket_idx = unmatched_left_bracket_indices.pop()
            matched_bracket_pairs.append((matched_left_bracket_idx, c_idx))
    for l, r in matched_bracket_pairs:
        bracketed_str = raw_triplets[l : r + 1]
        try:
            parsed_triple = ast.literal_eval(bracketed_str)
            if len(parsed_triple) == 3 and all([isinstance(t, str) for t in parsed_triple]):
                if all([e != "" and e != "_" for e in parsed_triple]):
                    collected_triples.append(parsed_triple)
            elif not all([type(x) == type(parsed_triple[0]) for x in parsed_triple]):
                for e_idx, e in enumerate(parsed_triple):
                    if isinstance(e, list):
                        parsed_triple[e_idx] = ", ".join(e)
                # print(parsed_triple)
                collected_triples.append(parsed_triple)
        except Exception as e:
            print(raw_triplets)
            print(str(e))
            print("ERROR!")
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

        relation_definition_dict[relation] = relation_description
    return relation_definition_dict


def is_model_openai(model_name):
    return "gpt" in model_name


def generate_completion_transformers(
    input: list,
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    device,
    batch_size=1,
    max_new_token=256,
    answer_prepend="",
):
    tokenizer.pad_token = tokenizer.eos_token
    completions = []
    if isinstance(input, str):
        input = [input]
    for i in range(0, len(input), batch_size):
        batch = input[i : i + batch_size]
        model_inputs = [
            tokenizer.apply_chat_template(entry, add_generation_prompt=True, tokenize=False) + answer_prepend
            for entry in batch
        ]
        model_inputs = tokenizer(model_inputs, return_tensors="pt", padding=True, add_special_tokens=False).to(device)

        generated_ids = model.generate(
            **model_inputs, max_new_tokens=max_new_token, do_sample=False, pad_token_id=tokenizer.eos_token_id
        )[:, model_inputs["input_ids"].shape[1] :]
        generated_texts = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
        completions += generated_texts
    return completions


def openai_chat_completion(model, system_prompt, history, temperature=0, max_tokens=512):
    openai.api_key = os.environ["OPENAI_KEY"]
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

    return response.choices[0].message.content
