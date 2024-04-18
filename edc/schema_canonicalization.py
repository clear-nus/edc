from typing import List
import os
from pathlib import Path
import edc.utils.llm_utils as llm_utils
import re
from edc.utils.e5_mistral_utils import MistralForSequenceEmbedding
from transformers import AutoModelForCausalLM, AutoTokenizer
import numpy as np
import copy
from tqdm import tqdm


class SchemaCanonicalizer:
    # The class to handle the last stage: Schema Canonicalization
    def __init__(
        self,
        target_schema_dict: dict,
        embedding_model,
        embedding_tokenizer,
        verifier_model: AutoTokenizer = None,
        verifier_tokenizer: AutoTokenizer = None,
        verifier_openai_model: AutoTokenizer = None,
    ) -> None:
        # The canonicalizer uses an embedding model to first fetch candidates from the target schema, then uses a verifier schema to decide which one to canonicalize to or not
        # canonoicalize at all.

        assert verifier_openai_model is not None or (verifier_model is not None and verifier_tokenizer is not None)
        self.verifier_model = verifier_model
        self.verifier_tokenizer = verifier_tokenizer
        self.verifier_openai_model = verifier_openai_model
        self.schema_dict = target_schema_dict

        self.embedding_model = embedding_model
        self.embedding_tokenizer = embedding_tokenizer

        # Embed the target schema

        self.schema_embedding_dict = {}

        print("Embedding target schema...")
        for relation, relation_definition in tqdm(target_schema_dict.items()):
            embedding = llm_utils.get_embedding_e5mistral(
                self.embedding_model,
                self.embedding_tokenizer,
                relation_definition,
            )
            self.schema_embedding_dict[relation] = embedding

        # Load the model
        pass

    def retrieve_similar_relations(self, query_relation_definition: str, top_k=5):
        target_relation_list = list(self.schema_embedding_dict.keys())
        target_relation_embedding_list = list(self.schema_embedding_dict.values())

        query_embedding = llm_utils.get_embedding_e5mistral(
            self.embedding_model,
            self.embedding_tokenizer,
            query_relation_definition,
            "Retrieve semantically similar text.",
        )
        scores = np.array([query_embedding]) @ np.array(target_relation_embedding_list).T

        scores = scores[0]
        highest_score_indices = np.argsort(-scores)

        return {
            target_relation_list[idx]: self.schema_dict[target_relation_list[idx]]
            for idx in highest_score_indices[:top_k]
        }, [scores[idx] for idx in highest_score_indices[:top_k]]

    def llm_verify(
        self,
        input_text_str: str,
        query_triplet: List[str],
        query_relation_definition: str,
        prompt_template_str: str,
        candidate_relation_definition_dict: dict,
        relation_example_dict: dict = None,
    ):
        canonicalized_triplet = copy.deepcopy(query_triplet)
        choice_letters_list = []
        choices = ""
        candidate_relations = list(candidate_relation_definition_dict.keys())
        candidate_relation_descriptions = list(candidate_relation_definition_dict.values())
        for idx, rel in enumerate(candidate_relations):
            choice_letter = chr(ord("@") + idx + 1)
            choice_letters_list.append(choice_letter)
            choices += f"{choice_letter}. '{rel}': {candidate_relation_descriptions[idx]}\n"
            if relation_example_dict is not None:
                choices += f"Example: '{relation_example_dict[candidate_relations[idx]]['triple']}' can be extracted from '{candidate_relations[idx]['sentence']}'\n"
        choices += f"{chr(ord('@')+idx+2)}. None of the above.\n"

        verification_prompt = prompt_template_str.format_map(
            {
                "input_text": input_text_str,
                "query_triplet": query_triplet,
                "query_relation": query_triplet[1],
                "query_relation_definition": query_relation_definition,
                "choices": choices,
            }
        )
        messages = [{"role": "user", "content": verification_prompt}]
        if self.verifier_openai_model is None:
            # llm_utils.generate_completion_transformers([messages], self.model, self.tokenizer, device=self.device)
            verificaiton_result = llm_utils.generate_completion_transformers(
                [messages],
                self.verifier_model,
                self.verifier_tokenizer,
                device=self.verifier_model.device,
                answer_prepend="Answer: ",
            )[0]
        else:
            verificaiton_result = llm_utils.openai_chat_completion(
                self.verifier_openai_model, None, messages, max_tokens=1
            )

        # print(verification_prompt)

        # print(verificaiton_result)
        # print(canonicalized_triplet)
        # print(choices)
        # print(verificaiton_result[0])
        # print(candidate_relations)
        if verificaiton_result[0] in choice_letters_list:
            canonicalized_triplet[1] = candidate_relations[choice_letters_list.index(verificaiton_result[0])]
        else:
            return None

        return canonicalized_triplet

    def canonicalize(
        self,
        input_text_str: str,
        open_triplet,
        open_relation_definition_dict: dict,
        verify_prompt_template: str,
        enrich=False,
    ):
        open_relation = open_triplet[1]

        if open_relation in self.schema_dict:
            # The relation is already canonical
            return open_triplet

        if len(self.schema_dict) != 0:
            candidate_relations, candidate_scores = self.retrieve_similar_relations(open_relation_definition_dict)

            if open_relation not in open_relation_definition_dict:
                canonicalized_triplet = None
            else:
                canonicalized_triplet = self.llm_verify(
                    input_text_str,
                    open_triplet,
                    open_relation_definition_dict[open_relation],
                    verify_prompt_template,
                    candidate_relations,
                    None,
                )
        else:
            canonicalized_triplet = None

        if canonicalized_triplet is None:
            # Cannot be canonicalized
            if enrich:
                self.schema_dict[open_relation] = open_relation_definition_dict[open_relation]
                embedding = llm_utils.get_embedding_e5mistral(
                    self.embedding_model, self.embedding_tokenizer, open_relation_definition_dict[open_relation]
                )
                self.schema_embedding_dict[open_relation] = embedding
                canonicalized_triplet = open_triplet
        return canonicalized_triplet
