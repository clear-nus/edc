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
from sentence_transformers import SentenceTransformer
import logging

logger = logging.getLogger(__name__)


class SchemaCanonicalizer:
    # The class to handle the last stage: Schema Canonicalization
    def __init__(
        self,
        target_schema_dict: dict,
        embedder: SentenceTransformer,
        verify_model: AutoTokenizer = None,
        verify_tokenizer: AutoTokenizer = None,
        verify_openai_model: AutoTokenizer = None,
    ) -> None:
        # The canonicalizer uses an embedding model to first fetch candidates from the target schema, then uses a verifier schema to decide which one to canonicalize to or not
        # canonoicalize at all.

        assert verify_openai_model is not None or (verify_model is not None and verify_tokenizer is not None)
        self.verifier_model = verify_model
        self.verifier_tokenizer = verify_tokenizer
        self.verifier_openai_model = verify_openai_model
        self.schema_dict = target_schema_dict

        self.embedder = embedder

        # Embed the target schema
        self.schema_embedding_dict = {}

        print("Embedding target schema...")
        for relation, relation_definition in tqdm(target_schema_dict.items()):
            embedding = self.embedder.encode(relation_definition)
            self.schema_embedding_dict[relation] = embedding

    def retrieve_similar_relations(self, query_relation_definition: str, top_k=5):
        target_relation_list = list(self.schema_embedding_dict.keys())
        target_relation_embedding_list = list(self.schema_embedding_dict.values())
        if "sts_query" in self.embedder.prompts:
            query_embedding = self.embedder.encode(query_relation_definition, prompt_name="sts_query")
        else:
            query_embedding = self.embedder.encode(query_relation_definition)

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
            verification_result = llm_utils.generate_completion_transformers(
                messages, self.verifier_model, self.verifier_tokenizer, answer_prepend="Answer: ", max_new_token=5
            )
        else:
            verification_result = llm_utils.openai_chat_completion(
                self.verifier_openai_model, None, messages, max_tokens=1
            )

        if verification_result[0] in choice_letters_list:
            canonicalized_triplet[1] = candidate_relations[choice_letters_list.index(verification_result[0])]
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
            # candidate_relations, candidate_scores = self.retrieve_similar_relations(
            #     open_relation_definition_dict[open_relation]
            # )
            return open_triplet, {}

        candidate_relations = []
        candidate_scores = []

        if len(self.schema_dict) != 0:
            if open_relation not in open_relation_definition_dict:
                canonicalized_triplet = None
            else:
                candidate_relations, candidate_scores = self.retrieve_similar_relations(
                    open_relation_definition_dict[open_relation]
                )
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
                if "sts_query" in self.embedder.prompts:
                    embedding = self.embedder.encode(
                        open_relation_definition_dict[open_relation], prompt_name="sts_query"
                    )
                else:
                    embedding = self.embedder.encode(open_relation_definition_dict[open_relation])
                self.schema_embedding_dict[open_relation] = embedding
                canonicalized_triplet = open_triplet
        return canonicalized_triplet, dict(zip(candidate_relations, candidate_scores))
