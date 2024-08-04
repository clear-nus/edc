from edc.extract import Extractor
from edc.schema_definition import SchemaDefiner
from edc.schema_canonicalization import SchemaCanonicalizer
from edc.entity_extraction import EntityExtractor
import edc.utils.llm_utils as llm_utils
from typing import List
from edc.utils.e5_mistral_utils import MistralForSequenceEmbedding
from transformers import AutoModelForCausalLM, AutoTokenizer
from edc.schema_retriever import SchemaRetriever
from tqdm import tqdm
import os
import csv
import pathlib
from functools import partial
import copy
import logging
from sentence_transformers import SentenceTransformer
from importlib import reload

reload(logging)
logging.basicConfig(level=logging.NOTSET)
logger = logging.getLogger(__name__)


class EDC:
    def __init__(self, **edc_configuration) -> None:
        # OIE module settings
        self.oie_llm_name = edc_configuration["oie_llm"]
        self.oie_prompt_template_file_path = edc_configuration["oie_prompt_template_file_path"]
        self.oie_few_shot_example_file_path = edc_configuration["oie_few_shot_example_file_path"]

        # Schema Definition module settings
        self.sd_llm_name = edc_configuration["sd_llm"]
        self.sd_template_file_path = edc_configuration["sd_prompt_template_file_path"]
        self.sd_few_shot_example_file_path = edc_configuration["sd_few_shot_example_file_path"]

        # Schema Canonicalization module settings
        self.sc_llm_name = edc_configuration["sc_llm"]
        self.sc_embedder_name = edc_configuration["sc_embedder"]
        self.sc_template_file_path = edc_configuration["sc_prompt_template_file_path"]

        # Refinement settings
        self.sr_adapter_path = edc_configuration["sr_adapter_path"]

        self.oie_r_prompt_template_file_path = edc_configuration["oie_refine_prompt_template_file_path"]
        self.oie_r_few_shot_example_file_path = edc_configuration["oie_refine_few_shot_example_file_path"]

        self.ee_llm_name = edc_configuration["ee_llm"]
        self.ee_template_file_path = edc_configuration["ee_prompt_template_file_path"]
        self.ee_few_shot_example_file_path = edc_configuration["ee_few_shot_example_file_path"]

        self.em_template_file_path = edc_configuration["em_prompt_template_file_path"]

        self.initial_schema_path = edc_configuration["target_schema_path"]
        self.enrich_schema = edc_configuration["enrich_schema"]

        if self.initial_schema_path is not None:
            reader = csv.reader(open(self.initial_schema_path, "r"))
            self.schema = {}
            for row in reader:
                relation, relation_definition = row
                self.schema[relation] = relation_definition
        else:
            self.schema = {}

        # Load the needed models and tokenizers
        self.needed_model_set = set(
            [self.oie_llm_name, self.sd_llm_name, self.sc_llm_name, self.sc_embedder_name, self.ee_llm_name]
        )

        self.loaded_model_dict = {}

        # print_flushed(f"Models used: {needed_model_set}")
        # print_flushed("Loading models...")
        logger.info(f"Model used: {self.needed_model_set}")

    def oie(
        self, input_text_list: List[str], previous_extracted_triplets_list: List[List[str]] = None, free_model=False
    ):
        if not llm_utils.is_model_openai(self.oie_llm_name):
            # Load the HF model for OIE
            logger.info(f"Loading OIE model {self.oie_llm_name}")
            if self.oie_llm_name not in self.loaded_model_dict:
                oie_model, oie_tokenizer = (
                    AutoModelForCausalLM.from_pretrained(self.oie_llm_name, device_map="auto"),
                    AutoTokenizer.from_pretrained(self.oie_llm_name),
                )
                self.loaded_model_dict[self.oie_llm_name] = (oie_model, oie_tokenizer)
            else:
                logger.info(f"Model {self.oie_llm_name} is already loaded, reusing it.")
                oie_model, oie_tokenizer = self.loaded_model_dict[self.oie_llm_name]
            extractor = Extractor(oie_model, oie_tokenizer)
        else:
            extractor = Extractor(openai_model=self.oie_llm_name)

        oie_triples_list = []
        if previous_extracted_triplets_list is not None:
            # Refined OIE
            logger.info("Running Refined OIE...")
            raise NotImplementedError
        else:
            # Normal OIE
            logger.info("Running OIE...")
            oie_few_shot_examples_str = open(self.oie_few_shot_example_file_path).read()
            oie_few_shot_prompt_template_str = open(self.oie_prompt_template_file_path).read()

            for input_text in tqdm(input_text_list):
                oie_triples = extractor.extract(input_text, oie_few_shot_examples_str, oie_few_shot_prompt_template_str)
                oie_triples_list.append(oie_triples)
                logger.debug(f"{input_text}\n -> {oie_triples}\n")

        logger.info("OIE finished.")

        if free_model:
            llm_utils.free_model(oie_model, oie_tokenizer)
            del self.loaded_model_dict[self.oie_llm_name]

        return oie_triples_list

    def schema_definition(self, input_text_list: List[str], oie_triplets_list: List[List[str]], free_model=False):
        assert len(input_text_list) == len(oie_triplets_list)

        if not llm_utils.is_model_openai(self.sd_llm_name):
            # Load the HF model for Schema Definition
            if self.sd_llm_name not in self.loaded_model_dict:
                logger.info(f"Loading Schema Definition model {self.sd_llm_name}")
                sd_model, sd_tokenizer = (
                    AutoModelForCausalLM.from_pretrained(self.sd_llm_name, device_map="auto"),
                    AutoTokenizer.from_pretrained(self.sd_llm_name),
                )
                self.loaded_model_dict[self.sd_llm_name] = (sd_model, sd_tokenizer)
            else:
                logger.info(f"Model {self.sd_llm_name} is already loaded, reusing it.")
                sd_model, sd_tokenizer = self.loaded_model_dict[self.sd_llm_name]
            schema_definer = SchemaDefiner(model=sd_model, tokenizer=sd_tokenizer)
        else:
            schema_definer = SchemaDefiner(openai_model=self.sd_llm_name)

        schema_definition_few_shot_prompt_template_str = open(self.sd_template_file_path).read()
        schema_definition_few_shot_examples_str = open(self.sd_few_shot_example_file_path).read()
        schema_definition_dict_list = []

        logger.info("Running Schema Definition...")
        for idx, oie_triplets in enumerate(tqdm(oie_triplets_list)):
            schema_definition_dict = schema_definer.define_schema(
                input_text_list[idx],
                oie_triplets,
                schema_definition_few_shot_examples_str,
                schema_definition_few_shot_prompt_template_str,
            )
            schema_definition_dict_list.append(schema_definition_dict)
            logger.debug(f"{input_text_list[idx]}, {oie_triplets}\n -> {schema_definition_dict}\n")

        logger.info("Schema Definition finished.")
        if free_model:
            llm_utils.free_model(sd_model)
            del self.loaded_model_dict[self.sd_llm_name]
        return schema_definition_dict_list

    def schema_canonicalization(
        self,
        input_text_list: List[str],
        oie_triplets_list: List[List[str]],
        schema_definition_dict_list: List[dict],
        free_model=False,
    ):
        assert len(input_text_list) == len(oie_triplets_list) and len(input_text_list) == len(
            schema_definition_dict_list
        )
        logger.info("Running Schema Canonicalization...")

        sc_verify_prompt_template_str = open(self.sc_template_file_path).read()

        sc_embedder = SentenceTransformer(self.sc_embedder_name)

        if not llm_utils.is_model_openai(self.sc_llm_name):
            if self.sc_llm_name not in self.loaded_model_dict:
                logger.info(f"Loading Schema Canonicalization model {self.sc_llm_name}")
                sc_verify_model, sc_verify_tokenizer = (
                    AutoModelForCausalLM.from_pretrained(self.sc_llm_name, device_map="auto"),
                    AutoTokenizer.from_pretrained(self.sc_llm_name),
                )
                self.loaded_model_dict[self.sc_llm_name] = (sc_verify_model, sc_verify_tokenizer)
            else:
                logger.info(f"Model {self.sc_llm_name} is already loaded, reusing it.")
                sc_verify_model, sc_verify_tokenizer = self.loaded_model_dict[self.sc_llm_name]
            schema_canonicalizer = SchemaCanonicalizer(self.schema, sc_embedder, sc_verify_model, sc_verify_tokenizer)
        else:
            schema_canonicalizer = SchemaCanonicalizer(self.schema, sc_embedder, verify_openai_model=self.sc_llm_name)

        canonicalized_triplets_list = []
        for idx, input_text in enumerate(tqdm(input_text_list)):
            oie_triplets = oie_triplets_list[idx]
            canonicalized_triplets = []
            sd_dict = schema_definition_dict_list[idx]
            for oie_triplet in oie_triplets:
                canonicalized_triplet = schema_canonicalizer.canonicalize(
                    input_text, oie_triplet, sd_dict, sc_verify_prompt_template_str, self.enrich_schema
                )
                canonicalized_triplets.append(canonicalized_triplet)
            canonicalized_triplets.append(canonicalized_triplets)
            logger.info(f"{input_text}\n, {oie_triplets} ->\n {canonicalized_triplets_list}")
        logger.info("Schema Canonicalization finished.")

        if free_model:
            llm_utils.free_model(sc_embedder)
            llm_utils.free_model(sc_verify_model)
            llm_utils.free_model(sc_verify_tokenizer)
            del self.loaded_model_dict[self.sc_llm_name]
        
        return canonicalized_triplets_list

    def construct_refinement_hint(
        self,
        input_text_list: List[str],
        extracted_triplets_list: List[List[List[str]]],
        include_relation_example="self",
        relation_top_k=10,
    ):
        entity_extraction_few_shot_examples_str = open(self.ee_few_shot_example_file_path).read()
        entity_extraction_prompt_template_str = open(self.ee_template_file_path).read()

        entity_merging_prompt_template_str = open(self.em_template_file_path).read()

        entity_hint_list = []
        relation_hint_list = []

        relation_example_dict = {}
        if include_relation_example == "self":
            # Include an example of where this relation can be extracted
            # print("Gathering relation examples...")
            for idx in range(len(input_text_list)):
                input_text_str = input_text_list[idx]
                extracted_triplets = extracted_triplets_list[idx]
                for triplet in extracted_triplets:
                    relation = triplet[1]
                    if relation not in relation_example_dict:
                        relation_example_dict[relation] = [{"text": input_text_str, "triplet": triplet}]
                    else:
                        relation_example_dict[relation].append({"text": input_text_str, "triplet": triplet})
        else:
            # Todo: allow to pass gold examples of relations
            pass

        for idx in range(len(input_text_list)):
            input_text_str = input_text_list[idx]
            extracted_triplets = extracted_triplets_list[idx]

            previous_relations = set()
            previous_entities = set()

            for triplet in extracted_triplets:
                previous_entities.add(triplet[0])
                previous_entities.add(triplet[2])
                previous_relations.add(triplet[1])

            previous_entities = list(previous_entities)
            previous_relations = list(previous_relations)

            # Obtain candidate entities
            extracted_entities = self.entity_extractor.extract_entities(
                input_text_str, entity_extraction_few_shot_examples_str, entity_extraction_prompt_template_str
            )
            merged_entities = self.entity_extractor.merge_entities(
                input_text_str, previous_entities, extracted_entities, entity_merging_prompt_template_str
            )
            entity_hint_list.append(str(merged_entities))

            # Obtain candidate relations
            hint_relations = previous_relations

            retrieved_relations = self.schema_retriever.retrieve_relevant_relations(input_text_str)

            counter = 0

            for relation in retrieved_relations:
                if counter >= relation_top_k:
                    break
                else:
                    if relation not in hint_relations:
                        hint_relations.append(relation)

            candidate_relation_str = ""
            for relation_idx, relation in enumerate(hint_relations):
                if relation not in self.schema:
                    continue

                relation_definition = self.schema[relation]

                candidate_relation_str += f"{relation_idx+1}. {relation}: {relation_definition}\n"
                if include_relation_example == "self":
                    if relation not in relation_example_dict:
                        candidate_relation_str += "Example: None.\n"
                        pass
                    else:
                        selected_example = None
                        for example in relation_example_dict[relation]:
                            # print(example)
                            if example["text"] != input_text_str:
                                selected_example = example
                                break
                        if selected_example is not None:
                            candidate_relation_str += f"""For example, {selected_example['triplet']} can be extracted from "{selected_example['text']}"\n"""
                        else:
                            candidate_relation_str += "Example: None.\n"
                            pass
            relation_hint_list.append(candidate_relation_str)
        return entity_hint_list, relation_hint_list

    def extract_kg(
        self,
        input_text_list: List[str],
        output_dir: str = None,
        refinement_iterations=0,
    ):
        if output_dir is not None:
            pathlib.Path(output_dir).mkdir(parents=True, exist_ok=True)

        output_kg_list = []

        # EDC run
        logger.info("EDC starts running...")

        required_model_dict = {
            "oie": self.oie_llm_name,
            "sd": self.sd_llm_name,
            "sc_embed": self.sc_embedder_name,
            "sc_verify": self.sc_llm_name,
            "ee": self.ee_llm_name,
        }

        for iteration in range(refinement_iterations + 1):
            required_model_dict_current_iteration = copy.deepcopy(required_model_dict)

            del required_model_dict_current_iteration["oie"]
            oie_triplets_list = self.oie(
                input_text_list,
                None,
                free_model=self.oie_llm_name not in required_model_dict_current_iteration.values()
                and iteration == refinement_iterations,
            )

            del required_model_dict_current_iteration["sd"]
            sd_dict_list = self.schema_definition(
                input_text_list,
                oie_triplets_list,
                free_model=self.sd_llm_name not in required_model_dict_current_iteration.values()
                and iteration == refinement_iterations,
            )

            del required_model_dict_current_iteration["sc_embed"]
            del required_model_dict_current_iteration["sc_verify"]
            canon_triplets_list = self.schema_canonicalization(
                input_text_list,
                oie_triplets_list,
                sd_dict_list,
                free_model=self.sc_llm_name not in required_model_dict_current_iteration.values()
                and iteration == refinement_iterations,
            )

        return canon_triplets_list

        # Determine if the model should be freed
