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

print_flushed = partial(print, flush=True)


class EDC:
    def __init__(self, **edc_configuration) -> None:

        # OIE module setting
        self.oie_llm_name = edc_configuration["oie_llm"]
        self.oie_prompt_template_file_path = edc_configuration["oie_prompt_template_file_path"]
        self.oie_few_shot_example_file_path = edc_configuration["oie_few_shot_example_file_path"]

        # Schema Definition module setting
        self.sd_llm_name = edc_configuration["sd_llm"]
        self.sd_template_file_path = edc_configuration["sd_prompt_template_file_path"]
        self.sd_few_shot_example_file_path = edc_configuration["sd_few_shot_example_file_path"]

        # Schema Canonicalization module setting
        self.sc_llm_name = edc_configuration["sc_llm"]
        self.sc_template_file_path = edc_configuration["sc_prompt_template_file_path"]

        # Refinement setting
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
        needed_model_set = set([self.oie_llm_name, self.sd_llm_name, self.sc_llm_name, self.ee_llm_name])

        print_flushed(f"Models used: {needed_model_set}")
        print_flushed("Loading models...")
        for model_name in needed_model_set:
            if not llm_utils.is_model_openai(model_name):
                needed_model_dict = {
                    model_name: (
                        AutoModelForCausalLM.from_pretrained(model_name, device_map="auto"),
                        AutoTokenizer.from_pretrained(model_name),
                    )
                }

        # Initialize the components

        # Initialize the extractor
        if not llm_utils.is_model_openai(self.oie_llm_name):
            extractor = Extractor(
                model=needed_model_dict[self.oie_llm_name][0], tokenizer=needed_model_dict[self.oie_llm_name][1]
            )
        else:
            extractor = Extractor(openai_model=self.oie_llm_name)

        # Initialize the schema definer
        if not llm_utils.is_model_openai(self.sd_llm_name):
            schema_definer = SchemaDefiner(
                model=needed_model_dict[self.sd_llm_name][0],
                tokenizer=needed_model_dict[self.sd_llm_name][1],
            )
        else:
            schema_definer = SchemaDefiner(openai_model=self.sd_llm_name)

        # Initialize the schema canonicalizer
        schema_canonicalization_embedding_model = MistralForSequenceEmbedding.from_pretrained(
            "intfloat/e5-mistral-7b-instruct", device_map="auto"
        )
        schema_canonicalization_embedding_tokenizer = AutoTokenizer.from_pretrained("intfloat/e5-mistral-7b-instruct")

        if not llm_utils.is_model_openai(self.sc_llm_name):
            schema_canonicalizer = SchemaCanonicalizer(
                self.schema,
                schema_canonicalization_embedding_model,
                schema_canonicalization_embedding_tokenizer,
                verifier_model=needed_model_dict[self.sc_llm_name][0],
                verifier_tokenizer=needed_model_dict[self.sc_llm_name][1],
            )
        else:
            schema_canonicalizer = SchemaCanonicalizer(
                self.schema,
                schema_canonicalization_embedding_model,
                schema_canonicalization_embedding_tokenizer,
                verifier_openai_model=self.sc_llm_name,
            )

        # Initialize the entity extractor
        if not llm_utils.is_model_openai(self.ee_llm_name):
            entity_extractor = EntityExtractor(
                model=needed_model_dict[self.ee_llm_name][0], tokenizer=needed_model_dict[self.ee_llm_name][1]
            )
        else:
            entity_extractor = EntityExtractor(openai_model=self.ee_llm_name)

        # Initialize the schema retriever
        schema_retriever_embedding_model = MistralForSequenceEmbedding.from_pretrained(
            "intfloat/e5-mistral-7b-instruct", device_map="auto"
        )
        if self.sr_adapter_path is not None:
            schema_retriever_embedding_model.load_adapter(self.sr_adapter_path)
        schema_retriever_embedding_tokenizer = AutoTokenizer.from_pretrained("intfloat/e5-mistral-7b-instruct")
        schema_retriever = SchemaRetriever(
            self.schema, schema_retriever_embedding_model, schema_retriever_embedding_tokenizer
        )

        self.entity_extractor = entity_extractor
        self.extractor = extractor
        self.definer = schema_definer
        self.canonicalizer = schema_canonicalizer
        self.schema_retriever = schema_retriever

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
        detail_log=True,
        refinement_iterations=0,
    ):
        if output_dir is not None:
            pathlib.Path(output_dir).mkdir(parents=True, exist_ok=True)

        output_kg_list = []

        # EDC run
        print_flushed("EDC running...")
        oie_triplets, schema_definition_dict_list, canonicalized_triplets_list = self.extract_kg_helper(input_text_list)
        output_kg_list.append(oie_triplets)
        output_kg_list.append(canonicalized_triplets_list)

        if output_dir is not None:
            with open(os.path.join(output_dir, "edc_output.txt"), "w") as f:
                for l in canonicalized_triplets_list:
                    f.write(str(l) + "\n")
                f.flush()

            if self.enrich_schema:
                with open(os.path.join(output_dir, "edc_updated_schema.csv"), "w") as f:
                    writer = csv.writer(f)
                    for relation, relation_definition in self.schema.items():
                        writer.writerow([relation, relation_definition])
                        f.flush()

        for iteration in range(refinement_iterations):
            print_flushed(f"EDC with Refinement iteration {iteration + 1} running...")
            oie_triplets, schema_definition_dict_list, canonicalized_triplets_list = self.extract_kg_helper(
                input_text_list, canonicalized_triplets_list
            )
            output_kg_list.append(canonicalized_triplets_list)
            if output_dir is not None:
                with open(os.path.join(output_dir, f"edc_output_refinement_{iteration + 1}.txt"), "w") as f:
                    for l in canonicalized_triplets_list:
                        f.write(str(l) + "\n")
                f.flush()
                if self.enrich_schema:
                    with open(os.path.join(output_dir, f"edc_updated_schema_{iteration + 1}.csv"), "w") as f:
                        writer = csv.writer(f)
                        for relation, relation_definition in self.schema.items():
                            writer.writerow([relation, relation_definition])
                            f.flush()
        return output_kg_list

    def extract_kg_helper(
        self,
        input_text_list: List[str],
        previous_extracted_triplets_list: List[List[str]] = None,
    ):
        oie_triplets_list = []
        if previous_extracted_triplets_list is None:
            oie_few_shot_examples_str = open(self.oie_few_shot_example_file_path).read()
            oie_few_shot_prompt_template_str = open(self.oie_prompt_template_file_path).read()

            # Extract a canonicalized KG with EDC
            print_flushed("Running OIE...")
            for input_text in tqdm(input_text_list):
                oie_triplets = self.extractor.extract(
                    input_text, oie_few_shot_examples_str, oie_few_shot_prompt_template_str
                )
                oie_triplets_list.append(oie_triplets)
                print_flushed(f"OIE: {input_text}\n -> {oie_triplets}\n")
        else:
            assert len(input_text_list) == len(
                previous_extracted_triplets_list
            ), "The number of given text does not match the number of triplets!"

            # Gather the refinement hint
            print_flushed("Putting together the refinement hint...")
            entity_hint_list, relation_hint_list = self.construct_refinement_hint(
                input_text_list, previous_extracted_triplets_list
            )

            oie_refinement_prompt_template_str = open(self.oie_r_prompt_template_file_path).read()
            oie_refinement_few_shot_examples_str = open(self.oie_r_few_shot_example_file_path).read()
            print_flushed("Running Refined OIE...")
            for idx in tqdm(range(len(input_text_list))):
                input_text = input_text_list[idx]
                entity_hint_str = entity_hint_list[idx]
                relation_hint_str = relation_hint_list[idx]
                refined_oie_triplets = self.extractor.extract(
                    input_text,
                    oie_refinement_few_shot_examples_str,
                    oie_refinement_prompt_template_str,
                    entity_hint_str,
                    relation_hint_str,
                )
                oie_triplets_list.append(refined_oie_triplets)
            print_flushed(f"Refined OIE: {input_text}\n -> {refined_oie_triplets}\n")

        schema_definition_dict_list = []
        schema_definition_few_shot_prompt_template_str = open(self.sd_template_file_path).read()
        schema_definition_few_shot_examples_str = open(self.sd_few_shot_example_file_path).read()

        # Define the relations in the induced open schema
        print_flushed("Running Schema Definition...")
        for idx, oie_triplets in enumerate(tqdm(oie_triplets_list)):
            schema_definition_dict = self.definer.define_schema(
                input_text_list[idx],
                oie_triplets,
                schema_definition_few_shot_examples_str,
                schema_definition_few_shot_prompt_template_str,
            )
            schema_definition_dict_list.append(schema_definition_dict)
            print_flushed(f"SD: {input_text}, {oie_triplets}\n -> {schema_definition_dict}\n")
        canonicalize_verify_prompt_template_str = open(self.sc_template_file_path).read()

        # Canonicalize
        canonicalized_triplets_list = []
        print_flushed("Running Schema Canonicalization...")
        for idx, oie_triplets in enumerate(tqdm(oie_triplets_list)):
            print_flushed(f"Schema Canonicalization: {input_text_list[idx]}")
            canonicalized_triplets = []
            for oie_triplet in oie_triplets:
                canonicalized_triplet = self.canonicalizer.canonicalize(
                    input_text_list[idx],
                    oie_triplet,
                    schema_definition_dict_list[idx],
                    canonicalize_verify_prompt_template_str,
                    enrich=self.enrich_schema,
                )
                print_flushed(f"{oie_triplet} -> {canonicalized_triplet}", flush=True)
                if canonicalized_triplet is not None:
                    canonicalized_triplets.append(canonicalized_triplet)
            canonicalized_triplets_list.append(canonicalized_triplets)

        # If schema may be changed, update the other modules that use embeddings
        if self.enrich_schema:
            self.schema_retriever.update_schema_embedding_dict()

        return oie_triplets_list, schema_definition_dict_list, canonicalized_triplets_list
