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


class EDC:
    def __init__(self, edc_configuration) -> None:
        extraction_llm_name = edc_configuration["extractor_llm"]
        schema_definition_llm_name = edc_configuration["schema_definition_llm"]
        schema_canonicalization_verifier_model_name = edc_configuration["canonicalization_verifier_model"]

        initial_target_schema = {
            "education": "The subject receives education at the institute specified by the object entity.",
            "gender": "The subject entity has the gender specified by the object entity.",
        }

        # Load the needed models and tokenizers
        needed_model_set = set(
            [extraction_llm_name, schema_definition_llm_name, schema_canonicalization_verifier_model_name]
        )

        print(f"Models used: {needed_model_set}")
        print("Loading models...")
        for model_name in needed_model_set:
            if not llm_utils.is_model_openai(model_name):
                needed_model_dict = {model_name: llm_utils.load_model(model_name)}

        # Initialize the components

        # Initialize the extractor
        if not llm_utils.is_model_openai(extraction_llm_name):
            extractor = Extractor(
                model=needed_model_dict[extraction_llm_name][0], tokenizer=needed_model_dict[extraction_llm_name][1]
            )
        else:
            extractor = Extractor(openai_model=extraction_llm_name)

        # Initialize the schema definer
        if not llm_utils.is_model_openai(schema_definition_llm_name):
            schema_definer = SchemaDefiner(
                model=needed_model_dict[schema_definition_llm_name][0],
                tokenizer=needed_model_dict[schema_definition_llm_name][1],
            )
        else:
            schema_definer = SchemaDefiner(openai_model=extraction_llm_name)

        # Initialize the schema canonicalizer
        schema_canonicalization_embedding_model = MistralForSequenceEmbedding.from_pretrained(
            "intfloat/e5-mistral-7b-instruct", device_map="cpu"
        )
        schema_canonicalization_embedding_tokenizer = AutoTokenizer.from_pretrained("intfloat/e5-mistral-7b-instruct")

        if not llm_utils.is_model_openai(schema_canonicalization_verifier_model_name):
            schema_canonicalizer = SchemaCanonicalizer(
                initial_target_schema,
                schema_canonicalization_embedding_model,
                schema_canonicalization_embedding_tokenizer,
                verifier_model=needed_model_dict[schema_canonicalization_verifier_model_name][0],
                verifier_tokenizer=needed_model_dict[schema_canonicalization_verifier_model_name][1],
            )
        else:
            schema_canonicalizer = SchemaCanonicalizer(
                initial_target_schema,
                schema_canonicalization_embedding_model,
                schema_canonicalization_embedding_tokenizer,
                verifier_openai_model=schema_canonicalization_verifier_model_name,
            )

        # Initialize the entity extractor
        if not llm_utils.is_model_openai(extraction_llm_name):
            entity_extractor = EntityExtractor(
                model=needed_model_dict[extraction_llm_name][0], tokenizer=needed_model_dict[extraction_llm_name][1]
            )
        else:
            entity_extractor = EntityExtractor(openai_model=extraction_llm_name)

        # Initialize the schema retriever
        schema_retriever = SchemaRetriever(
            initial_target_schema, schema_canonicalization_embedding_model, schema_canonicalization_embedding_tokenizer
        )

        self.entity_extractor = entity_extractor
        self.extractor = extractor
        self.definer = schema_definer
        self.canonicalizer = schema_canonicalizer
        self.schema_retriever = schema_retriever
        self.target_schema = initial_target_schema

    def construct_refinement_hint(
        self,
        input_text_list: List[str],
        extracted_triplets_list: List[List[List[str]]],
        include_relation_example="self",
        relation_top_k=10,
    ):
        entity_extraction_few_shot_examples_str = open("./prompt_templates/few_shot_examples.txt").read()
        entity_extraction_prompt_template_str = open("./prompt_templates/entity_extraction_template.txt").read()

        entity_merging_prompt_template_str = open("./prompt_templates/entity_merging_template.txt").read()

        entity_hint_list = []
        relation_hint_list = []

        relation_example_dict = {}
        if include_relation_example == "self":
            # Include an example of where this relation can be extracted
            print("Gathering relation examples...")
            for idx in tqdm(range(len(input_text_list))):
                input_text_str = input_text_list[idx]
                extracted_triplets = extracted_triplets_list[idx]
                for triplet in extracted_triplets:
                    relation = triplet[1]
                    if relation not in relation_example_dict:
                        relation_example_dict[relation] = {"text": input_text_str, "triplet": triplet}
                    else:
                        relation_example_dict[relation].append({"text": input_text_str, "triplet": triplet})
        else:
            # Todo: allow to pass gold examples of relations
            pass

        for idx in tqdm(range(len(input_text_list))):
            input_text_str = input_text_list[idx]
            extracted_triplets = extracted_triplets_list[idx]

            previous_relations = set()
            previous_entities = set()

            for triplet in extracted_triplets:
                print(triplet)
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
                    hint_relations.append(relation)

            candidate_relation_str = ""
            for relation_idx, relation in enumerate(hint_relations):
                if relation not in self.target_schema:
                    continue
                
                relation_definition = self.target_schema[relation]

                candidate_relation_str += f"{relation_idx+1}. {relation}: {relation_definition}\n"
                if include_relation_example == "self":
                    if relation not in relation_example_dict:
                        candidate_relation_str += "Example: None.\n"
                    else:
                        selected_example = None
                        for example in relation_example_dict[relation]:
                            if example["text"] != input_text_str:
                                selected_example = example
                                break
                        if selected_example is not None:
                            candidate_relation_str += f"""For example, {selected_example['triple']} can be extracted from "{selected_example['text']}"\n"""
                        else:
                            candidate_relation_str += "Example: None.\n"
            relation_hint_list.append(candidate_relation_str)
        return entity_hint_list, relation_hint_list

    # def refine_extracted_kg(self, input_text_list: List[str], extracted_triplets_list: List[List[str]]):
    #     assert len(input_text_list) == len(
    #         extracted_triplets_list
    #     ), "The number of given text does not match the number of triplets!"

    #     # Gather the refinement hint
    #     print("Putting together the refinement hint...")
    #     self.construct_refinement_hint(input_text_list, extracted_triplets_list)

    #     oie_refinement_prompt_template_str = open("./prompt_templates/extract_template_refined.txt").read()
    #     oie_refinement_prompt_template_str = open("./prompt_templates/extract_template_few_shot.txt").read()
    #     for idx in range(len(input_text_list)):
    #         input_text_str = input_text_list[idx]
    #         oie_few_shot_
    #         refined_oie_triplets = self.extractor.extract()

    def extract_kg(self, input_text_list: List[str], previous_extracted_triplets_list: List[List[str]] = None):

        oie_triplets_list = []
        if previous_extracted_triplets_list is None:
            oie_few_shot_examples_str = open("./prompt_templates/few_shot_examples.txt").read()
            oie_few_shot_prompt_template_str = open("./prompt_templates/extract_template_few_shot.txt").read()

            # if refine:
            #     # Perform extraction with refinement hint: entity hints and relation hints

            # Extract a canonicalized KG with EDC
            for input_text in input_text_list:
                oie_triplets = self.extractor.extract(
                    input_text, oie_few_shot_examples_str, oie_few_shot_prompt_template_str
                )
                oie_triplets_list.append(oie_triplets)
        else:
            assert len(input_text_list) == len(
                previous_extracted_triplets_list
            ), "The number of given text does not match the number of triplets!"

            # Gather the refinement hint
            print("Putting together the refinement hint...")
            entity_hint_list, relation_hint_list = self.construct_refinement_hint(
                input_text_list, previous_extracted_triplets_list
            )

            oie_refinement_prompt_template_str = open("./prompt_templates/extract_template_refined.txt").read()
            oie_refinement_few_shot_examples_str = open(
                "./prompt_templates/oie_refinement_few_shot_examples.txt"
            ).read()
            for idx in range(len(input_text_list)):
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

        schema_definition_dict_list = []
        schema_definition_few_shot_prompt_template_str = open(
            "./prompt_templates/schema_definition_template.txt"
        ).read()

        # Define the relations in the induced open schema
        for idx, oie_triplets in enumerate(oie_triplets_list):
            schema_definition_dict = self.definer.define_schema(
                input_text_list[idx], oie_triplets, "", schema_definition_few_shot_prompt_template_str
            )
            schema_definition_dict_list.append(schema_definition_dict)

        canonicalize_verify_prompt_template_str = open("./prompt_templates/canonicalize_verify_template.txt").read()

        # Canonicalize
        canonicalized_triplets_list = []
        for idx, oie_triplets in enumerate(oie_triplets_list):
            canonicalized_triplets = []
            for oie_triplet in oie_triplets:
                canonicalized_triplet = self.canonicalizer.canonicalize(
                    input_text_list[idx],
                    oie_triplet,
                    schema_definition_dict_list[idx],
                    canonicalize_verify_prompt_template_str,
                )
                canonicalized_triplets.append(canonicalized_triplet)
            canonicalized_triplets_list.append(canonicalized_triplets)

        return canonicalized_triplets
