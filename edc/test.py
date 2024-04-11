import edc.utils.llm_utils as llm_utils
from peft import PeftModel, PeftConfig
from edc.utils.e5_mistral_utils import MistralForSequenceEmbedding
from edc.extract import Extractor
from edc.schema_definition import SchemaDefiner
from edc.schema_canonicalization import SchemaCanonicalizer
from edc.schema_retriever import SchemaRetriever
from edc.edc_framework import EDC
from edc.utils.e5_mistral_utils import MistralForSequenceEmbedding
from transformers import AutoModelForCausalLM, AutoTokenizer
import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]=""

extraction_llm_name = "mistral7b"
schema_definition_llm_name = "mistral7b"
schema_canonicalization_verifier_model_name = "mistral7b"

edc_configuration = {'extractor_llm': extraction_llm_name, 'schema_definition_llm': schema_definition_llm_name, 'canonicalization_verifier_model': schema_canonicalization_verifier_model_name}

edc = EDC(edc_configuration)

edc.extract_kg(['Bowen is a student at National University of Singapore'], [[['Bowen', 'is', 'student']]])