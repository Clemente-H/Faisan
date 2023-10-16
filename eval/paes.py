import os
import json
from pprint import pprint
import bitsandbytes as bnb
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import transformers
from datasets import load_dataset
from peft import (
    LoraConfig,
    PeftConfig,
    PeftModel,
    get_peft_model,
    prepare_model_for_kbit_training,
)
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)

# Import the necessary libraries and modules
import pandas as pd
import math

# Define a function to process the dataset in chunks
def process_dataset_in_chunks(dataset, chunk_size, output_json_file):    
    # Model configuration
    PEFT_MODEL = "../finetuning/trained_model"
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
    )
    config = PeftConfig.from_pretrained(PEFT_MODEL)

    # Load the model with quantization config
    model = AutoModelForCausalLM.from_pretrained(
        config.base_model_name_or_path,
        return_dict=True,
        device_map="auto",
        quantization_config=bnb_config,
        trust_remote_code=True
    )

    model = PeftModel.from_pretrained(model, PEFT_MODEL)

    # Load the pre-trained tokenizer
    tokenizer = AutoTokenizer.from_pretrained(config.base_model_name_or_path, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token

    # Set the maximum number of tokens that the model can handle
    tokenizer.max_length = 4096 # You can adjust this value based on your available GPU memory


    generation_config = model.generation_config
    generation_config.max_new_tokens = 200
    generation_config.temperature = 0.7
    generation_config.top_p = 0.7
    generation_config.num_return_sequences = 1
    generation_config.pad_token_id = tokenizer.eos_token_id
    generation_config.eos_token_id = tokenizer.eos_token_id

    # Process the dataset in chunks
    num_chunks = math.ceil(len(dataset) / chunk_size)
    for chunk_idx in range(num_chunks):
        chunk_start = chunk_idx * chunk_size
        chunk_end = min((chunk_idx + 1) * chunk_size, len(dataset))
        chunk = dataset[chunk_start:chunk_end]

        with open(output_json_file, "a") as file:
            for i in range(len(chunk)):
                if len(chunk['options'][i]) == 5:
                    prompt = f"""
Given a <text>, a <question>, and some <options>, write the correct option in <response> and justify it in <explanation>, using the format of the following example:
<text>: The International Space Station (ISS) orbits the Earth approximately every 90 minutes.
<question>: How long does the International Space Station take to orbit the Earth?
<options>:
A: Every 60 minutes.
B: Every 45 minutes.
C: Every 90 minutes.
D: Every 120 minutes.
E: Every 150 minutes.
<response>:The correct option is C) Every 90 minutes.
<explanation>: The text explicitly states that the ISS orbits the Earth approximately every 90 minutes.
<texto> : {chunk["text"][i]}
<pregunta>:{chunk["question"][i]}
<opciones>:
A:{chunk["options"][i][0]['text']}
B:{chunk["options"][i][1]['text']}
C:{chunk["options"][i][2]['text']}
D:{chunk["options"][i][3]['text']}
E:{chunk["options"][i][4]['text']}
<respuesta>: La alternativa correcta es la
""".strip()
                else:
                    prompt = f"""
Given a <text>, a <question>, and some <options>, write the correct option in <response> and justify it in <explanation>, using the format of the following example:
<text>: The International Space Station (ISS) orbits the Earth approximately every 90 minutes.
<question>: How long does the International Space Station take to orbit the Earth?
<options>:
A: Every 60 minutes.
B: Every 45 minutes.
C: Every 90 minutes.
D: Every 120 minutes.
E: Every 150 minutes.
<response>: The correct option is C) Every 90 minutes.
<explanation>: The text explicitly states that the ISS orbits the Earth approximately every 90 minutes.
<texto> : {chunk["text"][i]}
<pregunta>:{chunk["question"][i]}
<opciones>:
A:{chunk["options"][i][0]['text']}
B:{chunk["options"][i][1]['text']}
C:{chunk["options"][i][2]['text']}
D:{chunk["options"][i][3]['text']}
<respuesta>: La alternativa correcta es la 
""".strip()
                device = "cuda:0" if torch.cuda.is_available() else "cpu"
                encoding = tokenizer(prompt, return_tensors="pt").to(device)
                
                # Enable mixed precision (automatic mixed precision)
                with torch.cuda.amp.autocast():
                    outputs = model.generate(
                        input_ids=encoding.input_ids,
                        attention_mask=encoding.attention_mask,
                        generation_config=generation_config,
                    )
                respuesta = {
                    'id_pregunta': i + chunk_start + 1,
                    'respuesta': tokenizer.decode(outputs[0], skip_special_tokens=True).strip()
                }
                json.dump(respuesta, file)
                file.write("\n")

# Load the dataset
dataset = pd.read_json('../paes_mod.json')

# Define the chunk size
chunk_size = 1  # You can adjust this value based on your available GPU memory

# Define the output JSON file path
output_json_file = './falcon_finetuned_paes.jsonl'

# Process the dataset in chunks
process_dataset_in_chunks(dataset, chunk_size, output_json_file)

