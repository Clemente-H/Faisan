import os
from pprint import pprint
import pandas as pd
import bitsandbytes as bnb
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import transformers
from datasets import load_dataset, load_from_disk
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
model_id = "../../falcon/falcon-7b"
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    load_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
)

model =AutoModelForCausalLM.from_pretrained(
    model_id,
    device_map="auto",
    trust_remote_code=True,
    quantization_config=bnb_config,
)

tokenizer = AutoTokenizer.from_pretrained(model_id)
tokenizer.pad_token = tokenizer.eos_token
model = prepare_model_for_kbit_training(model)

config = LoraConfig(
    r=8,
    lora_alpha=32,
    target_modules=["query_key_value", "dense","dense_h_to_4h","dense_4h_to_h"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
    modules_to_save=["word_embeddings", "lm_head"]
)

model = get_peft_model(model, config)
is_tokenized = True

if is_tokenized:
    dataset = load_from_disk('./tokenized_dataset/')
else:
    dataset = load_dataset('json', data_files='spanish_corpora/spanish-corpora/preprocessed/blocks_dataset.jsonl', split='train')

    def generate_prompt(data_point):
        return f"""
{data_point["text"]}
  """.strip()

    def generate_and_tokenize_prompt(data_point):
       full_prompt = generate_prompt(data_point)
       tokenized_full_prompt = tokenizer(full_prompt, padding=True, truncation=True)
       return tokenized_full_prompt

    dataset = dataset.shuffle().map(generate_and_tokenize_prompt)
    dataset.save_to_disk('./tokenized_dataset')


def print_trainable_parameters(model):
    """
    Prints the number of trainable parameters in the model.
    """
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    print(
        f"trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param}"
    )

print_trainable_parameters(model)
output_dir = "./falcon_pre_trained"

#entrenamiento
training_args = transformers.TrainingArguments(
    per_device_train_batch_size=25,
    gradient_accumulation_steps=4,
    num_train_epochs=1,
    learning_rate=2e-4,
    fp16=True,
    save_total_limit=3,
    logging_steps=1,
    output_dir=output_dir,
    save_strategy='epoch',
    optim="paged_adamw_8bit",
    lr_scheduler_type = 'cosine',
    warmup_ratio = 0.05,
    report_to="tensorboard",
)

trainer = transformers.Trainer(
    model=model,
    #train_dataset=train_dataset,
    train_dataset=dataset,
    args=training_args,
    #tokenizer=tokenizer,
    data_collator=transformers.DataCollatorForLanguageModeling(tokenizer, mlm=False),
)
model.config.use_cache = False  # silence the warnings. Please re-enable for inference!
trainer.train()

model.save_pretrained("falcon_pre_trained/trained_model")
