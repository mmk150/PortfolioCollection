import pandas as pd
import json
import os, sys
from pprint import pprint
import bitsandbytes as bnb
import torch
import torch.nn as nn
import transformers
import accelerate
from datasets import load_dataset, Dataset
from huggingface_hub import notebook_login
import time
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
import glob

os.environ["ROCR_VISIBLE_DEVICES"] = "0,1"

from read_data import collect_text_files_info, good_clusters_to_dict

MODEL_NAME = "unsloth/mistral-7b-bnb-4bit"
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
)

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    device_map="auto",
    trust_remote_code=True,
    quantization_config=bnb_config,
)

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
tokenizer.pad_token = tokenizer.eos_token

model = prepare_model_for_kbit_training(model)


import re


def get_num_layers(model):
    numbers = set()
    for name, _ in model.named_parameters():
        for number in re.findall(r"\d+", name):
            numbers.add(int(number))
    return max(numbers)


def get_last_layer_linears(model):
    names = []

    num_layers = get_num_layers(model)
    for name, module in model.named_modules():
        if str(num_layers) in name and not "encoder" in name:
            if isinstance(module, torch.nn.Linear):
                names.append(name)
    return names


config = LoraConfig(
    r=32,
    lora_alpha=128,
    target_modules=get_last_layer_linears(model),
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
)

model = get_peft_model(model, config)


def cast2bool(x):
    if x == "True":
        return True
    else:
        return False


df = collect_text_files_info("./Cleaned")


generated_df = pd.read_csv("./combined_results_a.csv")
generated_df2 = pd.read_csv("./combined_results_b.csv")
generated_df = pd.concat([generated_df, generated_df2])
generated_df["rewritten_text"] = generated_df["only_rewritten"]
generated_df["has_colon"] = generated_df["has_colon"].astype(bool)
generated_df = generated_df.drop(["placeholder", "date"], axis=1)

gen_df1 = generated_df.copy(deep=True)


gen_df1 = gen_df1[gen_df1["has_colon"]]

gen_df1["rewritten_text"] = gen_df1["colon_snip_after"]
gen_df2 = generated_df.drop(["colon_snip_before", "colon_snip_after", "has_colon"], axis=1)


SAMPLESIZE = 100

gen_df1 = gen_df1.sample(SAMPLESIZE)
gen_df2 = gen_df2.sample(SAMPLESIZE)
combined_df = pd.concat([gen_df1, gen_df2])
combined_df["RWText"] = combined_df["rewritten_text"]
combined_df["OGText"] = combined_df["original_text"]
combined_df["Prompt"] = combined_df["rewrite_prompt"]

# master_dict=good_clusters_to_dict()
# def classify_cluster(x):
#     global master_dict
#     try:
#         res=master_dict[x]
#     except:
#         res="GARBAGE"
#     return res

# combined_df['Answer']=combined_df['rewrite_prompt'].apply(classify_cluster)
# combined_df=combined_df[combined_df['Answer']!='GARBAGE']

combined_df = combined_df.drop(
    ["rewrite_prompt", "only_rewritten", "colon_snip_before", "colon_snip_after", "original_text", "rewritten_text", "has_colon"], axis=1
)

# df=df.drop(["Filename"],axis=1)
# df=pd.concat([df,combined_df]).sample(frac=1)

# df.to_csv("data_finetune.csv")

# print(df)

data = Dataset.from_pandas(combined_df)


generation_config = model.generation_config
generation_config.max_new_tokens = 10
generation_config.temperature = 0.7
generation_config.top_p = 0.7
generation_config.num_return_sequences = 1
generation_config.pad_token_id = tokenizer.eos_token_id
generation_config.eos_token_id = tokenizer.eos_token_id


device = "cuda"


def generate_prompt(data_point):
    return f""" <s>[INST]Context:You are an assistant that is helping derive rewrite prompts from the original text and the rewritten text. Please do your best to guess at what the rewrite prompt is from how the original text changes into the rewritten text!
            Original Text:{data_point["OGText"]}.
            Rewritten Text:{data_point["RWText"]}.
            Prompt:[/INST] {data_point["Prompt"]}</s>
            """.strip()


def generate_and_tokenize_prompt(data_point):
    full_prompt = generate_prompt(data_point)
    tokenized_full_prompt = tokenizer(full_prompt, padding=True, truncation=True)
    return tokenized_full_prompt


data = data.shuffle().map(generate_and_tokenize_prompt)


def proompt(prompt):
    encoding = tokenizer(prompt, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model.generate(
            input_ids=encoding.input_ids,
            attention_mask=encoding.attention_mask,
            generation_config=generation_config,
        )

    print(tokenizer.decode(outputs[0], skip_special_tokens=True))


training_args = transformers.TrainingArguments(
    per_device_train_batch_size=1,
    gradient_accumulation_steps=4,
    num_train_epochs=3,
    learning_rate=3e-4,
    fp16=True,
    output_dir="data",
    optim="paged_adamw_8bit",
    lr_scheduler_type="cosine",
    warmup_ratio=0.01,
    report_to="none",
)

trainer = transformers.Trainer(
    model=model,
    train_dataset=data,
    args=training_args,
    data_collator=transformers.DataCollatorForLanguageModeling(tokenizer, mlm=False),
)
model.config.use_cache = False
trainer.train()


model.save_pretrained("./data/trained-model-it-promptguess-2")
