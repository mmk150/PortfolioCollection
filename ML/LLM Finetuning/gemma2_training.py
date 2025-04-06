import pandas as pd
import numpy as np
from sklearn.metrics import log_loss
import torch
import matplotlib.pyplot as plt
import time
from tqdm.auto import tqdm

import torch
import transformers
from sklearn.metrics import accuracy_score
from transformers import (
    AutoTokenizer,
    LlamaModel,
    LlamaForSequenceClassification,
    GemmaForSequenceClassification,
    AutoModelForSequenceClassification,
    BitsAndBytesConfig,
    get_linear_schedule_with_warmup,
)
from peft import (
    get_peft_config,
    PeftModel,
    PeftConfig,
    get_peft_model,
    LoraConfig,
    TaskType,
)
import torch.nn.functional as F
import os
import random
import gc
from datasets import Dataset
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast, GradScaler
from tqdm import tqdm

DEVICE = "cuda"

from datasets import Dataset
from torch.utils.data import DataLoader


class Gemma2_9b_CFG:
    NUM_EPOCHS = 1
    BATCH_SIZE = 1
    DROPOUT = 0.05
    # MODEL_NAME = "unsloth/gemma-2-9b"
    MODEL_NAME = "unsloth/gemma-2-9b-bnb-4bit"
    SEED = 2024
    MAX_LENGTH = 1024
    NUM_WARMUP_STEPS = 128
    LR_MAX = 5e-5
    NUM_LABELS = 3
    LORA_RANK = 4
    LORA_ALPHA = 8
    LORA_MODULES = ["o_proj", "v_proj"]


class Gemma2_9b_in_CFG:
    NUM_EPOCHS = 1
    BATCH_SIZE = 1
    DROPOUT = 0.05
    MODEL_NAME = "google/gemma-2-9b-it"
    SEED = 2024
    MAX_LENGTH = 1024
    NUM_WARMUP_STEPS = 128
    LR_MAX = 5e-5
    NUM_LABELS = 3
    LORA_RANK = 4
    LORA_ALPHA = 8
    LORA_MODULES = ["o_proj", "v_proj"]


def set_seeds(seed):
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def get_token_lengths(texts, tokenizer):
    input_ids = tokenizer(texts.tolist(), return_tensors="np")["input_ids"]
    return [len(t) for t in input_ids]


bnb_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_use_double_quant=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.bfloat16)


def gemma2seq_train(train_df, model_params):
    CFG = Gemma2_9b_CFG
    set_seeds(seed=CFG.SEED)
    tokenizer = AutoTokenizer.from_pretrained(CFG.MODEL_NAME)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    tokenizer.add_eos_token = True
    tokenizer.save_pretrained("gemma2_tokenizer")

    train = prepare_train(train_df, tokenizer)

    tokens = tokenizer(
        train["text"].tolist(),
        padding="max_length",
        max_length=CFG.MAX_LENGTH,
        truncation=True,
        return_tensors="np",
    )

    INPUT_IDS = tokens["input_ids"]
    ATTENTION_MASKS = tokens["attention_mask"]
    LABELS = train[["winner_model_a", "winner_model_b", "winner_tie"]].values

    base_model = AutoModelForSequenceClassification.from_pretrained(
        CFG.MODEL_NAME, num_labels=CFG.NUM_LABELS, quantization_config=bnb_config, device_map="auto"
    )

    base_model.config.pretraining_tp = 1

    base_model.config.pad_token_id = tokenizer.pad_token_id

    peft_config = LoraConfig(
        task_type=TaskType.SEQ_CLS,
        inference_mode=False,
        r=CFG.LORA_RANK,
        lora_alpha=CFG.LORA_ALPHA,
        lora_dropout=CFG.DROPOUT,
        target_modules=CFG.LORA_MODULES,
    )

    model = get_peft_model(base_model, peft_config)
    model.print_trainable_parameters()

    optimizer = torch.optim.AdamW(model.parameters(), lr=CFG.LR_MAX)
    scheduler = transformers.get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=CFG.NUM_WARMUP_STEPS,
        num_training_steps=CFG.NUM_EPOCHS * len(train) // CFG.BATCH_SIZE,
    )

    dataset = torch.utils.data.TensorDataset(
        torch.tensor(INPUT_IDS, dtype=torch.long),
        torch.tensor(ATTENTION_MASKS, dtype=torch.long),
        torch.tensor(LABELS, dtype=torch.float),
    )
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=CFG.BATCH_SIZE, shuffle=True)

    model.train()
    for epoch in range(CFG.NUM_EPOCHS):
        print(f"Epoch {epoch + 1}/{CFG.NUM_EPOCHS}")
        progress_bar = tqdm(dataloader, total=len(dataloader))
        for batch in progress_bar:
            batch = tuple(t.to(DEVICE) for t in batch)
            input_ids, attention_mask, labels = batch

            optimizer.zero_grad()
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            loss.backward()
            optimizer.step()
            scheduler.step()

            progress_bar.set_postfix({"loss": loss.item()})

    model.save_pretrained("gemma2_finetuned_model")

    return model, tokenizer


def prepare_train(train, tokenizer):
    def process(input_str):
        stripped_str = input_str.strip("[]")
        sentences = [s.strip('"') for s in stripped_str.split('","')]
        return " ".join(sentences)

    train.loc[:, "prompt"] = train["prompt"].apply(process)
    train.loc[:, "response_a"] = train["response_a"].apply(process)
    train.loc[:, "response_b"] = train["response_b"].apply(process)

    indexes = train[(train.response_a == "null") & (train.response_b == "null")].index
    train.drop(indexes, inplace=True)
    train.reset_index(inplace=True, drop=True)

    train["text"] = "User prompt: " + train["prompt"] + "\n\nModel A :\n" + train["response_a"] + "\n\n--------\n\nModel B:\n" + train["response_b"]

    train.loc[:, "token_count"] = get_token_lengths(train["text"], tokenizer)

    train.loc[:, "label"] = np.argmax(train[["winner_model_a", "winner_model_b", "winner_tie"]].values, axis=1)

    return train


## TESTING
