from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from huggingface_hub import login
import torch
import json
import os
import pandas as pd
import numpy as np
from tqdm import tqdm
from datetime import datetime
from torch.utils.data import Dataset
import argparse

dev = "cuda"


def get_prompts(num, random, size=0):
    if size == 0:
        prompts_df = pd.read_csv("./prompts_max_80_tokens.csv")
    else:
        prompts_df = pd.read_csv("./prompts_max_200_tokens.csv")
    if random:
        sampled_df = prompts_df.sample(n=num)
        return sampled_df
    else:
        return prompts_df.head(num)


def get_text_data(num, random, size=0):
    if size == 0:
        texts_df = pd.read_csv("./texts_max_256_tokens.csv")
    else:
        texts_df = pd.read_csv("./texts_max_512_tokens.csv")
    if random:
        sampled_df = texts_df.sample(n=num)
        return sampled_df
    else:
        return texts_df.head(num)


def get_everything(num, random, param=0, size=0):
    texts_df = get_text_data(num, random, size=size)
    prompts_df = get_prompts(num, random, size=size)
    if param == 0:
        texts_df = texts_df.iloc[::2]
        prompts_df = prompts_df.iloc[::2]
    elif param == 1:
        texts_df = texts_df.iloc[1::2]
        prompts_df = prompts_df.iloc[1::2]

    return texts_df, prompts_df


# formats the prompt to be done with batching
def format_prompts(texts_df, prompts_df):

    curr_date = datetime.now().strftime("%Y-%m-%d")

    text_list = texts_df["text"].tolist()
    prompt_list = prompts_df["prompt"].tolist()
    final_prompt_list = []
    temp_data = []
    fin_prompt_len = []
    for curr_text in text_list:
        for curr_prompt in prompt_list:
            prompt = curr_prompt.replace("\n", " ")
            rtext = curr_text.replace("\n", " ")
            final_prompt = f"<start_of_turn>user \n {prompt}:\n {rtext}<end_of_turn>\n<start_of_turn>model"
            data = {
                "original_text": curr_text,
                "rewrite_prompt": curr_prompt,
                "final_prompt_string": final_prompt,
                "rewritten_text": None,
                "date": curr_date,
                "placeholder": None,
            }
            final_prompt_list.append(final_prompt)
            fin_prompt_len.append(len(curr_prompt))
            temp_data.append(data)
    data_df = pd.DataFrame(temp_data)

    return final_prompt_list, fin_prompt_len, data_df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Get size")
    parser.add_argument("num", type=int, help="An integer number, 0 for 256, otherwise 512")
    args = parser.parse_args()

    parity = args.num

    prompts_df = get_prompts(num=45, random=True, size=parity)
    texts_df = get_text_data(num=7, random=True, size=parity)

    tokenizer = AutoTokenizer.from_pretrained("google/gemma-7b-it", device_map=dev, device=dev, torch_dtype=torch.bfloat16)
    model = AutoModelForCausalLM.from_pretrained(
        "google/gemma-7b-it", device_map=dev, torch_dtype=torch.bfloat16, do_sample=True, temperature=4.20, top_k=50, top_p=0.69
    )

    pipe = pipeline(
        "text-generation",
        model=model,
        # model_kwargs={"torch_dtype": torch.float16,
        #               "do_sample":True},
        tokenizer=tokenizer,
        batch_size=4,
        return_full_text=False,
        do_sample=True,
        temperature=0.69,
        top_k=50,
        top_p=0.69,
    )

    final_prompts, fin_prompt_len, results_df = format_prompts(texts_df=texts_df, prompts_df=prompts_df)

    class MyDataset(Dataset):
        def __init__(self, arr):
            self.arr = arr

        def __len__(self):
            return len(self.arr)

        def __getitem__(self, i):
            return self.arr[i]

    test_final_prompts = MyDataset(final_prompts)
    count = 0
    results_texts = []

    for out in tqdm(
        pipe(
            test_final_prompts,
            batch_size=4,
            max_new_tokens=256,
            padding=True,
            do_sample=True,
            temperature=0.65,
            top_k=50,
            top_p=0.70,
        ),
        total=len(final_prompts),
    ):

        text = out[0]["generated_text"]
        results_texts.append(text)

        if count % 10 == 0:
            print(f"Count:{count}")
            print(f"Prompt:{final_prompts[count]}")
            print(f"Text:{text}")
        count += 1

    count = 0
    for idx, row in results_df.iterrows():
        row["rewritten_text"] = results_texts[count]
        count += 1

    results_df.to_csv("results15.csv", index=False, encoding="utf-8")
