import pandas as pd
import json
import os
from pprint import pprint
import bitsandbytes as bnb
import torch
import torch.nn as nn
import transformers
from datasets import load_dataset, Dataset
from huggingface_hub import notebook_login

from peft import LoraConfig, PeftConfig, PeftModel, get_peft_model, prepare_model_for_kbit_training
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"

MODEL_NAME = "TheBloke/Llama-2-7b-Chat-GPTQ"

model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.float16,
        trust_remote_code=True,
        device_map="auto",
    )

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
tokenizer.pad_token = tokenizer.eos_token

model = prepare_model_for_kbit_training(model)

import re
def get_num_layers(model):
    numbers = set()
    for name, _ in model.named_parameters():
        for number in re.findall(r'\d+', name):
            numbers.add(int(number))
    return max(numbers)

def get_last_layer_linears(model):
    names = []
    
    num_layers = get_num_layers(model)
    for name, module in model.named_modules():
        if str(num_layers) in name and not "encoder" in name:
            if isinstance(module, torch.nn.Linear):
                names.append(name)
    print(names)
    return names

config = LoraConfig(
    r=2,
    lora_alpha=32,
    # target_modules=get_last_layer_linears(model),
    target_modules = ["k_proj","o_proj","q_proj","v_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)

model = get_peft_model(model, config)

df = pd.read_csv("cleaned_data.csv")
df.columns = [str(q).strip() for q in df.columns]

df.dropna(subset=["Answer"], inplace=True)
df = df[["Question", "Context", "Answer", "Repo Url", "Repo"]]

def clean_text(text):
    # Define the regular expression pattern for HTTP URLs
    http_pattern = re.compile(r'http://[^\s]+')
    # Remove HTTP URLs
    text = http_pattern.sub('', str(text))

    https_pattern = re.compile(r'https://[^\s]+')
    # Remove HTTPS URLs
    text = https_pattern.sub('', str(text))
    
    # Define the regular expression pattern for <img> tags
    img_pattern = re.compile(r'<img[^>]*>')
    # Remove <img> tags
    text = img_pattern.sub('', str(text))
    
    return text

def clean_emoji(tx):
    emoji_pattern = re.compile("["
                           u"\U0001F600-\U0001F64F"  # emoticons
                           u"\U0001F300-\U0001F5FF"  # symbols 
                           u"\U0001F680-\U0001F6FF"  # transport 
                           u"\U0001F1E0-\U0001F1FF"  # flags 
                           u"\U00002702-\U000027B0"
                           u"\U000024C2-\U0001F251"
                           "]+", flags=re.UNICODE)

    return emoji_pattern.sub(r'', tx)

def text_cleaner(tx):

    text = re.sub(r"won\'t", "would not", tx)
    text = re.sub(r"im", "i am", tx)
    text = re.sub(r"Im", "I am", tx)
    text = re.sub(r"can\'t", "can not", text)
    text = re.sub(r"don\'t", "do not", text)
    text = re.sub(r"shouldn\'t", "should not", text)
    text = re.sub(r"needn\'t", "need not", text)
    text = re.sub(r"hasn\'t", "has not", text)
    text = re.sub(r"haven\'t", "have not", text)
    text = re.sub(r"weren\'t", "were not", text)
    text = re.sub(r"mightn\'t", "might not", text)
    text = re.sub(r"didn\'t", "did not", text)
    text = re.sub(r"n\'t", " not", text)
    text = re.sub(r"\'re", " are", text)
    text = re.sub(r"\'s", " is", text)
    text = re.sub(r"\'d", " would", text)
    text = re.sub(r"\'ll", " will", text)
    text = re.sub(r"\'t", " not", text)
    text = re.sub(r"\'ve", " have", text)
    text = re.sub(r"\'m", " am", text)
    text = re.sub('https?://\S+|www\.\S+', '', text)
    text = re.sub(r'[^a-zA-Z0-9\!\?\.\@]',' ' , text)
    text = re.sub(r'[!]+' , '!' , text)
    text = re.sub(r'[?]+' , '?' , text)
    text = re.sub(r'[.]+' , '.' , text)
    text = re.sub(r'[@]+' , '@' , text)
    text = re.sub(r'unk' , ' ' , text)
    text = re.sub('\n', '', text)
    text = text.lower()
    text = re.sub(r'[ ]+' , ' ' , text)

    return text

# df["Answer"] = df["Answer"].apply(clean_text)
# df["Answer"] = df["Answer"].apply(text_cleaner)
# df["Answer"] = df["Answer"].apply(clean_emoji)

project_name = df["Repo"].values[10820]
repository_url = df["Repo Url"].values[10820]
target_audience = "smart developer"
question = df["Question"].values[10820]
context = df["Context"].values[10820]
content_type = "docs"
prompt = f"""You are an AI assistant for a software project called {project_name}. You are trained on all the {content_type} that makes up this project.
    The {content_type} for the project is located at {repository_url}.
    You are given a repository which might contain several modules and each module will contain a set of files.
    Look at the source code in the repository and you have to generate content for the section of a README.md file following the heading given below. If you use any hyperlinks, they should link back to the github repository shared with you.
    You should only use hyperlinks that are explicitly listed in the context. Do NOT make up a hyperlink that is not listed.

    Assume the reader is a {target_audience} but is not deeply familiar with {project_name}.
    Assume the reader does not know anything about how the project is structured or which folders/files do what and what functions are written in which files and what these functions do.
    If you don't know how to fill up the readme.md file in one of its sections, leave that part blank. Don't try to make up any content.
    Do not include information that is not directly relevant to repository, even though the names of the functions might be common or is frequently used in several other places.
    Keep your response between 100 and 300 words. DO NOT RETURN MORE THAN 300 WORDS. Provide the answer in correct markdown format.

    Question: {question}
    Context:
    {context}

    Answer in Markdown:"""


def generate_prompt(data_point):
#     return f"""
#             {data_point["Question"]}. 
#             Answer as briefly as possible: {data_point["Answer"]}
#             """.strip()
    return f"""You are an AI assistant for a software project called {data_point["Repo"]}. You are trained on all the {content_type} that makes up this project.
    The docs for the project is located at {data_point["Repo Url"]}.
    You are given a repository which might contain several modules and each module will contain a set of files.
    Look at the source code in the repository and you have to generate content for the section of a README.md file following the heading given below. If you use any hyperlinks, they should link back to the github repository shared with you.
    You should only use hyperlinks that are explicitly listed in the context. Do NOT make up a hyperlink that is not listed.

    Assume the reader is a smart developer but is not deeply familiar with {data_point["Repo"]}.
    Assume the reader does not know anything about how the project is structured or which folders/files do what and what functions are written in which files and what these functions do.
    If you don't know how to fill up the readme.md file in one of its sections, leave that part blank. Don't try to make up any content.
    Do not include information that is not directly relevant to repository, even though the names of the functions might be common or is frequently used in several other places.
    Keep your response between 100 and 300 words. DO NOT RETURN MORE THAN 300 WORDS. Provide the answer in correct markdown format.

    Question: {data_point["Question"]}
    Context:
    {data_point["Context"]}

    Answer in Markdown:
    {data_point["Answer"]}
    """

def generate_and_tokenize_prompt(data_point):
    full_prompt = generate_prompt(data_point)
    tokenized_full_prompt = tokenizer(full_prompt, padding=True, truncation=True)
    return tokenized_full_prompt

data = Dataset.from_pandas(df)
data = data.shuffle().map(generate_and_tokenize_prompt)


import os

os.environ['CUDA_LAUNCH_BLOCKING']="1"
os.environ['TORCH_USE_CUDA_DSA']="1"

training_args = transformers.TrainingArguments(
    per_device_train_batch_size=1,
    gradient_accumulation_steps=1,
    num_train_epochs=1,
    learning_rate=1e-4,
    fp16=True,
    output_dir="outputs_llama2-7b-chat-gptq_v3",
    optim="paged_adamw_8bit",
    lr_scheduler_type="cosine",
    warmup_ratio=0.01,
    report_to="none"
)

trainer = transformers.Trainer(
    model=model,
    train_dataset=data,
    args=training_args,
    data_collator=transformers.DataCollatorForLanguageModeling(tokenizer, mlm=False)
)
model.config.use_cache = False
trainer.train()

model.save_pretrained("outputs_llama2-7b-chat-gptq_v3/trained-model")
# PEFT_MODEL = "outputs_llama2-7b-chat-gptq_v2/checkpoint-11000"

# config = PeftConfig.from_pretrained(PEFT_MODEL)
# model = AutoModelForCausalLM.from_pretrained(
#     config.base_model_name_or_path,
#     return_dict=True,
#     # quantization_config=bnb_config,
#     device_map="auto",
#     trust_remote_code=True
# )

# tokenizer=AutoTokenizer.from_pretrained(config.base_model_name_or_path)
# tokenizer.pad_token = tokenizer.eos_token

# model = PeftModel.from_pretrained(model, PEFT_MODEL)

# # %%
# generation_config = model.generation_config
# generation_config.max_new_tokens = 512
# generation_config.temperature = 0.7
# generation_config.top_p = 0.7
# generation_config.num_return_sequences = 1
# generation_config.repetition_penalty = 1.1
# generation_config.pad_token_id = tokenizer.eos_token_id
# generation_config.eos_token_id = tokenizer.eos_token_id

# # %%
# import numpy as np

# # %%
# %%time
# project_name = df["Repo"].values[10820]
# repository_url = df["Repo Url"].values[10820]
# target_audience = "smart developer"
# question = df["Question"].values[10820]
# context = df["Context"].values[10820]
# content_type = "docs"
# prompt = f"""You are an AI assistant for a software project called {project_name}. You are trained on all the {content_type} that makes up this project.
#     The {content_type} for the project is located at {repository_url}.
#     You are given a repository which might contain several modules and each module will contain a set of files.
#     Look at the source code in the repository and you have to generate content for the section of a README.md file following the heading given below. If you use any hyperlinks, they should link back to the github repository shared with you.
#     You should only use hyperlinks that are explicitly listed in the context. Do NOT make up a hyperlink that is not listed.

#     Assume the reader is a {target_audience} but is not deeply familiar with {project_name}.
#     Assume the reader does not know anything about how the project is structured or which folders/files do what and what functions are written in which files and what these functions do.
#     If you don't know how to fill up the readme.md file in one of its sections, leave that part blank. Don't try to make up any content.
#     Do not include information that is not directly relevant to repository, even though the names of the functions might be common or is frequently used in several other places.
#     Keep your response between 100 and 300 words. DO NOT RETURN MORE THAN 300 WORDS. Provide the answer in correct markdown format.

#     Question: {question}
#     Context:
#     {context}

#     Answer in Markdown:"""
    
# device = "cuda"
# encoding = tokenizer(prompt, return_tensors="pt").to(device)
# with torch.inference_mode():
#   outputs = model.generate(
#       input_ids = encoding.input_ids,
#       attention_mask = encoding.attention_mask,
#       generation_config = generation_config
#   )

# print(tokenizer.decode(outputs[0], skip_special_tokens=True))




