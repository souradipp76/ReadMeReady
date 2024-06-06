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

os.environ["CUDA_VISIBLE_DEVICES"] = "1"
os.environ['CUDA_LAUNCH_BLOCKING']="1"
os.environ['TORCH_USE_CUDA_DSA'] = "1"

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
    print(max(numbers))
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

df = pd.read_csv("readme_qa_cleaned_small_v4.csv")

target_audience = "smart developer"
content_type = "docs"

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

training_args = transformers.TrainingArguments(
    per_device_train_batch_size=1,
    gradient_accumulation_steps=1,
    num_train_epochs=3,
    learning_rate=1e-4,
    fp16=True,
    output_dir="outputs_llama2-7b-chat-gptq_v7",
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


# from trl import SFTTrainer
# trainer = SFTTrainer(
#     model=model,
#     args=training_args,
#     train_dataset=data,
#     peft_config=config,
#     dataset_text_field="text",
#     tokenizer=tokenizer,
#     packing=False,
#     data_collator=transformers.DataCollatorForLanguageModeling(tokenizer, mlm=False),
#     max_seq_length=512)

# model.config.use_cache = False
# trainer.train()

model.save_pretrained("outputs_llama2-7b-chat-gptq_v7/trained-model")
PEFT_MODEL = "outputs_llama2-7b-chat-gptq_v7/trained-model"

config = PeftConfig.from_pretrained(PEFT_MODEL)
model = AutoModelForCausalLM.from_pretrained(
    config.base_model_name_or_path,
    return_dict=True,
    # quantization_config=bnb_config,
    device_map="auto",
    trust_remote_code=True
)

tokenizer=AutoTokenizer.from_pretrained(config.base_model_name_or_path)
tokenizer.pad_token = tokenizer.eos_token

model = PeftModel.from_pretrained(model, PEFT_MODEL)

generation_config = model.generation_config
generation_config.max_new_tokens = 512
generation_config.temperature = 0.7
# generation_config.top_p = 0.7
# generation_config.num_return_sequences = 1
generation_config.repetition_penalty = 1.2
generation_config.pad_token_id = tokenizer.eos_token_id
generation_config.eos_token_id = tokenizer.eos_token_id

import numpy as np

project_name = df["Repo"].values[147]
repository_url = df["Repo Url"].values[147]
target_audience = "smart developer"
question = df["Question"].values[147]
context = df["Context"].values[147]
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
    
device = "cuda"
encoding = tokenizer(prompt, return_tensors="pt").to(device)
with torch.inference_mode():
  outputs = model.generate(
      input_ids = encoding.input_ids,
      attention_mask = encoding.attention_mask,
      generation_config = generation_config
  )

print(tokenizer.decode(outputs[0], skip_special_tokens=True))




