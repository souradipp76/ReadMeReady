import re
import bert_score
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from transformers import AutoTokenizer

import os
import subprocess
import pandas as pd

from readme_ready.query import query
from readme_ready.index import index
from readme_ready.types import (
    AutodocReadmeConfig,
    AutodocRepoConfig,
    AutodocUserConfig,
    LLMModels,
)

# Choose model from supported models
model = LLMModels.LLAMA2_7B_CHAT_GPTQ

# Clone repository to a local path
def git_clone(repo_url, clone_path):
    if os.path.exists(clone_path):
        subprocess.run(['rm', '-rf', clone_path], check=True)
    subprocess.run(['git', 'clone', repo_url, clone_path], check=True)
    subprocess.run(['rm', '-rf', os.path.join(clone_path,".git")], check=True)
    subprocess.run(['rm', '-rf', os.path.join(clone_path,".gitignore")], check=True)

# Parse the README.md content
def parse_markdown(md_file_path):
    with open(md_file_path, 'r', encoding='utf-8') as file:
        md_content = file.read()

    heading_pattern = re.compile(r'^(#+)\s*(.*)', re.MULTILINE)
    headings_contents = []
    current_heading = None
    current_content = []

    for line in md_content.split('\n'):
        match = heading_pattern.match(line)
        if match:
            if current_heading is not None:
                headings_contents.append([current_heading, ' '.join(current_content).strip()])
            current_heading = match.group(2).strip()
            current_content = []
        else:
            if line.strip():
                current_content.append(line.strip())

    if current_heading is not None:
        headings_contents.append([current_heading, ' '.join(current_content).strip()])

    df = pd.DataFrame(headings_contents, columns=['Title', 'Content'])
    return df, md_content

def clean_emoji(tx):
    emoji_pattern = re.compile("["
        u"\U0001F600-\U0001F64F"  # emoticons
        u"\U0001F300-\U0001F5FF"  # symbols
        u"\U0001F680-\U0001F6FF"  # transport
        u"\U0001F1E0-\U0001F1FF"  # flags
        u"\U00002702-\U000027B0"
        u"\U000024C2-\U0001F251"
        "]+", flags=re.UNICODE
    )

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
    # text = re.sub('https?://\S+|www\.\S+', ' ', text)
    text = re.sub(r'https?://[^\s\")]+', '', text)
    text = re.sub(r'http?://[^\s\")]+', '', text)
    text = re.sub(r'http%3A%2F%2F[^\s\")]+', '', text)
    text = re.sub(r'https%3A%2F%2F[^\s\")]+', '', text)
    text = re.sub(r'[^a-zA-Z0-9\!\?\.\@]',' ' , text)
    text = re.sub(r'[!]+' , '!' , text)
    text = re.sub(r'[?]+' , '?' , text)
    text = re.sub(r'[.]+' , '.' , text)
    text = re.sub(r'[@]+' , '@' , text)
    text = re.sub(r'unk' , '<UNK>' , text)
    # text = re.sub('\n', '<NL>', text)
    # text = re.sub('\t', '<TAB>', text)
    # text = re.sub(r'\s+', '<SP>', text)
    # text = re.sub(r'(<img[^>]*\bsrc=")[^"]*(")', '<img src=<IMG_SRC>', text)

    text = text.lower()
    text = re.sub(r'[ ]+' , ' ' , text)

    return text

def generate_readme(name, url, repo_root, output, headings):
    repo_config = AutodocRepoConfig (
        name = name,
        root = repo_root,
        repository_url = url,
        output = output,
        llms = [model],
        peft_model_path = None, # Set path to PEFT model
        ignore = [
            ".*",
            "*package-lock.json",
            "*package.json",
            "node_modules",
            "*dist*",
            "*build*",
            "*test*",
            "*.svg",
            "*.md",
            "*.mdx",
            "*.toml"
        ],
        file_prompt = "",
        folder_prompt = "",
        chat_prompt = "",
        content_type = "docs",
        target_audience = "smart developer",
        link_hosted = True,
        priority = None,
        max_concurrent_calls = 50,
        add_questions = False,
        device = "auto",
    )

    user_config = AutodocUserConfig(
        llms = [model]
    )

    readme_config = AutodocReadmeConfig(
        headings = headings
    )

    index.index(repo_config)
    query.generate_readme(repo_config, user_config, readme_config)

def get_score(url, readme_df, readme_content, generated_readme_df, generated_readme_content):
    scores = {'repo': url}
    # readme_df["Content"] = readme_df["Content"].apply(text_cleaner)
    # readme_df["Content"] = readme_df["Content"].apply(clean_emoji)
    # generated_readme_df["Content"] = generated_readme_df["Content"].apply(text_cleaner)
    # generated_readme_df["Content"] = generated_readme_df["Content"].apply(clean_emoji)

    # combined_df = pd.merge(readme_df, generated_readme_df, on="Title", suffixes=('_target', '_pred'))

    # pred = "\n".join(combined_df["Content_pred"].tolist())
    # target = "\n".join(combined_df["Content_target"].tolist())

    pred = generated_readme_content
    target = readme_content

    pred = re.sub(r' +', ' ', pred)
    target = re.sub(r' +', ' ', target)
    P, R, F1 = bert_score.score([pred], [target], lang='en', model_type='roberta-large', verbose=True)
    print(P,R,F1)

    scores['P'] = P.mean().item()
    scores['R'] = R.mean().item()
    scores['F1'] = F1.mean().item()

    tokenizer = AutoTokenizer.from_pretrained("TheBloke/Llama-2-7b-Chat-GPTQ")
    tokenizer.pad_token = tokenizer.eos_token

    def calculate_bleu(reference, candidate):
        reference_tokens = tokenizer.tokenize(reference)
        candidate_tokens = tokenizer.tokenize(candidate)
        smoothie = SmoothingFunction().method4
        return sentence_bleu([reference_tokens], candidate_tokens, smoothing_function=smoothie)

    bleu_score = calculate_bleu(target, pred)
    print(bleu_score)
    scores['bleu'] = bleu_score
    return scores

def main():
    repos = [
        "https://github.com/allenai/allennlp",
        "https://github.com/wting/autojump",
        "https://github.com/deezer/spleeter",
        "https://github.com/ddbourgin/numpy-ml",
        "https://github.com/eth-siplab/TouchPose"
    ]
    scores = []
    for repo in repos:
        name = repo.split("/")[-1]
        url = repo
        repo_root = f"./{name}"
        output = f"./output/{name}"
        
        git_clone(repo, f"./{name}")
        
        readme_df, readme_content = parse_markdown(f"{repo_root}/README.md")
        headings = ",".join(readme_df["Title"].tolist()[:2])
        
        generate_readme(name, url, repo_root, output, headings)
        generated_readme_df, generated_readme_content = parse_markdown(f"{output}/docs/data/README_LLAMA2_7B_CHAT_GPTQ.md")

        score = get_score(url, readme_df, readme_content, generated_readme_df, generated_readme_content)
        scores.append(score)
        scores_df = pd.DataFrame(scores, index=None)
        scores_df.to_csv("./scores.csv", index=False)
        subprocess.run(['rm', '-rf', repo_root], check=True)

if __name__ == "__main__":
    main()