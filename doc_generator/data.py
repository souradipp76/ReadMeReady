"""
I will provide a list of github repositories, this function should act on the readme file of each repository and produce one csv file. So if I provide a list of 50 github repo URLs then the output should be 50 csv files each for each repository's readme file."""

# pip install requests

import requests
import os
import re
import pandas as pd
from urllib.parse import urlparse, quote
from urllib.parse import urlparse

def parse_markdown_to_csv(md_content, csv_file_path):
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
    df.to_csv(csv_file_path, index=False, encoding='utf-8')

def fetch_and_convert_readme_to_csv(repo_urls, output_dir):
    # Ensure the output directory exists
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # GitHub API endpoint for fetching the contents of the README file
    for url in repo_urls:
        parsed_url = urlparse(url)
        parts = parsed_url.path.strip('/').split('/')
        repo_user, repo_name = parts[0], parts[1]
        api_url = f"https://api.github.com/repos/{repo_user}/{repo_name}/readme"
        
        # Set up appropriate headers for GitHub API including the token for authorization
        headers = {
            'Accept': 'application/vnd.github.v3.raw',
            'Authorization': 'token YOUR_GITHUB_TOKEN'  # Replace 'YOUR_GITHUB_TOKEN' with your actual GitHub token
        }
        
        response = requests.get(api_url, headers=headers)
        if response.status_code == 200:
            readme_content = response.text
            csv_file_path = os.path.join(output_dir, f"{repo_name}.csv")
            parse_markdown_to_csv(readme_content, csv_file_path)
            print(f"Processed {repo_name}.csv")
        else:
            print(f"Failed to fetch README for {repo_name}: {response.status_code}")

# Example usage:
repo_urls = [
    'https://github.com/user/repo1',
    'https://github.com/user/repo2',
    # Add more GitHub repository URLs here
]

fetch_and_convert_readme_to_csv(repo_urls, 'output_csv_files')


"""convert each github repository to a string and store is in a csv file. Suppose I provide a list of github URLs. For each repository, we will look at the source code, take each file and convert the entire thing into string. After doing this for each source code file, we join all the strings with "\n". Then store the huge concatenated string in a csv file. So for 10 github URLs, we will have 10 csv files, each one contains one string, corresponding to the entire source code from all files concatenated as one string."""



def fetch_and_concatenate_source_code(repo_urls, output_dir, token):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    headers = {
        'Authorization': f'token {token}',
        'Accept': 'application/vnd.github.v3+json'
    }

    for url in repo_urls:
        parsed_url = urlparse(url)
        parts = parsed_url.path.strip('/').split('/')
        repo_user, repo_name = parts[0], parts[1]

        # Fetch the default branch
        repo_info_url = f'https://api.github.com/repos/{repo_user}/{repo_name}'
        repo_info_response = requests.get(repo_info_url, headers=headers)
        if repo_info_response.status_code == 200:
            default_branch = repo_info_response.json()['default_branch']
        else:
            print(f'Failed to fetch repo info for {repo_name}: {repo_info_response.status_code}')
            continue

        api_url = f'https://api.github.com/repos/{repo_user}/{repo_name}/git/trees/{default_branch}?recursive=true'

        response = requests.get(api_url, headers=headers)
        if response.status_code == 200:
            data = response.json()
            all_files_content = []

            for file in data['tree']:
                if file['type'] == 'blob' and file['path'].endswith(('.py', '.c', '.cpp', '.java', '.js', '.ts', '.go')):
                    file_content_response = requests.get(file['url'], headers=headers)
                    if file_content_response.status_code == 200:
                        file_content = file_content_response.json()['content']
                        all_files_content.append(file_content)

            concatenated_content = "\n".join(all_files_content)
            df = pd.DataFrame([concatenated_content], columns=['SourceCode'])
            df.to_csv(os.path.join(output_dir, f'{repo_name}.csv'), index=False)
            print(f'Saved {repo_name}.csv')
        else:
            print(f'Failed to fetch repository data for {repo_name}: {response.status_code}')

# Example usage:
repo_urls = [
    'https://github.com/user/repo1',
    'https://github.com/user/repo2',
]
output_directory = 'output_csv_files'
github_token = 'YOUR_GITHUB_TOKEN'

fetch_and_concatenate_source_code(repo_urls, output_directory, github_token)
