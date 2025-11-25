#!/usr/bin/env python3
from huggingface_hub import HfApi
import os

# Read token from .env file
with open('../../.env', 'r') as f:
    for line in f:
        if line.startswith('HF_API_KEY'):
            token = line.split('=', 1)[1].strip()
            break

api = HfApi()
repo_id = 'professorsynapse/nexus-tools-v0.0.2'

# Create repository if it doesn't exist
try:
    api.create_repo(repo_id=repo_id, token=token, exist_ok=True, repo_type="model")
    print(f'✓ Repository {repo_id} ready')
except Exception as e:
    print(f'Repository may already exist: {e}')

# Upload Q4_K_M
print('Uploading Q4_K_M (4.1GB)...')
api.upload_file(
    path_or_fileobj='gguf_output/model-Q4_K_M.gguf',
    path_in_repo='model-Q4_K_M.gguf',
    repo_id=repo_id,
    token=token
)
print('✓ Q4_K_M uploaded')

# Upload Q5_K_M
print('Uploading Q5_K_M (3.1GB)...')
api.upload_file(
    path_or_fileobj='gguf_output/model-Q5_K_M.gguf',
    path_in_repo='model-Q5_K_M.gguf',
    repo_id=repo_id,
    token=token
)
print('✓ Q5_K_M uploaded')

# Upload Q8_0
print('Uploading Q8_0 (4.7GB)...')
api.upload_file(
    path_or_fileobj='gguf_output/model-Q8_0.gguf',
    path_in_repo='model-Q8_0.gguf',
    repo_id=repo_id,
    token=token
)
print('✓ Q8_0 uploaded')

print('\nAll GGUF files uploaded successfully!')
