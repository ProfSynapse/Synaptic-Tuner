#!/usr/bin/env python3
import sys

with open('train.ps1', 'r', encoding='utf-8') as f:
    content = f.read()

# Replace all smart quotes with straight quotes
content = content.replace('\u2018', "'")  # Left single quote
content = content.replace('\u2019', "'")  # Right single quote
content = content.replace('\u201C', '"')  # Left double quote
content = content.replace('\u201D', '"')  # Right double quote

with open('train.ps1', 'w', encoding='utf-8') as f:
    f.write(content)

print('Fixed all smart quotes')
