# -*- coding: utf-8 -*-
"""
Created on Mon Feb 10 18:52:27 2025

@author: vanessa
"""

import json
import pdfplumber
import re
import os

# Folder paths
input_folder = "D:\\llm\\text\\government"
output_folder = "D:\\llm\\results"

# Ensure output folder exists
os.makedirs(output_folder, exist_ok=True)

# Function to extract text and clean it
def extract_text_from_pdf(file_path):
    text_data = []
    with pdfplumber.open(file_path) as pdf:
        for page in pdf.pages:
            text = page.extract_text()
            if text:
                paragraphs = re.split(r'\n{2,}', text)  # Split by double newline for paragraphs
                text_data.extend([p.strip() for p in paragraphs if p.strip()])
    return text_data

# Convert extracted text to JSONL format
jsonl_data = []
for filename in os.listdir(input_folder):
    if filename.endswith(".pdf"):
        file_path = os.path.join(input_folder, filename)
        match = re.search(r'_(\d{4})', filename)
        year = match.group(1) if match else "Unknown"
        doc_name = filename.replace(".pdf", "")
        
        text_paragraphs = extract_text_from_pdf(file_path)
        
        for paragraph in text_paragraphs:
            jsonl_data.append({
                "time": year,
                "document": doc_name,
                "content": paragraph
            })

# Save JSONL file
jsonl_file_path = os.path.join(output_folder, "cleaned_text_paragraphs.jsonl")
with open(jsonl_file_path, "w", encoding="utf-8") as f:
    for entry in jsonl_data:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")

print(f"JSONL file saved: {jsonl_file_path}")
