
import json
import os
from pathlib import Path

import requests


def get_llm_response(test_data):
     return requests.post(
        "https://rahulshettyacademy.com/rag-llm/ask",
        json={
            "question": test_data["question"],
            "chat_history": [],
        },
    ).json() 


def load_test_data(file_name):
    project_directory = Path(__file__).parent.absolute()
    file_path = project_directory/"testdata"/file_name
    with open(file_path, 'r') as f:
        return json.load(f)