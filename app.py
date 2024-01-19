import os
import requests

#API_TOKEN = "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"

API_TOKEN=os.environ["API_TOKEN"] #Set a API_TOKEN environment variable before running
API_URL = "https://api-inference.huggingface.co/models/openchat/openchat-3.5-0106"
headers = {"Authorization": f"Bearer {API_TOKEN}"}

def query(prompt):
    payload = {
        "inputs": prompt,
        "parameters": { #Try and experiment with the parameters
            "max_new_tokens": 1024,
            "temperature": 0.6,
            "top_p": 0.9,
            "do_sample": False,
            "return_full_text": False
        }
    }
    response = requests.post(API_URL, headers=headers, json=payload)
    
    return response.json()[0]['generated_text']

question = "Who is the mayor of Jacksonville, Florida?"
context = "Donna Deagon became the mayor of Jacksonville FL in 2023."
prompt = f"""Use the following context to answer the question at the end.

{context}

Question: {question}
"""

print(prompt)
print(query(prompt))
