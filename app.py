import os
import requests
from context import getContext
import chromadb

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

def getQuestion():
    print("")
    question=input("Enter your question (quit to stop): ")
    return question

##########Main################
chroma_client= chromadb.PersistentClient(path="./chromadb")

try:
    collection=chroma_client.create_collection(name="countries")
    print("Created Countries collection ")
    #chroma_client.delete_collection("countries")
except chromadb.db.base.UniqueConstraintError:
    print("Countries collection already exists")
    collection=chroma_client.get_collection(name="countries")



context=getContext()

collection.add(documents=[context],             metadatas=[{"type":"country"}],             ids=["amaze"])

con= collection.query(query_texts=["amaze"],
                    n_results=1)




while 1:
    question=getQuestion()
    if question.lower()=="quit":
        break
    #print(question)
    #question = "Who is the mayor of Jacksonville, Florida?"
    #context = "Donna Deagon became the mayor of Jacksonville FL in 2023."
    prompt = f"""Use the following context to answer the question at the end.

    {context}

    Question: {question}
    """
    #print(prompt)
    print("HF Model : ")
    print(query(prompt))
