import os
import requests
from context import getContext
import chromadb

#API_TOKEN = "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"

API_TOKEN=os.environ["API_TOKEN"] #Set a API_TOKEN environment variable before running
API_URL = "https://api-inference.huggingface.co/models/openchat/openchat-3.5-0106"

#API_URL = "https://api-inference.huggingface.co/models/HuggingFaceH4/zephyr-7b-beta"

headers = {"Authorization": f"Bearer {API_TOKEN}"}

def query(prompt):
    payload = {
        "inputs": prompt,
        "parameters": { #Try and experiment with the parameters
            "max_new_tokens": 1024,
            "temperature": 0.6,
            "top_p": 0.5,
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
getContext()
chroma_client= chromadb.PersistentClient(path="./chromadb")
collection=chroma_client.get_collection(name="countries")
#print(collection.peek())

from transformers import AutoTokenizer, AutoModelForTokenClassification
ner_model_id = 'dslim/bert-base-NER'
tokenizer = AutoTokenizer.from_pretrained(ner_model_id)
ner_model = AutoModelForTokenClassification.from_pretrained(ner_model_id)

from transformers import pipeline
ner_pipeline = pipeline("ner", model=ner_model, tokenizer=tokenizer)



while 1:
    question=getQuestion()
    if question.lower()=="quit":
        break
    #print(question)
    #question = "Who is the mayor of Jacksonville, Florida?"
    #context = "Donna Deagon became the mayor of Jacksonville FL in 2023."
    #print(question)
    #country=ner_pipeline(question)[0]["word"]

    context=collection.query(query_texts=question,n_results=1)["documents"]
    #print(context)

    prompt = f"""Use the following context to answer the question at the end. Stop when you've answered the question. Do not generate any more than that.

    {context}

    Question: {question}
    """
    #print(prompt)
    print("HF Model : ")
    print(query(prompt))






