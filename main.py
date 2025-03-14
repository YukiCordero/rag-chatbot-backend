from fastapi import FastAPI
from langchain.vectorstores import FAISS
from langchain.embeddings.huggingface import HuggingFaceEmbeddings
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

app = FastAPI()

# Load DeepSeek R1
model_name = "deepseek-ai/deepseek-moe-16b"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16, device_map="auto")

# Load FAISS vector database
embeddings = HuggingFaceEmbeddings(model_name="deepseek-ai/deepseek-text-7b")
vectorstore = FAISS.load_local("medical_vector_db", embeddings)
retriever = vectorstore.as_retriever()

def generate_response(query):
    inputs = tokenizer(query, return_tensors="pt").to("cuda")
    outputs = model.generate(**inputs, max_length=500)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

@app.get("/chat/")
def chat(query: str):
    retrieved_docs = retriever.get_relevant_documents(query)
    
    if retrieved_docs:
        response = retrieved_docs[0].page_content  # Use database response
    else:
        response = generate_response(f"User asked: {query}")  # Use AI model

    return {"response": response}
