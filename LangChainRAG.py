# app.py
import os
import re
import requests
import chainlit as cl
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_core.embeddings import Embeddings

# Custom embedding class for LM Studio's Qwen
class QwenLocalEmbeddings(Embeddings):
    def __init__(self, model="text-embedding-qwen3-embedding-0.6b", base_url="http://127.0.0.1:1234/v1"):
        self.model = model
        self.base_url = base_url
        self.api_key = "lm-studio"

    def embed_documents(self, texts):
        return [self.embed_query(text) for text in texts]

    def embed_query(self, text):
        payload = {"model": self.model, "input": text}
        headers = {"Authorization": f"Bearer {self.api_key}"}
        response = requests.post(f"{self.base_url}/embeddings", json=payload, headers=headers)
        if response.status_code != 200:
            raise Exception(f"Embedding failed: {response.status_code} - {response.text}")
        return response.json()["data"][0]["embedding"]

# Build FAISS index (once)
def build_faiss_vectorstore(doc_path="docs/doc1.txt", save_path="faiss_index"):
    with open(doc_path, "r", encoding="utf-8") as f:
        raw_text = f.read()

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=700, chunk_overlap=50)
    documents = text_splitter.create_documents([raw_text])
    texts = [doc.page_content for doc in documents]

    embedding_model = QwenLocalEmbeddings()
    vectorstore = FAISS.from_texts(texts, embedding_model)
    vectorstore.save_local(save_path)

# Answer generation using LM Studio (Qwen)
async def generate_answer_stream(query, docs, msg: cl.Message):
    context = "\n\n".join([doc.page_content for doc in docs])
    prompt = f"""You are a helpful assistant. Use the context below to answer the question. /no_think

Context:
{context}

Question: {query}
Answer:"""

    payload = {
        "model": "qwen3-0.6b",
        "messages": [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt}
        ]
    }
    headers = {"Authorization": "Bearer lm-studio"}
    response = requests.post("http://localhost:1234/v1/chat/completions", json=payload, headers=headers)

    if response.status_code != 200:
        raise Exception(f"LLM call failed: {response.status_code} - {response.text}")

    answer = response.json()["choices"][0]["message"]["content"]

    # Clean <think> tags
    answer = re.sub(r'<think>.*?</think>', '', answer, flags=re.DOTALL)
    answer = re.sub(r'<think>', '', answer)

    # Simulate streaming: send chunk-by-chunk
    for chunk in answer.split(". "):  # You can also split by \n if you prefer
        await msg.stream_token(chunk.strip() + ". ")
        await cl.sleep(0.05)  # Small delay for smoother streaming

    await msg.send()  # Finalize the message box


# Run on user input
@cl.on_message
async def main(message: cl.Message):
    query = message.content.strip()
    embedding_model = QwenLocalEmbeddings()
    vectorstore = FAISS.load_local("faiss_index", embedding_model, allow_dangerous_deserialization=True)
    results = vectorstore.similarity_search(query, k=5)

    # Show retrieved chunks
    # await cl.Message(content="🔎 Top matching chunks:").send()
    # for i, doc in enumerate(results, 1):
    #     await cl.Message(content=f"**[{i}]** {doc.page_content.strip()}").send()

    # Generate and show answer
    msg = cl.Message("")
    await msg.send()  # Show empty message container first
    await generate_answer_stream(query, results, msg)
