# app.py
import os
import re
import requests
import chainlit as cl
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_core.embeddings import Embeddings


import google.generativeai as genai
import pathlib

from dotenv import load_dotenv
load_dotenv()

genai.configure(api_key=os.getenv("GEMINI_API_KEY"))  # Replace with your actual key



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

def extract_text_from_pdf_to_file(pdf_file_path: str, output_file_path: str = "docs/doc1.txt") -> str:
    pdf_bytes = pathlib.Path(pdf_file_path).read_bytes()

    model = genai.GenerativeModel("models/gemini-1.5-flash")  # or 2.5-flash
    response = model.generate_content([
        {
            "inline_data": {
                "mime_type": "application/pdf",
                "data": pdf_bytes
            }
        },
        {
            "text": "Give me the plain text of the document in detailed format and do not miss any information at all. You can give me in the full and understandable form"
        }
    ])

    with open(output_file_path, "w", encoding="utf-8") as f:
        f.write(response.text)

    return response.text


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

@cl.on_chat_start
def on_chat_start():
    cl.user_session.set("counter", 0)

# Run on user input
@cl.on_message
async def main(message: cl.Message):
    counter = cl.user_session.get("counter", 0)
    counter += 1
    cl.user_session.set("counter", counter)

    # If the user uploaded a file (no prompt)
    if message.elements:
        for element in message.elements:
            if isinstance(element, cl.File) and element.name.endswith(".pdf"):
                file_path = element.path
                await cl.Message(content=f"File received: {element.name}. Processing...").send()

                # Gemini extracts to doc1.txt
                extract_text_from_pdf_to_file(file_path)

                # Rebuild FAISS index
                build_faiss_vectorstore()

                await cl.Message(content="Document processed. You can now ask questions.").send()
                return

    # If it's a normal query
    query = message.content.strip()

    # Load vector store and search
    embedding_model = QwenLocalEmbeddings()
    vectorstore = FAISS.load_local("faiss_index", embedding_model, allow_dangerous_deserialization=True)
    results = vectorstore.similarity_search(query, k=5)

    # Generate streaming answer
    msg = cl.Message("")
    await msg.send()
    await generate_answer_stream(query, results, msg)