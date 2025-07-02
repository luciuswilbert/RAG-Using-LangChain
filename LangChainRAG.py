import os
import requests
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_core.embeddings import Embeddings

# Custom embedding class using LM Studio (Qwen)
class QwenLocalEmbeddings(Embeddings):
    def __init__(self, model="text-embedding-qwen3-embedding-0.6b", base_url="http://127.0.0.1:1234/v1"):
        self.model = model
        self.base_url = base_url
        self.api_key = "lm-studio"  

    def embed_documents(self, texts):
        return [self.embed_query(text) for text in texts]

    def embed_query(self, text):
        payload = {
            "model": self.model,
            "input": text
        }
        headers = {
            "Authorization": f"Bearer {self.api_key}"
        }
        response = requests.post(f"{self.base_url}/embeddings", json=payload, headers=headers)
        if response.status_code != 200:
            raise Exception(f"Embedding failed: {response.status_code} - {response.text}")
        return response.json()["data"][0]["embedding"]

# Main function to process, embed, and save FAISS index
def build_faiss_vectorstore(doc_path="docs/doc1.txt", save_path="faiss_index"):
    # Load text
    with open(doc_path, "r", encoding="utf-8") as f:
        raw_text = f.read()

    # Chunk 
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    documents = text_splitter.create_documents([raw_text])
    texts = [doc.page_content for doc in documents]

    # Embed with custom Qwen embeddings
    embedding_model = QwenLocalEmbeddings()

    # Store in FAISS
    vectorstore = FAISS.from_texts(texts, embedding_model)
    vectorstore.save_local(save_path)
    print(f"FAISS Index saved to `{save_path}`")

def retrieve_from_faiss(query, index_path="faiss_index"):
    # Reuse the same embedding model for querying
    embedding_model = QwenLocalEmbeddings()

    # Load FAISS index
    vectorstore = FAISS.load_local(index_path, embedding_model, allow_dangerous_deserialization=True)


    # Run similarity search
    results = vectorstore.similarity_search(query, k=20)

    # Print top matches
    print("\nTop matching chunks:")
    for i, doc in enumerate(results, 1):
        print(f"\n[{i}] {doc.page_content.strip()}")

    generate_answer(query, results)

def generate_answer(query, docs):
    # Build the prompt: context + query
    context = "\n\n".join([doc.page_content for doc in docs])
    prompt = f"""You are a helpful assistant. Use the context below to answer the question.

Context:
{context}

Question: {query}
Answer:"""

    # Send to LM Studio's Qwen chat model
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
    print("\n🤖 AI Answer:")
    print(answer)


# Run 
if __name__ == "__main__":
    build_faiss_vectorstore()
    retrieve_from_faiss(input("Enter your question: ") + "/no_think")
