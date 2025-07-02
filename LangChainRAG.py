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
    results = vectorstore.similarity_search(query, k=3)

    # Print top matches
    print("\nTop matching chunks:")
    for i, doc in enumerate(results, 1):
        print(f"\n[{i}] {doc.page_content.strip()}")

# Run 
if __name__ == "__main__":
    build_faiss_vectorstore()
    retrieve_from_faiss("Where does Lucius Wilbert Tjoa work?")
