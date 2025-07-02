import chainlit as cl
import re
import numpy as np
import faiss  
import fitz

from openai import AsyncOpenAI

client = AsyncOpenAI(
    base_url="http://127.0.0.1:1234/v1",
    api_key="not-needed"
)

settings = {
    "model": "qwen3-0.6b",
    "temperature": 0.9,
    "stream": True,
}

documents = [
    "Chainlit makes LLM apps interactive.",
    "FAISS is used for fast vector similarity search.",
    "Qwen is a multilingual open-source LLM.",
    "Embeddings are numerical representations of text.",
    "User name is Alan David Wilson"
]

async def get_embedding(text: str, model="text-embedding-qwen3-embedding-0.6b") -> list:
    response = await client.embeddings.create(
        input=[text],
        model=model
    )
    return response.data[0].embedding

@cl.on_chat_start
async def start_chat():
    cl.user_session.set(
        "message_history",
        [{"role": "system", "content": "You are a helpful assistant."}],
    )

    # Embed the default documents and build the FAISS index
    embeddings = []
    for doc in documents:
        embedding = await get_embedding(doc)
        embeddings.append(embedding)
    embedding_dim = len(embeddings[0])
    index = faiss.IndexFlatL2(embedding_dim)
    index.add(np.array(embeddings).astype("float32"))
    cl.user_session.set("index", index)
    cl.user_session.set("documents", documents)

@cl.on_message
async def on_message(message: cl.Message):
    message_history = cl.user_session.get("message_history")
    files = message.elements

    # If a file is uploaded, rebuild the index and documents
    if files:
        uploaded_file = files[0]

        with fitz.open(uploaded_file.path) as doc:
            text = ""
            for page in doc:
                text += page.get_text()

        if not text.strip():
            await cl.Message(content="⚠️ No readable text found in the PDF.").send()
            return

        new_documents = [chunk.strip() for chunk in text.split("\n\n") if chunk.strip()]
        embeddings = []
        for doc in new_documents:
            embedding = await get_embedding(doc)
            embeddings.append(embedding)
        if embeddings:
            embedding_dim = len(embeddings[0])
            index = faiss.IndexFlatL2(embedding_dim)
            index.add(np.array(embeddings).astype("float32"))
            cl.user_session.set("index", index)
            cl.user_session.set("documents", new_documents)
        else:
            await cl.Message(content="⚠️ No text extracted from the PDF. Please upload a readable PDF.").send()
            return

    # Always use the session's index and documents
    index = cl.user_session.get("index")
    documents = cl.user_session.get("documents")

    # Step 1: Embed the user's query
    query_embedding = await get_embedding(message.content)

    # Step 2: Search FAISS index
    D, I = index.search(np.array([query_embedding]).astype("float32"), k=3)

    # Step 3: Retrieve top-matching documents
    retrieved_docs = [documents[i] for i in I[0]]
    context = "\n\n".join(retrieved_docs)

    # Step 4: Inject context into prompt
    prompt = f"""Use the following documents as context to help answer the question:\n\n{context}\n\nQuestion: {message.content} /no_think"""

    message_history.append({"role": "user", "content": prompt})

    msg = cl.Message(content="")
    full_response = ""

    stream = await client.chat.completions.create(
        messages=message_history, **settings
    )

    async for part in stream:
        if token := (part.choices[0].delta.content or ""):
            full_response += token
            # Remove thinking tags from the accumulated response
            cleaned_response = re.sub(r'<think>.*?</think>', '', full_response, flags=re.DOTALL)
            cleaned_response = re.sub(r'<think>', '', cleaned_response)
            # Stream the cleaned content
            if len(cleaned_response) > len(msg.content):
                new_content = cleaned_response[len(msg.content):]
                await msg.stream_token(new_content)

    # Save the assistant's reply to the history
    message_history.append({"role": "assistant", "content": msg.content})
    await msg.update()
    print("🔍 Extracted text preview:", text[:300])
