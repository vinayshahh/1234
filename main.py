import os
import json
from datetime import date
from dotenv import load_dotenv

from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_groq import ChatGroq

load_dotenv()

# ---------- setup

embedding_function = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

vector_store = Chroma(
    embedding_function=embedding_function,
    persist_directory="my_chroma_db",
    collection_name="sample",
    collection_metadata={"hnsw:space": "cosine"},
)

# pdf loading + chunking


def load_and_add_pdfs():
    loader = DirectoryLoader(
        path="data",
        glob="*.pdf",
        loader_cls=PyPDFLoader,
    )
    docs = loader.load()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=100,
    )
    finaldocs = splitter.split_documents(docs)

    vector_store.add_documents(finaldocs)


load_and_add_pdfs()  # only run this once when adding new PDFs, then comment it back


# ----retriever + model   #####

retriever = vector_store.as_retriever(
    search_type="similarity",
    search_kwargs={
        "k": 3,
    },
)

chat_model = ChatGroq(
    model="openai/gpt-oss-120b",
    temperature=0,
    api_key=os.environ.get("GROQ_API_KEY"),
)

SYSTEM_PROMPT = """You are a helpful assistant that answers questions using the context provided below, which comes from the user's uploaded documents.
- Answer using the context provided. If the context does not actually contain the answer to the question, say "I couldn't find this information in the documents" — don't make up an answer.
- If the user's message is just a greeting, ignore the context and reply normally and briefly.
- Keep answers clear and concise.
- Don't mention that you're an AI or explain how you're generating the answer."""

# ---------- memory load

messages = []
today = date.today().strftime("%B %d, %Y")

if os.path.exists("memory.json"):
    with open("memory.json", "r") as f:
        old_memory = json.load(f)
        messages.append({"role": "system", "content": old_memory["summary"]})

# conversation main loop
print("""
_____________________________________________
            PHOENIX AI ASSISTANT
______________________________________________""")

while True:
    user_input = input("\nYOU: ")

    if not user_input:
        continue

    if user_input.lower() == "quit":
        print("See u soon sir!!!")

        messages.append(
            {
                "role": "system",
                "content": f"Write a summary in JSON format with keys: title, "
                f'date ("{today}"), summary (what the user asked about), '
                f"and messages (the conversation so far). "
                f"Output ONLY the JSON, nothing else.",
            }
        )

        summary_response = chat_model.invoke(messages)

        parsed_memory = json.loads(summary_response.content)
        with open("memory.json", "w") as f:
            json.dump(parsed_memory, f, indent=4)

        break

    # 1. retrieve relevant chunks for THIS message
    retrieved_docs = retriever.invoke(user_input)

    if not retrieved_docs:
        context = "No relevant information found in the documents."
    else:
        context = "\n\n".join(doc.page_content for doc in retrieved_docs)

    # 2. build the messages for the model (system rules + context + question)
    messages_for_query = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "system", "content": f"Context:\n{context}"},
        {"role": "user", "content": user_input},
    ]

    # 3. ask the model
    response = chat_model.invoke(messages_for_query)
    print(f"\nPhoenix Assistant: {response.content}\n")

    # 4. remember this turn
    messages.append({"role": "user", "content": user_input})
    messages.append({"role": "assistant", "content": response.content})
