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


#   load_and_add_pdfs()


# ----retriever + model   #####

retriever = vector_store.as_retriever(search_type="mmr", search_kwargs={"k": 2})

chat_model = ChatGroq(
    model="openai/gpt-oss-120b",
    temperature=0,
    api_key=os.environ.get("GROQ_API_KEY"),
)

# ---------- memory load

messages = []
today = date.today().strftime("%B %d, %Y")

if os.path.exists("memory.json"):
    with open("memory.json", "r") as f:
        old_memory = json.load(f)
        messages.append({"role": "system", "content": old_memory["summary"]})

# conversation main loop

while True:
    user_input = input("\n\nYOU: ")

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

    retrieved_docs = retriever.invoke(user_input)
    context = "\n\n".join(doc.page_content for doc in retrieved_docs)

    prompt = f"""Answer the question using ONLY the context below.

Context:
{context}

Question: {user_input}

Answer:"""

    response = chat_model.invoke(prompt)
    print(f"\nPhoenix Assistant: {response.content}\n")

    messages.append({"role": "user", "content": user_input})
    messages.append({"role": "assistant", "content": response.content})
