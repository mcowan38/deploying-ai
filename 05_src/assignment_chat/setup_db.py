#### Preamble ####
# Purpose: One-time script to chunk chesterton.txt and embed it into
#          a ChromaDB PersistentClient collection for semantic search.
# Author: Mike Cowan
# Date: 27 February 2026
# Contact: m.cowan@utoronto.ca
# License: MIT
# Pre-requisites:
# - .secrets file with API_GATEWAY_KEY
# - chesterton.txt in ./documents/
# References:
# - [https://docs.trychroma.com/docs/run-chroma/persistent-client]

#### Workspace Setup ####

import os
from dotenv import load_dotenv
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
import chromadb
from chromadb.utils.embedding_functions import OpenAIEmbeddingFunction

# Load environment variables
load_dotenv(dotenv_path=".env")
load_dotenv(dotenv_path=".secrets")

# Load the text
loader = TextLoader("./documents/chesterton.txt")
documents = loader.load()
text = documents[0].page_content

# Split into chunks
splitter = RecursiveCharacterTextSplitter(
    chunk_size=2000, chunk_overlap=200, length_function=len
)
chunks = splitter.split_text(text)
print(f"Created {len(chunks)} chunks")

# ChromaDB setup
embedding_fn = OpenAIEmbeddingFunction(
    api_key="any value",
    model_name="text-embedding-3-small",
    api_base="https://k7uffyg03f.execute-api.us-east-1.amazonaws.com/prod/openai/v1",
    default_headers={"x-api-key": os.getenv("API_GATEWAY_KEY")},
)

chroma = chromadb.PersistentClient(path="./assignment_chat/chroma_data")
collection = chroma.get_or_create_collection(
    name="father_brown", embedding_function=embedding_fn
)

# Add chunks in batches of 100
batch_size = 100
for i in range(0, len(chunks), batch_size):
    batch = chunks[i : i + batch_size]
    ids = [f"chunk_{j}" for j in range(i, i + len(batch))]
    collection.add(documents=batch, ids=ids)
    print(f"Added batch {i // batch_size + 1}: chunks {i}-{i + len(batch) - 1}")

print(f"\nDone! Total documents in collection: {collection.count()}")
