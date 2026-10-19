import os
import time
from dotenv import load_dotenv

# Import Pinecone client
from pinecone import Pinecone, ServerlessSpec

# Import LangChain integrations for Google Gemini & Pinecone
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Load environment variables
load_dotenv()

# Initialize Pinecone
pc = Pinecone(api_key=os.environ.get("PINECONE_API_KEY"))
index_name = os.environ.get("PINECONE_INDEX_NAME")

# Create index automatically if it doesn't exist yet
existing_indexes = [index_info["name"] for index_info in pc.list_indexes()]

if index_name not in existing_indexes:
    pc.create_index(
        name=index_name,
        dimension=768,
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region="us-east-1"),
    )
    while not pc.describe_index(index_name).status["ready"]:
        time.sleep(1)

index = pc.Index(index_name)

# 1. Load PDFs from documents directory
loader = PyPDFDirectoryLoader("documents/")
raw_documents = loader.load()

# 2. Split document text into chunks
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=800,
    chunk_overlap=100,
    length_function=len,
    is_separator_regex=False,
)
documents = text_splitter.split_documents(raw_documents)

print(f"Loaded {len(raw_documents)} pages and split into {len(documents)} chunks.")

# 3. Initialize Gemini Embeddings with explicit 768 output dimensionality
embeddings = GoogleGenerativeAIEmbeddings(
    model="gemini-embedding-2-preview",
    output_dimensionality=768,
    google_api_key=os.environ.get("GEMINI_API_KEY")
)

# 4. Generate unique IDs for each chunk
uuids = [f"id{i+1}" for i in range(len(documents))]

# 5. Initialize vector store & upload to Pinecone
vector_store = PineconeVectorStore(index=index, embedding=embeddings)

print("Uploading document vectors to Pinecone...")
vector_store.add_documents(documents=documents, ids=uuids)

print("Ingestion complete! Your documents are stored in Pinecone.")