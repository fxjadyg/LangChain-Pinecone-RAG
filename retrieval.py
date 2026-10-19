# import basics
import os
from dotenv import load_dotenv

# import pinecone
from pinecone import Pinecone

# import langchain
from langchain_pinecone import PineconeVectorStore
from langchain_google_genai import GoogleGenerativeAIEmbeddings

load_dotenv()

# Handle API key mapping for Gemini
gemini_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
if gemini_key:
    os.environ["GOOGLE_API_KEY"] = gemini_key

# initialize pinecone database
pc = Pinecone(api_key=os.environ.get("PINECONE_API_KEY"))

# set the pinecone index
index_name = os.environ.get("PINECONE_INDEX_NAME") 
index = pc.Index(index_name)

# initialize Gemini embeddings model (768 dimensions matching your Pinecone index)
embeddings = GoogleGenerativeAIEmbeddings(
    model="gemini-embedding-2-preview",
    output_dimensionality=768,
    google_api_key=gemini_key
)

vector_store = PineconeVectorStore(index=index, embedding=embeddings)

# retrieval
retriever = vector_store.as_retriever(
    search_type="similarity_score_threshold",
    search_kwargs={"k": 5, "score_threshold": 0.5},
)
results = retriever.invoke("what is retrieval augmented generation?")

# show results
print("RESULTS:")

for res in results:
    print(f"* {res.page_content} [{res.metadata}]")