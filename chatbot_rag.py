import os
import re
import streamlit as st
from dotenv import load_dotenv

from pinecone import Pinecone
from langchain_pinecone import PineconeVectorStore
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Load environment variables
load_dotenv()

# Set up API keys
gemini_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
if gemini_key:
    os.environ["GOOGLE_API_KEY"] = gemini_key

# Page Config
st.set_page_config(
    page_title="RAG // CORE TERMINAL ALPHA",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- MODERN TECH-CYBER ANIMATION & STYLE OVERRIDE ---
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;600;800&family=Inter:wght@300;400;600;700&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

/* Keyframes for Animations */
@keyframes fadeInUp {
    from {
        opacity: 0;
        transform: translateY(12px);
    }
    to {
        opacity: 1;
        transform: translateY(0);
    }
}

@keyframes cyberPulse {
    0% {
        box-shadow: 0 0 5px rgba(56, 189, 248, 0.2), 0 0 10px rgba(56, 189, 248, 0.1);
    }
    50% {
        box-shadow: 0 0 15px rgba(56, 189, 248, 0.6), 0 0 25px rgba(56, 189, 248, 0.3);
    }
    100% {
        box-shadow: 0 0 5px rgba(56, 189, 248, 0.2), 0 0 10px rgba(56, 189, 248, 0.1);
    }
}

@keyframes glowPulse {
    0%, 100% { opacity: 0.85; filter: drop-shadow(0 0 2px rgba(56, 189, 248, 0.2)); }
    50% { opacity: 1; filter: drop-shadow(0 0 8px rgba(56, 189, 248, 0.8)); }
}

@keyframes scanLine {
    0% { transform: translateY(-100%); }
    100% { transform: translateY(100%); }
}

/* Application Container Animations */
.stApp {
    animation: fadeInUp 0.75s cubic-bezier(0.16, 1, 0.3, 1) forwards;
}

/* Header & Banner Animation */
.tech-header-container {
    animation: fadeInUp 0.5s cubic-bezier(0.16, 1, 0.3, 1) forwards;
}

.tech-header {
    font-family: 'JetBrains Mono', monospace;
    background: linear-gradient(135deg, #00F2FE 0%, #4FACFE 50%, #C084FC 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    font-size: 2.3rem;
    font-weight: 800;
    letter-spacing: -0.5px;
    margin-bottom: 4px;
}

.tech-subtext {
    color: #64748B;
    font-size: 0.85rem;
    font-family: 'JetBrains Mono', monospace;
    margin-bottom: 1.5rem;
    display: flex;
    align-items: center;
    gap: 8px;
}

/* Dashboard Modules */
.status-container {
    background: rgba(15, 23, 42, 0.55);
    border: 1px solid rgba(56, 189, 248, 0.15);
    border-radius: 8px;
    padding: 12px 16px;
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.8rem;
    color: #CBD5E1;
    backdrop-filter: blur(8px);
    transition: all 0.3s cubic-bezier(0.16, 1, 0.3, 1);
    display: flex;
    align-items: center;
}

.status-container:hover {
    transform: translateY(-2px);
    border-color: rgba(56, 189, 248, 0.4);
    box-shadow: 0 4px 20px rgba(56, 189, 248, 0.1);
}

.status-dot {
    height: 8px;
    width: 8px;
    background-color: #10B981;
    border-radius: 50%;
    display: inline-block;
    margin-right: 10px;
    animation: glowPulse 2s infinite ease-in-out;
}

/* Glassmorphism Chat Containers */
div[data-testid="stChatMessage"] {
    animation: fadeInUp 0.45s cubic-bezier(0.16, 1, 0.3, 1) forwards;
    border-radius: 12px;
    padding: 1rem;
    transition: transform 0.2s ease, border-color 0.2s ease;
    border: 1px solid transparent;
}

div[data-testid="stChatMessage"]:hover {
    border-color: rgba(255, 255, 255, 0.05);
    background: rgba(255, 255, 255, 0.015);
}

/* Tech Citation & Chunks dropdown Card Animation */
.citation-card {
    background: rgba(15, 23, 42, 0.4);
    border-left: 3px solid #38BDF8;
    padding: 12px 14px;
    border-radius: 4px;
    margin-bottom: 1rem;
    animation: fadeInUp 0.4s ease-out forwards;
}

/* Custom Badges */
.source-badge {
    background: rgba(56, 189, 248, 0.08);
    border: 1px solid rgba(56, 189, 248, 0.25);
    color: #38BDF8;
    padding: 3px 8px;
    border-radius: 4px;
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.72rem;
    font-weight: 600;
    transition: all 0.2s ease;
}

.source-badge:hover {
    background: rgba(56, 189, 248, 0.15);
    border-color: rgba(56, 189, 248, 0.5);
    box-shadow: 0 0 10px rgba(56, 189, 248, 0.15);
}

.page-badge {
    background: rgba(168, 85, 247, 0.08);
    border: 1px solid rgba(168, 85, 247, 0.25);
    color: #C084FC;
    padding: 3px 8px;
    border-radius: 4px;
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.72rem;
    font-weight: 600;
    transition: all 0.2s ease;
}

.page-badge:hover {
    background: rgba(168, 85, 247, 0.15);
    border-color: rgba(168, 85, 247, 0.5);
    box-shadow: 0 0 10px rgba(168, 85, 247, 0.15);
}

/* Interactive Upload Area */
div[data-testid="stFileUploader"] {
    transition: all 0.3s ease;
}
div[data-testid="stFileUploader"]:hover {
    box-shadow: 0 0 15px rgba(56, 189, 248, 0.08);
}

/* Input Fields & Glowing Interactive Buttons */
div.stButton > button {
    background: transparent;
    border-radius: 6px;
    border: 1px solid rgba(56, 189, 248, 0.25);
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.8rem;
    letter-spacing: 0.5px;
    transition: all 0.25s cubic-bezier(0.16, 1, 0.3, 1);
}

div.stButton > button:hover {
    background: rgba(56, 189, 248, 0.06);
    border-color: #38BDF8;
    color: #38BDF8;
    box-shadow: 0 0 12px rgba(56, 189, 248, 0.2);
    transform: translateY(-1px);
}

div.stButton > button:active {
    transform: translateY(0px);
}

/* Progress bar cyber skin */
div[data-testid="stProgress"] > div > div > div {
    background: linear-gradient(90deg, #38BDF8, #A855F7);
    animation: cyberPulse 2s infinite ease-in-out;
}
</style>
""", unsafe_allow_html=True)

# Top Cinematic Header
st.markdown("""
<div class="tech-header-container">
    <div class="tech-header">⚡ RAG CORE // ALPHA-01</div>
    <div class="tech-subtext">
        <span>[ONLINE]</span> • <span>CONSTITUTIONAL DOCUMENT SEARCH & COMPREHENSION MATRIX</span>
    </div>
</div>
""", unsafe_allow_html=True)

# Cyber Status Metrics Grid
col1, col2, col3 = st.columns(3)
with col1:
    st.markdown('<div class="status-container"><span class="status-dot"></span><b>ENGINE:</b> Gemini Pro Active</div>', unsafe_allow_html=True)
with col2:
    st.markdown('<div class="status-container"><span class="status-dot"></span><b>DB CLUSTER:</b> Pinecone Secure</div>', unsafe_allow_html=True)
with col3:
    st.markdown('<div class="status-container"><span class="status-dot"></span><b>DOCUMENT LAYER:</b> PyMuPDF Active</div>', unsafe_allow_html=True)

st.markdown("---")

# Initialize Pinecone and Vector Store
@st.cache_resource
def get_vector_store():
    pc = Pinecone(api_key=os.environ.get("PINECONE_API_KEY"))
    index_name = os.environ.get("PINECONE_INDEX_NAME")
    index = pc.Index(index_name)

    embeddings = GoogleGenerativeAIEmbeddings(
        model="gemini-embedding-2-preview",
        output_dimensionality=768,
        google_api_key=gemini_key
    )
    return PineconeVectorStore(index=index, embedding=embeddings)

try:
    vector_store = get_vector_store()
except Exception as e:
    st.error(f"Failed to connect to Pinecone/Embeddings: {e}")
    st.stop()

# --- HELPER FUNCTIONS ---

def extract_text(content):
    """Safely extracts a plain string whether LLM returns a string or list of content blocks."""
    if isinstance(content, str):
        return content
    elif isinstance(content, list):
        return "".join(
            block.get("text", "") if isinstance(block, dict) else str(block)
            for block in content
        )
    return str(content)


def rephrase_query(chat_messages, latest_prompt, llm):
    """Safely rephrases follow-up questions with fallbacks if quota is exceeded."""
    history = [msg for msg in chat_messages if not isinstance(msg, SystemMessage)]
    if len(history) <= 1:
        return latest_prompt

    history_text = ""
    for msg in history[-6:-1]:
        role = "User" if isinstance(msg, HumanMessage) else "Assistant"
        history_text += f"{role}: {msg.content}\n"

    rephrase_prompt = f"""Given the following conversation history and a follow-up question, rephrase the follow-up question into a single, self-contained standalone search query. 
Do NOT answer the question. Only output the reformulated question.

Chat History:
{history_text}

Follow-up Question: {latest_prompt}

Standalone Question:"""

    try:
        response = llm.invoke([HumanMessage(content=rephrase_prompt)])
        raw_text = extract_text(response.content)
        standalone_q = raw_text.strip()
        return standalone_q if standalone_q else latest_prompt
    except Exception:
        return latest_prompt


def rerank_documents(query, candidates, llm, top_n=3):
    """Safely reranks context chunks with fallback to standard top candidates on quota limit."""
    if len(candidates) <= top_n:
        return candidates

    context_snippets = "\n".join(
        [f"[{i+1}] {doc.page_content[:350]}..." for i, doc in enumerate(candidates)]
    )

    rerank_prompt = f"""You are a search reranking assistant.
Given search query: "{query}"
Candidate chunks:
{context_snippets}

Select indices of the {top_n} most relevant chunks. Return ONLY a comma-separated list of numbers (e.g., 1, 3, 5):"""

    try:
        response = llm.invoke([HumanMessage(content=rerank_prompt)])
        raw_text = extract_text(response.content).strip()
        selected_indices = [int(num) - 1 for num in re.findall(r'\d+', raw_text) if 0 <= int(num) - 1 < len(candidates)]
        
        seen = set()
        unique_indices = [i for i in selected_indices if not (i in seen or seen.add(i))]
        
        reranked = [candidates[i] for i in unique_indices if i < len(candidates)]
        return reranked[:top_n] if reranked else candidates[:top_n]
    except Exception:
        return candidates[:top_n]


# --- SIDEBAR: Configuration Card Deck ---
with st.sidebar:
    st.header("⚙️ Model & Quota Settings")
    
    selected_model = st.selectbox(
        "Select Active Gemini Model",
        options=["gemini-3.8-flash", "gemini-3.6-flash", "gemini-3.5-flash-lite"],
        index=0
    )

    st.caption("💡 **Speed & Core Customizers (Keep off for sub-second execution):**")
    enable_rephrase = st.checkbox("Enable Query Rephrasing", value=False)
    enable_rerank = st.checkbox("Enable Smart Reranking", value=False)

    st.markdown("---")
    st.header("📄 File Ingestion")
    uploaded_file = st.file_uploader("Upload PDF File for Mapping", type=["pdf"])

    if uploaded_file and st.button("⚡ Process & Index PDF"):
        with st.spinner("Executing structural PDF analysis with PyMuPDF..."):
            temp_path = f"temp_{uploaded_file.name}"
            with open(temp_path, "wb") as f:
                f.write(uploaded_file.getbuffer())

            loader = PyMuPDFLoader(temp_path)
            raw_docs = loader.load()
            
            for doc in raw_docs:
                doc.metadata["source"] = uploaded_file.name

            text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
            split_docs = text_splitter.split_documents(raw_docs)

            total_chunks = len(split_docs)
            st.info(f"Generated **{total_chunks}** segment vectors. Starting database batch loading...")

            progress_bar = st.progress(0)
            status_text = st.empty()

            batch_size = 100
            for i in range(0, total_chunks, batch_size):
                batch = split_docs[i:i + batch_size]
                vector_store.add_documents(batch)
                
                current_count = min(i + batch_size, total_chunks)
                progress_bar.progress(current_count / total_chunks)
                status_text.text(f"Processed {current_count} // {total_chunks} matrix segments...")

            st.success(f"Parsing complete. Database mapped {total_chunks} segments from `{uploaded_file.name}`!")

            if os.path.exists(temp_path):
                os.remove(temp_path)

    st.markdown("---")
    if st.button("🗑️ Clear Terminal Buffer"):
        st.session_state.messages = [
            SystemMessage("You are an assistant for question-answering tasks.")
        ]
        st.rerun()

# --- CHAT BUFFER Matrix ---
if "messages" not in st.session_state:
    st.session_state.messages = [
        SystemMessage("You are an assistant for question-answering tasks.")
    ]

# Render historic message cascading matrix
for message in st.session_state.messages:
    if isinstance(message, HumanMessage):
        with st.chat_message("user"):
            st.markdown(message.content)
    elif isinstance(message, AIMessage):
        with st.chat_message("assistant"):
            st.markdown(message.content)

# Standard prompt evaluation mapping
prompt = st.chat_input("Inject query into search matrix...")

if prompt:
    with st.chat_message("user"):
        st.markdown(prompt)
    st.session_state.messages.append(HumanMessage(prompt))

    with st.chat_message("assistant"):
        try:
            llm = ChatGoogleGenerativeAI(
                model=selected_model,
                google_api_key=gemini_key,
                temperature=0.3,
                request_timeout=30
            )

            # 1. Historical re-rephrasing evaluation
            if enable_rephrase:
                with st.spinner("Analyzing memory parameters..."):
                    standalone_query = rephrase_query(st.session_state.messages, prompt, llm)
                    if standalone_query != prompt:
                        st.caption(f"🔍 *Rephrased standalone parameters:* **\"{standalone_query}\"**")
            else:
                standalone_query = prompt

            # 2. Extract Document Fragments
            with st.spinner("Querying vector databases..."):
                retriever = vector_store.as_retriever(
                    search_type="similarity",
                    search_kwargs={"k": 8 if enable_rerank else 3}
                )
                candidate_docs = retriever.invoke(standalone_query)

                if enable_rerank:
                    docs = rerank_documents(standalone_query, candidate_docs, llm, top_n=3)
                else:
                    docs = candidate_docs[:3]

                docs_text = "\n\n".join(d.page_content for d in docs)

            # 3. Formulate structural system query prompt
            system_prompt_fmt = (
                "You are an assistant for question-answering tasks. "
                "Use the following pieces of retrieved context to answer the question concisely. "
                "If you don't know the answer, say you don't know.\n\n"
                f"Context:\n{docs_text}"
            )

            history_msgs = [msg for msg in st.session_state.messages if not isinstance(msg, SystemMessage)]
            input_messages = [SystemMessage(system_prompt_fmt)] + history_msgs

            # 4. Cascaded generation streaming block
            def stream_response():
                for chunk in llm.stream(input_messages):
                    yield extract_text(chunk.content)

            result = st.write_stream(stream_response)

            # 5. Cyber-accented metadata visualization
            with st.expander("🔍 REVIEW RETRIEVED SEGMENTS & METRIC ORIGINS"):
                for idx, doc in enumerate(docs, 1):
                    raw_page = doc.metadata.get("page", 0)
                    page_num = raw_page + 1 if isinstance(raw_page, int) else raw_page
                    source_name = doc.metadata.get("source", "Uploaded Document")

                    st.markdown(f"""
                    <div class="citation-card">
                        <strong>MATRIX SEGMENT {idx}</strong> <br>
                        <span class="source-badge" style="margin-top: 6px; display: inline-block;">📄 SRC: {source_name}</span> 
                        <span class="page-badge" style="margin-top: 6px; display: inline-block;">🏷️ PG: {page_num}</span>
                        <div style="margin-top: 8px; color: #94A3B8; font-size: 0.82rem; line-height: 1.4;">
                            {doc.page_content}
                        </div>
                    </div>
                    """, unsafe_allow_html=True)

            st.session_state.messages.append(AIMessage(result))

        except Exception as e:
            st.error(f"An error occurred: {e}")