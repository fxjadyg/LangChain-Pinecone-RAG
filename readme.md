# ⚡ RAG Core // Production PDF Intelligence Terminal

An enterprise-grade Retrieval-Augmented Generation (RAG) conversational pipeline built with **Streamlit**, **LangChain**, **Pinecone Vector Database**, and **Google Gemini LLMs**. Designed to ingest complex PDF documents, perform high-dimensional vector search, and stream real-time, context-grounded answers with exact page citations.

---

## 🚀 Key Features

- **⚡ Fast PDF Processing**: High-speed document parsing using `PyMuPDF` (`fitz`) with automatic batch vector embedding and real-time indexing progress tracking.
- **🌊 Real-Time Token Streaming**: Instant response output powered by `st.write_stream` for minimum Time To First Token (TTFT).
- **🧠 History-Aware Conversational Search**: Automatically rephrases follow-up queries using past conversation context to preserve multi-turn retrieval precision.
- **🎯 Multi-Stage Context Reranking**: Optional Gemini LLM reranking stage to filter candidate chunks from Pinecone before response synthesis.
- **📄 Page-Level Source Citations**: Displays source document filenames and exact page numbers for transparent fact-checking.
- **🛡️ Dynamic Model & Quota Controls**: Sidebar controls to switch active Gemini models (`gemini-3.8-flash`, `gemini-3.6-flash`, `gemini-3.5-flash-lite`) and toggle execution passes to optimize latency and API usage.
- **💻 Cyber-Terminal Theme**: Custom CSS injected UI featuring live system status indicators and animated tech badges.

---

## 🛠️ Tech Stack

- **Frontend & UI**: Streamlit (Custom Cyber-Terminal Theme)
- **Framework & Orchestration**: LangChain
- **Vector Store**: Pinecone Index (768 Dimensions)
- **LLMs & Embeddings**: Google Gemini (`gemini-embedding-2-preview`, `gemini-3.8-flash`, `gemini-3.6-flash`)
- **Document Engine**: PyMuPDF (`fitz`)

---

## 📁 Project Structure

```text
LangChain-Pinecone-RAG/
├── chatbot_rag.py           # Main Streamlit Cyber-Terminal App
├── ingestion.py             # CLI Document Ingestion Script
├── retrieval.py             # Standalone Vector Query Script
├── requirements.txt         # Cleaned Project Dependencies
├── readme.md                # Project Documentation
├── .gitignore               # Git Exclusions (.env, venv, temporary uploads)
└── .env                     # Environment Keys (API Keys)