# 💹 FinRAG: Financial Retrieval-Augmented Generation System

FinRAG is an intelligent assistant for financial investors and analysts. It uses a **Hybrid RAG pipeline** (BM25 + Vector Search) combined with **LLM-based reasoning (GPT-4o)** to search, extract, and summarize financial insights from the web in real time.

---

## 🚀 Features

- 🔍 **Web Search Integration** – Real-time search via DuckDuckGo
- 🧹 **Smart Web Scraping** – Clean HTML-to-text using BeautifulSoup
- ⚖️ **Hybrid Retrieval** – BM25 + FAISS for keyword + semantic search
- 🧠 **LLM Agent** – Uses GPT-4o with LangChain to summarize results
- 📌 **Source Tracking** – Cites top sources in each response
- 🧾 **Natural Query Handling** – Ask questions like “Compare Q1 results of Google vs Amazon”
- 🖥️ **Streamlit Web App** – Simple and clean UI for user interaction

---

## 🧱 Architecture Overview

```text
[User Query]
     |
     v
[DuckDuckGo Search (DDGS)]
     |
     v
[Top URLs] --> [Web Scraper (BeautifulSoup)] --> [Text Documents]
     |
     v
[Hybrid Retriever]
   ├─ BM25 (Keyword Match)
   ├─ FAISS (Vector Match)
   └─ Score Normalization + Fusion
     |
     v
[Top-K Relevant Docs]
     |
     v
[LLM Agent (GPT-4o via LangChain)]
     |
     v
[Final Answer + Source URLs]
     |
     v
[Streamlit UI Display]
