import os
from dotenv import load_dotenv
import streamlit as st
import requests
from bs4 import BeautifulSoup
from duckduckgo_search import DDGS
from rank_bm25 import BM25Okapi
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain.agents import AgentExecutor, create_openai_tools_agent
from langchain.tools import Tool
from langchain.prompts import ChatPromptTemplate
import time

# Load environment variables from .env file
load_dotenv()

# Access the OpenAI API key
openai_api_key = os.getenv("OPENAI_API_KEY")
if not openai_api_key:
    raise ValueError("OPENAI_API_KEY is not set in .env file.")

# Set up Streamlit UI
st.set_page_config(page_title="RAG Search Assistant", layout="wide")
st.title("Fin-RAG Enhancing Financial Intelligence with GenAI")

# Initialize session state
if "retriever" not in st.session_state:
    st.session_state.retriever = None

# Sidebar for configurations
with st.sidebar:
    st.header("Configuration")
    os.environ["OPENAI_API_KEY"] = openai_api_key
    search_results_num = st.slider("Number of Search Results", 1, 10, 3)

# Web Search and Document Loading Functions
def web_search(query, num_results=3):
    """Fetches search results from DuckDuckGo."""
    with DDGS() as ddgs:
        results = list(ddgs.text(query, max_results=num_results))
        if not results:
            print("[ERROR] No search results found.")
            return []
    return results

def page_scrape(url):
    """Fetches and scrapes webpage content."""
    try:
        headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                                 "(KHTML, like Gecko) Chrome/119.0.0.0 Safari/537.36"}
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()  # Raise an HTTPError for bad responses (4xx and 5xx)
        soup = BeautifulSoup(response.text, "html.parser")
        paras = [p.get_text() for p in soup.find_all("p")]
        return "\n".join(paras).strip()
    except requests.exceptions.Timeout:
        print(f"Timeout occurred while fetching {url}.")
    except requests.exceptions.RequestException as e:
        print(f"Error fetching {url}: {e}")
    return None  # Return None if an error occurs

def load_documents(query):
    search_results = web_search(query, search_results_num)
    documents = []
    for result in search_results:
        if content := page_scrape(result["href"]):
            documents.append({
                "url": result["href"],
                "title": result["title"],
                "content": content
            })
        time.sleep(1)
    return documents

# Retriever Classes
class BM25Retriever:
    def __init__(self, documents):
        self.documents = documents
        self.texts = [doc["content"] for doc in documents]
        self.bm25 = BM25Okapi([text.split() for text in self.texts])

    def retrieve(self, query, k=3):
        query_tokens = query.split()
        scores = self.bm25.get_scores(query_tokens)
        return scores, self.documents

# Hybrid Retriever combining BM25 and FAISS
class HybridRetriever:
    def __init__(self, documents, bm25_weight=0.5):
        self.documents = documents
        self.bm25_retriever = BM25Retriever(documents)
        self.embeddings = OpenAIEmbeddings()
        texts = [doc["content"] for doc in documents]
        if texts:
            self.vectorstore = FAISS.from_texts(texts, self.embeddings)
        else:
            self.vectorstore = None
        self.bm25_weight = bm25_weight

    def retrieve(self, query, k=3):
        if not self.documents:
            return []

        # Get BM25 scores
        bm25_scores, docs = self.bm25_retriever.retrieve(query)

        # Get vector similarity scores
        if self.vectorstore:
            vector_results = self.vectorstore.similarity_search_with_score(query, k=len(self.documents))
            vector_scores = [1.0 - score for _, score in vector_results]  # Convert distance to similarity
        else:
            vector_scores = [0] * len(self.documents)

        # Normalize scores
        if max(bm25_scores) > 0:
            bm25_scores = [score / max(bm25_scores) for score in bm25_scores]
        if max(vector_scores) > 0:
            vector_scores = [score / max(vector_scores) for score in vector_scores]

        # Combine scores
        combined_scores = [self.bm25_weight * bm25 + (1 - self.bm25_weight) * vector
                           for bm25, vector in zip(bm25_scores, vector_scores)]

        # Get top k documents
        top_indices = sorted(range(len(combined_scores)), key=lambda i: combined_scores[i], reverse=True)[:k]
        return [self.documents[i] for i in top_indices]

# Retrieve Documents using Hybrid Search
def search_docs(query, hybrid_retriever):
    """Fetches relevant documents using hybrid search."""
    top_docs = hybrid_retriever.retrieve(query)
    if not top_docs:
        return "No relevant information found."
    return "\n\n".join([f"[Source: {doc['url']}]\n{doc['content'][:500]}..." for doc in top_docs])

def setup_agent_executor():
    tools = [
        Tool(
            name="WebContentSearch",
            func=lambda q: search_docs(q, st.session_state.retriever),
            description="Searches for information in the web content database using hybrid search."
        )
    ]
    # Initialize LLM
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

    # Define Prompt for LLM Agent
    system_prompt = """You are an AI assistant that finds information from web sources.
    - Use the WebContentSearch tool to find relevant information.
    - If no relevant information is found, return 'No relevant information available.'
    - Do not call WebContentSearch more than once per query.
    - Do not repeat queries or enter a loop."""

    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "{input}"),
        ("ai", "{agent_scratchpad}")
    ])

    # Create LLM Agent
    agent = create_openai_tools_agent(llm, tools, prompt)
    
    return AgentExecutor(
        agent=agent,
        tools=tools,
        verbose=True,
        handle_parsing_errors=True,
        max_iterations=10
    )

# Main UI Components
query = st.text_input("Enter your research question:", placeholder="Type your question here...")

if st.button("Search and Analyze"):
    if not query:
        st.warning("Please enter a question!")
        st.stop()
        
    with st.spinner("🔍 Searching web and processing documents..."):
        documents = load_documents(query)
        st.session_state.retriever = HybridRetriever(documents)
        
    with st.spinner("🤖 Analyzing results..."):
        try:
            agent_executor = setup_agent_executor()
            response = agent_executor.invoke({"input": query, "agent_scratchpad": ""})
            st.subheader("Final Answer")
            st.markdown(f"**{response['output']}**")
            
            st.subheader("Retrieved Documents")
            for doc in st.session_state.retriever.documents:
                with st.expander(f"📄 {doc['title']}"):
                    st.markdown(f"**Source:** {doc['url']}")
                    st.markdown(doc['content'][:1000] + "...")
                    
        except Exception as e:
            st.error(f"Error processing request: {str(e)}")
