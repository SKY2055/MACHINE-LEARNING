# Setting up environment

import os
from dotenv import load_dotenv
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from crewai_tools import SerperDevTool,ScrapeWebsiteTool
from crewai import Agent, Task, Crew, LLM

load_dotenv()

# API Keys
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
SERPER_API_KEY = os.getenv("SERPER_API_KEY")
GEMINI = os.getenv("GEMINI")

# Initialiizing the LLMs
    
llm = ChatGroq(
model="llama-3.3-70b-versatile",
temperature=0,
max_tokens=500,
timeout=None,
max_retries=2,
)

crew_llm = LLM(
model="gemini/gemini-2.5-flash",
api_key=GEMINI,
max_tokens=500,
temperature=0.7
)

# Decision Maker

def check_local_knowledge(query, context):
    prompt = '''Role: Question-Answering Assistant
Task: Determine whether the system can answer the user's question based on the provided text.
Output Format: Answer: Yes/No
User Question: {query}
Text: {text}'''
    
    formatted_prompt = prompt.format(text=context, query=query)
    response = llm.invoke(formatted_prompt)
    
    return response.content.strip().lower().startswith("yes")

# Web Scraping and Scraping Agent

def setup_web_scraping_agent(query):
    search_tool = SerperDevTool()
    scrape_website = ScrapeWebsiteTool()
    
    web_search_agent = Agent(
        role="Expert Web Search Agent",
        goal=f"Identify and retrieve relevant web data for {query}",
        backstory="You are a meticulous researcher who specializes in finding technical definitions and explaining complex AI concepts accurately.",
        llm=crew_llm,
        tools=[search_tool, scrape_website],
        verbose=True
        )
    
    search_task = Task(
        description=f"Search the web to find a detailed explanation of {query}.",
        expected_output="A comprehensive summary of the topic based on web results.",
        agent=web_search_agent
        )
    
    # Removed the unused web_scraper_agent block
    
    return Crew(
        agents=[web_search_agent],
        tasks=[search_task],
        verbose=True
    )

def get_web_content(query):
    crew = setup_web_scraping_agent(query)
    result = crew.kickoff(inputs={"topic": query})
    
    return result.raw

# Creating the Vector Database

def setup_vector_db(pdf_path):
    loader = PyPDFLoader (pdf_path)
    documents = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000,chunk_overlap=50)
    chunks = text_splitter.split_documents (documents)
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")
    vector_db = FAISS.from_documents(chunks,embeddings)
    return vector_db

def get_local_content(vector_db, query):
    docs = vector_db.similarity_search(query, k = 5 )
    return " ".join([doc.page_content for doc in docs])

# Generating the Final Answer

def generate_final_answer (context, query):
    messages = [
    ("system", "You are a helpful assistant. Use the provided context to answer the query accurately."),
    ("system", f"Context: {context}"),
    ("human", query),
    ]
    response = llm.invoke(messages)
    return response.content

def process_query(query, vector_db, local_context):
    can_answer_locally = check_local_knowledge (query,local_context)
    context = get_local_content(vector_db, query) if can_answer_locally else get_web_content(query)
    return generate_final_answer(context, query)
    
    
def main():
    pdf_path = "/Users/suraj/Machine Learning/GenAI/What is an AI agent.pdf"
    vector_db = setup_vector_db(pdf_path)
    local_context = get_local_content(vector_db, "")
    query = "What is Agentic RAG?"
    result = process_query(query, vector_db,
    local_context)
    print("\nFinal Answer:")
    print(result)

if __name__ == "__main__":
    main()