'''
Streamlit is an open-source Python library for building interactive web apps using only Python. 
It's ideal for creating dashboards, data-driven web apps, reporting tools and interactive user interfaces 
without needing HTML, CSS or JavaScript.

OS module in Python provides functions for interacting with the operating system. 
OS comes under Python's standard utility modules. 
This module provides a portable way of using operating system-dependent functionality.

An LLM wrapper is a software application that builds on top of foundation models (like GPT-4 or Claude) 
via APIs to provide specialized functionality, acting as a user interface or middleware.

LangChain is an open source orchestration framework for application development using large language models (LLMs).
'''

# Import Streamlit for building the web app UI
import streamlit as st
# Import OS module to interact with environment variables
import os
# Import dotenv to load API keys from a .env file
from dotenv import load_dotenv
# Import Groq LLM wrapper (for using Groq models like LLaMA)
from langchain_groq import ChatGroq
# Import Google Gemini LLM wrapper
from langchain_google_genai import ChatGoogleGenerativeAI
# Loader to fetch and parse webpage content
from langchain_community.document_loaders import WebBaseLoader
# Prebuilt summarization chain from LangChain
from langchain_classic.chains.summarize import load_summarize_chain
# Used to define custom prompts for the LLM
from langchain_core.prompts import PromptTemplate

# Load environment variables from .env file (API keys, etc.)
load_dotenv()
# Set a user agent for web requests (helps avoid being blocked by websites)
os.environ["USER_AGENT"] = "MyWebSummarizerApp/1.0"

# Configure Streamlit page (title and icon in browser tab)
st.set_page_config(page_title="Bullet Point Summarizer", page_icon="⚡")
# Main title displayed in the app
st.title("⚡ Smart Web Research Assistant")

# -------------------------------
# 1. Sidebar Setup (left panel)
# -------------------------------
with st.sidebar:
    st.header("System Status")
    # Check if API keys exist in environment variables
    groq_ready = bool(os.getenv("GROQ_API_KEY"))
    gemini_ready = bool(os.getenv("GOOGLE_API_KEY"))
    
    # Display whether each service is ready or missing
    st.write(f"Groq: {'Ready' if groq_ready else 'Missing'}")
    st.write(f"Gemini: {'Ready' if gemini_ready else 'Missing'}")
    
    # Dropdown to choose summarization method
    # "stuff" = simple summarization
    # "map_reduce" = better for large documents
    summary_type = st.selectbox("Summarization Style", ["stuff", "map_reduce"])

# -------------------------------
# 2. Define the Bullet Point Prompt
# -------------------------------

# This template tells the AI exactly how to format the output
bullet_prompt_template = """
Write a concise summary of the following text in clear bullet points:
"{text}"
CONCISE SUMMARY IN BULLET POINTS:
"""

# Create a LangChain prompt object with input variable "text"
BULLET_PROMPT = PromptTemplate(template=bullet_prompt_template, input_variables=["text"])

# Input box for user to enter a webpage URL
url = st.text_input("Enter URL to summarize:")

# When user clicks the button
if st.button("Generate Summary"):
    # -------------------------------
    # Input Validation
    # -------------------------------
    
    # Reject unsupported file types (PDFs, images, etc.)
    if url.lower().endswith(('.pdf', '.docx', '.jpg', '.png', '.jpeg')):
        st.error("Policy: PDF, Image, and Docx files are ignored.")
        
    # If no URL is entered
    elif not url:
        st.warning("Please enter a URL.")
        
    else:
        # -------------------------------
        # Model Configuration (Failover Setup)
        # -------------------------------

        # List of models to try in order
        model_configs = [
            {"name": "Groq", "class": ChatGroq, "kwargs": {"model_name": "llama-3.3-70b-versatile"}},
            {"name": "Gemini", "class": ChatGoogleGenerativeAI, "kwargs": {"model": "gemini-2.0-flash"}}
        ]

        success = False # Tracks if any model succeeds
        errors = [] # Stores errors from failed attempts

        try:
            # -------------------------------
            # Step 1: Fetch Web Content
            # -------------------------------
            with st.spinner("Fetching web content..."):
                loader = WebBaseLoader(url) # Create loader for URL
                docs = loader.load() # Extract webpage text into documents
            
            
            # -------------------------------
            # Step 2: Try Each Model (Failover)
            # -------------------------------
            for config in model_configs:
                try:
                    with st.spinner(f"Attempting {config['name']}..."):
                        # Initialize the LLM with given config
                        llm = config["class"](**config["kwargs"])
                        
                        # -------------------------------
                        # Step 3: Create Summarization Chain
                        # -------------------------------

                        # Apply the custom bullet point prompt to the chain
                        if summary_type == "stuff":
                            # "stuff" = pass all text at once to model
                            chain = load_summarize_chain(llm, chain_type="stuff", prompt=BULLET_PROMPT)
                        else:
                            # "map_reduce" = split text → summarize chunks → combine
                            chain = load_summarize_chain(
                                llm, 
                                chain_type="map_reduce", 
                                map_prompt=BULLET_PROMPT, 
                                combine_prompt=BULLET_PROMPT
                            )
                        
                        # -------------------------------
                        # Step 4: Run the Chain
                        # -------------------------------
                        result = chain.invoke(docs)
                        
                        
                        # -------------------------------
                        # Step 5: Display Results
                        # -------------------------------
                        st.success(f"Success via {config['name']}")
                        st.subheader("Key Takeaways")
                        st.write(result["output_text"])
                        
                        success = True
                        break  # Stop trying other models if success
                    
                except Exception as e:
                    # If model fails, log error and try next one
                    errors.append(f"{config['name']}: {str(e)}")
                    st.warning(f"{config['name']} encountered an issue. Trying failover...")
            
            # -------------------------------
            # Step 6: If All Models Fail
            # -------------------------------
            if not success:
                st.error("All models failed to process the request.")
                for err in errors:
                    st.expander("Technical Error Log").write(err)

        # -------------------------------
        # Step 7: Catch Critical Errors
        # -------------------------------
        except Exception as e:
            st.error(f"Critical Scraper Error: {e}")