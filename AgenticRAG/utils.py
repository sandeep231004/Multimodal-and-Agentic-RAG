import os
import time
import random
import re
from dotenv import load_dotenv
from typing import Optional, List, Dict, Any
from llama_index.llms.groq import Groq
from llama_index.llms.openai import OpenAI

from llama_index.core import (
    SimpleDirectoryReader,
    VectorStoreIndex,
    SummaryIndex,
    StorageContext,
    load_index_from_storage,
)

from llama_index.core import VectorStoreIndex, Settings
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.tools import FunctionTool, QueryEngineTool
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core.vector_stores import MetadataFilters, FilterCondition


PERSIST_DIR = "./storage"

# Rate limiting helpers
_last_request_time = 0
_request_count = 0
_cooldown_until = 0

def extract_wait_time(error_message: str) -> Optional[float]:
    """Extract wait time in seconds from Groq rate limit error message."""
    # Try to match "Please try again in 2m9.183s"
    match = re.search(r'Please try again in (\d+)m(\d+\.\d+)s', error_message)
    if match:
        minutes = int(match.group(1))
        seconds = float(match.group(2))
        return minutes * 60 + seconds
    
    # Try to match "Please try again in 30.5s"
    match = re.search(r'Please try again in (\d+\.\d+)s', error_message)
    if match:
        return float(match.group(1))
    
    return None

# Create a wrapper class for rate limiting
class RateLimitedLLMWrapper:
    """Wrapper for LLM that adds rate limiting functionality."""
    
    def __init__(self, llm):
        self.llm = llm
        
    def _handle_rate_limit(self, func, *args, **kwargs):
        """Apply rate limiting logic to any function call."""
        global _last_request_time, _request_count, _cooldown_until
        
        current_time = time.time()
        
        # Check if we're in a cooldown period
        if current_time < _cooldown_until:
            sleep_time = _cooldown_until - current_time
            print(f"Rate limiting: waiting for {sleep_time:.1f}s due to previous rate limit...")
            time.sleep(sleep_time)
            current_time = time.time()
            _cooldown_until = 0
            _request_count = 0
        
        # Reset counter if more than 60s has passed
        if current_time - _last_request_time > 60:
            _request_count = 0
        
        # If we've made several requests in the last minute, add a delay
        if _request_count > 5:
            delay = min(5, _request_count * 0.5)  # Scale delay with request count
            print(f"Proactive rate limiting: adding {delay:.1f}s delay...")
            time.sleep(delay)
        
        # Now attempt the completion
        max_retries = 5
        retry_count = 0
        base_delay = 2
        
        while retry_count < max_retries:
            try:
                result = func(*args, **kwargs)
                # Update request tracking
                _last_request_time = time.time()
                _request_count += 1
                return result
            
            except Exception as e:
                error_msg = str(e)
                if "rate_limit_exceeded" in error_msg and retry_count < max_retries - 1:
                    retry_count += 1
            
                    wait_time = extract_wait_time(error_msg) # Try to extract wait time from error message
                    if wait_time is None:
                        wait_time = base_delay * (2 ** retry_count) + random.uniform(0, 1) # Default backoff
                    wait_time += 3 # Add buffer
                    
                    print(f"Rate limit reached. Waiting {wait_time:.1f}s before retry {retry_count}/{max_retries}...")
                    
                    _cooldown_until = time.time() + wait_time # Set the cooldown period
                    time.sleep(wait_time)
                    _request_count = 0
                
                else:                    
                    raise   # If it's not a rate limit error or we're out of retries, re-raise

        raise Exception(f"Failed after {max_retries} retries due to rate limiting.")
    
    # Forward all attribute accesses to the wrapped LLM
    def __getattr__(self, name):
        attr = getattr(self.llm, name)

        if callable(attr): # If it's a callable, wrap it with rate limiting
            def wrapper(*args, **kwargs):
                return self._handle_rate_limit(attr, *args, **kwargs)
            return wrapper
        
        return attr # Otherwise just return the attribute


def get_doc_tools(
        file_path: str,
        name: str,
):
    """
    Get vector query and summary query tools from a document with persistence.
    """
    vector_index = None

    if not os.path.exists(PERSIST_DIR):
        print("Creating data base")
        documents = SimpleDirectoryReader(input_files=[file_path]).load_data()
        splitter = SentenceSplitter(chunk_size=2024)
        nodes = splitter.get_nodes_from_documents(documents)
        vector_index = VectorStoreIndex(nodes)
        vector_index.storage_context.persist(persist_dir=PERSIST_DIR)

    else:
        print("Using stored data!")
        storage_context = StorageContext.from_defaults(persist_dir=PERSIST_DIR)
        vector_index = load_index_from_storage(storage_context)

    if vector_index is None:
        raise ValueError("VectorStoreIndex is not properly initialized")
    
    def vector_query(query: str, page_numbers: Optional[List[str]] = None) -> str:
        """
        Get vector query tool from a document for indexed retrieval.
        """

        try:
            query_engine = None
            page_numbers = page_numbers or []
            
            if page_numbers:
                metadata_filters = MetadataFilters() # Create metadata filters for page numbers if provided
                
                # Add each page number as a filter
                for page in page_numbers: 
                    metadata_filters.add_filter("page_label", page, FilterCondition.EQ)
                
                ## Create query engine with filters
                query_engine = vector_index.as_query_engine(
                    similarity_top_k=1, 
                    filters=metadata_filters
                )
            else:
                ## Create query engine without filters
                query_engine = vector_index.as_query_engine(similarity_top_k=1 )

            response = query_engine.query(query)
            return str(response)
        
        except Exception as e:
            print(f"Error in vector_query: {str(e)}")
            return f"Encountered error: {str(e)}"
    
    vector_query_tool = FunctionTool.from_defaults(
        name = f'vector_tool_{name}',
        fn=vector_query
    )

    # To create a summary_index, we need nodes, which are part of the documents, not vector_index
    documents = SimpleDirectoryReader(input_files=[file_path]).load_data()
    splitter = SentenceSplitter(chunk_size=1024)
    nodes = splitter.get_nodes_from_documents(documents)

    summary_index = SummaryIndex(nodes)
    summary_query_engine = summary_index.as_query_engine(
        response_mode="tree_summarize",
        use_async = True,
    )

    def summary_query(input: str) -> str:
        """
        Get summary from document based on input query
        """
        try:
            response = summary_query_engine.query(input)
            return str(response)
        except Exception as e:
            print(f"Error in summary_query: {str(e)}")
            return f"Encountered error in summary: {str(e)}"

    summary_tool = FunctionTool.from_defaults(
        name = f"summary_tool_{name}",
        fn=summary_query,
        description=f"Useful for summarization questions related to {name}",
    )

    return vector_query_tool, summary_tool 


def clean_text(text: str) -> str:
    """
    Clean the raw text by removing unnecessary whitespace and new lines.
    """
    cleaned_text = ' '.join(line.strip() for line in text.split('\n') if line.strip())
    return cleaned_text

# In utils.py
def initialize_settings(llm_provider: str = "openai"):
    """
    Initializes the LLM, embedding model, and node parser.
    """

    # Load environment variables from .env
    load_dotenv()

    # Common settings
    Settings.embed_model = HuggingFaceEmbedding(model_name="sentence-transformers/all-MiniLM-L6-v2")
    Settings.node_parser = SentenceSplitter(chunk_size=1024, chunk_overlap=200)

    if llm_provider == "openai":
        openai_key = os.getenv("OPENAI_API_KEY")
        if not openai_key:
            raise ValueError("OPENAI_API_KEY is not set in your .env file!")
        Settings.llm = OpenAI(api_key=openai_key)

    elif llm_provider == "groq":
        groq_key = os.getenv("GROQ_API_KEY")
        if not groq_key:
            raise ValueError("GROQ_API_KEY is not set in your .env file!")
        
        # Define a custom class that inherits from Groq and adds rate limiting
        class RateLimitedGroq(Groq):
            def complete(self, prompt, **kwargs):
                return self._rate_limited_call(super().complete, prompt, **kwargs)
                
            def chat(self, messages, **kwargs):
                return self._rate_limited_call(super().chat, messages, **kwargs)
                
            def _rate_limited_call(self, original_func, *args, **kwargs):
                global _last_request_time, _request_count, _cooldown_until
                
                current_time = time.time()
                
                # Check if we're in a cooldown period
                if current_time < _cooldown_until:
                    sleep_time = _cooldown_until - current_time
                    print(f"Rate limiting: waiting for {sleep_time:.1f}s due to previous rate limit...")
                    time.sleep(sleep_time)
                    current_time = time.time()
                    _cooldown_until = 0
                    _request_count = 0
                
                # Reset counter if more than 60s has passed
                if current_time - _last_request_time > 60:
                    _request_count = 0
                
                # If we've made several requests in the last minute, add a delay
                if _request_count > 5:
                    delay = min(5, _request_count * 0.5)  # Scale delay with request count
                    print(f"Proactive rate limiting: adding {delay:.1f}s delay...")
                    time.sleep(delay)
                
                # Now attempt the completion
                max_retries = 5
                retry_count = 0
                base_delay = 5  # Increased from 2 to 5
                
                while retry_count < max_retries:
                    try:
                        result = original_func(*args, **kwargs)
                        # Update request tracking
                        _last_request_time = time.time()
                        _request_count += 1
                        return result
                    except Exception as e:
                        error_msg = str(e)
                        if "rate_limit_exceeded" in error_msg and retry_count < max_retries - 1:
                            retry_count += 1
                            
                            # Try to extract wait time from error message
                            wait_time = extract_wait_time(error_msg)
                            if wait_time is None:
                                # Default backoff
                                wait_time = base_delay * (2 ** retry_count) + random.uniform(0, 1)
                            
                            # Add buffer
                            wait_time += 5  # Increased from 3 to 5
                            
                            print(f"Rate limit reached. Waiting {wait_time:.1f}s before retry {retry_count}/{max_retries}...")
                            
                            # Set the cooldown period
                            _cooldown_until = time.time() + wait_time
                            time.sleep(wait_time)
                            _request_count = 0
                        else:
                            # If it's not a rate limit error or we're out of retries, re-raise
                            raise
        
                raise Exception(f"Failed after {max_retries} retries due to rate limiting.")
        
        # Create our custom rate-limited Groq instance
        Settings.llm = RateLimitedGroq(model="llama3-8b-8192", api_key=groq_key, max_tokens=512)
        print("Created rate-limited Groq LLM")

    else:
        raise ValueError("Invalid LLM provider. Use 'openai' or 'groq'.")