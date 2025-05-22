import os
import time
import random
import re
import logging
from dotenv import load_dotenv
from typing import Optional, List, Dict, Any, Tuple

from llama_index.llms.groq import Groq
from llama_index.llms.openai import OpenAI

from llama_index.core import (
    SimpleDirectoryReader,
    VectorStoreIndex,
    SummaryIndex,
    StorageContext,
    load_index_from_storage,
    Settings,
)
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.tools import FunctionTool
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core.vector_stores import MetadataFilters, FilterCondition
from llama_index.core.vector_stores import MetadataFilters, FilterCondition, MetadataFilter
# --- Configuration ---
PERSIST_DIR = "./storage"

# --- Logging Setup ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

load_dotenv()

# --- Rate Limiting Helper ---
def extract_wait_time(error_message: str) -> Optional[float]:
    """
    Extracts the recommended wait time in seconds from a Groq rate limit error message.
    """
    match_min_sec = re.search(r'Please try again in (\d+)m(\d+\.\d+)s', error_message)
    if match_min_sec:
        minutes = int(match_min_sec.group(1))
        seconds = float(match_min_sec.group(2))
        return minutes * 60 + seconds
    
    match_sec = re.search(r'Please try again in (\d+\.\d+)s', error_message)
    if match_sec:
        return float(match_sec.group(1))
    
    logger.warning(f"Could not extract specific wait time from error: {error_message}")
    return None

# --- Custom Rate-Limited Groq LLM Class ---
class RateLimitedGroq(Groq):
    """
    A custom Groq LLM class that implements robust rate limiting with exponential backoff,
    proactive delays, and error message parsing for wait times.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._last_request_time = 0.0
        self._request_count = 0
        self._cooldown_until = 0.0
        self._max_retries = 5
        self._base_delay = 5

    def _rate_limited_call(self, original_func, *args, **kwargs):
        """Internal method to apply rate limiting logic to a given function call."""
        current_time = time.time()
        
        if current_time < self._cooldown_until:
            sleep_time = self._cooldown_until - current_time
            logger.info(f"Rate limiting: Waiting for {sleep_time:.1f}s due to previous API cooldown.")
            time.sleep(sleep_time)
            current_time = time.time()
            self._cooldown_until = 0.0
            self._request_count = 0
        
        if current_time - self._last_request_time > 60:
            self._request_count = 0
        
        if self._request_count > 5:
            delay = min(5, self._request_count * 0.5)
            logger.info(f"Proactive rate limiting: Adding {delay:.1f}s delay (requests in last min: {self._request_count}).")
            time.sleep(delay)
        
        retry_count = 0
        while retry_count < self._max_retries:
            try:
                result = original_func(*args, **kwargs)
                self._last_request_time = time.time()
                self._request_count += 1
                return result
            
            except Exception as e:
                error_msg = str(e)
                if "rate_limit_exceeded" in error_msg and retry_count < self._max_retries - 1:
                    retry_count += 1
                    
                    wait_time = extract_wait_time(error_msg)
                    if wait_time is None:
                        wait_time = self._base_delay * (2 ** retry_count) + random.uniform(0, 1)
                    
                    wait_time += 5
                    
                    logger.warning(f"Rate limit reached. Waiting {wait_time:.1f}s before retry {retry_count}/{self._max_retries}...")
                    
                    self._cooldown_until = time.time() + wait_time
                    time.sleep(wait_time)
                    self._request_count = 0
                else:
                    raise
        
        raise Exception(f"Failed after {self._max_retries} retries due to persistent rate limiting or other errors.")

    def complete(self, prompt: str, **kwargs: Any) -> Any:
        """Overrides the base complete method to apply rate limiting."""
        return self._rate_limited_call(super().complete, prompt, **kwargs)
            
    def chat(self, messages: List[Dict[str, Any]], **kwargs: Any) -> Any:
        """Overrides the base chat method to apply rate limiting."""
        return self._rate_limited_call(super().chat, messages, **kwargs)

# --- Document Indexing and Tool Creation ---
def get_doc_tools(
    file_path: str,
    name: str,
) -> Tuple[FunctionTool, FunctionTool]:
    """
    Creates and returns LlamaIndex FunctionTools for vector-based retrieval and
    summary-based retrieval from a document, with index persistence.
    """
    vector_index: Optional[VectorStoreIndex] = None
    
    doc_persist_dir = os.path.join(PERSIST_DIR, f"{name}_index")

    if not os.path.exists(doc_persist_dir):
        logger.info(f"Creating new index for '{name}' at {doc_persist_dir}")
        documents = SimpleDirectoryReader(input_files=[file_path]).load_data()
        splitter = SentenceSplitter(chunk_size=2024, chunk_overlap=200)
        nodes = splitter.get_nodes_from_documents(documents)
        
        vector_index = VectorStoreIndex(nodes)
        vector_index.storage_context.persist(persist_dir=doc_persist_dir)
        logger.info(f"Index for '{name}' created and persisted.")
    else:
        logger.info(f"Loading existing index for '{name}' from {doc_persist_dir}")
        storage_context = StorageContext.from_defaults(persist_dir=doc_persist_dir)
        vector_index = load_index_from_storage(storage_context)
        logger.info(f"Index for '{name}' loaded successfully.")

    if vector_index is None:
        raise ValueError("VectorStoreIndex could not be initialized or loaded.")
    
    def vector_query(query: str, page_numbers: Optional[List[str]] = None) -> str:
        """
        Performs a vector-based similarity search on the document.
        Useful for finding specific information or details within the document.
        """
        try:
            query_engine = None
            
            if page_numbers:
                metadata_filters = MetadataFilters(
                    filters=[
                        MetadataFilter(key="page_label", operator="==", value=page)
                        for page in page_numbers
                    ]
                )
                query_engine = vector_index.as_query_engine(
                    similarity_top_k=1,
                    filters=metadata_filters
                )
            else:
                query_engine = vector_index.as_query_engine(similarity_top_k=1)

            response = query_engine.query(query)
            return str(response)
        
        except Exception as e:
            logger.error(f"Error in vector_query for '{name}': {str(e)}")
            return f"An error occurred during vector query: {str(e)}"
    
    vector_query_tool = FunctionTool.from_defaults(
        name = f'vector_tool_{name}',
        fn=vector_query,
        description=f"Useful for questions requiring specific details or retrieving relevant sections from the {name} document. "
                    f"Optionally, provide 'page_numbers' (e.g., ['1', '2']) to search only within specific pages."
    )

    documents_for_summary = SimpleDirectoryReader(input_files=[file_path]).load_data()
    summary_splitter = SentenceSplitter(chunk_size=1024, chunk_overlap=200)
    summary_nodes = summary_splitter.get_nodes_from_documents(documents_for_summary)

    summary_index = SummaryIndex(summary_nodes)
    summary_query_engine = summary_index.as_query_engine(
        response_mode="tree_summarize",
        use_async=True,
    )

    def summary_query(input_query: str) -> str:
        """
        Generates a high-level summary of the document based on a given input query.
        Useful for general understanding or questions requiring an overview.
        """
        try:
            response = summary_query_engine.query(input_query)
            return str(response)
        except Exception as e:
            logger.error(f"Error in summary_query for '{name}': {str(e)}")
            return f"An error occurred during summary query: {str(e)}"

    summary_tool = FunctionTool.from_defaults(
        name = f"summary_tool_{name}",
        fn=summary_query,
        description=f"Useful for summarization questions or getting a general overview related to the {name} document."
    )

    return vector_query_tool, summary_tool 

# --- LLM and Embedding Settings Initialization ---
def initialize_settings(llm_provider: str = "groq"):
    """
    Initializes global LlamaIndex settings for the LLM, embedding model, and node parser.
    """
    Settings.embed_model = HuggingFaceEmbedding(model_name="sentence-transformers/all-MiniLM-L6-v2")
    logger.info("Initialized embedding model: sentence-transformers/all-MiniLM-L6-v2")
    
    Settings.node_parser = SentenceSplitter(chunk_size=1024, chunk_overlap=200)
    logger.info("Initialized node parser: SentenceSplitter (chunk_size=1024, chunk_overlap=200)")

    if llm_provider == "openai":
        openai_key = os.getenv("OPENAI_API_KEY")
        if not openai_key:
            raise ValueError("OPENAI_API_KEY environment variable is not set!")
        Settings.llm = OpenAI(api_key=openai_key, model="gpt-3.5-turbo", temperature=0.0)
        logger.info("Initialized OpenAI LLM: gpt-3.5-turbo")

    elif llm_provider == "groq":
        groq_key = os.getenv("GROQ_API_KEY")
        if not groq_key:
            raise ValueError("GROQ_API_KEY environment variable is not set!")
        
        Settings.llm = RateLimitedGroq(api_key=groq_key, model="llama3-8b-8192", max_tokens=512, temperature=0.0)
        logger.info("Initialized Rate-Limited Groq LLM: llama3-8b-8192")

    else:
        raise ValueError(f"Invalid LLM provider: '{llm_provider}'. Please choose 'openai' or 'groq'.")

# --- Utility for cleaning text ---
def clean_text(text: str) -> str:
    """
    Cleans raw text by removing excessive whitespace and consolidating newlines.
    """
    cleaned_text = ' '.join(line.strip() for line in text.split('\n') if line.strip())
    return cleaned_text

# Example of how to use it
if __name__ == "__main__":
    logger.info("--- Running utils.py demonstration ---")
    
    try:
        initialize_settings(llm_provider="groq")
        logger.info("LlamaIndex settings initialized successfully with Groq.")
    except ValueError as e:
        logger.error(f"Failed to initialize settings: {e}")
        logger.info("Please set your GROQ_API_KEY in a .env file.")
        exit()
        
    dummy_file_path = "data/sample_document.txt"
    os.makedirs(os.path.dirname(dummy_file_path), exist_ok=True)
    with open(dummy_file_path, "w") as f:
        f.write("This is a sample document for testing purposes. It contains some information.\n")
        f.write("Page 1. This is the first paragraph.\n")
        f.write("Page 2. This is the second paragraph with more details.")
    
    logger.info(f"Created dummy document: {dummy_file_path}")

    try:
        vector_tool, summary_tool = get_doc_tools(
            file_path=dummy_file_path,
            name="SampleDoc"
        )
        logger.info(f"Document tools created: {vector_tool.metadata.name}, {summary_tool.metadata.name}")

        logger.info("\n--- Testing Vector Tool ---")
        vector_result = vector_tool("What is the purpose of this document?", page_numbers=['1'])
        logger.info(f"Vector Query Result (Page 1): {vector_result}")

        logger.info("\n--- Testing Summary Tool ---")
        summary_result = summary_tool("Summarize the document.")
        logger.info(f"Summary Query Result: {summary_result}")
        
    except Exception as e:
        logger.error(f"Error during tool creation or demonstration: {e}")
    finally:
        if os.path.exists(dummy_file_path):
            os.remove(dummy_file_path)
        dummy_index_dir = os.path.join(PERSIST_DIR, "SampleDoc_index")
        if os.path.exists(dummy_index_dir):
            import shutil
            shutil.rmtree(dummy_index_dir)
        logger.info("Cleaned up dummy files and storage.")
    