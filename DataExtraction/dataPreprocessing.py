import warnings
warnings.filterwarnings("ignore") # Suppress minor warnings for cleaner output
import os
import glob
import easyocr
import time
import numpy as np
import logging
from concurrent.futures import ThreadPoolExecutor # For parallel processing of images
from groq import RateLimitError # Specific error for Groq API rate limits
from PIL import Image, UnidentifiedImageError # For image manipulation
from typing import Optional, List, Tuple, Dict, Any # For type hinting
from dotenv import load_dotenv # To load environment variables (like API keys)
from transformers import pipeline # For image captioning (HuggingFace Transformers)
from langchain_groq import ChatGroq # LangChain integration for Groq LLMs
from langchain_core.prompts import ChatPromptTemplate # For structuring LLM prompts
from DataExtraction.ExtractionUtils import tables_text # Custom utility for PDF parsing
from langchain_core.output_parsers import StrOutputParser # To parse LLM output as a string
import cv2 # OpenCV for image preprocessing

# --- Logging Setup ---
# Configures basic logging to show INFO level messages and above, with timestamp and level.
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Load environment variables (e.g., API keys from a .env file)
load_dotenv()

# --- LLM Chain Setup ---
def get_chain(api_key: Optional[str] = None, model_name: str = "llama-3.1-8b-instant", temperature: float = 0.3):
    """
    Creates a LangChain LLM chain specifically for summarizing text or tables.
    Returns: A LangChain Runnable object configured for summarization.
    """
    prompt_text = """
    You are an expert document analyzer tasked with summarizing information from a document.

    Given the following table or text chunk, provide a concise and informative summary that:
    1. Captures the key information, facts, or statistics
    2. Preserves important numerical data and relationships
    3. Maintains the original context and meaning

    If analyzing a table:
    - Identify the main subject of the table
    - Note key trends, patterns or significant values
    - Include important column/row relationships

    If analyzing text:
    - Focus on main ideas and supporting details
    - Preserve any critical facts, figures, or quantities
    - Maintain the logical flow of information

    Respond ONLY with the summary itself. No introduction or explanation.

    Table or text chunk: {element}
    """
    prompt = ChatPromptTemplate.from_template(prompt_text)

    try:
        # Initialize the ChatGroq model with the given parameters
        model = ChatGroq(
            temperature=temperature,
            model=model_name,
            groq_api_key=api_key, # Uses provided key or falls back to env var
            timeout=60 # Set a timeout for API calls
        )
        
        # Define the chain: input -> prompt -> model -> output parser
        chain = {"element": lambda x: x} | prompt | model | StrOutputParser()
        return chain
    
    except Exception as e:
        logger.error(f"Error initializing model chain: {str(e)}")
        raise # Re-raise the exception to indicate a critical failure

# --- API Call and Retry Logic ---
def process_with_retry(chain, content_batch: List[Any], max_retries: int = 5, initial_concurrency: int = 5) -> List[str]:
    """
    Processes a batch of content using the LLM chain with automatic retries and adaptive concurrency.
    This helps to manage API rate limits and temporary network issues

    Returns:List[str]: A list of processed results (summaries) for the batch.
    """
    concurrency = initial_concurrency
    retry_count = 0
    
    while retry_count < max_retries:
        try:
            logger.info(f"Processing batch with concurrency: {concurrency}")
            # Use chain.batch for parallel processing of the batch
            results = chain.batch(content_batch, {"max_concurrency": concurrency})
            return results
        
        except RateLimitError as e:
            retry_count += 1
            # Reduce concurrency and wait longer on rate limit errors to avoid being blocked
            concurrency = max(1, concurrency // 2) # Halve concurrency, minimum 1
            wait_time = min(2 ** retry_count * 5, 120) # Exponential backoff, max 120s
            logger.warning(f"Rate limit hit. Reducing concurrency to {concurrency}. Waiting {wait_time}s before retry {retry_count}/{max_retries}")
            time.sleep(wait_time)
        
        except Exception as e:
            logger.error(f"Error processing batch: {str(e)}")
            retry_count += 1
            # Wait with exponential backoff for other errors too
            wait_time = min(2 ** retry_count * 3, 60) # Exponential backoff, max 60s
            logger.warning(f"Error occurred. Waiting {wait_time}s before retry {retry_count}/{max_retries}")
            time.sleep(wait_time)
    
    # If all retries fail, log an error and return empty strings for the batch
    logger.error(f"Failed to process batch after {max_retries} retries")
    return [""] * len(content_batch) # Return empty strings to maintain list length

def chunk_list(lst: List[Any], chunk_size: int = 10) -> List[List[Any]]:
    """
    Splits a given list into smaller chunks of a specified size.
    Useful for batch processing to manage API limits and memory.
    """
    return [lst[i:i + chunk_size] for i in range(0, len(lst), chunk_size)]

# --- Summarization Logic ---
def get_summaries(tables: List[Any], texts: List[Any], api_key: Optional[str] = None) -> Tuple[List[str], List[str]]:
    """
    Generates concise summaries for both extracted text chunks and tables using an LLM.
    Returns:
        Tuple[List[str], List[str]]: A tuple containing two lists:
            - text_summary (List[str]): Summaries for each text chunk.
            - table_summary (List[str]): Summaries for each table.
    """
    chain = get_chain(api_key) # Initialize the LLM chain

    logger.info(f"Processing {len(texts)} text chunks and {len(tables)} tables for summarization.")
    
    # Process text summaries in batches
    text_summary = []
    if texts:
        text_batches = chunk_list(texts, 20) # Chunk text into batches of 20
        for i, batch in enumerate(text_batches):
            batch_results = process_with_retry(chain, batch)
            text_summary.extend(batch_results)
            # Add a small delay between batches to be kind to the API
            if i < len(text_batches) - 1:
                time.sleep(2)
    
    # Process table summaries
    table_summary = []
    if tables: # Check if there are tables to process
        try:
            html_tables = []
            # Prepare tables for LLM input, preferring HTML representation
            for i, table in enumerate(tables):
                if hasattr(table, 'metadata') and hasattr(table.metadata, 'text_as_html'):
                    html_tables.append(table.metadata.text_as_html)
                else:
                    # Fallback to string representation if HTML metadata is missing
                    logger.warning(f"Table at index {i} does not have 'metadata.text_as_html'. Falling back to string representation.")
                    html_tables.append(str(table))
            
            # Process table HTMLs in batches
            table_batches = chunk_list(html_tables, 10) # Chunk tables into batches of 10
            for i, batch in enumerate(table_batches):
                # Use a slightly lower max_retries for tables as they might be more complex
                batch_results = process_with_retry(chain, batch, max_retries=2)
                table_summary.extend(batch_results)
                
                if i < len(table_batches) - 1:
                    time.sleep(2)
        
        except Exception as e:
            logger.error(f"Error processing tables for summarization: {str(e)}")
    else:
        logger.info("No tables found in the document for summarization.")
    
    # Filter out any empty summaries that might have resulted from failed retries
    text_summary = [summary for summary in text_summary if summary.strip()]
    table_summary = [summary for summary in table_summary if summary.strip()]
    
    # Log length information and warnings for debugging
    logger.info(f"Generated {len(text_summary)} text summaries and {len(table_summary)} table summaries.")
    
    if len(texts) != len(text_summary):
        logger.warning(f"Warning: Number of original text chunks ({len(texts)}) does not match generated text summaries ({len(text_summary)}).")
    
    if len(tables) != len(table_summary):
        logger.warning(f"Warning: Number of original tables ({len(tables)}) does not match generated table summaries ({len(table_summary)}).")
    
    return text_summary, table_summary

# --- Image Processing Logic ---
def preprocess_image(image_path: str) -> Optional[np.ndarray]:
    """
    Improves image quality for better OCR (Optical Character Recognition) results
    by converting to grayscale, applying adaptive thresholding, and noise removal.
    Returns:
        Optional[np.ndarray]: The preprocessed image as a NumPy array (OpenCV format),
                              or None if the image cannot be read or an error occurs.
    """
    try:
        img = cv2.imread(image_path)
        if img is None:
            logger.warning(f"Could not read image at {image_path}. Skipping preprocessing.")
            return None
            
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) # Convert to grayscale
        
        # Apply adaptive thresholding to convert to black and white, enhancing text
        thresh = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
            cv2.THRESH_BINARY, 11, 2
        )
        
        # Apply morphological opening to remove small noise
        kernel = np.ones((1, 1), np.uint8)
        opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
        
        return opening
    except Exception as e:
        logger.error(f"Error preprocessing image {image_path}: {str(e)}")
        return None

def process_image(img_path: str, reader: easyocr.Reader, caption_pipe) -> Tuple[str, str]:
    """
    Processes a single image by performing OCR to extract text and generating a caption
    using an image-to-text model.
    Returns:
        Tuple[str, str]: A tuple containing:
            - extracted_text (str): Text found in the image via OCR.
            - caption (str): A descriptive caption generated for the image.
    """
    try:
        # Preprocess the image for better OCR results
        preprocessed = preprocess_image(img_path)
        
        # Perform OCR: use preprocessed image if successful, otherwise original
        if preprocessed is not None:
            results = reader.readtext(preprocessed)
        else:
            results = reader.readtext(img_path)
            
        extracted_text = " ".join([text for _, text, _ in results]) # Concatenate OCR results
        
        # Generate image caption using the HuggingFace pipeline
        img = Image.open(img_path).convert('RGB') # Open image for captioning
        caption = caption_pipe(img)[0]['generated_text']
        
        return extracted_text, caption
    
    except Exception as e:
        logger.error(f"Error processing image {img_path}: {str(e)}")
        return "", "" # Return empty strings on failure

def encode_images(pipe, ocr_lang: List[str] = ['en'], num_workers: int = 4) -> Tuple[List[str], List[str]]:
    """
    Extracts text from image files (e.g., figures, charts) using OCR and generates
    descriptive captions for them using an image-to-text model.
    Returns:
        Tuple[List[str], List[str]]: A tuple containing:
            - extracted_texts (List[str]): OCR text from each image.
            - captions (List[str]): Generated captions for each image.
    """
    # Find all image files generated by Unstructured (stored in 'extracted_new' directory)
    img_path_list = glob.glob("extracted_new/figure*.jpg") + \
                    glob.glob("extracted_new/figure*.png") + \
                    glob.glob("extracted_new/chart*.jpg") + \
                    glob.glob("extracted_new/chart*.png")

    logger.info(f"Number of image files found for processing: {len(img_path_list)}")

    if not img_path_list:
        logger.warning("No image files found in 'extracted_new' directory. Skipping image processing.")
        return [], []

    # Initialize EasyOCR reader
    reader = easyocr.Reader(ocr_lang, gpu=False) # gpu=False for CPU-only processing

    extracted_texts = []
    captions = []
    
    # Process images in parallel using a ThreadPoolExecutor
    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        # map applies process_image to each path in img_path_list
        results = list(executor.map(
            lambda path: process_image(path, reader, pipe),
            img_path_list
        ))
    
    # Separate the results (OCR text and captions) into their respective lists
    for ocr_text, caption in results:
        extracted_texts.append(ocr_text)
        captions.append(caption)

    logger.info(f"Successfully processed {len(img_path_list)} images.")
    return extracted_texts, captions

# --- Main Data Processing Orchestration ---
def text_table_img(path: str, api_key: Optional[str] = None) -> Tuple[List[Any], List[Any], List[str], List[str], List[str], List[str]]:
    """
    Orchestrates the full process of extracting content (tables, text, images)
    from a document, applying OCR, image captioning, and generating summaries.
    Returns:
        Tuple[List[Any], List[Any], List[str], List[str], List[str], List[str]]: A tuple containing:
            - tables (List[Any]): Original extracted table objects.
            - texts (List[Any]): Original extracted text chunk objects.
            - images (List[str]): OCR text from images.
            - text_summary (List[str]): Summaries of text chunks.
            - table_summary (List[str]): Summaries of tables.
            - images_summary (List[str]): Captions generated for images.
    """
    if not os.path.exists(path):
        logger.error(f"Error: Document file not found at {path}. Please check the path.")
        return [], [], [], [], [], [] # Return empty lists if file not found
    
    # Step 1: Extract tables and text chunks using ExtractionUtils_improved
    try:
        tables, texts = tables_text(path)
        logger.info(f"Initial extraction complete. Found {len(tables)} tables and {len(texts)} text chunks.")
    except Exception as e:
        logger.error(f"Failed to extract tables and text from {path}: {str(e)}")
        tables, texts = [], [] # Ensure lists are empty on failure
    
    # Step 2: Set up image processing pipeline and extract image text/captions
    # Note: Images are saved to the 'extracted_new' directory by ExtractionUtils_improved
    try:
        # Initialize HuggingFace image-to-text pipeline (downloads model if not present)
        image_captioning_pipeline = pipeline("image-to-text", model="Salesforce/blip-image-captioning-large", device=-1) # device=-1 uses CPU
        images_ocr_text, images_captions = encode_images(image_captioning_pipeline)
        logger.info(f"Image processing complete. Extracted text from {len(images_ocr_text)} images and generated {len(images_captions)} captions.")
    except Exception as e:
        logger.error(f"Failed to process images: {str(e)}")
        images_ocr_text, images_captions = [], []

    # Step 3: Generate summaries for text and tables using LLM
    try:
        text_summary, table_summary = get_summaries(tables, texts, api_key)
        logger.info("Text, Table, and Image Summaries generated successfully.")
    except Exception as e:
        logger.error(f"Failed to generate summaries for text and tables: {str(e)}")
        text_summary, table_summary = [], []

    # Return all processed data
    return tables, texts, images_ocr_text, text_summary, table_summary, images_captions

# --- Main Execution Block ---
if __name__ == "__main__":
    # Configure logging for the main execution block
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    
    document_path = 'data/document.pdf' # Path to the document you want to process
    groq_api_key = os.getenv("GROQ_API_KEY") # Get API key from environment variables

    logger.info(f"Starting data preprocessing for: {document_path}")
    
    # Run the full extraction and processing pipeline
    extracted_tables, extracted_texts, ocr_images, text_summaries, table_summaries, image_captions = \
        text_table_img(document_path, groq_api_key)
    
    # Output a final summary of the processed data counts
    logger.info(f"""
    --- Processing Complete ---
    Original Tables: {len(extracted_tables)}
    Original Text Chunks: {len(extracted_texts)}
    OCR Text from Images: {len(ocr_images)}
    ---------------------------
    Text Summaries: {len(text_summaries)}
    Table Summaries: {len(table_summaries)}
    Image Captions: {len(image_captions)}
    """)