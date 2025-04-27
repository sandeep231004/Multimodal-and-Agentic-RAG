import warnings
warnings.filterwarnings("ignore")
import os
import glob
import easyocr
import time
import numpy as np
import logging
from concurrent.futures import ThreadPoolExecutor
from groq import RateLimitError
from PIL import Image, UnidentifiedImageError
from typing import Optional, List, Tuple, Dict, Any
from dotenv import load_dotenv
from transformers import pipeline
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from DataExtraction.ExtractionUtils_improved import tables_text
from langchain_core.output_parsers import StrOutputParser
import cv2

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

load_dotenv()

def get_chain(api_key: Optional[str] = None, model_name: str = "llama-3.1-8b-instant", temperature: float = 0.3):
    """
    Creates a language model chain for summarization with improved prompt.
    Args:
        temperature: Creativity parameter (lower for more factual responses)  
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
        if api_key:
            model = ChatGroq(
                temperature=temperature,
                model=model_name,
                groq_api_key=api_key,
                timeout=60
            )
        else:
            model = ChatGroq(
                temperature=temperature,
                model=model_name,
                timeout=60
            )
        
        chain = {"element": lambda x: x} | prompt | model | StrOutputParser()
        return chain
    
    except Exception as e:
        logger.error(f"Error initializing model chain: {str(e)}")
        raise

def process_with_retry(chain, content_batch, max_retries=5, initial_concurrency=5):
    """
    Process content with automatic retries and adaptive concurrency.
    Args:
        chain: The LLM chain to use
        content_batch: List of text chunks to process
        max_retries: Maximum number of retry attempts
        initial_concurrency: Starting concurrency level
    Returns:
        List of processed results
    """
    concurrency = initial_concurrency
    retry_count = 0
    
    while retry_count < max_retries:
        try:
            logger.info(f"Processing batch with concurrency: {concurrency}")
            results = chain.batch(content_batch, {"max_concurrency": concurrency})
            return results
        
        except RateLimitError as e:
            retry_count += 1
            # Reduce concurrency on rate limit errors
            concurrency = max(1, concurrency // 2)
            wait_time = min(2 ** retry_count * 5, 120)
            logger.warning(f"Rate limit hit. Reducing concurrency to {concurrency}. Waiting {wait_time}s before retry {retry_count}/{max_retries}")
            time.sleep(wait_time)
        
        except Exception as e:
            logger.error(f"Error processing batch: {str(e)}")
            retry_count += 1
            wait_time = min(2 ** retry_count * 3, 60)
            logger.warning(f"Error occurred. Waiting {wait_time}s before retry {retry_count}/{max_retries}")
            time.sleep(wait_time)
    
    # If we exhaust retries, return empty results for this batch
    logger.error(f"Failed to process batch after {max_retries} retries")
    return [""] * len(content_batch)

def chunk_list(lst, chunk_size=10):
    """Split list into chunks of specified size"""
    return [lst[i:i + chunk_size] for i in range(0, len(lst), chunk_size)]

def get_summaries(tables, texts, api_key: Optional[str] = None):
    """
    Generates table and text summaries used for efficient data retrieval.
    Args:
        tables: list of tables in document
        texts: list of text paragraphs in document       
    Returns:
        Tuple of (text_summaries, table_summaries)
    """
    chain = get_chain(api_key)

    logger.info(f"Processing {len(texts)} text chunks and {len(tables)} tables")
    
    # Process text summaries in smaller batches
    text_summary = []
    if texts:
        # Process in manageable chunks to avoid overwhelming the API
        text_batches = chunk_list(texts, 20)
        for i, batch in enumerate(text_batches):
            batch_results = process_with_retry(chain, batch)
            text_summary.extend(batch_results)
            # Small delay between batches
            if i < len(text_batches) - 1:
                time.sleep(2)
    
    # Process table summaries
    table_summary = []
    if tables and len(tables) > 0:
        try:
            # Check if tables have the expected structure
            if all(hasattr(table, 'metadata') and hasattr(table.metadata, 'text_as_html') for table in tables):
                html_tables = [t.metadata.text_as_html for t in tables]
            else:
                # Create fallback representation for tables without expected structure
                missing_metadata = [i for i, table in enumerate(tables) if not hasattr(table, 'metadata') or not hasattr(table.metadata, 'text_as_html')]
                logger.warning(f"Tables don't have the expected structure with metadata.text_as_html. Issues at indices: {missing_metadata}")
                html_tables = []
                for table in tables:
                    if hasattr(table, 'metadata') and hasattr(table.metadata, 'text_as_html'):
                        html_tables.append(table.metadata.text_as_html)
                    else:
                        # Fallback to string representation or another appropriate format
                        html_tables.append(str(table))
            
            # Process tables in smaller batches
            table_batches = chunk_list(html_tables, 10)
            for i, batch in enumerate(table_batches):
                batch_results = process_with_retry(chain, batch, max_retries=2)
                table_summary.extend(batch_results)
                
                if i < len(table_batches) - 1:
                    time.sleep(2)
        
        except Exception as e:
            logger.error(f"Error processing tables: {str(e)}")
    else:
        logger.info("No tables found in the document")
    
    # Filter out empty summaries to prevent downstream issues
    text_summary = [summary for summary in text_summary if summary.strip()]
    table_summary = [summary for summary in table_summary if summary.strip()]
    
    # Log length information for better debugging
    logger.info(f"Generated {len(text_summary)} text summaries and {len(table_summary)} table summaries")
    
    # Check if there's still a mismatch in lengths
    if len(texts) != len(text_summary):
        logger.warning(f"Warning: texts ({len(texts)}) and text_summaries ({len(text_summary)}) have different lengths")
    
    if len(tables) != len(table_summary):
        logger.warning(f"Warning: tables ({len(tables)}) and table_summaries ({len(table_summary)}) have different lengths")
    
    return text_summary, table_summary

def preprocess_image(image_path):
    """
    Improve image quality for better OCR results.
    """
    try:
        # Read image
        img = cv2.imread(image_path)
        if img is None:
            logger.warning(f"Could not read image at {image_path}")
            return None
            
        # Convert to grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Apply adaptive thresholding
        thresh = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
            cv2.THRESH_BINARY, 11, 2
        )
        
        # Noise removal
        kernel = np.ones((1, 1), np.uint8)
        opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
        
        return opening
    except Exception as e:
        logger.error(f"Error preprocessing image {image_path}: {str(e)}")
        return None

def process_image(img_path, reader, caption_pipe):
    """
    Process a single image with OCR and captioning.
    """
    try:
        # Prepare image for OCR
        preprocessed = preprocess_image(img_path)
        
        # Extract text with OCR
        if preprocessed is not None:
            results = reader.readtext(preprocessed)   # Use preprocessed image for OCR
        else:
            results = reader.readtext(img_path)  # Fallback to original image
            
        extracted_text = " ".join([text for _, text, _ in results])
        
        # Generate caption
        img = Image.open(img_path).convert('RGB')
        caption = caption_pipe(img)[0]['generated_text']
        
        return extracted_text, caption
    
    except Exception as e:
        logger.error(f"Error processing image {img_path}: {str(e)}")
        return "", ""

def encode_images(pipe, ocr_lang=['en'], num_workers=4):
    """
    Extracts text from images using OCR and image-to-text models.
    """
    img_path_list = glob.glob("extracted_new/figure*.jpg") + glob.glob("extracted_new/figure*.png") + glob.glob("extracted_new/chart*.jpg") + glob.glob("extracted_new/chart*.png")

    logger.info(f"Number of image files found: {len(img_path_list)}")

    if not img_path_list:
        logger.warning("No image files found.")
        return [], []

    reader = easyocr.Reader(ocr_lang, gpu=False)
    
    extracted_texts = []
    captions = []
    
    # Process images in parallel
    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        results = list(executor.map(
            lambda path: process_image(path, reader, pipe),
            img_path_list
        ))
    
    # Separate the results
    for ocr_text, caption in results:
        extracted_texts.append(ocr_text)
        captions.append(caption)

    logger.info(f"Processed {len(img_path_list)} images")
    return extracted_texts, captions

def text_table_img(path, api_key: Optional[str] = None):
    """
    Extract and process tables, text, and images from a document.
    """
    # Check if file exists
    if not os.path.exists(path):
        logger.error(f"Error: File not found at {path}")
        return [], [], [], [], [], []
    
    try:
        tables, texts = tables_text(path)
        logger.info(f"Extraction complete. Found {len(tables)} tables and {len(texts)} text chunks")
    except Exception as e:
        logger.error(f"Error extracting tables and text: {str(e)}")
        tables, texts = [], []
    
    # Setup image processing and extract images
    try:
        pipe = pipeline("image-to-text", model="Salesforce/blip-image-captioning-large", device=-1)
        images, images_summary = encode_images(pipe)
    except Exception as e:
        logger.error(f"Error processing images: {str(e)}")
        images, images_summary = [], []

    # Generate summaries
    try:
        text_summary, table_summary = get_summaries(tables, texts, api_key)
        logger.info("Text, Table and Image Summaries extracted")
    except Exception as e:
        logger.error(f"Error generating summaries: {str(e)}")
        text_summary, table_summary = [], []

    return tables, texts, images, text_summary, table_summary, images_summary

if __name__ == "__main__" : 
    # Configure more verbose logging for standalone execution
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    
    path = 'data/document.pdf'
    api_key = os.getenv("GROQ_API_KEY")
    
    # Run the extraction process
    tables, texts, images, text_summary, table_summary, image_summary = text_table_img(path, api_key)
    
    # Output results summary
    logger.info(f"Extraction complete: {len(tables)} tables, {len(texts)} text chunks, {len(images)} images, {len(table_summary)} table summaries, {len(text_summary)} text summaries, {len(image_summary)} image summaries" )