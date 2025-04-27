import warnings
warnings.filterwarnings("ignore")
import tqdm
import os
import logging
from pathlib import Path
from typing import List, Tuple, Dict, Any
from dotenv import load_dotenv
from unstructured.partition.pdf import partition_pdf
from unstructured.documents.elements import Table, Image, CompositeElement, Element

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

load_dotenv()

# Check for optional dependencies
try:
    import pytesseract
    import cv2
    HAS_OCR = True
except ImportError:
    HAS_OCR = False
    logger.info("OCR capabilities limited: cv2/pytesseract not installed")

try:
    import pdfplumber
    HAS_PDFPLUMBER = True
except ImportError:
    HAS_PDFPLUMBER = False
    logger.info("Advanced table extraction limited: pdfplumber not installed")

def partition(file_path):
    """
    Creates chunks of given document used for creating embeddings for RAG agent.
    """
    logger.info(f"Partitioning document: {file_path}")
    
    # Enhanced extraction settings for better chunk quality
    chunks = partition_pdf(
        filename=file_path, 
        infer_table_structure=True,         # Maintain table structure
        strategy='hi_res',                  # High quality OCR for better text extraction
        ocr_languages=["eng"],              # Specify language for OCR (default: English)
        extract_images=True,                # Extract images
        extract_image_block_types=["Image", "Figure", "Chart"], # Extract various visual elements
        extract_image_block_output_dir="extracted_new", 
        extract_image_block_to_payload=True, # Include image data in payload for processing
        
        # Chunking parameters optimized for RAG
        chunking_strategy="by_title",       # Uses titles/headings for logical splits
        max_characters=4000,  
        overlap_characters=400,              # Reduced from 10000 for better context window fit
        combine_text_under_n_chars=1000,    # Reduced to create more cohesive chunks
        new_after_n_chars=3000,             # Slightly reduced for more manageable chunks
        
        # Table and structure settings
        extract_tables=True,                # Enable table extraction
        include_page_breaks=True,           # Preserve page context
        include_metadata=True,              # Keep metadata for better context
    )
    
    logger.info(f"Document partitioned into {len(list(chunks))} chunks")
    return chunks # return: chunks

def extract_tables_with_pdfplumber(path):
    """Extract tables using pdfplumber for better table detection."""
    if not HAS_PDFPLUMBER:
        return []
        
    additional_tables = []
    try:
        with pdfplumber.open(path) as pdf:
            for i, page in enumerate(pdf.pages):
                page_tables = page.extract_tables()
                for table_data in page_tables:
                    if not table_data:  # Skip empty tables
                        continue
                        
                    # Convert to HTML format
                    html = '<table border="1">'
                    for row in table_data:
                        html += '<tr>'
                        for cell in row:
                            cell_content = str(cell).replace('None', '') if cell else ''
                            html += f'<td>{cell_content}</td>'
                        html += '</tr>'
                    html += '</table>'
                    
                    # Create text representation
                    text_content = '\n'.join([
                        '\t'.join([str(cell) if cell else '' for cell in row])
                        for row in table_data
                    ])
                    
                    # Create enhanced table object
                    class EnhancedTable(Table):
                        def __init__(self, html_content, text_content, page_number):
                            self.metadata = type('obj', (object,), {'text_as_html': html_content})
                            self.text = text_content
                            self.page_number = page_number
                    
                    table_obj = EnhancedTable(html, text_content, i+1)
                    additional_tables.append(table_obj)
    except Exception as e:
        logger.error(f"Error in pdfplumber extraction: {e}")
    
    return additional_tables

def is_likely_table(text):
    """Detect if text likely represents tabular data."""
    text_lower = text.lower()
    
    # Check for common table indicators
    if ('|' in text and ('-' * 3) in text):
        return True
    elif text.count('\t') > 3:
        return True
    elif any(text.count(sep) > 3 for sep in [',', '\t', '|', ';']):
        return True
    elif text.count('\n') > 3 and len(set([line.count(':') for line in text.split('\n') if line.strip()])) <= 2:
        return True
    return False

def create_table_like(content):
    """Create a table-like object from text content."""
    class TableLike:
        def __init__(self, content):
            self.metadata = type('obj', (object,), {'text_as_html': f"<table><tr><td>{content}</td></tr></table>"})
            self.text = content
    
    return TableLike(content)

def process_image_with_ocr(image_path):
    """Extract text from image using OCR."""
    if not HAS_OCR:
        return ""
        
    try:
        img = cv2.imread(image_path)
        if img is None:
            return ""
        return pytesseract.image_to_string(img)
    except Exception as e:
        logger.error(f"OCR error on {image_path}: {e}")
        return ""

def tables_text(path):
    """
    Extract tables and text from document with enhanced detection for
    charts, pictures, and tables including text within tables.
    """
    if not os.path.exists(path):
        logger.error(f"File not found: {path}")
        return [], []
        
    logger.info("Starting document partitioning")
    chunks = partition(file_path=path)
    logger.info(f"Document partitioned into {len(chunks)} chunks")
    
    # Create output directories
    os.makedirs("extracted/tables", exist_ok=True)
    os.makedirs("extracted/images", exist_ok=True)
    os.makedirs("extracted/charts", exist_ok=True)
    
    tables = []
    texts = []
    
    # Extract tables with pdfplumber for better detection
    additional_tables = extract_tables_with_pdfplumber(path)
    
    # Process all chunks from unstructured
    for idx, chunk in enumerate(chunks):
        # Table detection
        if isinstance(chunk, Table) or 'table' in str(type(chunk)).lower():
            tables.append(chunk)
            
            # Save table reference
            if hasattr(chunk, 'metadata') and hasattr(chunk.metadata, 'text_as_html'):
                with open(f"extracted_new/tables/table_{idx}.html", 'w', encoding='utf-8') as f:
                    f.write(chunk.metadata.text_as_html)
        
        # Image detection
        elif isinstance(chunk, Image) or 'image' in str(type(chunk)).lower():
            # Extract text from image with OCR
            if HAS_OCR and hasattr(chunk, 'metadata') and hasattr(chunk.metadata, 'image_path'):
                ocr_text = process_image_with_ocr(chunk.metadata.image_path)
                if ocr_text.strip():
                    texts.append(ocr_text)
        
        # Chart detection
        elif any(x in str(chunk).lower() for x in ['chart', 'figure', 'graph', 'plot']):
            texts.append(chunk)  # Include as text to capture descriptions
        
        # Handle composite elements
        elif isinstance(chunk, CompositeElement):
            has_table = False
            
            # Check for tables or images in child elements
            if hasattr(chunk, 'elements'):
                for element in chunk.elements:
                    if isinstance(element, Table):
                        tables.append(element)
                        has_table = True
                    elif isinstance(element, Image):
                        ocr_text = process_image_with_ocr(element.metadata.image_path)
                        if ocr_text.strip():
                            texts.append(ocr_text)
            
            # Check for HTML table tags
            chunk_str = str(chunk).lower()
            if any(tag in chunk_str for tag in ['<table>', '<tr>', '<td>', '|----']):
                if not has_table:
                    tables.append(chunk)
                    continue
            
            # Otherwise, treat as text
            texts.append(chunk)
        
        # Everything else is text
        else:
            # Check if text might represent a table
            if is_likely_table(str(chunk)):
                tables.append(create_table_like(str(chunk)))
            else:
                texts.append(chunk)
    
    # Add tables from pdfplumber
    tables.extend(additional_tables)
    
    logger.info(f"Extraction complete: {len(texts)} text chunks, {len(tables)} tables")
    return tables, texts

if __name__ == '__main__':
    path = 'data/document.pdf'
    tables, texts = tables_text(path)
    logger.info(f"Final count - Tables: {len(tables)}, Text chunks: {len(texts)}")