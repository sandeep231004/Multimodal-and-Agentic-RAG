import warnings
warnings.filterwarnings("ignore")
import tqdm
import os
from dotenv import load_dotenv
from unstructured.partition.pdf import partition_pdf # function from Unstructured that extracts structured text from PDFs.
from unstructured.documents.elements import Table
load_dotenv()

def partition(file_path): 
    """
    Creates chunks of given document used for creating embeddings for RAG agent.
    :param file_path: path of file/doc
    :return: chunks
    """

    chunks = partition_pdf(  # extracts structured content from the PDF.
        filename=file_path, 
        infer_table_structure=True, # Detects and maintains table structure.
        strategy='hi_res', # Uses high-resolution OCR for accurate text extraction.

        extract_image_block_types=["Image"],
        extract_image_block_output_dir="extracted", # Saves them in the "extracted" directory.
        extract_image_block_to_payload=False, # Doesn’t return images in the output payload.  

        chunking_strategy="by_title",           # Ensures proper chunking based on title. Uses titles/headings to logically split the text.
        max_characters=10000,
        combine_text_under_n_chars=2000,
        new_after_n_chars=6000,

        extract_tables=True,  # Explicitly enable table extraction
        include_page_breaks=True,  # May help with context
    )
    return chunks # Returns the structured PDF chunks.

def tables_text(path):
    """
    Extracts Tabular and textual information from the document
    :param file_path: path of file/doc
    :return: tables and texts
    """
    print(f"Processing document: {path}")
    if not os.path.exists(path):
        print(f"Error: File not found at {path}")
        return [], []
        
    print("Chunking started")
    chunks = list(partition(file_path=path))
    print(f"Chunking Ended - {len(chunks)} chunks found")
    
    # Print all chunk types to debug
    chunk_types = [(type(chunk).__name__, chunk.__class__.__module__) for chunk in chunks]
    unique_types = set(chunk_types)
    print(f"Found chunk types: {unique_types}")

    tables = []
    texts = []

    # Process all chunks
    for chunk in chunks:
        chunk_type = type(chunk).__name__
        
        # Try to identify tables within CompositeElements
        if chunk_type == 'CompositeElement':
            # Check if it has children that might be tables
            if hasattr(chunk, 'elements'):
                for element in chunk.elements:
                    if isinstance(element, Table) or 'table' in str(type(element)).lower():
                        tables.append(element)
                        print(f"Found table inside CompositeElement: {type(element).__name__}")
            
            # Check if it contains HTML table tags
            chunk_str = str(chunk).lower()
            if '<table>' in chunk_str or '<tr>' in chunk_str or '<td>' in chunk_str:
                tables.append(chunk)
                print(f"Found table by HTML tags in CompositeElement")
                continue
            
            # Otherwise, treat it as text
            texts.append(chunk)
        else:
            # Direct table check for non-composite elements
            if isinstance(chunk, Table) or 'table' in str(type(chunk)).lower():
                tables.append(chunk)
                print(f"Found direct table: {chunk_type}")
            else:
                texts.append(chunk)
    
    # If no tables found, try alternative detection methods
    if not tables:
        print("\nNo tables found with primary methods. Trying alternative detection...")
        for chunk in chunks:
            # Look for table-like content in the text
            chunk_str = str(chunk).lower()
            
            # Check for common table indicators
            if ('|' in chunk_str and ('-' * 3) in chunk_str) or \
               (chunk_str.count('\t') > 5) or \
               any(chunk_str.count(sep) > 3 for sep in [',', '\t', '|']):
                print(f"Found potential table by text pattern analysis")
                
                # Create a table-like object
                class TableLike:
                    def __init__(self, content):
                        self.metadata = type('obj', (object,), {'text_as_html': f"<table><tr><td>{content}</td></tr></table>"})
                
                tables.append(TableLike(str(chunk)))
    
    print(f"Text and Tables extracted - {len(texts)} text chunks and {len(tables)} tables")
    return tables, texts

if __name__ == '__main__':
    path = 'data/document.pdf'
    tables, texts = tables_text(path)
    print(f"Final count - Tables: {len(tables)}, Text chunks: {len(texts)}")
    