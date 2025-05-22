import uuid
import os
import sys
import logging
from langchain_chroma import Chroma
from langchain.storage import InMemoryStore
from langchain.schema.document import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.retrievers.multi_vector import MultiVectorRetriever

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Add parent directory to path to enable imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

def create_retriever(id_key="doc_id"):
    """
    Create a MultiVectorRetriever with Chroma vector store and in-memory document store.
    Args:
        id_key: Key to use for document IDs
    Returns:
        Tuple of (id_key, retriever)
    """
    vectorstore = Chroma(
        collection_name="multi_modal_rag",
        embedding_function=HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    )

    # Storage Layer for parent docs
    store = InMemoryStore()

    retriever = MultiVectorRetriever(
        vectorstore=vectorstore,
        docstore=store,
        id_key=id_key,
    )
    return id_key, retriever

def makeVectorStore(texts, tables, images, text_summaries, table_summaries, image_summaries):
    """
    Create vector store with summaries as vectors and original content in document store.
    Ensures all available content is stored even when there are length mismatches.
    
    Args:
        texts: List of text chunks
        tables: List of table elements
        images: List of image elements
        text_summaries: List of summaries for text chunks
        table_summaries: List of summaries for tables
        image_summaries: List of summaries for images
    
    Returns:
        MultiVectorRetriever instance
    """
    id_key, retriever = create_retriever()

    # ----- HANDLE TEXT SUMMARIES -----
    logger.info(f"Processing {len(texts)} texts with {len(text_summaries)} summaries")
    
    # Make sure every text has a summary by adding placeholders if needed
    if len(texts) > len(text_summaries):
        logger.warning(f"Adding {len(texts) - len(text_summaries)} placeholder summaries for texts")
        text_summaries.extend(["No summary available"] * (len(texts) - len(text_summaries)))
    
    # Truncate summaries if there are more summaries than texts
    if len(text_summaries) > len(texts):
        logger.warning(f"Truncating {len(text_summaries) - len(texts)} excess text summaries")
        text_summaries = text_summaries[:len(texts)]
    
    # Generate IDs for all texts
    text_doc_ids = [str(uuid.uuid4()) for _ in texts]
    
    # Create documents for non-empty summaries
    valid_text_indices = [i for i, summary in enumerate(text_summaries) if summary and len(str(summary).strip()) > 0]
    if len(valid_text_indices) < len(text_summaries):
        logger.warning(f"Found {len(text_summaries) - len(valid_text_indices)} empty text summaries")
    
    if valid_text_indices:
        text_documents = [
            Document(page_content=str(text_summaries[i]), metadata={id_key: text_doc_ids[i]}) 
            for i in valid_text_indices
        ]
        
        # Store all texts that have valid summaries
        texts_to_store = [(text_doc_ids[i], texts[i]) for i in valid_text_indices]
        
        try:
            retriever.vectorstore.add_documents(text_documents)
            retriever.docstore.mset(texts_to_store)
            logger.info(f"Successfully added {len(text_documents)} text documents")
        except Exception as e:
            logger.error(f"Error adding text documents: {str(e)}")

    # ----- HANDLE TABLE SUMMARIES -----
    logger.info(f"Processing {len(tables)} tables with {len(table_summaries)} summaries")
    
    # Make sure every table has a summary by adding placeholders if needed
    if len(tables) > len(table_summaries):
        logger.warning(f"Adding {len(tables) - len(table_summaries)} placeholder summaries for tables")
        table_summaries.extend(["No table summary available"] * (len(tables) - len(table_summaries)))
    
    # Truncate summaries if there are more summaries than tables
    if len(table_summaries) > len(tables):
        logger.warning(f"Truncating {len(table_summaries) - len(tables)} excess table summaries")
        table_summaries = table_summaries[:len(tables)]
    
    # Generate IDs for all tables
    table_doc_ids = [str(uuid.uuid4()) for _ in tables]
    
    # Create documents for non-empty summaries
    valid_table_indices = [i for i, summary in enumerate(table_summaries) if summary and len(str(summary).strip()) > 0]
    if len(valid_table_indices) < len(table_summaries):
        logger.warning(f"Found {len(table_summaries) - len(valid_table_indices)} empty table summaries")
    
    if valid_table_indices:
        table_documents = [
            Document(page_content=str(table_summaries[i]), metadata={id_key: table_doc_ids[i]}) 
            for i in valid_table_indices
        ]
        
        # Store all tables that have valid summaries
        tables_to_store = [(table_doc_ids[i], tables[i]) for i in valid_table_indices]
        
        try:
            retriever.vectorstore.add_documents(table_documents)
            retriever.docstore.mset(tables_to_store)
            logger.info(f"Successfully added {len(table_documents)} table documents")
        except Exception as e:
            logger.error(f"Error adding table documents: {str(e)}")
    
    # ----- HANDLE IMAGE SUMMARIES -----
    logger.info(f"Processing {len(images)} images with {len(image_summaries)} summaries")
    
    # Make sure every image has a summary by adding placeholders if needed
    if len(images) > len(image_summaries):
        logger.warning(f"Adding {len(images) - len(image_summaries)} placeholder summaries for images")
        image_summaries.extend(["No image description available"] * (len(images) - len(image_summaries)))
    
    # Truncate summaries if there are more summaries than images
    if len(image_summaries) > len(images):
        logger.warning(f"Truncating {len(image_summaries) - len(images)} excess image summaries")
        image_summaries = image_summaries[:len(images)]
    
    # Generate IDs for all images
    image_doc_ids = [str(uuid.uuid4()) for _ in images]
    
    # Create documents for non-empty summaries
    valid_image_indices = [i for i, summary in enumerate(image_summaries) if summary and len(str(summary).strip()) > 0]
    if len(valid_image_indices) < len(image_summaries):
        logger.warning(f"Found {len(image_summaries) - len(valid_image_indices)} empty image summaries")
    
    if valid_image_indices:
        image_documents = [
            Document(page_content=str(image_summaries[i]), metadata={id_key: image_doc_ids[i]}) 
            for i in valid_image_indices
        ]
        
        # Store all images that have valid summaries
        images_to_store = [(image_doc_ids[i], images[i]) for i in valid_image_indices]
        
        try:
            retriever.vectorstore.add_documents(image_documents)
            retriever.docstore.mset(images_to_store)
            logger.info(f"Successfully added {len(image_documents)} image documents")
        except Exception as e:
            logger.error(f"Error adding image documents: {str(e)}")
    
    return retriever

if __name__ == '__main__':
    path = 'data/document.pdf'
    
    # Import the extraction function
    from DataExtraction.dataPreprocessing import text_table_img
    
    # Extract content from document
    tables, texts, images, text_summaries, table_summaries, image_summaries = text_table_img(path)
    
    # Fix: correct parameter order in the function call
    retriever = makeVectorStore(texts, tables, images, text_summaries, table_summaries, image_summaries)
    
    # Test the retriever
    docs = retriever.invoke("what is the document about?")
    logger.info(f"Retrieved {len(docs)} documents")
    for i, doc in enumerate(docs):
        logger.info(f"Document {i+1} type: {type(doc)}")
        content = getattr(doc, "page_content", str(doc))
        logger.info(f"Document {i+1}: {content[:100]}...")
