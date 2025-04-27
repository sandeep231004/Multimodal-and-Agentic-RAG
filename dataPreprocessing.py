import warnings
warnings.filterwarnings("ignore")
import os
import glob
import easyocr
import time
from groq import RateLimitError
from PIL import Image
from typing import Optional
from dotenv import load_dotenv
from transformers import pipeline
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from ExtractionUtils import tables_text
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

def get_chain(api_key : Optional[str] = None):

    prompt_text = """
    You are an assistant tasked with summarizing tables and text.
    Give a concise summary of the table or text.

    Respond only with the summary, no additionnal comment.
    Do not start your message by saying "Here is a summary" or anything like that.
    Just give the summary as it is.

    Table or text chunk: {element}
    """

    prompt = ChatPromptTemplate.from_template(prompt_text) # Converts the prompt_text into a structured prompt template.

    if api_key:

        model = ChatGroq(
            temperature=0.5,
            model="llama-3.1-8b-instant",
            groq_api_key = api_key 
        )

    else:
        
        model = ChatGroq(
            temperature=0.5,
            model="llama-3.1-8b-instant"
        )
    
    chain = {"element": lambda x: x} | prompt | model | StrOutputParser()

    return chain

def get_summaries(tables, texts, api_key: Optional[str] = None):
    """
    Generates table and text summaries used for efficient data retrieval.
    :param tables: list of tables in document
    :param texts: list of text paragraphs in document
    :return: summaries of text and tables
    """
    chain = get_chain(api_key)

    # Print debug information
    print(f"Number of text chunks to process: {len(texts)}")
    print(f"Number of tables to process: {len(tables)}")
    
    # Process text summaries with retries
    text_summary = []
    max_retries = 3
    
    try:
        text_summary = chain.batch(texts, {"max_concurrency": 1})  # Reduced concurrency
    except RateLimitError as e:
        retry_count = 0
        while retry_count < max_retries:
            wait_time = min(2 ** retry_count * 5, 60)  # Exponential backoff, max 60 seconds
            print(f"Rate limit hit. Waiting for {wait_time} seconds before retrying...")
            time.sleep(wait_time)
            try:
                text_summary = chain.batch(texts, {"max_concurrency": 1})
                break
            except RateLimitError:
                retry_count += 1
                if retry_count >= max_retries:
                    raise
    
    # Check if tables exist and process them
    table_summary = []
    if tables and len(tables) > 0:
        # Check if tables have the expected structure
        if hasattr(tables[0], 'metadata') and hasattr(tables[0].metadata, 'text_as_html'):
            html_tables = [i.metadata.text_as_html for i in tables]
            try:
                table_summary = chain.batch(html_tables, {"max_concurrency": 1})
            except RateLimitError as e:
                retry_count = 0
                while retry_count < max_retries:
                    wait_time = min(2 ** retry_count * 5, 60)
                    print(f"Rate limit hit. Waiting for {wait_time} seconds before retrying...")
                    time.sleep(wait_time)
                    try:
                        table_summary = chain.batch(html_tables, {"max_concurrency": 1})
                        break
                    except RateLimitError:
                        retry_count += 1
                        if retry_count >= max_retries:
                            raise
        else:
            print("Tables don't have the expected structure with metadata.text_as_html")
    else:
        print("No tables found in the document")
    
    return text_summary, table_summary
def encode_images(pipe):
    """
    Extracts text from images using OCR and image-to-text models.
    Uses OCR model to extract text and Hugging Face's image-to-text model to generate image captions.
    """
    img_ls = []
    imgs = []
    img_path_list = glob.glob("extracted/figure*jpg") # Finds all image files in the extracted/ directory.

    # Debug: Check if image files exist
    print(f"Number of image files found: {len(img_path_list)}")

    if not img_path_list:
        print("No image files found. Check the path and file pattern.")
        return [], []

    reader = easyocr.Reader(['en'])  # Initializes EasyOCR for English text extraction.

    for i in img_path_list:
        img = Image.open(i).convert('RGB')
        img_ls.append(pipe(img)[0]['generated_text'])

        results = reader.readtext(i)
        extracted_text = ""

        for _,text, _ in results:
            extracted_text += f"{text}"
        imgs.append(extracted_text)
    return imgs, img_ls # Returns OCR-extracted text and AI-generated image captions.

def text_table_img(path, api_key : Optional[str] = None):

    # Debug: Check if file exists
    if not os.path.exists(path):
        print(f"Error: File not found at {path}")
        return [], [], [], [], [], []
    
    # Extract tables and text
    print(f"Extracting tables and text from {path}")
    tables, texts = tables_text(path)
    print(f"Extraction complete. Found {len(tables)} tables and {len(texts)} text chunks")

    # Setup image processing
    device = -1  # Use CPU instead of GPU since CUDA isn't available
    pipe = pipeline("image-to-text", model="Salesforce/blip-image-captioning-large", device=device)
    
    # Process images
    images, images_summary = encode_images(pipe)
    print("Images extracted")

    # Generate summaries
    text_summary, table_summary = get_summaries(tables, texts, api_key)
    print("Text, Table and Image Summaries extracted")

    return tables, texts, images, text_summary, table_summary, images_summary

if __name__ == "__main__" : 

    path = 'data/document.pdf'
    api_key = os.getenv("GROQ_API_KEY")
    tables, texts, images , text_summary, table_summary, image_summary= text_table_img(path, api_key)
    print(f"tables : {len(tables)}")
    print(f"tables : {len(table_summary)}")