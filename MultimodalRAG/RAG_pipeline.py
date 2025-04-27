import warnings
warnings.filterwarnings("ignore")

import os
from typing import Optional
from dotenv import load_dotenv
from langchain_groq import ChatGroq

from RAGUtils import makeVectorStore
from DataExtraction import text_table_img

from langchain_core.messages import HumanMessage
from langchain.chains import create_retrieval_chain
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains.history_aware_retriever import create_history_aware_retriever


load_dotenv()

def load_api_key():
    load_dotenv()
    return os.getenv('GROQ_API_KEY')

def initialize_retriever(path, api_key : Optional[str] = None):
    """
    Takes path (PDF file) and api_key (optional).
    Calls text_table_img() to extract text, tables, and images.
    Passes extracted data to makeVectoreStore() to create vector embeddings.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Error: {path} not found. Please check the file path.")
    try:
        tables, texts, images, text_summary, table_summary, images_summary = text_table_img(path, api_key)
        return makeVectorStore(tables, texts, images, table_summary, text_summary, images_summary)
    except Exception as e:
        print(f"Error initializing retriever: {e}")
        return None

def initialize_llm(temperature = 0.0):

    try:
        return ChatGroq(
            model = "llama3-8b-8192",
            temperature = temperature,
            max_retries = 2, 
        )
    except:
        return ChatGroq(
            model="llama3-8b-8192",
            temperature= temperature,
            max_retries=2,
            groq_api_key= input("enter groq api key")
        )

def parse_docs(docs): # Takes retrieved documents and returns them as a dictionary.
    """
    Extracts text from retrieved documents
    Returns structured output
    """
    if not docs:
        return {"texts": ["No relevant information found."]}
    return {"texts": docs}

def build_prompt(kwargs): #  Build the Prompt for LLM
    docs_by_type = kwargs["context"]
    user_question = kwargs["question"] # Retrieves the question (user_question) and context (docs_by_type).
    context_text = "".join(text_element.text for text_element in docs_by_type["texts"]) #   Joins all retrieved text into a single context block.
    prompt_template = f"Answer the question based only on the following context, which can include text and tables given below.\nContext: {context_text}\nQuestion: {user_question}" # Formats a structured prompt to prevent AI from hallucinating.
    return ChatPromptTemplate.from_messages([
        HumanMessage(content=[{"type": "text", "text": prompt_template}])  # Returns a properly formatted ChatPromptTemplate.
        ])

def create_chain(retriever, llm):
    """
    Chain Steps:

    Retrieve relevant documents (retriever → parse_docs).
    Pass the user question as-is (RunnablePassthrough).
    Format a structured prompt (RunnableLambda(build_prompt)).
    Send to the LLM (llm).
    Extract structured output (StrOutputParser).
    """

    chain = (
        {
            "context": retriever | RunnableLambda(parse_docs),
            "question": RunnablePassthrough(),
        }
        | RunnableLambda(build_prompt)
        | llm
        | StrOutputParser()
    )
    return chain

def create_rephrase_history_chain(llm_pipeline, retriever, system_prompt):  # Enhances Retrieval by rephrasing user queries using chat history.

    """
    Creates a history aware retriever. It is needed to generate comprehensive summary of the chat history while also considering the user 
    queries. 
    """

    contextualize_query_prompt = ChatPromptTemplate.from_messages([("system", system_prompt),
                                                               MessagesPlaceholder("chat_history"),
                                                               ("human", "{input}")])

    history_aware_retriever = create_history_aware_retriever(llm_pipeline, retriever, contextualize_query_prompt)

    return history_aware_retriever

def create_qa_RAG_chain_history(llm_pipeline, retriever, system_prompt): # Stores past interactions for better responses in a conversational RAG pipeline.
    """
    Performs RAG storing the chat history for future queries needed in conversational RAG.
    """
    qa_prompt = ChatPromptTemplate.from_messages([("system", system_prompt),
                                                  MessagesPlaceholder("chat_history"),
                                                  ("human", "{input}")])

    question_answer_chain = create_stuff_documents_chain(llm_pipeline, qa_prompt)
    rag_chain = create_retrieval_chain(retriever, question_answer_chain)

    return rag_chain

def main():
    api_key = load_api_key()
    path = "data/document.pdf"
    retriever = initialize_retriever(path, api_key)
    llm = initialize_llm()
    chain = create_chain(retriever, llm)
    queries = [
        "What is the e-Tender notice number and the purpose of the tender mentioned in the document?",
        "What are the eligibility criteria for bidders to participate in this tender?",
        "What are the deadlines for submitting the online bids and physically submitting the tender fee and EMD?",
        "What is the role of Annexure-G in determining supplier eligibility, and how is local content defined?",
        "What is the payment structure for the successful supplier as mentioned in the document?",
        "What are the warranty obligations for suppliers as outlined in Annexure-F?",
        "How is the technical bid evaluated, and what criteria are used for shortlisting bidders?",
        "What penalties are imposed for delays in delivery or non-performance by the supplier?",
        "What does Annexure-E specify about the blacklisting or debarment of suppliers?",
        "What does Annexure-D require from suppliers regarding manufacturer authorization?"
    ]
    responses = {}