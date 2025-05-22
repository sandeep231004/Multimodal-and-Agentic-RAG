import argparse
import os
import sys
import logging
from typing import List, Dict, Any, Optional
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, AIMessage

from RAGUtils import makeVectorStore
from DataExtraction import text_table_img


def load_api_key():
    """Load API key from environment variables."""
    load_dotenv()
    api_key = os.getenv('GROQ_API_KEY')
    if not api_key:
        logging.warning("GROQ_API_KEY not found in environment variables.")
    return api_key


def initialize_retriever(path, api_key: Optional[str] = None):
    """
    Initialize the retriever with document content.
    
    Args:
        path: Path to the PDF file
        api_key: Optional API key for services that need it
        
    Returns:
        Initialized retriever or None if initialization fails
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Error: {path} not found. Please check the file path.")
    
    try:
        logging.info(f"Extracting content from {path}")
        tables, texts, images, text_summary, table_summary, images_summary = text_table_img(path, api_key)
        
        logging.info(f"Creating vector store with {len(texts)} texts, {len(tables)} tables, {len(images)} images")
        return makeVectorStore(texts, tables, images, text_summary, table_summary, images_summary)
    except Exception as e:
        logging.error(f"Error initializing retriever: {str(e)}")
        return None


def initialize_llm(temperature=0.0):
    """
    Initialize the LLM with error handling and fallback options.
    
    Args:
        temperature: Temperature parameter for the LLM
        
    Returns:
        Initialized LLM
    """
    from langchain_groq import ChatGroq
    
    try:
        return ChatGroq(
            model="llama3-8b-8192",
            temperature=temperature,
            max_retries=2,
        )
    except Exception as e:
        logging.warning(f"Error initializing LLM with environment variables: {str(e)}")
        api_key = input("Enter Groq API key: ")
        return ChatGroq(
            model="llama3-8b-8192",
            temperature=temperature,
            max_retries=2,
            groq_api_key=api_key
        )


def parse_docs(docs):
    """
    Extract and organize content from retrieved documents.
    
    Args:
        docs: Retrieved documents
        
    Returns:
        Dictionary with organized document content
    """
    if not docs:
        return {"texts": ["No relevant information found."]}
    
    # Enhanced to handle different document types more explicitly
    result = {"texts": []}
    
    for doc in docs:
        # Handle different document types appropriately
        if hasattr(doc, "page_content"):
            result["texts"].append({"text": doc.page_content, "metadata": doc.metadata})
        else:
            result["texts"].append({"text": str(doc)})
            
    return result


def build_prompt(kwargs):
    """
    Build a prompt for the LLM with retrieved context.
    
    Args:
        kwargs: Dictionary containing question and context
        
    Returns:
        Formatted prompt template
    """
    from langchain_core.prompts import ChatPromptTemplate
    
    docs_by_type = kwargs["context"]
    user_question = kwargs["question"]
    
    # Enhanced to better format the context
    context_pieces = []
    for text_element in docs_by_type["texts"]:
        if isinstance(text_element, dict) and "text" in text_element:
            context_pieces.append(text_element["text"])
        else:
            context_pieces.append(str(text_element))
    
    context_text = "\n\n".join(context_pieces)
    
    system_message = """You are an intelligent assistant that answers questions based solely on the provided context.
    Only use information from the context to answer questions.
    If the context doesn't contain relevant information, simply state that you don't have enough information to answer.
    Do not make up information or use external knowledge."""
    
    prompt_template = f"""Context: {context_text}\n\nQuestion: {user_question}"""
    
    return ChatPromptTemplate.from_messages([
        ("system", system_message),
        HumanMessage(content=[{"type": "text", "text": prompt_template}])
    ])


def create_chain(retriever, llm):
    """
    Create a RAG chain that retrieves documents and answers questions.
    
    Args:
        retriever: Document retriever
        llm: Language model
        
    Returns:
        Runnable chain
    """
    from langchain_core.output_parsers import StrOutputParser
    from langchain_core.runnables import RunnablePassthrough, RunnableLambda
    
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


def create_rephrase_history_chain(llm, retriever):
    """
    Creates a history-aware retriever for better query understanding.
    
    Args:
        llm: Language model
        retriever: Base retriever
        
    Returns:
        History-aware retriever
    """
    from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
    from langchain.chains.history_aware_retriever import create_history_aware_retriever
    
    system_prompt = """Given the conversation history and the latest user question, 
    create a search query that will help retrieve relevant information to answer the user's question.
    The query should be focused on finding specific information from documents that would help answer the question.
    If the question is a follow-up, make sure to include relevant context from the conversation history."""
    
    contextualize_query_prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}")
    ])
    
    history_aware_retriever = create_history_aware_retriever(llm, retriever, contextualize_query_prompt)
    
    return history_aware_retriever


def create_qa_RAG_chain_history(llm, retriever):
    """
    Creates a QA chain that incorporates chat history for conversational RAG.
    
    Args:
        llm: Language model
        retriever: Document retriever
        
    Returns:
        RAG chain with history awareness
    """
    from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
    from langchain.chains import create_retrieval_chain
    from langchain.chains.combine_documents import create_stuff_documents_chain
    
    system_prompt = """You are a helpful assistant that answers questions based on the retrieved documents and conversation history.
    Always ground your answers in the information provided in the retrieved documents.
    If you cannot find the answer in the retrieved documents, politely say so instead of making up information.
    Use the conversation history to provide context for your answers, but prioritize the information in the retrieved documents."""
    
    qa_prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
        ("human", "Retrieved documents: {documents}")
    ])
    
    question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)
    rag_chain = create_retrieval_chain(retriever, question_answer_chain)
    
    return rag_chain


def run_interactive_mode(retriever, llm):
    """
    Run the chatbot in interactive mode with conversation history.
    
    Args:
        retriever: Document retriever
        llm: Language model
    """
    print("\n=== Multi-Modal RAG Chatbot ===")
    print("Type 'exit' or 'quit' to end the conversation.\n")
    
    # Set up history-aware components
    history_aware_retriever = create_rephrase_history_chain(llm, retriever)
    history_chain = create_qa_RAG_chain_history(llm, history_aware_retriever)
    simple_chain = create_chain(retriever, llm)
    
    chat_history = []
    
    while True:
        user_input = input("\nYou: ").strip()
        if user_input.lower() in ["exit", "quit", "q"]:
            print("\nThank you for using the Multi-Modal RAG Chatbot. Goodbye!")
            break
        
        try:
            # First attempt with history chain
            response = history_chain.invoke({
                "input": user_input,
                "chat_history": chat_history
            })
            
            # Fallback to simple chain if history chain fails
            if not response or response.strip() == "":
                print("Using fallback chain...")
                response = simple_chain.invoke(user_input)
                
            print(f"\nBot: {response}")
            
            # Update chat history
            chat_history.append(HumanMessage(content=user_input))
            chat_history.append(AIMessage(content=response))
            
            # Keep history manageable by limiting size
            if len(chat_history) > 10:
                chat_history = chat_history[-10:]
                
        except Exception as e:
            logging.error(f"Error generating response: {str(e)}")
            print("\nBot: I encountered an error processing your request. Please try again.")


def run_predefined_queries(retriever, llm, queries):
    """
    Run a list of predefined queries and return responses.
    
    Args:
        retriever: Document retriever
        llm: Language model
        queries: List of query strings
        
    Returns:
        Dictionary of query-response pairs
    """
    chain = create_chain(retriever, llm)
    responses = {}
    
    for i, query in enumerate(queries, 1):
        print(f"\nQuery {i}/{len(queries)}: {query}")
        try:
            response = chain.invoke(query)
            responses[query] = response
            print(f"Response: {response[:100]}...")  # Preview first 100 chars
        except Exception as e:
            logging.error(f"Error processing query '{query}': {str(e)}")
            responses[query] = f"Error: {str(e)}"
    
    return responses


def main():
    """
    Enhanced main function with command-line argument support and flexible modes.
    """
    # Configure logging
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description='Multi-Modal RAG Chatbot')
    parser.add_argument('--path', type=str, default="data/document.pdf", help='Path to the document')
    parser.add_argument('--query', type=str, help='Single query to process (optional)')
    parser.add_argument('--interactive', action='store_true', help='Run in interactive mode')
    parser.add_argument('--predefined', action='store_true', help='Run predefined queries')
    parser.add_argument('--temp', type=float, default=0.0, help='LLM temperature (0.0-1.0)')
    args = parser.parse_args()
    
    # Load API key and initialize components
    api_key = load_api_key()
    
    try:
        # Initialize retriever
        logging.info(f"Initializing retriever with document: {args.path}")
        retriever = initialize_retriever(args.path, api_key)
        if not retriever:
            logging.error("Failed to initialize retriever. Exiting.")
            return
        
        # Initialize LLM
        logging.info("Initializing language model...")
        llm = initialize_llm(temperature=args.temp)
        
        # Determine mode of operation
        if args.query:
            # Single query mode
            logging.info(f"Processing single query: {args.query}")
            chain = create_chain(retriever, llm)
            response = chain.invoke(args.query)
            print(f"\nQuery: {args.query}")
            print(f"Response: {response}")
        
        elif args.interactive:
            # Interactive mode
            logging.info("Starting interactive mode")
            run_interactive_mode(retriever, llm)
        
        elif args.predefined or not (args.query or args.interactive):
            # Default to predefined queries if no mode specified
            logging.info("Running predefined queries")
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
            responses = run_predefined_queries(retriever, llm, queries)
            
            # Save responses to file
            with open("rag_responses.txt", "w") as f:
                for query, response in responses.items():
                    f.write(f"Q: {query}\n\n")
                    f.write(f"A: {response}\n\n")
                    f.write("-" * 80 + "\n\n")
            
            logging.info("Responses saved to rag_responses.txt")
    
    except Exception as e:
        logging.error(f"Error in main function: {str(e)}")
        print(f"An error occurred: {str(e)}")


if __name__ == "__main__":
    main()