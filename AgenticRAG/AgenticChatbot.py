from pathlib import Path
import time
import logging
from AgenticRAG.AgentUtils import get_doc_tools, initialize_settings

from llama_index.core.agent import AgentRunner
from llama_index.core.objects import ObjectIndex
from llama_index.core import VectorStoreIndex, Settings
from llama_index.core.agent import FunctionCallingAgentWorker

# Set up logging
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def ChatBot():
    """
    Creates an Agent instance with specified context and tools.
    This function initializes the chat agent for inference.
    """
    files = ["data/document.pdf"]
    
    initialize_settings(llm_provider="groq")
    
    llm = Settings.llm
    logger.info(f"Using LLM of type: {type(llm).__name__}")

    all_tools = []
    for file in files:
        logger.info(f"Getting tools for file: {file}")
        vector_tool, summary_tool = get_doc_tools(file, Path(file).stem)
        all_tools.extend([vector_tool, summary_tool])
    
    logger.info(f"Created tools for {len(files)} files")

    obj_index = ObjectIndex.from_objects(
        all_tools,
        index_cls=VectorStoreIndex,
    )
    obj_retriever = obj_index.as_retriever(similarity_top_k=1)
    
    logger.info("Creating agent worker...")
    
    agent_worker = FunctionCallingAgentWorker.from_tools(
        tool_retriever=obj_retriever,
        llm=llm,
        system_prompt=
        """
        You are an intelligent assistant specializing in retrieving and answering queries based on provided documents. 
        Use the retrieved context to generate accurate and concise answers. 
        If the information is not in the document, explicitly state, "The document does not provide this information." 
        Ensure all answers are derived only from the retrieved content and are highly relevant to the query.
        
        Always follow this process:
        1. Carefully review all retrieved document content
        2. Extract only relevant information that directly answers the query
        3. Format your response clearly and concisely
        4. If tool results are incomplete or unclear, acknowledge this limitation
        
        Example Queries:

        **Query** : What is the e-Tender notice number and its purpose?
        **Answer** : The e-Tender notice number is NITJ/DRC/PUR/TT/36/2024. Its purpose is the fabrication of a machine for continuous production of textile waste-based composite materials for the Department of Textile Technology.
        
        **Query** : How is the technical bid evaluated?
        **Answer** : The technical bid is evaluated to ensure compliance with essential eligibility criteria, submission of EMD and Tender Fee, completion of required documents, adherence to equipment specifications, and validity of service and warranty policies. Only technically qualified bids proceed to the financial evaluation stage.
        """,
        verbose=True
    )
    
    logger.info("Creating agent runner...")
    return AgentRunner(agent_worker)

def chat(query: str) -> str:
    """Generates responses to input queries using the initialized chatbot agent."""
    try:
        logger.info("Initializing ChatBot...")
        agent = ChatBot()
        logger.info(f"Querying agent with: {query}")
        resp = agent.query(query)
        
        if resp and resp.response:
            response_text = resp.response
            if len(response_text.strip()) < 5 or response_text.strip().upper() == "ISU":
                return "The system couldn't retrieve a proper response from the document. Please try rephrasing your query."
            return response_text
        else:
            return "No response was generated. Please try again with a different query."
            
    except Exception as e:
        logger.error(f"Error in chat: {str(e)}", exc_info=True)
        return f"An error occurred: {str(e)}"

def main():
    """
    Main function to run a series of predefined queries through the chatbot.
    Includes delays to manage API rate limits.
    """
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
    
    for idx, query in enumerate(queries):
        logger.info(f"Processing query {idx+1}/{len(queries)}: {query}")
        try:
            response = chat(query)
            print(f"\nQ) {query}")
            print(f"A) {response}")
            print("-" * 80)
            
        except Exception as e:
            logger.error(f"Error processing query: {str(e)}", exc_info=True)
            print(f"Failed to process query: {query}")
            print(f"Error: {str(e)}")
        
        if idx < len(queries) - 1:
            delay = 30
            logger.info(f"Pausing for {delay} seconds between queries to avoid rate limiting...")
            time.sleep(delay)

if __name__ == "__main__":
    main()

    