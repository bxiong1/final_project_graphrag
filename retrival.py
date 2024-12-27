from langchain.graphs import Neo4jGraph
from langchain.chains import GraphCypherQAChain
from langchain_openai import ChatOpenAI, OpenAI
from langchain_community.chat_models import ChatMlflow

def basic_retrieval(question):
    # Define Neo4j connection details
    host = "neo4j+s://6bb08534.databases.neo4j.io"  # Replace with your Neo4j host
    username = "neo4j"              # Replace with your Neo4j username
    password = "LzdFRD0j0fqPVk2tOsfCmN037fjD18P9_-550ULIi6c"      # Replace with your Neo4j password
    openai_key = "<YOUR-API>"
    # Initialize the Neo4jGraph object
    graph = Neo4jGraph(
        url=host,
        username=username,
        password=password
    )
    
    
    
    # Initialize the LLM (e.g., OpenAI GPT-4)
    llm = ChatOpenAI(temperature=0, model="gpt-4", api_key=openai_key)
    
    # Create a GraphCypherQAChain
    
    qa_chain = GraphCypherQAChain.from_llm(
        llm,
        graph=graph,
        verbose=False,
        validate_cypher= True,
    )
    # Ask a question about the graph
    query = question
    response = qa_chain.invoke(query)
    return response