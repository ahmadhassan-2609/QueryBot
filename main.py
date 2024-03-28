# Import necessary libraries
import os  
import streamlit as st
from dotenv import load_dotenv 
from langchain_community.vectorstores import FAISS
from langchain.document_loaders.csv_loader import CSVLoader
from langchain.chains import RetrievalQA
from langchain.memory import ConversationBufferMemory
from langchain_community.chat_models import ChatOpenAI
from langchain.chains import LLMChain
from langchain_openai import OpenAIEmbeddings
from langchain.agents import AgentExecutor, Tool, ZeroShotAgent

# Load environment variables from .env file
load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")

# Set Streamlit page configuration
st.set_page_config(page_title='🤖Querybot', layout='wide')

# Set up the Streamlit app layout
st.title("QueryBot🤖")
st.subheader("Powered by LangChain 🦜🔗+ OpenAI + Streamlit")

# Define function to get user input
def get_text():
    """
    Get the user input text.

    Returns:
        (str): The text entered by the user
    """
    input_text = st.text_input("You: ",
                               placeholder="How may I help you?", 
                               label_visibility='hidden')
    
    return input_text

# Create an OpenAI instance
llm = ChatOpenAI(temperature=0,
                 openai_api_key=openai_api_key,
                 verbose=False)

# Create OpenAIEmbeddings
embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)

# Load data and create a vector database for retrieval
loader = CSVLoader(file_path='codebasics_faqs.csv', source_column="prompt")
data = loader.load()

# Create a FAISS instance for vector database from 'data'
vectordb = FAISS.from_documents(documents=data,
                                embedding=embeddings)

# Save vector database locally
vectordb_file_path = "faiss_index"
vectordb.save_local(vectordb_file_path)

# Load the vector database from the local folder
vectordb = FAISS.load_local(vectordb_file_path, embeddings, allow_dangerous_deserialization=True)

# Create a retriever for querying the vector database
retriever = vectordb.as_retriever(score_threshold=0.7)

qa = RetrievalQA.from_chain_type(
                llm=llm,
                chain_type="stuff",   # small data size
                retriever=retriever,
            )

# Set up the conversational agent
tools = [
    Tool(
        name="CodeBasics QA System",
        func=qa.run,
        description="Useful for when you need to answer questions about the aspects asked. Input may be a partial or fully formed question.",
    )
]
prefix = """You are an AI assistant designed to engage in conversations with humans. Your goal is to provide helpful and informative responses based on the information provided to you. You should consider both the context and memory.
            Remember to avoid generating responses that are not grounded in the available information. Hallucinating information or providing inaccurate responses is not acceptable. Your responses should be coherent, relevant, and based on the context provided.
            Please engage in the conversation naturally, maintaining a helpful and respectful tone throughout the interaction. Your primary objective is to assist the human and provide valuable insights or assistance whenever possible. Answer in a detailed manner.
            You have access to a single tool:"""
suffix = """Begin!

{chat_history}
Question: {input}
{agent_scratchpad}"""

prompt = ZeroShotAgent.create_prompt(
    tools,
    prefix=prefix,
    suffix=suffix,
    input_variables=["input", "chat_history", "agent_scratchpad"],
)

if "memory" not in st.session_state:
    st.session_state.memory = ConversationBufferMemory(
        memory_key="chat_history"
    )

# Initialize session states to display chat history
if "generated" not in st.session_state:
    st.session_state["generated"] = []
if "past" not in st.session_state:
    st.session_state["past"] = []

llm_chain = LLMChain(
    llm=ChatOpenAI(temperature=0,
                   openai_api_key=openai_api_key, 
                   model_name="gpt-3.5-turbo"
                   ),
    prompt=prompt,
)
agent = ZeroShotAgent(llm_chain=llm_chain, tools=tools, verbose=True)
agent_chain = AgentExecutor.from_agent_and_tools(
    agent=agent, 
    tools=tools, 
    verbose=True, 
    memory=st.session_state.memory,
    handle_parsing_errors=True
)

# Allow the user to enter a query and generate a response
query = get_text()

if query:
    with st.spinner(
        "Generating Answer to your Query : `{}` ".format(query)
    ):
        res = agent_chain.run(query)
        st.success(res, icon="🤖") 
        st.session_state.past.append(query)  
        st.session_state.generated.append(res) 

# Allow to download as well
download_str = []
# Display the conversation history using an expander, and allow the user to download it
if st.session_state['generated']:
    with st.expander("Conversation History", expanded=False):
        for i in range(len(st.session_state['generated'])-1, -1, -1):
            st.info(st.session_state["past"][i],icon="👤")
            st.success(st.session_state["generated"][i], icon="🤖")
            download_str.append(st.session_state["past"][i])
            download_str.append(st.session_state["generated"][i])
        
        # Can throw error - requires fix
        download_str = '\n'.join(download_str)
        if download_str:
            st.download_button('Download', download_str, file_name='conversation.txt', mime='text/plain')

# # Allow the user to view the conversation history and other information stored in the agent's memory
# with st.sidebar.expander("History/Memory", expanded=False):
#     st.session_state.memory
