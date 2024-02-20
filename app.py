
from  langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.tools.retriever import create_retriever_tool
from langchain.agents import OpenAIFunctionsAgent, AgentExecutor, create_openai_tools_agent

from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage,HumanMessage, AIMessage
from langchain.memory import ConversationBufferMemory
import streamlit as st
from langchain_core.prompts import MessagesPlaceholder
from langchain_community.callbacks import StreamlitCallbackHandler
from langchain_community.document_loaders import AzureBlobStorageContainerLoader
import os


st.set_page_config(
    page_title="ServiceNow Knowledge Base Chatbot",
    page_icon="🦜",
    layout="wide",
    initial_sidebar_state="collapsed",
)

"# Chat🦜🔗"
from dotenv import load_dotenv
CONN_STR = os.getenv("BLOB_CONN_STRING")


load_dotenv()

def search_knowledge_base():
    """Searches Servicenow Knowledge Base (KB) for KB articles and returns answers based on the issue and resolution notes i the article."""   
    #loader = PyPDFDirectoryLoader(  path="./files")
    CONN_STR="DefaultEndpointsProtocol=https;AccountName=snowkbsujit;AccountKey=Mjg8YiKmuZwItLE7fcOZ/J8qEJUOSaIxdMwwzSjId3numC2vHSB7ml0eg/IFc968B9sCYbZJTaSF+ASt7ZAXgw==;EndpointSuffix=core.windows.net"
    loader = AzureBlobStorageContainerLoader(
                    conn_str=CONN_STR, 
                    container="servicenowkb" 
                    #, prefix="<prefix>"
                    ) 
   
    documents = loader.load()
    docs = loader.load()
    embeddings = OpenAIEmbeddings()
   
    vectorstore = FAISS.from_documents(documents, embeddings)
    return vectorstore.as_retriever(search_kwargs={"k": 2})

kbsearch_tool = create_retriever_tool(
    search_knowledge_base(),
    "search_SNow_Knowledge_Base",
    "Searches in ServiceNow Knowledge Base or KB and returns the summary including issue and resolution for that or similar issues",
)

tools = [kbsearch_tool]

#setup llm
llm= ChatOpenAI(model='gpt-4', temperature=0, streaming=True )

#Set up System Prompt template
message= SystemMessage(
    content= (
    """
        Based on the user query, use the search_knowledge_base  tool to  search ServiceNow or Snow Knowledgebase or KB when the user asks help on an issue and returns a summary of issue and resolution from same or similar issue
    """
    )
    )

#Create Prompt
prompt=  OpenAIFunctionsAgent.create_prompt (
    system_message=message,
    extra_prompt_messages=[MessagesPlaceholder(variable_name="history")],

)

#setup Agent
agent=create_openai_tools_agent(llm=llm, prompt=prompt, tools=tools) 

#setup Agent executor
agent_executor= AgentExecutor(
    agent=agent,
    tools=tools,
    verbose=True,
    return_intermediate_steps=True


)

#setup Memory
memory=ConversationBufferMemory(llm=llm,memoty_key="history", return_messages=True,output_key='output')


starter_message="I can provide resolutions from ServiceNow Tickets or Knowledge Base"
st.session_state["messages"] =[]
if "messages" not in st.session_state :
    st.session_state.messages=AIMessage(content=starter_message)
 
#Write messages from session to chat in UI
for msg in st.session_state.messages:
    if (isinstance(msg,AIMessage)):
        st.chat_message("assistant").write(msg.content)
    if (isinstance(msg, HumanMessage)):
        st.chat_message("user").write(msg.content)

    memory.chat_memory.add_message(msg)

if prompt := st.chat_input(placeholder=starter_message):
    #first write back prompt
    st.chat_message("user").write(prompt)
    with (st.chat_message("assistant")):
        #Register callback
        st_callback=StreamlitCallbackHandler(st.container())
        #Now actually run agent executor
        response=agent_executor (
                {"input": prompt, "history":st.session_state.messages},
                callbacks=[st_callback],
                include_run_info=True
        )
        #Write response to field
        st.write(response["output"])
        #store response in session state
        st.session_state.messages.append(AIMessage(content=response["output"]))
        #store input and response in context memory
        memory.save_context({"input":prompt},response)

        #Save conversation in state session
        #st.session_state["messages"]=memory.buffer
        #run_id=response["__run"].run_id