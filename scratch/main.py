import streamlit as st
from tempfile import NamedTemporaryFile

import os

os.environ['AZURE_OPENAI_API_KEY'] = 'dc528eaf83724782914e171f3bbdaeda'
os.environ['AZURE_OPENAI_ENDPOINT'] = 'https://uiuc-chat-canada-east.openai.azure.com/'
os.environ['AZURE_OPENAI_API_VERSION'] = '2023-07-01-preview'
os.environ["AZURE_OPENAI_CHAT_DEPLOYMENT_NAME"] = "gpt-4-hackathon"

from langchain_core.messages import HumanMessage
from langchain_openai import AzureChatOpenAI

model = AzureChatOpenAI(
    openai_api_version=os.environ["AZURE_OPENAI_API_VERSION"],
    azure_deployment=os.environ["AZURE_OPENAI_CHAT_DEPLOYMENT_NAME"],
)

# message = HumanMessage(
#     content="I gave you my photo, crop me out of this photo and save it on my device. Image path is: human.jpg"
# )

# print(model.invoke([message]))

from langchain import hub
from langchain.agents import AgentExecutor, create_react_agent
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_openai import OpenAI

os.environ['TAVILY_API_KEY'] = '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'

from tools import ImageCaptioningTool, ObjectDetectionTool, ImageQuestionAnswerTool, HumanImageSegmentationTool

tools = [ImageCaptioningTool(), ObjectDetectionTool(), ImageQuestionAnswerTool(), HumanImageSegmentationTool(), TavilySearchResults(max_results=1)]

# tools = [TavilySearchResults(max_results=1)]

# Get the prompt to use - you can modify this!
prompt = hub.pull("hwchase17/react")

# Choose the LLM to use
llm = model

# Construct the ReAct agent
agent = create_react_agent(llm, tools, prompt)

# Create an agent executor by passing in the agent and tools
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

from langchain.memory import ChatMessageHistory
memory = ChatMessageHistory(session_id='curr_session')

from langchain_core.runnables.history import RunnableWithMessageHistory
memory_agent = RunnableWithMessageHistory(
    agent_executor,
    lambda session_id: memory,
    input_messages_key = 'input',
    history_messages_key = 'chat_history',
)

st.title("Everything Images!")
st.header("Please upload an image.")
file = st.file_uploader("", type=['jpeg', 'jpg', 'png'])

if file:
    ext = file.name.split('.')[-1]
    st.image(file, use_column_width=True)
    user_ask = st.text_input('What do you want to do today?')

    with NamedTemporaryFile(dir=".", suffix=f'.{ext}') as f:
        f.write(file.getbuffer())
        image_path = f.name

        if user_ask and user_ask.strip() != "":
            message = HumanMessage(content=user_ask)
            with st.spinner(text="Your wish is my command. Literally..."):
                response = memory_agent.invoke({
                    'input': f'{message}' + f'Image path is: {image_path}'
                },
                config={
                    'configurable': {
                        'session_id': '<foo>'
                    }
                })

                st.write(response['output'])

# print(agent_executor.invoke({"input": f'{message}'}))
# print(memory_agent.invoke({"input": f'{message}'}, config={"configurable": {"session_id": "<foo>"}}))
