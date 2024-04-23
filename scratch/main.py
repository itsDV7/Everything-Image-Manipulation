from tempfile import NamedTemporaryFile
from dotenv import load_dotenv
import json

import streamlit as st
from langchain.agents import initialize_agent
from langchain_community.chat_models import AzureChatOpenAI
from langchain.chains.conversation.memory import ConversationBufferWindowMemory

from tools import ImageCaptioningTool, ObjectDetectionTool, ImageQuestionAnswerTool

tools = [ImageCaptioningTool(), ObjectDetectionTool(), ImageQuestionAnswerTool()]

conversational_memory = ConversationBufferWindowMemory(
    memory_key = 'chat_history',
    k = 5,
    return_messages = True
)

load_dotenv(dotenv_path='.env', override=True)

llm = AzureChatOpenAI(
    temperature = 0,
    model = "gpt-4-hackathon",
)

agent = initialize_agent(
    agent="chat-conversational-react-description",
    tools=tools,
    llm=llm,
    max_iterations=5,
    verbose=True,
    memory=conversational_memory,
    early_stopping_method='generate'
)

image_path = "dogs_playing.jpg"

user_question = input("Q: ")

response = agent.run('{}, The image path is: {}'.format(user_question, image_path))

print(response)

# st.title('Ask a question to an image')
#
# st.header("Please upload an image")
#
# file = st.file_uploader("", type=["jpeg", "jpg", "png"])
#
# if file:
#     st.image(file, use_column_width=True)
#
#     user_question = st.text_input('Ask a question about your image:')
#
#     with NamedTemporaryFile(dir='.') as f:
#         f.write(file.getbuffer())
#         image_path = f.name
#
#         # write agent response
#         if user_question and user_question != "":
#             with st.spinner(text="In progress..."):
#                 response = agent.run('{}, this is the image path: {}'.format(user_question, image_path))
#                 st.write(response)
