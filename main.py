import os
from langchain_community.llms import OpenAI
import openai
import gradio as gr
import sys
import api_secret
key = api_secret.API_KEY
openai.api_key = key
os.environ["OPENAI_API_KEY"] = key
from llama_index.core.readers.file.base import SimpleDirectoryReader
from llama_index.core.indices.list.base import GPTListIndex
from llama_index.indices.vector_store import GPTVectorStoreIndex
from llama_index.llms import LLMPredictor
from llama_index.prompts import PromptHelper
from llama_index.indices.loading import load_index_from_storage

from llama_index.core.indices.list.base import GPTListIndex
def construct_index(directory_path):
    max_input_size = 4096
    num_outputs = 512
    max_chunk_overlap = 0.1 #20
    chunk_size_limit = 600

    prompt_helper = PromptHelper(max_input_size, num_outputs, max_chunk_overlap, chunk_size_limit=chunk_size_limit)
    llm_predictor = LLMPredictor(llm=OpenAI(temperature=0.7, model_name="gpt-3.5-turbo", max_tokens=num_outputs))
    documents = SimpleDirectoryReader(directory_path).load_data()
    index = GPTVectorStoreIndex(documents, llm_predictor=llm_predictor, prompt_helper=prompt_helper)
    index.storage_context.persist(persist_dir="index.json")
    return index

index = construct_index("docs")


def chatbot(input_text):
    query_engine = index.as_query_engine()
    response = query_engine.query(input_text)
    return response.response

iface = gr.Interface(fn=chatbot,
                     inputs=gr.Textbox(lines=7, label="Enter your text"),
                     outputs="text",
                     title="One Energy AI Chatbot")

iface.launch(share=True)