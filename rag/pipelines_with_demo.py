from langchain.vectorstores import Pinecone
from langchain.prompts import PromptTemplate
from langchain.schema.runnable import RunnablePassthrough, RunnableLambda
from pinecone import Pinecone, ServerlessSpec
from langchain_huggingface.embeddings import HuggingFaceEndpointEmbeddings
from langchain_community.chat_models import ChatOllama
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_teddynote.messages import stream_response
from contextlib import redirect_stdout

# 주의사항: Ollama set up and pull EEVE model 필수! 서버에서 할게요

# Step 1: Set up Pinecone index and embedding model
pinecone_api_key = 'your api key'
hf_api_token = "your api key"
pinecone_namespace = '2024 1 SKKU  international student handbook.pdf'

# Initialize Pinecone client
pc = Pinecone(api_key=pinecone_api_key)
index = pc.Index('iskku-data')
llm = ChatOllama(model='hf.co/heegyu/EEVE-Korean-Instruct-10.8B-v1.0-GGUF:Q4_K_M', temperature = 0.1)

# Define the embeddings model
model_name = "intfloat/multilingual-e5-large-instruct"
hf_embeddings = HuggingFaceEndpointEmbeddings(
    model=model_name,
    task="feature-extraction",
    huggingfacehub_api_token=hf_api_token,
)

# Step 2: Retriever function to extract top_k matches from Pinecone
def retrieve(query, top_k=5):
    embedded_query = hf_embeddings.embed_query(query) # 쿼리로 변환해서 넘기는 것임!
    results = index.query(vector=embedded_query, top_k=top_k, namespace=pinecone_namespace, include_metadata=True)
    return results['matches']

# Step 3: Define the generator component using ChatOllama
def generate_response(query, context):
    global llm

    prompt = ChatPromptTemplate.from_template("""A chat between a curious user and an artificial intelligence assistant. The assistant gives helpful, very detailed, and polite answers to the user's questions.
Human: Use given context, answer the following question in detail as much as you can. This context is about SungKyunKwan University's information(성균관대학교). write based on user's language. Context: {context} \n Question: {query}
Assistant:\n""")

    chain = prompt | llm | StrOutputParser()
    
    answer = chain.stream({"query": query, "context": context})

    return answer


# Step 4: RAG pipeline combining retriever and generator
def rag_pipeline(query):
    # Retrieve relevant contexts from Pinecone
    matches = retrieve(query)
    retrieved_context = "\n".join([match['metadata']['text'] for match in matches])

    # Generate answer based on retrieved context
    response = generate_response(query, retrieved_context)
    return response, matches

def writter(query):
  response, matches = rag_pipeline(query)

  with open("response_output.txt", "w") as f:
    with redirect_stdout(f):
        response, matches = rag_pipeline(query)
        stream_response(response)

  with open("response_output.txt", "r") as f:
    saved_output = f.read()
    return saved_output

import gradio as gr

def rag_chatbot(message, history):
    response = writter(message)

    history.append((message, response))
    return history, history

with gr.Blocks() as demo:
    chatbot = gr.Chatbot()
    msg = gr.Textbox(label="Your Message")
    clear = gr.Button("Clear")

    msg.submit(rag_chatbot, [msg, chatbot], [chatbot, chatbot])
    clear.click(lambda: None, None, chatbot)

demo.launch(share=True)
